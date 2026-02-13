"""Input/output validation and guardrails for the RAG pipeline."""

import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, List, Optional

from app.models import QueryResponse, SourceReference

logger = logging.getLogger(__name__)

# --- Constants ---

QUERY_MIN_LENGTH = 3
QUERY_MAX_LENGTH = 500

# Suspicious patterns that may indicate injection attempts
INJECTION_PATTERNS = [
    r"(?i)(\b(or|and)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?)",
    r"(?i)\b(drop|delete|truncate|insert|update)\s+(table|from|into)\b",
    r"(?i)\b(exec|xec|execute)\s*\(",
    r"(?i)\b(union|select)\s+(.*?\s+)?from\b",
    r"(?i);\s*(drop|delete|--)\b",
    r"(?i)\{\{.*?\}\}",  # Template injection
    r"(?i)\$\{.*?\}",   # Variable expansion
    r"(?i)\b(script|javascript|on\w+=)\b",
    r"<[^>]*script[^>]*>",
    r"<\?php|\<\?=",
    r"(?i)\b(sys\.|os\.|__import__|eval\s*\()\b",
]

# Basic profanity/inappropriate content (conservative list - common workplace-inappropriate terms)
PROFANITY_PATTERNS = [
    r"(?i)\b(f\*\*k|fuck|s\*\*t|shit|a\*\*hole|asshole)\b",
    r"(?i)\b(dumb\s*ass|piece\s+of\s+shit)\b",
    r"(?i)\b(kill\s+yourself| kys)\b",
    r"(?i)\b(hate\s+you|f\*\*k\s+you)\b",
]


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    """Compile regex patterns for efficiency."""
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error:
            logger.warning("Invalid regex pattern in guardrails: %s", p)
    return compiled


_INJECTION_COMPILED = _compile_patterns(INJECTION_PATTERNS)
_PROFANITY_COMPILED = _compile_patterns(PROFANITY_PATTERNS)

# Citation format: [Source N] or [Source N, M]
CITATION_PATTERN = re.compile(r"\[Source\s*(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE)


# --- Input Validation ---


def validate_query(query: str) -> dict[str, Any]:
    """Validate user query before retrieval.

    Checks:
    - Length (min 3 chars, max 500 chars)
    - Potential injection attempts (suspicious patterns)
    - Profanity/inappropriate content (basic regex)

    Args:
        query: Raw user query string.

    Returns:
        Dict with keys:
        - valid: bool
        - reason: Optional[str] - human-readable reason when invalid
        - sanitized_query: str - stripped/sanitized query for safe logging
    """
    if not isinstance(query, str):
        logger.warning("Query validation failed: non-string input")
        return {
            "valid": False,
            "reason": "Invalid input",
            "sanitized_query": "",
        }

    sanitized = query.strip()
    # For logging: truncate to avoid exposing full queries
    log_safe = sanitized[:80] + "..." if len(sanitized) > 80 else sanitized

    # Length checks
    if len(sanitized) < QUERY_MIN_LENGTH:
        logger.info("Query validation failed: too short (len=%d)", len(sanitized))
        return {
            "valid": False,
            "reason": "Query too short",
            "sanitized_query": sanitized,
        }

    if len(sanitized) > QUERY_MAX_LENGTH:
        logger.info("Query validation failed: too long (len=%d)", len(sanitized))
        return {
            "valid": False,
            "reason": "Query too long",
            "sanitized_query": sanitized[:QUERY_MAX_LENGTH],
        }

    # Content safety
    safety = check_content_safety(sanitized)
    if not safety["safe"]:
        logger.info("Query validation failed: content safety issues=%s", safety["issues"])
        return {
            "valid": False,
            "reason": "Inappropriate content detected",
            "sanitized_query": "",
        }

    return {
        "valid": True,
        "reason": None,
        "sanitized_query": sanitized,
    }


# --- Output Validation ---


def validate_response(response: str, sources: List[dict]) -> dict[str, Any]:
    """Validate LLM response before returning to user.

    Checks:
    - Response has content (not empty)
    - Citation format is valid ([Source N])
    - No harmful/inappropriate content

    Args:
        response: Raw LLM response text.
        sources: List of source dicts with doc_id, doc_name, etc.

    Returns:
        Dict with keys:
        - valid: bool
        - reason: Optional[str]
        - safe_response: str - sanitized response or empty if invalid
    """
    if not isinstance(response, str):
        logger.warning("Response validation failed: non-string output")
        return {
            "valid": False,
            "reason": "Invalid response",
            "safe_response": "",
        }

    stripped = response.strip()
    if not stripped:
        logger.info("Response validation failed: empty response")
        return {
            "valid": False,
            "reason": "Invalid response",
            "safe_response": "",
        }

    # Verify citation format: [Source N] where N is 1..len(sources)
    for m in CITATION_PATTERN.finditer(stripped):
        indices = m.group(1)
        for part in indices.split(","):
            try:
                idx = int(part.strip())
                if idx < 1 or idx > len(sources):
                    logger.info(
                        "Response validation: out-of-range citation [Source %d] "
                        "(max=%d)",
                        idx,
                        len(sources),
                    )
                    # Allow it; we sanitize in parse - just note it
            except ValueError:
                logger.info("Response validation: malformed citation")
                # Still allow - parse will handle

    # Content safety
    safety = check_content_safety(stripped)
    if not safety["safe"]:
        logger.info("Response validation failed: unsafe content=%s", safety["issues"])
        return {
            "valid": False,
            "reason": "Invalid response",
            "safe_response": "",
        }

    return {
        "valid": True,
        "reason": None,
        "safe_response": stripped,
    }


# --- Rate Limiting ---


class RateLimiter:
    """Per-user rate limiter with sliding window."""

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed per window.
            window_seconds: Time window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # user_id -> list of timestamps (request times)
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check_rate_limit(self, user_id: str = "default") -> bool:
        """Check if user is within rate limit.

        Resets old requests outside the window. Returns True if allowed,
        False if rate limit exceeded.

        Args:
            user_id: User identifier (e.g. IP, session id). Default: "default".

        Returns:
            True if request is allowed, False if rate limit exceeded.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds

        # Remove stale timestamps
        timestamps = self._requests[user_id]
        timestamps[:] = [t for t in timestamps if t > window_start]

        if len(timestamps) >= self.max_requests:
            logger.info(
                "Rate limit exceeded for user=%s (count=%d in window)",
                user_id[:20] if len(user_id) > 20 else user_id,
                len(timestamps),
            )
            return False

        timestamps.append(now)
        return True

    def reset(self, user_id: Optional[str] = None) -> None:
        """Reset rate limit for a user or all users.

        Args:
            user_id: If provided, reset only this user. Else reset all.
        """
        if user_id is not None:
            self._requests.pop(user_id, None)
        else:
            self._requests.clear()


# --- Content Safety ---


def check_content_safety(text: str) -> dict[str, Any]:
    """Check text for potentially harmful or inappropriate content.

    Uses basic regex patterns for profanity and injection attempts.
    Conservative: better safe than sorry.

    Args:
        text: Text to check.

    Returns:
        Dict with keys:
        - safe: bool
        - issues: List[str] - descriptions of found issues
    """
    if not text:
        return {"safe": True, "issues": []}

    issues: List[str] = []

    for pat in _INJECTION_COMPILED:
        if pat.search(text):
            issues.append("suspicious_pattern")
            break

    for pat in _PROFANITY_COMPILED:
        if pat.search(text):
            issues.append("inappropriate_content")
            break

    return {
        "safe": len(issues) == 0,
        "issues": issues,
    }


# --- Fallback Responses ---


def get_fallback_response(reason: str) -> QueryResponse:
    """Return a professional fallback QueryResponse for validation failures.

    Args:
        reason: One of:
            - "Query too long"
            - "Query too short"
            - "Inappropriate content detected"
            - "Rate limit exceeded"
            - "No relevant information found"
            - "Invalid input" (generic)

    Returns:
        QueryResponse with user-friendly message and empty sources.
    """
    messages = {
        "Query too long": (
            "Your question is too long. Please limit it to 500 characters "
            "and try again."
        ),
        "Query too short": (
            "Please enter a question with at least 3 characters."
        ),
        "Inappropriate content detected": (
            "Your question contains content that we cannot process. "
            "Please rephrase in a professional manner."
        ),
        "Rate limit exceeded": (
            "You've made too many requests. Please wait a moment and try again."
        ),
        "No relevant information found": (
            "I could not find relevant documents to answer your question. "
            "Please ensure your resume and/or job description are uploaded."
        ),
        "Invalid input": (
            "Unable to process your request. Please try rephrasing your question."
        ),
        "Invalid response": (
            "We encountered an issue generating a response. Please try again."
        ),
    }
    answer = messages.get(reason, messages["Invalid input"])
    return QueryResponse(
        answer=answer,
        sources=[],
        confidence=0.0,
        generated_at=datetime.now(timezone.utc),
    )
