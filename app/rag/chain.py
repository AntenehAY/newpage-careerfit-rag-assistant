"""RAG chain orchestrating retrieval and LLM generation for career advice."""

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

from langchain_anthropic import ChatAnthropic

from app.config import settings
from app.models import QueryResponse, SourceReference

from .guardrails import (
    RateLimiter,
    get_fallback_response,
    validate_query,
    validate_response,
)
from .prompts import get_prompt_for_type

logger = logging.getLogger(__name__)


# Heuristics for query type detection
SKILL_GAP_KEYWORDS = (
    "skill", "skills", "missing", "gap", "gaps", "lack", "need",
    "qualification", "qualified", "requirement", "requirements",
)
EXPERIENCE_ALIGNMENT_KEYWORDS = (
    "experience", "align", "alignment", "match", "fit", "relevant",
    "years", "background", "qualify", "suitable",
)
INTERVIEW_PREP_KEYWORDS = (
    "interview", "prep", "prepare", "expect", "question", "ask",
    "talking point", "strength", "weakness", "discuss",
)


class RAGChain:
    """RAG chain: retrieve context, select prompt, call LLM, parse response."""

    def __init__(
        self,
        retriever: Any,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """Initialize the RAG chain.

        Args:
            retriever: Retriever instance (app.retrieval.retriever.Retriever).
            llm_model: Override LLM model name. Defaults to settings.LLM_MODEL_NAME.
            api_key: Override Anthropic API key. Defaults to settings.anthropic_api_key.
            temperature: Override temperature. Defaults to settings.llm_temperature (0.3).
            rate_limiter: Optional RateLimiter for request throttling.
        """
        self.retriever = retriever
        self.rate_limiter = rate_limiter
        self.llm_model = llm_model or settings.llm_model_name
        self.api_key = api_key or settings.anthropic_api_key
        self.temperature = temperature if temperature is not None else settings.llm_temperature

        self.llm = ChatAnthropic(
            model=self.llm_model,
            anthropic_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=settings.llm_max_tokens,
        )
        logger.info(
            "RAGChain initialized: model=%s, temperature=%s",
            self.llm_model,
            self.temperature,
        )

    def answer_query(
        self,
        query: str,
        filter_doc_type: Optional[str] = None,
        filter_doc_id: Optional[str] = None,
        user_id: str = "default",
    ) -> QueryResponse:
        """Answer a query using retrieval-augmented generation.

        Args:
            query: User question.
            filter_doc_type: Optional filter (resume, job_description).
            filter_doc_id: Optional filter by document ID.
            user_id: User identifier for rate limiting. Default: "default".

        Returns:
            QueryResponse with answer, sources, and metadata.
        """
        t_total = time.perf_counter()

        # Step 0: Input validation
        validation = validate_query(query)
        if not validation["valid"]:
            return get_fallback_response(validation["reason"])

        # Step 0b: Rate limiting
        if self.rate_limiter is not None:
            if not self.rate_limiter.check_rate_limit(user_id):
                return get_fallback_response("Rate limit exceeded")

        query_to_use = validation["sanitized_query"]

        # Step 1: Retrieve context
        t_ret = time.perf_counter()
        context_result = self.retriever.get_context_for_llm(
            query=query_to_use,
            filter_doc_type=filter_doc_type,
            filter_doc_id=filter_doc_id,
        )
        elapsed_ret = time.perf_counter() - t_ret
        context = context_result["context"]
        sources_raw = context_result["sources"]

        if not context or not context.strip():
            logger.warning("No context retrieved for query (len=%d)", len(query_to_use))
            return get_fallback_response("No relevant information found")

        # Step 2: Detect query type and select prompt
        query_type = self._detect_query_type(query_to_use)
        prompt_template = self._select_prompt_template(query_type)

        # Step 3: Format prompt with context
        messages = prompt_template.format_messages(context=context, query=query_to_use)

        # Step 4: Call LLM
        t_llm = time.perf_counter()
        try:
            response = self.llm.invoke(messages)
            llm_output = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            err_msg = str(e).lower()
            if "rate" in err_msg or "429" in err_msg or "overloaded" in err_msg:
                logger.warning("LLM rate limit or overload: %s", e)
                raise RuntimeError(
                    "The AI service is currently busy. Please try again in a few moments."
                ) from e
            logger.exception("LLM invocation failed: %s", e)
            raise
        elapsed_llm = time.perf_counter() - t_llm

        # Optional: log token usage if available
        if hasattr(response, "response_metadata") and response.response_metadata:
            usage = response.response_metadata.get("usage", {})
            if usage:
                logger.info(
                    "LLM usage: input_tokens=%s, output_tokens=%s",
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                )

        logger.info(
            "RAG answer_query: retrieval=%.3fs, llm=%.3fs, total=%.3fs",
            elapsed_ret,
            elapsed_llm,
            time.perf_counter() - t_total,
        )

        # Step 5: Output validation
        output_validation = validate_response(llm_output, sources_raw)
        if not output_validation["valid"]:
            logger.warning("Response validation failed: %s", output_validation["reason"])
            return get_fallback_response(output_validation["reason"])

        # Step 6: Parse response and map citations to sources
        parsed = self._parse_response(output_validation["safe_response"], sources_raw)

        # Step 7: Return QueryResponse
        return QueryResponse(
            answer=parsed["answer"],
            sources=parsed["sources"],
            confidence=parsed.get("confidence"),
            generated_at=datetime.now(timezone.utc),
        )

    def answer_query_stream(
        self,
        query: str,
        filter_doc_type: Optional[str] = None,
        filter_doc_id: Optional[str] = None,
        user_id: str = "default",
    ):
        """Stream the LLM response for a query (generator of text chunks).

        Yields:
            str: Text chunks as they are generated by the LLM.
        """
        # Input validation
        validation = validate_query(query)
        if not validation["valid"]:
            fallback = get_fallback_response(validation["reason"])
            yield fallback.answer
            return

        # Rate limiting
        if self.rate_limiter is not None:
            if not self.rate_limiter.check_rate_limit(user_id):
                fallback = get_fallback_response("Rate limit exceeded")
                yield fallback.answer
                return

        query_to_use = validation["sanitized_query"]
        context_result = self.retriever.get_context_for_llm(
            query=query_to_use,
            filter_doc_type=filter_doc_type,
            filter_doc_id=filter_doc_id,
        )
        context = context_result["context"]
        if not context or not context.strip():
            fallback = get_fallback_response("No relevant information found")
            yield fallback.answer
            return

        query_type = self._detect_query_type(query_to_use)
        prompt_template = self._select_prompt_template(query_type)
        messages = prompt_template.format_messages(context=context, query=query_to_use)

        for chunk in self.llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    def _detect_query_type(self, query: str) -> str:
        """Classify query into: skill_gap, experience_alignment, interview_prep, general."""
        q = query.lower().strip()
        if any(k in q for k in SKILL_GAP_KEYWORDS):
            return "skill_gap"
        if any(k in q for k in EXPERIENCE_ALIGNMENT_KEYWORDS):
            return "experience_alignment"
        if any(k in q for k in INTERVIEW_PREP_KEYWORDS):
            return "interview_prep"
        return "general"

    def _select_prompt_template(self, query_type: str) -> Any:
        """Return the ChatPromptTemplate for the given query type."""
        return get_prompt_for_type(query_type)

    def _parse_response(self, llm_output: str, sources_raw: List[dict]) -> dict:
        """Extract answer text, map source citations to SourceReference objects."""
        answer = llm_output.strip()

        # Extract cited source indices from [Source N] or [Source N, M] patterns
        cite_pattern = re.compile(r"\[Source\s*(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE)
        cited_indices: set[int] = set()
        for m in cite_pattern.finditer(answer):
            for part in m.group(1).split(","):
                try:
                    idx = int(part.strip())
                    if 1 <= idx <= len(sources_raw):
                        cited_indices.add(idx - 1)
                except ValueError:
                    pass

        # Build SourceReference list from sources_raw
        # Include only cited sources if we found citations; otherwise include all
        if cited_indices:
            refs = [
                SourceReference(
                    doc_id=s.get("doc_id", ""),
                    doc_name=s.get("doc_name", s.get("doc_id", "unknown")),
                    doc_type=s.get("doc_type", "unknown"),
                    chunk_text=s.get("chunk_text", ""),
                    relevance_score=float(s.get("relevance_score", 0.0)),
                )
                for i, s in enumerate(sources_raw)
                if i in cited_indices
            ]
        else:
            refs = [
                SourceReference(
                    doc_id=s.get("doc_id", ""),
                    doc_name=s.get("doc_name", s.get("doc_id", "unknown")),
                    doc_type=s.get("doc_type", "unknown"),
                    chunk_text=s.get("chunk_text", ""),
                    relevance_score=float(s.get("relevance_score", 0.0)),
                )
                for s in sources_raw
            ]

        # Optional confidence: based on presence of citations and source count
        confidence = None
        if sources_raw:
            cited_ratio = len(cited_indices) / len(sources_raw) if sources_raw else 0
            if cited_indices:
                confidence = min(1.0, 0.5 + 0.5 * cited_ratio)
            else:
                confidence = 0.7  # All sources used as context

        return {
            "answer": answer,
            "sources": refs,
            "confidence": confidence,
        }
