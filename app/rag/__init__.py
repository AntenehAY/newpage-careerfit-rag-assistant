"""RAG pipeline: prompts, chain orchestration, and guardrails."""

from app.rag.chain import RAGChain
from app.rag.prompts import (
    EXPERIENCE_ALIGNMENT_PROMPT,
    GENERAL_CAREER_PROMPT,
    INTERVIEW_PREP_PROMPT,
    SKILL_GAP_PROMPT,
    get_prompt_for_type,
)

__all__ = [
    "RAGChain",
    "SKILL_GAP_PROMPT",
    "EXPERIENCE_ALIGNMENT_PROMPT",
    "INTERVIEW_PREP_PROMPT",
    "GENERAL_CAREER_PROMPT",
    "get_prompt_for_type",
]
