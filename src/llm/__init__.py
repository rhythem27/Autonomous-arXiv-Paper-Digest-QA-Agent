"""LLM integration module for Gemini 2.5 Flash, structured briefings, and grounded QA."""

from src.llm.prompts import (
    ANTI_HALLUCINATION_REFUSAL_PREFIX,
    BRIEFING_SYSTEM_PROMPT,
    BRIEFING_USER_PROMPT_TEMPLATE,
    GROUNDED_QA_SYSTEM_PROMPT,
    GROUNDED_QA_USER_PROMPT_TEMPLATE,
    format_briefing_markdown,
)
from src.llm.gemini_client import (
    answer_grounded_qa,
    generate_briefing,
    get_gemini_client,
)

__all__ = [
    "ANTI_HALLUCINATION_REFUSAL_PREFIX",
    "BRIEFING_SYSTEM_PROMPT",
    "BRIEFING_USER_PROMPT_TEMPLATE",
    "GROUNDED_QA_SYSTEM_PROMPT",
    "GROUNDED_QA_USER_PROMPT_TEMPLATE",
    "format_briefing_markdown",
    "answer_grounded_qa",
    "generate_briefing",
    "get_gemini_client",
]
