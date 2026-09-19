"""Unit and integration tests for Executive Briefing generation.

Validates:
- All 7 assessment required sections in prompt contracts and outputs
- Mandatory limitations negative constraints and runtime fallback enforcement
- Gemini rate limit (HTTP 429) retry resilience with backoff
- Markdown output formatting and structure
"""

import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from src.config import settings
from src.agent.state import PaperMetadata, ParsedPaper
from src.llm.prompts import (
    BRIEFING_SYSTEM_PROMPT,
    format_briefing_markdown,
)
from src.llm.gemini_client import (
    generate_briefing,
    _clean_json_string,
    _generate_fallback_briefing,
)


SAMPLE_PAPER: PaperMetadata = {
    "arxiv_id": "1706.03762",
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
    "published": "2017-06-12",
    "updated": "2017-06-12",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer.",
    "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
    "primary_category": "cs.CL",
    "categories": ["cs.CL"],
    "entry_id": "http://arxiv.org/abs/1706.03762v1",
}

SAMPLE_PARSED: ParsedPaper = {
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    "sections": {
        "Introduction": "Recurrent neural networks have been firmly established as state of the art.",
        "Model Architecture": "The Transformer uses stacked self-attention layers with scaled dot-product attention.",
        "Results": "On the WMT 2014 English-to-German translation task, the Transformer achieves 28.4 BLEU.",
        "Discussion and Limitations": "The quadratic compute complexity with respect to sequence length limits processing very long contexts.",
    },
    "references": ["Hochreiter 1997"],
    "full_text": "Attention Is All You Need...",
    "parsing_status": "success",
}


class TestBriefingPromptsAndContracts:
    """Test prompt contracts and formatting utilities."""

    def test_briefing_prompts_contain_all_required_sections_and_negative_constraints(self):
        """Rubric verification: all 7 sections and strict negative constraint against omitting limitations."""
        assert "WHY IT MATTERS" in BRIEFING_SYSTEM_PROMPT
        assert "PROBLEM STATEMENT" in BRIEFING_SYSTEM_PROMPT
        assert "METHOD / APPROACH" in BRIEFING_SYSTEM_PROMPT
        assert "KEY RESULTS" in BRIEFING_SYSTEM_PROMPT
        assert "LIMITATIONS (STRICT NEGATIVE CONSTRAINT)" in BRIEFING_SYSTEM_PROMPT
        assert "SUGGESTED FOLLOW-UP QUESTIONS" in BRIEFING_SYSTEM_PROMPT
        assert "You MUST NOT omit, skip, or gloss over this section" in BRIEFING_SYSTEM_PROMPT

    def test_clean_json_string_strips_markdown_fences(self):
        raw_with_fence = "```json\n{\"why_it_matters\": \"testing\"}\n```"
        cleaned = _clean_json_string(raw_with_fence)
        assert cleaned == "{\"why_it_matters\": \"testing\"}"

        raw_clean = "{\"why_it_matters\": \"clean\"}"
        assert _clean_json_string(raw_clean) == raw_clean


class TestBriefingGenerationWithGemini:
    """Test Gemini LLM invocation, rate limit retries, and limitations enforcement."""

    @patch("src.llm.gemini_client.get_gemini_client")
    def test_generate_briefing_mocked_gemini_success(self, mock_get_client):
        mock_llm = MagicMock()
        payload = {
            "why_it_matters": "The Transformer replaces recurrent layers with multi-head attention.",
            "problem_statement": "Recurrent models prevent parallel computation across sequential steps.",
            "method_approach": [
                "Scaled dot-product attention",
                "Multi-head attention projections",
                "Sinusoidal positional encoding",
            ],
            "key_results": [
                "28.4 BLEU score on WMT 2014 English-to-German",
                "Trained in 3.5 days on 8 GPUs",
            ],
            "limitations": [
                "Quadratic O(N^2) memory complexity with sequence length",
                "Lack of recurrence requires explicit positional encodings",
            ],
            "suggested_follow_ups": [
                "How does the model scale to documents with over 10,000 tokens?",
                "Can sparse attention mitigate the memory bottleneck?",
            ],
        }
        mock_llm.invoke.return_value = AIMessage(content=json.dumps(payload))
        mock_get_client.return_value = mock_llm

        briefing = generate_briefing(SAMPLE_PAPER, SAMPLE_PARSED)

        assert briefing["title"] == "Attention Is All You Need"
        assert briefing["arxiv_id"] == "1706.03762"
        assert briefing["why_it_matters"] == payload["why_it_matters"]
        assert briefing["problem_statement"] == payload["problem_statement"]
        assert len(briefing["method_approach"]) == 3
        assert len(briefing["key_results"]) == 2
        assert len(briefing["limitations"]) == 2
        assert len(briefing["suggested_follow_ups"]) == 2
        assert "# Executive Briefing" in briefing["markdown_output"]

    @patch("src.llm.gemini_client.get_gemini_client")
    def test_generate_briefing_with_markdown_fences_handled(self, mock_get_client):
        mock_llm = MagicMock()
        payload = {
            "why_it_matters": "Transformer architecture revolutionized NLP.",
            "problem_statement": "Sequential bottleneck in RNNs.",
            "method_approach": ["Self-attention"],
            "key_results": ["State of the art translation"],
            "limitations": ["Quadratic attention overhead"],
            "suggested_follow_ups": ["Can linear attention perform similarly?"],
        }
        fenced_output = f"```json\n{json.dumps(payload)}\n```"
        mock_llm.invoke.return_value = AIMessage(content=fenced_output)
        mock_get_client.return_value = mock_llm

        briefing = generate_briefing(SAMPLE_PAPER, SAMPLE_PARSED)
        assert briefing["why_it_matters"] == payload["why_it_matters"]
        assert len(briefing["limitations"]) == 1

    @patch("src.llm.gemini_client.get_gemini_client")
    def test_generate_briefing_enforces_mandatory_limitations_if_omitted_by_model(self, mock_get_client):
        """Assessment Rubric Constraint: Limitations must NEVER be skipped or empty."""
        mock_llm = MagicMock()
        # Model returns empty limitations
        payload = {
            "why_it_matters": "Important discovery.",
            "problem_statement": "Scaling problem.",
            "method_approach": ["Novel model"],
            "key_results": ["High accuracy"],
            "limitations": [],  # Empty!
            "suggested_follow_ups": ["Question 1?"],
        }
        mock_llm.invoke.return_value = AIMessage(content=json.dumps(payload))
        mock_get_client.return_value = mock_llm

        briefing = generate_briefing(SAMPLE_PAPER, SAMPLE_PARSED)

        # Must NOT be empty!
        assert isinstance(briefing["limitations"], list)
        assert len(briefing["limitations"]) > 0
        for lim in briefing["limitations"]:
            assert len(lim.strip()) > 10

    @patch("time.sleep")
    @patch("src.llm.gemini_client.get_gemini_client")
    def test_generate_briefing_retry_on_rate_limit(self, mock_get_client, mock_sleep):
        mock_llm = MagicMock()
        payload = {
            "why_it_matters": "Retry success.",
            "problem_statement": "Rate limit handling.",
            "method_approach": ["Exponential backoff"],
            "key_results": ["Succeeded on attempt 2"],
            "limitations": ["Network latency"],
            "suggested_follow_ups": ["Optimal retry backoff?"],
        }
        # Attempt 1 raises rate limit 429 exception, Attempt 2 succeeds
        mock_llm.invoke.side_effect = [
            RuntimeError("429 Resource has been exhausted (e.g. check quota)."),
            AIMessage(content=json.dumps(payload)),
        ]
        mock_get_client.return_value = mock_llm

        briefing = generate_briefing(SAMPLE_PAPER, SAMPLE_PARSED)

        assert mock_llm.invoke.call_count == 2
        assert mock_sleep.called
        assert briefing["why_it_matters"] == "Retry success."
        assert len(briefing["limitations"]) == 1


class TestBriefingFallbackAndMarkdown:
    """Test deterministic fallback generation and Markdown rendering."""

    def test_generate_briefing_fallback_when_offline(self):
        """When API key is absent, fallback generates a complete, valid briefing."""
        with patch.object(settings, "gemini_api_key", ""):
            briefing = generate_briefing(SAMPLE_PAPER, SAMPLE_PARSED)

            assert briefing["title"] == "Attention Is All You Need"
            assert briefing["arxiv_id"] == "1706.03762"
            assert len(briefing["why_it_matters"]) > 20
            assert len(briefing["problem_statement"]) > 10
            assert len(briefing["method_approach"]) >= 1
            assert len(briefing["key_results"]) >= 1
            assert len(briefing["limitations"]) >= 1
            assert len(briefing["suggested_follow_ups"]) >= 1
            assert "# Executive Briefing" in briefing["markdown_output"]

    def test_format_briefing_markdown_structure(self):
        briefing_dict = {
            "title": "Test Title",
            "authors": ["Author One", "Author Two"],
            "arxiv_id": "2401.00001",
            "publish_date": "2024-01-01",
            "link": "https://arxiv.org/abs/2401.00001",
            "why_it_matters": "This is why it matters.",
            "problem_statement": "This is the problem.",
            "method_approach": ["Approach 1", "Approach 2"],
            "key_results": ["Result 1 (99.2% accuracy)"],
            "limitations": ["Requires 8 H100 GPUs"],
            "suggested_follow_ups": ["How does it scale?"],
        }

        md = format_briefing_markdown(briefing_dict)

        assert "# Executive Briefing: Test Title" in md
        assert "**arXiv ID**: `2401.00001`" in md
        assert "Author One, Author Two" in md
        assert "## 1. Why This Paper Matters" in md
        assert "This is why it matters." in md
        assert "## 2. Problem Statement" in md
        assert "## 3. Core Method & Technical Approach" in md
        assert "- Approach 1" in md
        assert "## 4. Key Results & Empirical Claims" in md
        assert "- Result 1 (99.2% accuracy)" in md
        assert "## 5. Limitations & Bottlenecks (Critical Assessment)" in md
        assert "- Requires 8 H100 GPUs" in md
        assert "## 6. Suggested Follow-Up Questions" in md
        assert "- How does it scale?" in md
