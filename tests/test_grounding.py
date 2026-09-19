"""Unit and integration tests for Grounded QA RAG Loop and Anti-Hallucination Guardrails.

Verifies:
- Retrieved chunk context formatting with section headers and chunk indices
- Grounded answering with verifiable section citations ([Section: <Name>])
- Strict anti-hallucination refusal: 'The provided text from this paper does not contain information regarding [topic].'
- Multi-turn conversational history persistence via SQLite checkpointer
"""

from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.config import settings
from src.agent.state import create_initial_state, PaperMetadata
from src.agent.nodes import node_qa_answer
from src.agent.persistence import get_sqlite_saver, get_thread_config
from src.agent.graph import build_agent_graph
from src.llm.prompts import (
    ANTI_HALLUCINATION_REFUSAL_PREFIX,
    GROUNDED_QA_SYSTEM_PROMPT,
)
from src.llm.gemini_client import answer_grounded_qa


SAMPLE_PAPER: PaperMetadata = {
    "arxiv_id": "1706.03762",
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "published": "2017-06-12",
    "updated": "2017-06-12",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer.",
    "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
    "primary_category": "cs.CL",
    "categories": ["cs.CL"],
    "entry_id": "http://arxiv.org/abs/1706.03762v1",
}

RETRIEVED_CHUNKS = [
    {
        "text": "The Transformer uses multi-head attention with 8 parallel attention heads and key dimension d_k=64.",
        "section_name": "Model Architecture",
        "chunk_index": 2,
        "score": 0.89,
    },
    {
        "text": "On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU.",
        "section_name": "Results",
        "chunk_index": 5,
        "score": 0.84,
    },
]


class TestGroundingPromptAndFormatting:
    """Test grounding prompt rules and chunk context formatting."""

    def test_grounding_system_prompt_contains_mandatory_rules(self):
        """Rubric verification: Strict anti-hallucination refusal and section citation rules."""
        assert "The provided text from this paper does not contain information regarding [topic]" in GROUNDED_QA_SYSTEM_PROMPT
        assert "Do NOT guess, extrapolate, or introduce external training facts" in GROUNDED_QA_SYSTEM_PROMPT
        assert "[Section: <Section Name>]" in GROUNDED_QA_SYSTEM_PROMPT

    def test_empty_question_returns_refusal(self):
        answer = answer_grounded_qa("", RETRIEVED_CHUNKS)
        assert ANTI_HALLUCINATION_REFUSAL_PREFIX in answer

    def test_empty_chunks_returns_exact_anti_hallucination_refusal(self):
        answer = answer_grounded_qa("What is the architecture?", [])
        assert answer == f"{ANTI_HALLUCINATION_REFUSAL_PREFIX} this topic."


class TestGroundedAnsweringAndAntiHallucination:
    """Test grounded answers with section citations and strict refusals."""

    @patch("src.llm.gemini_client.get_gemini_client")
    def test_grounded_answer_with_mocked_gemini(self, mock_get_client):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="According to [Section: Model Architecture], the model uses 8 parallel attention heads."
        )
        mock_get_client.return_value = mock_llm

        answer = answer_grounded_qa("How many heads are used?", RETRIEVED_CHUNKS)

        assert "[Section: Model Architecture]" in answer
        assert "8 parallel attention heads" in answer

    def test_grounded_answer_offline_mode_cites_section(self):
        """In offline mode, keyword-matching chunks produce grounded response with section citation."""
        with patch.object(settings, "gemini_api_key", ""):
            answer = answer_grounded_qa("What were the BLEU score results on translation?", RETRIEVED_CHUNKS)

            assert "[Section: Model Architecture]" in answer or "[Section: Results]" in answer
            assert "28.4" in answer

    def test_anti_hallucination_refusal_for_unrelated_query(self):
        """Rubric verification: Must refuse queries that cannot be answered from retrieved chunks."""
        unrelated_question = "What was the stock price of Apple Inc. in fiscal year 2021?"
        answer = answer_grounded_qa(unrelated_question, RETRIEVED_CHUNKS)

        # Must explicitly decline rather than invent facts
        assert ANTI_HALLUCINATION_REFUSAL_PREFIX in answer


class TestQaAnswerNodeAndStatePersistence:
    """Test node_qa_answer and multi-turn state persistence."""

    def test_node_qa_answer_skips_when_no_human_message(self):
        state = create_initial_state(raw_query="1706.03762")
        state["qa_messages"] = []

        update = node_qa_answer(state)
        assert update["status"] == "qa_skipped"

    @patch("src.agent.nodes.search_chunks")
    def test_node_qa_answer_appends_grounded_ai_message(self, mock_search):
        mock_search.return_value = RETRIEVED_CHUNKS

        state = create_initial_state(raw_query="1706.03762")
        state["selected_paper"] = SAMPLE_PAPER
        state["qa_messages"] = [
            HumanMessage(content="What BLEU score did the model achieve on WMT 2014?")
        ]

        update = node_qa_answer(state)

        assert update["status"] == "qa_answered"
        assert len(update["qa_messages"]) == 1
        reply = update["qa_messages"][0]
        assert isinstance(reply, AIMessage)
        assert "[Section:" in reply.content

    @patch("src.agent.nodes.search_chunks")
    def test_multi_turn_persisted_qa_across_consecutive_questions(self, mock_search):
        """Verify conversational persistence across multiple sequential QA turns via SQLite."""
        mock_search.return_value = RETRIEVED_CHUNKS

        checkpointer = get_sqlite_saver(":memory:")
        graph = build_agent_graph(checkpointer=checkpointer)
        session_id = "test_multi_turn_grounding_thread"
        config = get_thread_config(session_id)

        # Initialize thread state
        init_state = create_initial_state("1706.03762", session_id=session_id)
        init_state["selected_paper"] = SAMPLE_PAPER
        init_state["briefing"] = {
            "title": SAMPLE_PAPER["title"],
            "authors": SAMPLE_PAPER["authors"],
            "arxiv_id": SAMPLE_PAPER["arxiv_id"],
            "publish_date": SAMPLE_PAPER["published"],
            "link": SAMPLE_PAPER["pdf_url"],
            "why_it_matters": "...",
            "problem_statement": "...",
            "method_approach": ["..."],
            "key_results": ["..."],
            "limitations": ["Quadratic memory overhead"],
            "suggested_follow_ups": ["..."],
            "markdown_output": "...",
        }
        init_state["status"] = "summarized"

        # Turn 1: Question 1
        turn1_input = {
            **init_state,
            "qa_messages": [HumanMessage(content="How many attention heads are used?")],
        }
        turn1_state = graph.invoke(turn1_input, config=config)
        assert turn1_state["status"] == "qa_answered"
        assert len(turn1_state["qa_messages"]) >= 2
        assert isinstance(turn1_state["qa_messages"][-1], AIMessage)

        # Turn 2: Question 2 on same thread
        turn2_input = {
            "qa_messages": [HumanMessage(content="What translation results were achieved?")]
        }
        turn2_state = graph.invoke(turn2_input, config=config)
        assert turn2_state["status"] == "qa_answered"
        # Total messages must now be >= 4 (Q1, A1, Q2, A2)
        assert len(turn2_state["qa_messages"]) >= 4
        assert isinstance(turn2_state["qa_messages"][-1], AIMessage)
