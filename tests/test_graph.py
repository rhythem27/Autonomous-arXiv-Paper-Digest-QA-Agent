"""Unit and integration tests for LangGraph state graph routing, edges, and multi-turn persistence.

Verifies end-to-end execution across all conditional branches:
- Direct arXiv ID lookup vs topic search
- Zero-results recovery branch
- Corrupt/scanned PDF metadata fallback branch
- Multi-turn conversational QA on persisted SQLite sessions
"""

import pytest
from unittest.mock import MagicMock, patch
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.state import create_initial_state, PaperMetadata, ParsedPaper
from src.agent.graph import (
    build_agent_graph,
    route_entrypoint,
    route_post_retrieval,
    route_post_parse,
)
from src.agent.persistence import get_sqlite_saver, get_thread_config


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

SAMPLE_PARSED: ParsedPaper = {
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models...",
    "sections": {
        "Introduction": "Recurrent neural networks have been firmly established.",
        "Model Architecture": "The Transformer uses stacked self-attention layers.",
        "Results": "Achieves 28.4 BLEU on English-to-German.",
        "Limitations": "Quadratic compute overhead with respect to sequence length.",
    },
    "references": [],
    "full_text": "Attention Is All You Need...",
    "parsing_status": "success",
}


class TestRoutingFunctions:
    """Test pure conditional routing functions."""

    def test_route_entrypoint_initial_ingestion(self):
        state = create_initial_state(raw_query="1706.03762")
        assert route_entrypoint(state) == "query_understanding"

    def test_route_entrypoint_follow_up_qa(self):
        state = create_initial_state(raw_query="1706.03762")
        state["selected_paper"] = SAMPLE_PAPER
        state["briefing"] = {
            "title": "Attention Is All You Need",
            "authors": ["A. Vaswani"],
            "arxiv_id": "1706.03762",
            "publish_date": "2017-06-12",
            "link": "https://arxiv.org/pdf/1706.03762.pdf",
            "why_it_matters": "...",
            "problem_statement": "...",
            "method_approach": ["..."],
            "key_results": ["..."],
            "limitations": ["..."],
            "suggested_follow_ups": ["..."],
            "markdown_output": "...",
        }
        state["qa_messages"] = [
            HumanMessage(content="What is the computational complexity?")
        ]

        assert route_entrypoint(state) == "qa_answer"

    def test_route_post_retrieval_branches(self):
        # 1. Zero results
        state_zero = create_initial_state(raw_query="missing")
        state_zero["candidate_papers"] = []
        assert route_post_retrieval(state_zero) == "handle_zero_results"

        # 2. Direct ID
        state_id = create_initial_state(raw_query="1706.03762")
        state_id["candidate_papers"] = [SAMPLE_PAPER]
        state_id["query_type"] = "arxiv_id"
        assert route_post_retrieval(state_id) == "fetch_parse"

        # 3. Topic search
        state_topic = create_initial_state(raw_query="transformers")
        state_topic["candidate_papers"] = [SAMPLE_PAPER]
        state_topic["query_type"] = "topic_search"
        assert route_post_retrieval(state_topic) == "selection_ranking"

    def test_route_post_parse_branches(self):
        # 1. Success
        state_ok = create_initial_state(raw_query="1706.03762")
        state_ok["parsed_paper"] = SAMPLE_PARSED
        assert route_post_parse(state_ok) == "chunk_embed"

        # 2. Scanned / metadata only
        state_meta = create_initial_state(raw_query="1706.03762")
        state_meta["parsed_paper"] = {
            "title": "T",
            "abstract": "A",
            "sections": {},
            "references": [],
            "full_text": "",
            "parsing_status": "metadata_only",
        }
        assert route_post_parse(state_meta) == "metadata_fallback"


class TestGraphCompilation:
    """Test StateGraph compilation and structural integrity."""

    def test_graph_compiles_with_memory_saver(self):
        checkpointer = MemorySaver()
        graph = build_agent_graph(checkpointer=checkpointer)

        assert graph is not None
        # Verify registered nodes
        expected_nodes = {
            "query_understanding",
            "arxiv_retrieval",
            "selection_ranking",
            "fetch_parse",
            "metadata_fallback",
            "chunk_embed",
            "summarize",
            "qa_answer",
            "handle_zero_results",
        }
        nodes = set(graph.get_graph().nodes.keys())
        for node in expected_nodes:
            assert node in nodes

    def test_graph_compiles_with_sqlite_saver(self):
        checkpointer = get_sqlite_saver(":memory:")
        graph = build_agent_graph(checkpointer=checkpointer)
        assert graph is not None


class TestGraphEndToEndExecution:
    """Test full execution of compiled state graph through different execution paths."""

    @patch("src.agent.nodes.index_chunks")
    @patch("src.agent.nodes.chunk_paper")
    @patch("src.agent.nodes.parse_pdf")
    @patch("src.agent.nodes.download_pdf")
    @patch("src.agent.nodes.get_arxiv_client")
    def test_graph_direct_id_pipeline(
        self,
        mock_get_client,
        mock_download,
        mock_parse,
        mock_chunk,
        mock_index,
    ):
        mock_client = MagicMock()
        mock_client.fetch_by_id.return_value = SAMPLE_PAPER
        mock_get_client.return_value = mock_client
        mock_download.return_value = "/tmp/1706.03762.pdf"
        mock_parse.return_value = SAMPLE_PARSED
        mock_chunk.return_value = [{"text": "Chunk", "section_name": "Intro", "chunk_index": 0, "metadata": {}}]
        mock_index.return_value = "arxiv_1706_03762"

        checkpointer = MemorySaver()
        graph = build_agent_graph(checkpointer=checkpointer)
        thread_config = get_thread_config("test_thread_direct_id")

        initial_state = create_initial_state("1706.03762v2", session_id="test_thread_direct_id")
        final_state = graph.invoke(initial_state, config=thread_config)

        assert final_state["status"] == "summarized"
        assert final_state["query_type"] == "arxiv_id"
        assert final_state["extracted_id"] == "1706.03762"
        assert final_state["selected_paper"]["title"] == "Attention Is All You Need"
        assert final_state["briefing"] is not None
        assert len(final_state["briefing"]["limitations"]) >= 1

    @patch("src.agent.nodes.index_chunks")
    @patch("src.agent.nodes.chunk_paper")
    @patch("src.agent.nodes.parse_pdf")
    @patch("src.agent.nodes.download_pdf")
    @patch("src.agent.nodes.get_arxiv_client")
    def test_graph_topic_search_pipeline(
        self,
        mock_get_client,
        mock_download,
        mock_parse,
        mock_chunk,
        mock_index,
    ):
        mock_client = MagicMock()
        mock_client.search_papers.return_value = [SAMPLE_PAPER]
        mock_get_client.return_value = mock_client
        mock_download.return_value = "/tmp/1706.03762.pdf"
        mock_parse.return_value = SAMPLE_PARSED
        mock_chunk.return_value = [{"text": "Chunk", "section_name": "Intro", "chunk_index": 0, "metadata": {}}]
        mock_index.return_value = "arxiv_1706_03762"

        checkpointer = MemorySaver()
        graph = build_agent_graph(checkpointer=checkpointer)
        thread_config = get_thread_config("test_thread_topic")

        initial_state = create_initial_state("attention transformer architecture", session_id="test_thread_topic")
        final_state = graph.invoke(initial_state, config=thread_config)

        assert final_state["status"] == "summarized"
        assert final_state["query_type"] == "topic_search"
        assert final_state["selected_paper"] is not None

    @patch("src.agent.nodes.get_arxiv_client")
    def test_graph_zero_results_pipeline(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.search_papers.return_value = []
        mock_get_client.return_value = mock_client

        checkpointer = MemorySaver()
        graph = build_agent_graph(checkpointer=checkpointer)
        thread_config = get_thread_config("test_thread_zero")

        initial_state = create_initial_state("asdfghjklnonexistent123", session_id="test_thread_zero")
        final_state = graph.invoke(initial_state, config=thread_config)

        assert final_state["status"] == "zero_results"
        assert "No arXiv papers were found" in final_state["error_message"]

    @patch("src.agent.nodes.parse_pdf")
    @patch("src.agent.nodes.download_pdf")
    @patch("src.agent.nodes.get_arxiv_client")
    def test_graph_metadata_fallback_pipeline(
        self,
        mock_get_client,
        mock_download,
        mock_parse,
    ):
        mock_client = MagicMock()
        mock_client.fetch_by_id.return_value = SAMPLE_PAPER
        mock_get_client.return_value = mock_client
        mock_download.return_value = "/tmp/corrupt.pdf"
        mock_parse.return_value = {
            "title": SAMPLE_PAPER["title"],
            "abstract": SAMPLE_PAPER["abstract"],
            "sections": {},
            "references": [],
            "full_text": "",
            "parsing_status": "metadata_only",
        }

        checkpointer = MemorySaver()
        graph = build_agent_graph(checkpointer=checkpointer)
        thread_config = get_thread_config("test_thread_fallback")

        initial_state = create_initial_state("1706.03762", session_id="test_thread_fallback")
        final_state = graph.invoke(initial_state, config=thread_config)

        assert final_state["status"] == "summarized"
        assert final_state["briefing"] is not None

    @patch("src.agent.nodes.search_chunks")
    @patch("src.agent.nodes.index_chunks")
    @patch("src.agent.nodes.chunk_paper")
    @patch("src.agent.nodes.parse_pdf")
    @patch("src.agent.nodes.download_pdf")
    @patch("src.agent.nodes.get_arxiv_client")
    def test_graph_multi_turn_persisted_qa(
        self,
        mock_get_client,
        mock_download,
        mock_parse,
        mock_chunk,
        mock_index,
        mock_search,
    ):
        # 1. Mock Ingestion Tools
        mock_client = MagicMock()
        mock_client.fetch_by_id.return_value = SAMPLE_PAPER
        mock_get_client.return_value = mock_client
        mock_download.return_value = "/tmp/1706.03762.pdf"
        mock_parse.return_value = SAMPLE_PARSED
        mock_chunk.return_value = [{"text": "Chunk", "section_name": "Intro", "chunk_index": 0, "metadata": {}}]
        mock_index.return_value = "arxiv_1706_03762"
        mock_search.return_value = [
            {
                "text": "The Transformer achieves 28.4 BLEU on English-to-German.",
                "section_name": "Results",
                "chunk_index": 0,
                "score": 0.9,
            }
        ]

        # Use SQLite checkpointer in memory
        checkpointer = get_sqlite_saver(":memory:")
        graph = build_agent_graph(checkpointer=checkpointer)
        session_id = "multi_turn_session_42"
        config = get_thread_config(session_id)

        # Turn 1: Ingest Paper
        initial_state = create_initial_state("1706.03762", session_id=session_id)
        state_after_ingest = graph.invoke(initial_state, config=config)
        assert state_after_ingest["status"] == "summarized"

        # Turn 2: User asks a follow-up question on the same persisted thread
        qa_input = {
            "qa_messages": [HumanMessage(content="What BLEU score was achieved on translation?")]
        }
        state_after_qa = graph.invoke(qa_input, config=config)

        assert state_after_qa["status"] == "qa_answered"
        # Verify qa_messages has both human question and AI answer
        assert len(state_after_qa["qa_messages"]) >= 2
        last_msg = state_after_qa["qa_messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "[Section: Results]" in last_msg.content
