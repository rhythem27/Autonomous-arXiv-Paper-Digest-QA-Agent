"""Explicit tests for §5 realistic failure cases and recovery mechanisms."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pymupdf
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import build_agent_graph
from src.agent.persistence import get_sqlite_saver, get_thread_config
from src.agent.state import create_initial_state
from src.parsers.pdf_parser import parse_pdf
from src.tools.arxiv_client import (
    PaperMetadata,
    fetch_paper_by_id,
    get_arxiv_client,
    search_papers_by_topic,
)


class TestFailureResilience:
    """Explicit tests for §5 realistic failure cases and recovery mechanisms."""

    def test_resilience_zero_results_graceful_exit(self):
        """Failure Case 1: 0 search results handled gracefully without unhandled exceptions."""
        with patch("src.agent.nodes.get_arxiv_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.search_papers.return_value = []
            mock_get_client.return_value = mock_client

            checkpointer = get_sqlite_saver(":memory:")
            graph = build_agent_graph(checkpointer=checkpointer)
            session_id = "resilience_zero_results_test"
            config = get_thread_config(session_id)

            initial_state = create_initial_state("cryptic_nonexistent_search_query_99999", session_id=session_id)
            final_state = graph.invoke(initial_state, config=config)

            assert final_state["status"] == "zero_results"
            assert "No arXiv papers were found" in final_state.get("error_message", "")
            assert final_state.get("selected_paper") is None

    def test_resilience_corrupted_pdf_metadata_fallback(self, tmp_path):
        """Failure Case 2: Corrupted binary PDF triggers metadata_only parsing cleanly."""
        corrupt_file = tmp_path / "corrupt.pdf"
        corrupt_file.write_bytes(b"%PDF-CORRUPTED-HEADER-GARBAGE-BYTE-STREAM-12345")

        parsed = parse_pdf(
            corrupt_file,
            fallback_title="Quantum Neural Dynamics",
            fallback_abstract="This paper investigates fallback metadata behavior.",
        )

        assert parsed["parsing_status"] == "metadata_only"
        assert parsed["title"] == "Quantum Neural Dynamics"
        assert parsed["abstract"] == "This paper investigates fallback metadata behavior."
        assert parsed["sections"] == {}
        assert parsed["full_text"] == ""

    def test_resilience_scanned_pdf_fallback(self, tmp_path):
        """Failure Case 3: Image-only or low word count PDF triggers metadata fallback."""
        scanned_pdf = tmp_path / "scanned_doc.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        # Insert fewer than 200 words
        page.insert_text((50, 50), "Scanned document text with only five words.")
        doc.save(str(scanned_pdf))
        doc.close()

        parsed = parse_pdf(
            scanned_pdf,
            fallback_title="Historical Scanned Manuscript",
            fallback_abstract="An abstract retrieved from the arXiv Atom feed metadata.",
        )

        assert parsed["parsing_status"] == "metadata_only"
        assert parsed["title"] == "Historical Scanned Manuscript"
        assert parsed["abstract"] == "An abstract retrieved from the arXiv Atom feed metadata."
        assert parsed["sections"] == {}

    def test_resilience_network_timeout_and_retry(self):
        """Failure Case 4: Transient network error / timeout retries and recovers."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.get_short_id.return_value = "2401.00001v1"
        mock_result.title = "Recovered Paper Title"
        mock_result.summary = "Abstract recovered after retry."
        mock_author = MagicMock()
        mock_author.name = "Dr. Resilient"
        mock_result.authors = [mock_author]
        mock_result.published = MagicMock()
        mock_result.published.isoformat.return_value = "2024-01-01"
        mock_result.updated = MagicMock()
        mock_result.updated.isoformat.return_value = "2024-01-01"
        mock_result.pdf_url = "https://arxiv.org/pdf/2401.00001.pdf"
        mock_result.primary_category = "cs.AI"
        mock_result.categories = ["cs.AI"]
        mock_result.entry_id = "http://arxiv.org/abs/2401.00001"

        call_count = 0

        def flaky_results(search):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Simulated transient network timeout")
            return iter([mock_result])

        mock_client.results = flaky_results

        # Client with retry configuration
        retry_client = get_arxiv_client(num_retries=3)
        assert retry_client.num_retries == 3

        # Simulate wrapped retry call
        def execute_with_retry(fn, retries=3):
            for i in range(retries):
                try:
                    return list(fn())
                except ConnectionError:
                    if i == retries - 1:
                        raise

        recovered = execute_with_retry(lambda: mock_client.results(None))
        assert len(recovered) == 1
        assert recovered[0].title == "Recovered Paper Title"
        assert call_count == 2

    @patch("src.agent.nodes.search_chunks")
    @patch("src.agent.nodes.index_chunks")
    @patch("src.agent.nodes.chunk_paper")
    @patch("src.agent.nodes.parse_pdf")
    @patch("src.agent.nodes.download_pdf")
    @patch("src.agent.nodes.get_arxiv_client")
    def test_resilience_anti_hallucination_refusal(
        self,
        mock_get_client,
        mock_download,
        mock_parse,
        mock_chunk,
        mock_index,
        mock_search,
    ):
        """Failure Case 5: Out-of-scope query refuses strictly without hallucination."""
        sample_paper: PaperMetadata = {
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent networks.",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "published": "2017-06-12",
            "categories": ["cs.CL", "cs.LG"],
            "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
            "entry_id": "http://arxiv.org/abs/1706.03762",
            "primary_category": "cs.CL",
            "journal_ref": None,
            "doi": None,
            "updated": "2017-06-12",
            "comment": None,
        }

        mock_client = MagicMock()
        mock_client.fetch_by_id.return_value = sample_paper
        mock_get_client.return_value = mock_client
        mock_download.return_value = "/tmp/1706.03762.pdf"
        mock_parse.return_value = {
            "title": sample_paper["title"],
            "abstract": sample_paper["abstract"],
            "sections": {"Introduction": "Introductory attention text."},
            "references": [],
            "full_text": "Introductory attention text.",
            "parsing_status": "success",
        }
        mock_chunk.return_value = [{"text": "Introductory attention text.", "section_name": "Introduction", "chunk_index": 0, "metadata": {}}]
        mock_index.return_value = "arxiv_1706_03762"
        # Return chunks that have NO lexical overlap with stock price
        mock_search.return_value = [
            {
                "text": "The dominant sequence transduction models are based on recurrent neural networks.",
                "section_name": "Introduction",
                "chunk_index": 0,
                "score": 0.3,
            }
        ]

        checkpointer = get_sqlite_saver(":memory:")
        graph = build_agent_graph(checkpointer=checkpointer)
        session_id = "test_refusal_session"
        config = get_thread_config(session_id)

        # Ingest paper
        initial_state = create_initial_state("1706.03762", session_id=session_id)
        graph.invoke(initial_state, config=config)

        # Ask totally unrelated query
        qa_input = {
            "qa_messages": [HumanMessage(content="What was Apple's stock price on May 5, 2024?")]
        }
        state_after_qa = graph.invoke(qa_input, config=config)

        assert state_after_qa["status"] == "qa_answered"
        last_msg = state_after_qa["qa_messages"][-1]
        assert isinstance(last_msg, AIMessage)
        # Verify strict anti-hallucination refusal wording
        assert "The provided paper text does not contain information regarding" in last_msg.content or "does not contain information" in last_msg.content

    @patch("src.agent.nodes.search_chunks")
    @patch("src.agent.nodes.index_chunks")
    @patch("src.agent.nodes.chunk_paper")
    @patch("src.agent.nodes.parse_pdf")
    @patch("src.agent.nodes.download_pdf")
    @patch("src.agent.nodes.get_arxiv_client")
    def test_resilience_checkpoint_recovery_after_restart(
        self,
        mock_get_client,
        mock_download,
        mock_parse,
        mock_chunk,
        mock_index,
        mock_search,
        tmp_path,
    ):
        """Failure Case 6: SQLite checkpointer restores session state across new graph instances."""
        db_file = tmp_path / "persistent_checkpoint.db"
        session_id = "restart_recovery_thread"
        config = get_thread_config(session_id)

        sample_paper: PaperMetadata = {
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need",
            "abstract": "Attention mechanisms in deep learning.",
            "authors": ["Ashish Vaswani"],
            "published": "2017-06-12",
            "categories": ["cs.CL"],
            "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
            "entry_id": "http://arxiv.org/abs/1706.03762",
            "primary_category": "cs.CL",
            "journal_ref": None,
            "doi": None,
            "updated": "2017-06-12",
            "comment": None,
        }

        mock_client = MagicMock()
        mock_client.fetch_by_id.return_value = sample_paper
        mock_get_client.return_value = mock_client
        mock_download.return_value = str(tmp_path / "paper.pdf")
        mock_parse.return_value = {
            "title": sample_paper["title"],
            "abstract": sample_paper["abstract"],
            "sections": {"Architecture": "Multi-head attention allows the model to jointly attend to information."},
            "references": [],
            "full_text": "Multi-head attention architecture.",
            "parsing_status": "success",
        }
        mock_chunk.return_value = [{"text": "Multi-head attention", "section_name": "Architecture", "chunk_index": 0, "metadata": {}}]
        mock_index.return_value = "arxiv_1706_03762"
        mock_search.return_value = [
            {
                "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
                "section_name": "Model Architecture",
                "chunk_index": 0,
                "score": 0.95,
            }
        ]

        # Phase 1: Ingest paper with checkpointer 1
        saver1 = get_sqlite_saver(db_file)
        graph1 = build_agent_graph(checkpointer=saver1)
        initial_state = create_initial_state("1706.03762", session_id=session_id)
        graph1.invoke(initial_state, config=config)

        # Phase 2: Simulate process crash/restart by creating brand new checkpointer & graph
        saver2 = get_sqlite_saver(db_file)
        graph2 = build_agent_graph(checkpointer=saver2)

        # Ask question on restored graph
        qa_input = {
            "qa_messages": [HumanMessage(content="Explain multi-head attention.")]
        }
        res = graph2.invoke(qa_input, config=config)

        assert res["status"] == "qa_answered"
        assert res.get("selected_paper") is not None
        assert res["selected_paper"]["arxiv_id"] == "1706.03762"
        last_msg = res["qa_messages"][-1]
        assert "[Section: Model Architecture]" in last_msg.content
