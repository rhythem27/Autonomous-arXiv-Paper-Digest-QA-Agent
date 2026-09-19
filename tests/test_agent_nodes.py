"""Unit and integration tests for LangGraph agent node functions.

Validates all 7 assessment lifecycle stages, fallback routes, strict limitations enforcement,
and anti-hallucination guardrails.
"""

from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.state import create_initial_state, PaperMetadata, ParsedPaper
from src.agent.nodes import (
    node_query_understanding,
    node_arxiv_retrieval,
    node_selection_ranking,
    node_fetch_parse,
    node_metadata_fallback,
    node_chunk_embed,
    node_summarize,
    node_qa_answer,
    node_handle_zero_results,
)
from src.llm.prompts import ANTI_HALLUCINATION_REFUSAL_PREFIX


SAMPLE_PAPER: PaperMetadata = {
    "arxiv_id": "1706.03762",
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
    "published": "2017-06-12",
    "updated": "2017-06-12",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer.",
    "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
    "primary_category": "cs.CL",
    "categories": ["cs.CL", "cs.LG"],
    "entry_id": "http://arxiv.org/abs/1706.03762v1",
}

SAMPLE_PARSED: ParsedPaper = {
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models are based on recurrent or convolutional neural networks.",
    "sections": {
        "Abstract": "The dominant sequence transduction models...",
        "Introduction": "Recurrent neural networks have been firmly established as state of the art.",
        "Model Architecture": "The Transformer follows this overall architecture using stacked self-attention and point-wise fully connected layers.",
        "Results": "On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU.",
        "Limitations and Discussion": "The quadratic compute complexity with respect to sequence length limits processing very long contexts without approximations.",
    },
    "references": ["Hochreiter & Schmidhuber, 1997", "Bahdanau et al., 2014"],
    "full_text": "Attention Is All You Need\n\nAbstract...",
    "parsing_status": "success",
}


class TestQueryUnderstandingNode:
    """Test Node 1: query_understanding."""

    def test_direct_id_query(self):
        state = create_initial_state(raw_query="1706.03762v2")
        update = node_query_understanding(state)

        assert update["query_type"] == "arxiv_id"
        assert update["extracted_id"] == "1706.03762"
        assert update["status"] == "query_understood"

    def test_url_query(self):
        state = create_initial_state(raw_query="https://arxiv.org/abs/2301.00234")
        update = node_query_understanding(state)

        assert update["query_type"] == "arxiv_id"
        assert update["extracted_id"] == "2301.00234"

    def test_topic_search_query(self):
        state = create_initial_state(raw_query="state space models vs transformers")
        update = node_query_understanding(state)

        assert update["query_type"] == "topic_search"
        assert update["extracted_id"] is None
        assert "state space models" in update["search_keywords"]


class TestArxivRetrievalNode:
    """Test Node 2: arxiv_retrieval."""

    @patch("src.agent.nodes.get_arxiv_client")
    def test_retrieval_by_id_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.fetch_by_id.return_value = SAMPLE_PAPER
        mock_get_client.return_value = mock_client

        state = create_initial_state(raw_query="1706.03762")
        state["query_type"] = "arxiv_id"
        state["extracted_id"] = "1706.03762"

        update = node_arxiv_retrieval(state)

        assert update["status"] == "retrieved"
        assert len(update["candidate_papers"]) == 1
        assert update["candidate_papers"][0]["arxiv_id"] == "1706.03762"
        assert update["selected_paper"] == SAMPLE_PAPER

    @patch("src.agent.nodes.get_arxiv_client")
    def test_retrieval_by_topic_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.search_papers.return_value = [SAMPLE_PAPER]
        mock_get_client.return_value = mock_client

        state = create_initial_state(raw_query="attention mechanisms")
        state["query_type"] = "topic_search"
        state["search_keywords"] = "attention mechanisms"

        update = node_arxiv_retrieval(state)

        assert update["status"] == "retrieved"
        assert len(update["candidate_papers"]) == 1
        assert update["candidate_papers"][0]["title"] == "Attention Is All You Need"

    @patch("src.agent.nodes.get_arxiv_client")
    def test_retrieval_zero_results(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.search_papers.return_value = []
        mock_get_client.return_value = mock_client

        state = create_initial_state(raw_query="asdfghjklnonexistentquery123")
        state["query_type"] = "topic_search"

        update = node_arxiv_retrieval(state)

        assert update["status"] == "zero_results"
        assert update["candidate_papers"] == []
        assert update["selected_paper"] is None

    @patch("src.agent.nodes.get_arxiv_client")
    def test_retrieval_network_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.search_papers.side_effect = RuntimeError("Connection timeout")
        mock_get_client.return_value = mock_client

        state = create_initial_state(raw_query="test query")
        update = node_arxiv_retrieval(state)

        assert update["status"] == "retrieval_error"
        assert "Connection timeout" in update["error_message"]


class TestSelectionRankingNode:
    """Test Node 3: selection_ranking."""

    def test_selected_paper_already_set(self):
        state = create_initial_state(raw_query="1706.03762")
        state["selected_paper"] = SAMPLE_PAPER

        update = node_selection_ranking(state)
        assert update["status"] == "paper_selected"
        assert update["selected_paper"]["arxiv_id"] == "1706.03762"

    def test_empty_candidates_yields_zero_results(self):
        state = create_initial_state(raw_query="empty")
        state["candidate_papers"] = []

        update = node_selection_ranking(state)
        assert update["status"] == "zero_results"
        assert update["selected_paper"] is None

    def test_single_candidate_selection(self):
        state = create_initial_state(raw_query="transformer")
        state["candidate_papers"] = [SAMPLE_PAPER]

        update = node_selection_ranking(state)
        assert update["status"] == "paper_selected"
        assert update["selected_paper"]["title"] == "Attention Is All You Need"

    def test_multiple_candidates_semantic_ranking(self):
        other_paper: PaperMetadata = {
            "arxiv_id": "9999.00001",
            "title": "Unrelated Quantum Biology Simulation",
            "authors": ["Scientist X"],
            "published": "2020-01-01",
            "updated": "2020-01-01",
            "abstract": "We study enzymatic reaction kinetics under quantum coherence conditions in chloroplasts.",
            "pdf_url": "https://arxiv.org/pdf/9999.00001.pdf",
            "primary_category": "quant-ph",
            "categories": ["quant-ph"],
            "entry_id": "http://arxiv.org/abs/9999.00001",
        }
        state = create_initial_state(raw_query="self-attention transformer architecture for language translation")
        state["candidate_papers"] = [other_paper, SAMPLE_PAPER]

        update = node_selection_ranking(state)
        assert update["status"] == "paper_selected"
        # Attention Is All You Need must be ranked higher than Quantum Biology
        assert update["selected_paper"]["arxiv_id"] == "1706.03762"


class TestFetchParseNode:
    """Test Node 4: fetch_parse and metadata fallback."""

    def test_fetch_parse_no_selected_paper(self):
        state = create_initial_state(raw_query="test")
        update = node_fetch_parse(state)

        assert update["status"] == "parse_failed"
        assert "No paper selected" in update["error_message"]

    @patch("src.agent.nodes.parse_pdf")
    @patch("src.agent.nodes.download_pdf")
    def test_fetch_parse_success(self, mock_download, mock_parse):
        mock_download.return_value = "/tmp/1706.03762.pdf"
        mock_parse.return_value = SAMPLE_PARSED

        state = create_initial_state(raw_query="1706.03762")
        state["selected_paper"] = SAMPLE_PAPER

        update = node_fetch_parse(state)

        assert update["status"] == "parsed"
        assert update["parsed_paper"]["title"] == "Attention Is All You Need"
        assert "Model Architecture" in update["parsed_paper"]["sections"]

    @patch("src.agent.nodes.parse_pdf")
    @patch("src.agent.nodes.download_pdf")
    def test_fetch_parse_metadata_only_fallback(self, mock_download, mock_parse):
        mock_download.return_value = "/tmp/scanned.pdf"
        mock_parse.return_value = {
            "title": "Attention Is All You Need",
            "abstract": "...",
            "sections": {},
            "references": [],
            "full_text": "...",
            "parsing_status": "metadata_only",
        }

        state = create_initial_state(raw_query="1706.03762")
        state["selected_paper"] = SAMPLE_PAPER

        update = node_fetch_parse(state)
        assert update["status"] == "metadata_only"

    def test_metadata_fallback_node(self):
        state = create_initial_state(raw_query="1706.03762")
        state["selected_paper"] = SAMPLE_PAPER

        update = node_metadata_fallback(state)

        assert update["status"] == "metadata_fallback"
        assert update["parsed_paper"]["parsing_status"] == "metadata_only"
        assert update["parsed_paper"]["title"] == "Attention Is All You Need"
        assert "Abstract" in update["parsed_paper"]["sections"]


class TestChunkEmbedNode:
    """Test Node 5: chunk_embed."""

    def test_chunk_embed_skipped_when_data_missing(self):
        state = create_initial_state(raw_query="test")
        update = node_chunk_embed(state)
        assert update["status"] == "chunk_embed_skipped"

    @patch("src.agent.nodes.index_chunks")
    @patch("src.agent.nodes.chunk_paper")
    def test_chunk_embed_success(self, mock_chunk, mock_index):
        mock_chunk.return_value = [
            {"text": "Chunk 1", "section_name": "Intro", "chunk_index": 0, "metadata": {}}
        ]
        mock_index.return_value = "arxiv_1706_03762"

        state = create_initial_state(raw_query="1706.03762")
        state["selected_paper"] = SAMPLE_PAPER
        state["parsed_paper"] = SAMPLE_PARSED

        update = node_chunk_embed(state)

        assert update["status"] == "indexed"
        assert update["qdrant_collection_name"] == "arxiv_1706_03762"


class TestSummarizeNode:
    """Test Node 6: summarize (Executive Briefing generation)."""

    def test_summarize_missing_data(self):
        state = create_initial_state(raw_query="test")
        update = node_summarize(state)
        assert update["status"] == "summarize_failed"

    def test_summarize_generates_valid_briefing_with_mandatory_limitations(self):
        state = create_initial_state(raw_query="1706.03762")
        state["selected_paper"] = SAMPLE_PAPER
        state["parsed_paper"] = SAMPLE_PARSED

        update = node_summarize(state)

        assert update["status"] == "summarized"
        briefing = update["briefing"]
        assert briefing is not None
        assert briefing["title"] == "Attention Is All You Need"
        assert briefing["arxiv_id"] == "1706.03762"
        assert len(briefing["why_it_matters"]) > 20
        assert len(briefing["problem_statement"]) > 10
        assert len(briefing["method_approach"]) >= 1
        assert len(briefing["key_results"]) >= 1

        # CRITICAL RUBRIC TEST: Limitations MUST NEVER be empty
        assert isinstance(briefing["limitations"], list)
        assert len(briefing["limitations"]) >= 1
        for lim in briefing["limitations"]:
            assert len(lim.strip()) > 10

        assert len(briefing["suggested_follow_ups"]) >= 1
        assert "# Executive Briefing" in briefing["markdown_output"]


class TestQaAnswerNode:
    """Test Node 7: qa_answer (grounded answering and anti-hallucination)."""

    def test_qa_answer_skipped_without_question(self):
        state = create_initial_state(raw_query="test")
        update = node_qa_answer(state)
        assert update["status"] == "qa_skipped"

    @patch("src.agent.nodes.search_chunks")
    def test_qa_answer_grounded_response(self, mock_search):
        mock_search.return_value = [
            {
                "text": "The Transformer uses multi-head attention with 8 parallel attention heads.",
                "section_name": "Model Architecture",
                "chunk_index": 0,
                "score": 0.88,
            }
        ]

        state = create_initial_state(raw_query="1706.03762")
        state["selected_paper"] = SAMPLE_PAPER
        state["qa_messages"] = [
            HumanMessage(content="How many attention heads does the Transformer use?")
        ]

        update = node_qa_answer(state)

        assert update["status"] == "qa_answered"
        assert len(update["qa_messages"]) == 1
        reply = update["qa_messages"][0]
        assert isinstance(reply, AIMessage)
        assert "[Section: Model Architecture]" in reply.content

    @patch("src.agent.nodes.search_chunks")
    def test_qa_answer_anti_hallucination_refusal(self, mock_search):
        # When retrieval finds no relevant chunks or question is outside scope
        mock_search.return_value = []

        state = create_initial_state(raw_query="1706.03762")
        state["selected_paper"] = SAMPLE_PAPER
        state["qa_messages"] = [
            HumanMessage(content="What was the stock price of Apple in 2017?")
        ]

        update = node_qa_answer(state)

        assert update["status"] == "qa_answered"
        reply = update["qa_messages"][0]
        assert isinstance(reply, AIMessage)
        # CRITICAL RUBRIC TEST: Anti-hallucination refusal requirement
        assert ANTI_HALLUCINATION_REFUSAL_PREFIX in reply.content


class TestZeroResultsNode:
    """Test Node 8: handle_zero_results."""

    def test_zero_results_guidance(self):
        state = create_initial_state(raw_query="qwertyuiopasdfghjkl")
        update = node_handle_zero_results(state)

        assert update["status"] == "zero_results"
        assert "No arXiv papers were found" in update["error_message"]
        assert "qwertyuiopasdfghjkl" in update["error_message"]
