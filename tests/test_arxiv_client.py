"""Unit and integration tests for the official arXiv API client and metadata extraction."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

import arxiv

from src.tools.arxiv_client import (
    PaperMetadata,
    _result_to_metadata,
    fetch_paper_by_id,
    get_arxiv_client,
    search_papers_by_topic,
)


class TestArxivClientUnit:
    """Unit tests using mocked arXiv results."""

    def test_result_to_metadata_conversion(self):
        """Verify proper field extraction and sanitization from arxiv.Result."""
        mock_result = MagicMock(spec=arxiv.Result)
        mock_result.get_short_id.return_value = "2401.12345v2"
        mock_result.title = "Sample Paper Title:\n With Newlines   and Spaces"
        mock_result.summary = "This is a summary\nwith multiple lines and spaces."
        
        mock_author1 = MagicMock()
        mock_author1.name = "Alice Smith"
        mock_author2 = MagicMock()
        mock_author2.name = "Bob Jones"
        mock_result.authors = [mock_author1, mock_author2]
        
        mock_result.published = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_result.updated = datetime(2024, 1, 20, 15, 30, 0, tzinfo=timezone.utc)
        mock_result.pdf_url = "https://arxiv.org/pdf/2401.12345v2"
        mock_result.primary_category = "cs.CL"
        mock_result.categories = ["cs.CL", "cs.AI"]
        mock_result.entry_id = "http://arxiv.org/abs/2401.12345v2"

        metadata = _result_to_metadata(mock_result)

        assert metadata["arxiv_id"] == "2401.12345"  # Version stripped
        assert metadata["title"] == "Sample Paper Title: With Newlines and Spaces"
        assert metadata["abstract"] == "This is a summary with multiple lines and spaces."
        assert metadata["authors"] == ["Alice Smith", "Bob Jones"]
        assert "2024-01-15" in metadata["published"]
        assert "2024-01-20" in metadata["updated"]
        assert metadata["pdf_url"] == "https://arxiv.org/pdf/2401.12345v2"
        assert metadata["primary_category"] == "cs.CL"
        assert metadata["categories"] == ["cs.CL", "cs.AI"]
        assert metadata["entry_id"] == "http://arxiv.org/abs/2401.12345v2"

    def test_fetch_paper_by_id_empty_input(self):
        """Verify empty or blank ID returns None without API call."""
        assert fetch_paper_by_id("") is None
        assert fetch_paper_by_id("   ") is None

    def test_fetch_paper_by_id_handles_exception(self):
        """Verify exceptions are caught defensively and return None."""
        mock_client = MagicMock(spec=arxiv.Client)
        mock_client.results.side_effect = Exception("Simulated network timeout")

        result = fetch_paper_by_id("1706.03762", client=mock_client)
        assert result is None

    def test_search_papers_by_topic_empty_input(self):
        """Verify empty topic search returns empty list."""
        assert search_papers_by_topic("") == []
        assert search_papers_by_topic("   ") == []

    def test_search_papers_by_topic_handles_exception(self):
        """Verify exceptions in search are caught defensively and return empty list."""
        mock_client = MagicMock(spec=arxiv.Client)
        mock_client.results.side_effect = Exception("Simulated arXiv 503 error")

        results = search_papers_by_topic("deep learning", client=mock_client)
        assert results == []

    def test_get_arxiv_client_config(self):
        """Verify client creation adheres to rate limits."""
        client = get_arxiv_client(page_size=20, delay_seconds=3.0, num_retries=5)
        assert isinstance(client, arxiv.Client)
        assert client.page_size == 20
        assert client.delay_seconds == 3.0
        assert client.num_retries == 5


class TestArxivClientIntegration:
    """Integration tests connecting to live arXiv API."""

    def test_fetch_known_paper_by_id(self):
        """Verify retrieval of 'Attention Is All You Need' (1706.03762)."""
        metadata = fetch_paper_by_id("1706.03762")
        assert metadata is not None
        assert metadata["arxiv_id"] == "1706.03762"
        assert "Attention Is All You Need" in metadata["title"]
        assert any("Vaswani" in author for author in metadata["authors"])
        assert metadata["pdf_url"].startswith("http")
        assert len(metadata["abstract"]) > 50

    def test_fetch_paper_by_url(self):
        """Verify retrieval when passed a full arXiv abs URL."""
        metadata = fetch_paper_by_id("https://arxiv.org/abs/1706.03762v7")
        assert metadata is not None
        assert metadata["arxiv_id"] == "1706.03762"
        assert "Attention" in metadata["title"]

    def test_fetch_nonexistent_paper_returns_none(self):
        """Verify searching for non-existent paper returns None gracefully."""
        metadata = fetch_paper_by_id("0000.00000")
        assert metadata is None

    def test_search_papers_by_topic(self):
        """Verify topic search returns structured candidate papers."""
        papers = search_papers_by_topic("KV-cache compression", max_results=3)
        assert isinstance(papers, list)
        assert len(papers) > 0
        assert len(papers) <= 3
        
        for paper in papers:
            assert "arxiv_id" in paper
            assert "title" in paper
            assert "abstract" in paper
            assert "pdf_url" in paper
            assert len(paper["authors"]) > 0

    def test_search_gibberish_topic_returns_empty_or_handled(self):
        """Verify searching for nonsensical query handles gracefully."""
        papers = search_papers_by_topic("xjqpwkzzzznonexistentquery987654321", max_results=2)
        assert isinstance(papers, list)
