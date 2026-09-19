"""Unit tests for arXiv query parsing and intent classification."""

import pytest

from src.tools.arxiv_client import ParsedQuery, parse_query, strip_arxiv_version


class TestStripArxivVersion:
    """Tests for version suffix stripping helper."""

    def test_strip_modern_version(self):
        assert strip_arxiv_version("1706.03762v1") == "1706.03762"
        assert strip_arxiv_version("1706.03762v12") == "1706.03762"
        assert strip_arxiv_version("2401.12345") == "2401.12345"

    def test_strip_legacy_version(self):
        assert strip_arxiv_version("quant-ph/0201082v1") == "quant-ph/0201082"
        assert strip_arxiv_version("cs/0112017") == "cs/0112017"


class TestQueryParsingModernIds:
    """Tests for recognizing modern arXiv IDs."""

    def test_acceptance_criteria_id(self):
        result = parse_query("2401.12345")
        assert result == ("arxiv_id", "2401.12345", None)
        assert result.query_type == "arxiv_id"
        assert result.clean_id == "2401.12345"
        assert result.search_query is None

    def test_modern_id_with_version(self):
        result = parse_query("1706.03762v2")
        assert result == ("arxiv_id", "1706.03762", None)

    def test_modern_id_with_whitespace(self):
        result = parse_query("   2310.06825v1   ")
        assert result == ("arxiv_id", "2310.06825", None)

    def test_modern_4_digit_number(self):
        result = parse_query("1501.0001")
        assert result == ("arxiv_id", "1501.0001", None)

    def test_arxiv_prefixed_modern_id(self):
        assert parse_query("arXiv: 2401.12345") == ("arxiv_id", "2401.12345", None)
        assert parse_query("arxiv:1706.03762v3") == ("arxiv_id", "1706.03762", None)
        assert parse_query("ARXIV: 2312.00752") == ("arxiv_id", "2312.00752", None)


class TestQueryParsingLegacyIds:
    """Tests for recognizing legacy arXiv IDs."""

    def test_legacy_category_id_with_version(self):
        result = parse_query("quant-ph/0201082v1")
        assert result == ("arxiv_id", "quant-ph/0201082", None)

    def test_legacy_category_id_without_version(self):
        result = parse_query("math.PR/0501001")
        assert result == ("arxiv_id", "math.PR/0501001", None)

    def test_legacy_cs_id(self):
        result = parse_query("cs/0112017")
        assert result == ("arxiv_id", "cs/0112017", None)

    def test_prefixed_legacy_id(self):
        assert parse_query("arXiv: quant-ph/0201082v1") == ("arxiv_id", "quant-ph/0201082", None)


class TestQueryParsingUrls:
    """Tests for recognizing arXiv web URLs."""

    def test_acceptance_criteria_abs_url(self):
        result = parse_query("https://arxiv.org/abs/1706.03762v1")
        assert result == ("arxiv_id", "1706.03762", None)

    def test_pdf_url_with_extension(self):
        result = parse_query("https://arxiv.org/pdf/2401.12345.pdf")
        assert result == ("arxiv_id", "2401.12345", None)

    def test_pdf_url_without_extension(self):
        result = parse_query("https://arxiv.org/pdf/2401.12345")
        assert result == ("arxiv_id", "2401.12345", None)

    def test_http_legacy_url(self):
        result = parse_query("http://arxiv.org/abs/cs/0112017v1")
        assert result == ("arxiv_id", "cs/0112017", None)

    def test_url_with_fragments_and_query_params(self):
        result = parse_query("https://arxiv.org/abs/2401.12345?context=cs#sec1")
        assert result == ("arxiv_id", "2401.12345", None)


class TestQueryParsingTopicSearch:
    """Tests for natural-language research topic classification."""

    def test_acceptance_criteria_topic(self):
        result = parse_query("flash attention algorithms")
        assert result == ("topic_search", None, "flash attention algorithms")
        assert result.query_type == "topic_search"
        assert result.clean_id is None
        assert result.search_query == "flash attention algorithms"

    def test_long_natural_language_query(self):
        query = "recent work on KV-cache compression for LLMs"
        result = parse_query(query)
        assert result == ("topic_search", None, query)

    def test_collapses_multiple_spaces(self):
        result = parse_query("  reinforcement   learning   from   human   feedback  ")
        assert result == ("topic_search", None, "reinforcement learning from human feedback")

    def test_query_with_punctuation(self):
        query = "What is LoRA, and how does QLoRA improve memory efficiency?"
        result = parse_query(query)
        assert result == ("topic_search", None, query)


class TestQueryParsingEdgeCases:
    """Tests for edge cases, whitespace, and unpacking."""

    def test_empty_string(self):
        result = parse_query("")
        assert result == ("topic_search", None, "")

    def test_whitespace_only(self):
        result = parse_query("    \t\n  ")
        assert result == ("topic_search", None, "")

    def test_tuple_unpacking(self):
        query_type, clean_id, search_query = parse_query("2401.12345")
        assert query_type == "arxiv_id"
        assert clean_id == "2401.12345"
        assert search_query is None

    def test_named_tuple_instance(self):
        result = parse_query("2401.12345")
        assert isinstance(result, ParsedQuery)
        assert isinstance(result, tuple)
