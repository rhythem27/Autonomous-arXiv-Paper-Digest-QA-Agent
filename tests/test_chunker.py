"""Unit tests for custom section-aware document chunking."""

import pytest

from src.parsers.pdf_parser import ParsedPaper
from src.vectorstore.chunker import DocumentChunk, chunk_paper


@pytest.fixture
def sample_parsed_paper() -> ParsedPaper:
    """Fixture providing a structured ParsedPaper with distinct sections."""
    return ParsedPaper(
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        sections={
            "Introduction": (
                "Recurrent neural networks, long short-term memory and gated recurrent neural networks "
                "have been firmly established as state of the art approaches in sequence modeling. "
                "Sequential computation inhibits parallelization."
            ),
            "Methodology": (
                "The Transformer follows this overall architecture using stacked self-attention and "
                "point-wise, fully connected layers for both the encoder and decoder. "
                "Multi-head attention allows the model to jointly attend to information from different representation subspaces."
            ),
            "Conclusion": (
                "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, "
                "replacing recurrent layers most commonly used in encoder-decoder architectures."
            ),
        },
        references=[
            "[1] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. 1997.",
            "[2] Dzmitry Bahdanau et al. Neural machine translation by jointly learning to align and translate. 2014.",
        ],
        full_text="Full paper text...",
        parsing_status="success",
    )


class TestSectionAwareChunker:
    """Test suite verifying section boundary preservation, headers, and metadata."""

    def test_section_boundary_preservation(self, sample_parsed_paper):
        """Verify chunks strictly never cross section boundaries."""
        chunks = chunk_paper(sample_parsed_paper, arxiv_id="1706.03762", chunk_size=150, chunk_overlap=30)
        assert len(chunks) > 0

        for chunk in chunks:
            if chunk.section_name == "Introduction":
                assert "Transformer follows this overall architecture" not in chunk.text
                assert "presented the Transformer" not in chunk.text
            elif chunk.section_name == "Methodology":
                assert "Recurrent neural networks" not in chunk.text
                assert "presented the Transformer" not in chunk.text
            elif chunk.section_name == "Conclusion":
                assert "Recurrent neural networks" not in chunk.text
                assert "Multi-head attention" not in chunk.text

    def test_contextual_header_injection(self, sample_parsed_paper):
        """Verify each chunk contains an explicit paper and section contextual header."""
        chunks = chunk_paper(sample_parsed_paper, arxiv_id="1706.03762")
        assert len(chunks) >= 4  # Abstract + 3 sections

        for chunk in chunks:
            expected_header = f"[Paper: {sample_parsed_paper['title']} | Section: {chunk.section_name}]"
            assert chunk.text.startswith(expected_header)
            assert "\n\n" in chunk.text

    def test_abstract_included_as_first_section(self, sample_parsed_paper):
        """Verify abstract is extracted and chunked as the primary section."""
        chunks = chunk_paper(sample_parsed_paper, arxiv_id="1706.03762")
        first_chunk = chunks[0]
        assert first_chunk.section_name == "Abstract"
        assert "dominant sequence transduction models" in first_chunk.text

    def test_metadata_payload(self, sample_parsed_paper):
        """Verify rich metadata payload is correctly populated on all chunks."""
        chunks = chunk_paper(sample_parsed_paper, arxiv_id="1706.03762")

        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_id.startswith("1706.03762_")
            assert chunk.arxiv_id == "1706.03762"
            assert chunk.chunk_index == idx
            assert chunk.metadata["title"] == "Attention Is All You Need"
            assert chunk.metadata["section_name"] == chunk.section_name
            assert chunk.metadata["word_count"] > 0
            assert chunk.metadata["char_count"] > 0
            assert chunk.metadata["metadata_only"] is False

    def test_sliding_window_overlap(self):
        """Verify that long sections produce overlapping chunks."""
        long_section_text = (
            "Sentence one describes the baseline algorithm. "
            "Sentence two discusses computational complexity. "
            "Sentence three introduces hardware bottlenecks. "
            "Sentence four proposes asynchronous pipelining. "
            "Sentence five details memory footprint reduction. "
            "Sentence six reports speedup on synthetic benchmarks. "
            "Sentence seven concludes the empirical analysis."
        )
        paper = ParsedPaper(
            title="Pipelined Training",
            abstract="Abstract text here.",
            sections={"Methodology": long_section_text},
            references=[],
            full_text=long_section_text,
            parsing_status="success",
        )

        # Use small chunk size (120 chars) and overlap (60 chars) to force multiple chunks
        chunks = chunk_paper(paper, arxiv_id="2401.00001", chunk_size=120, chunk_overlap=60)
        method_chunks = [c for c in chunks if c.section_name == "Methodology"]
        assert len(method_chunks) >= 3

        # Verify that chunk N and chunk N+1 share text
        for i in range(len(method_chunks) - 1):
            body_1 = method_chunks[i].text.split("\n\n", 1)[1]
            body_2 = method_chunks[i + 1].text.split("\n\n", 1)[1]
            words_1 = set(body_1.split())
            words_2 = set(body_2.split())
            shared_words = words_1.intersection(words_2)
            assert len(shared_words) > 0  # Demonstrates overlap retention

    def test_metadata_only_paper_fallback(self):
        """Verify scanned/corrupt paper produces fallback metadata chunk."""
        fallback_paper = ParsedPaper(
            title="Scanned Legacy Document",
            abstract="Summary from arXiv Atom feed.",
            sections={},
            references=[],
            full_text="",
            parsing_status="metadata_only",
        )

        chunks = chunk_paper(fallback_paper, arxiv_id="quant-ph_0201082")
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.section_name == "Abstract"
        assert chunk.metadata["metadata_only"] is True
        assert "[Paper: Scanned Legacy Document | Section: Abstract]" in chunk.text
        assert "Summary from arXiv Atom feed." in chunk.text
