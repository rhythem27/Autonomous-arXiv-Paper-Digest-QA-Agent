"""Unit and integration tests for PyMuPDF structural section parsing."""

from pathlib import Path
import pytest
import pymupdf

from src.parsers.pdf_parser import (
    ParsedPaper,
    download_pdf,
    parse_pdf,
)


def _generate_synthetic_academic_pdf(target_path: Path) -> Path:
    """Generate a multi-section synthetic PDF with academic formatting for testing."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)  # Standard Letter size

    # Margin noise: Header and footer
    page.insert_text((50, 20), "arXiv:2401.12345v1 [cs.AI] 15 Jan 2024", fontsize=8)
    page.insert_text((300, 775), "1", fontsize=9)

    # Title
    page.insert_text((50, 70), "Attention Mechanisms in Deep Neural Networks", fontsize=18, fontname="helv")

    # Abstract
    page.insert_text((50, 110), "Abstract", fontsize=11, fontname="hebo")
    abstract_text = (
        "This paper investigates multi-head attention mechanisms and their mathematical properties "
        "in sequence modeling tasks. We show that self-attention layers provide superior representation "
        "capacity while enabling massively parallel training on modern hardware accelerators."
    )
    page.insert_textbox(pymupdf.Rect(50, 120, 560, 180), abstract_text, fontsize=10)

    # 1. Introduction
    page.insert_text((50, 200), "1. Introduction", fontsize=13, fontname="hebo")
    intro_text = (
        "Recurrent neural networks and gated architectures have long been the foundation of natural language "
        "processing and sequence transduction problems. However, their inherently sequential computation prevents "
        "efficient batch training across long sequence lengths. In this work, we propose a novel model architecture "
        "that dispenses entirely with recurrence and convolutions, relying solely on an attention mechanism to model "
        "global dependencies between input and output representations."
    )
    page.insert_textbox(pymupdf.Rect(50, 215, 560, 300), intro_text, fontsize=10)

    # 2. Methodology
    page.insert_text((50, 320), "2. Methodology", fontsize=13, fontname="hebo")
    method_text = (
        "Our architecture follows an encoder-decoder paradigm using stacked self-attention and point-wise fully "
        "connected layers. For each attention head, queries, keys, and values are linearly projected into dimension d_k. "
        "We compute scaled dot-product attention where the scaling factor sqrt(d_k) stabilizes gradient flow during training. "
        "Residual connections and layer normalization are applied after each sub-layer to facilitate deep optimization."
    )
    page.insert_textbox(pymupdf.Rect(50, 335, 560, 420), method_text, fontsize=10)

    # 3. Results & Evaluation
    page.insert_text((50, 440), "3. Results & Evaluation", fontsize=13, fontname="hebo")
    results_text = (
        "We evaluate our model on standard machine translation benchmarks including WMT 2014 English-to-German "
        "and English-to-French tasks. Experimental results indicate that our approach establishes a new state of the art "
        "BLEU score of 28.4 while requiring a fraction of the training wall-clock time compared to competitive architectures."
    )
    page.insert_textbox(pymupdf.Rect(50, 455, 560, 530), results_text, fontsize=10)

    # 4. Conclusion
    page.insert_text((50, 550), "4. Conclusion", fontsize=13, fontname="hebo")
    conclusion_text = (
        "In this work, we introduced a purely attention-driven sequence model that eliminates recurrence. "
        "We demonstrated remarkable empirical gains on translation tasks and established strong scaling behaviors."
    )
    page.insert_textbox(pymupdf.Rect(50, 565, 560, 620), conclusion_text, fontsize=10)

    # References
    page.insert_text((50, 640), "References", fontsize=13, fontname="hebo")
    ref_1 = "[1] Vaswani, A., et al. Attention is all you need. Advances in Neural Information Processing Systems, 2017."
    ref_2 = "[2] Devlin, J., et al. BERT: Pre-training of deep bidirectional transformers. NAACL-HLT, 2019."
    page.insert_textbox(pymupdf.Rect(50, 655, 560, 750), f"{ref_1}\n{ref_2}", fontsize=9)

    doc.save(str(target_path))
    doc.close()
    return target_path


class TestPdfParserSynthetic:
    """Unit tests using generated academic PDF structures."""

    def test_parse_structured_sections(self, tmp_path):
        """Verify section detection, title, abstract, and reference isolation."""
        pdf_path = tmp_path / "synthetic_paper.pdf"
        _generate_synthetic_academic_pdf(pdf_path)

        parsed = parse_pdf(pdf_path)

        assert parsed["parsing_status"] == "success"
        assert "Attention Mechanisms" in parsed["title"]
        assert len(parsed["abstract"]) > 50
        assert "multi-head attention" in parsed["abstract"].lower()

        # Sections verification
        sections = parsed["sections"]
        assert "Introduction" in sections
        assert "Methodology" in sections
        assert "Results" in sections
        assert "Conclusion" in sections

        assert "sequence transduction" in sections["Introduction"].lower()
        assert "scaled dot-product" in sections["Methodology"].lower()
        assert "state of the art" in sections["Results"].lower()

        # References verification
        references = parsed["references"]
        assert len(references) == 2
        assert "Vaswani" in references[0]
        assert "Devlin" in references[1]

        # Full text includes all sections
        assert "Introduction" in parsed["full_text"]
        assert "Methodology" in parsed["full_text"]

    def test_noise_line_filtering(self, tmp_path):
        """Verify margin headers and arXiv stamps are omitted from parsed text."""
        pdf_path = tmp_path / "synthetic_noise.pdf"
        _generate_synthetic_academic_pdf(pdf_path)

        parsed = parse_pdf(pdf_path)

        # Header watermark should NOT appear in section body or full text
        assert "arXiv:2401.12345v1" not in parsed["full_text"]
        for section_content in parsed["sections"].values():
            assert "arXiv:2401.12345v1" not in section_content


class TestPdfParserEdgeCases:
    """Tests for scanned documents, corrupt files, and fallback handling."""

    def test_scanned_pdf_under_word_count_fallback(self, tmp_path):
        """Verify PDF with < 200 words triggers metadata_only status."""
        scanned_pdf = tmp_path / "scanned.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        # Insert only 10 words
        page.insert_text((50, 50), "Scanned document placeholder with only a few visible words.")
        doc.save(str(scanned_pdf))
        doc.close()

        parsed = parse_pdf(
            scanned_pdf,
            fallback_title="Fallback Scanned Paper",
            fallback_abstract="Fallback Abstract Summary",
        )

        assert parsed["parsing_status"] == "metadata_only"
        assert parsed["title"] == "Fallback Scanned Paper"
        assert parsed["abstract"] == "Fallback Abstract Summary"
        assert parsed["sections"] == {}
        assert parsed["references"] == []
        assert parsed["full_text"] == ""

    def test_nonexistent_pdf_returns_metadata_only(self, tmp_path):
        """Verify non-existent PDF returns metadata_only cleanly without raising."""
        parsed = parse_pdf(
            tmp_path / "missing.pdf",
            fallback_title="Missing Title",
            fallback_abstract="Missing Abstract",
        )
        assert parsed["parsing_status"] == "metadata_only"
        assert parsed["title"] == "Missing Title"

    def test_corrupt_binary_pdf_returns_metadata_only(self, tmp_path):
        """Verify broken binary file returns metadata_only cleanly."""
        corrupt_file = tmp_path / "corrupt.pdf"
        corrupt_file.write_bytes(b"%PDF-corrupt-data-1234567890")

        parsed = parse_pdf(corrupt_file, fallback_title="Corrupt Title")
        assert parsed["parsing_status"] == "metadata_only"
        assert parsed["title"] == "Corrupt Title"


class TestPdfParserLivePaper:
    """Integration test parsing a real academic paper."""

    def test_live_arxiv_paper_parsing(self, tmp_path):
        """Download and parse 'Attention Is All You Need' (1706.03762)."""
        pdf_path = download_pdf("1706.03762", cache_dir=tmp_path)
        assert pdf_path.exists()

        parsed = parse_pdf(pdf_path)

        assert parsed["parsing_status"] == "success"
        assert len(parsed["abstract"]) > 50
        assert "Introduction" in parsed["sections"]
        assert len(parsed["references"]) >= 5
        assert len(parsed["full_text"].split()) >= 1000
