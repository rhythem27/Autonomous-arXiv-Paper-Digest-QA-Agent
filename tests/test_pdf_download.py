"""Unit and integration tests for PDF downloading, caching, and atomic writing."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.parsers.pdf_parser import (
    PDFDownloadError,
    download_pdf,
    get_pdf_cache_path,
    sanitize_arxiv_id,
)


class TestSanitizeArxivId:
    """Tests for arXiv ID filesystem sanitization."""

    def test_modern_id_sanitization(self):
        assert sanitize_arxiv_id("1706.03762") == "1706.03762"
        assert sanitize_arxiv_id("1706.03762v2") == "1706.03762"

    def test_legacy_id_with_slashes(self):
        assert sanitize_arxiv_id("quant-ph/0201082") == "quant-ph_0201082"
        assert sanitize_arxiv_id("quant-ph/0201082v1") == "quant-ph_0201082"
        assert sanitize_arxiv_id("math.PR/0501001") == "math.PR_0501001"

    def test_full_url_sanitization(self):
        assert sanitize_arxiv_id("https://arxiv.org/abs/1706.03762v7") == "1706.03762"
        assert sanitize_arxiv_id("https://arxiv.org/pdf/2401.12345.pdf") == "2401.12345"

    def test_empty_string(self):
        assert sanitize_arxiv_id("") == ""
        assert sanitize_arxiv_id("   ") == ""


class TestPdfCachingUnit:
    """Unit tests for cache validation, atomic writes, and error handling."""

    def test_cache_hit_avoids_network(self, tmp_path):
        """Verify pre-existing valid PDF in cache is returned without HTTP requests."""
        cached_file = tmp_path / "1706.03762.pdf"
        # Write dummy valid PDF header and > 1024 bytes
        dummy_content = b"%PDF-1.4\n" + (b"0" * 2000)
        cached_file.write_bytes(dummy_content)

        with patch("httpx.Client") as mock_client:
            result_path = download_pdf("1706.03762", cache_dir=tmp_path)
            assert result_path == cached_file
            assert result_path.exists()
            mock_client.assert_not_called()

    def test_empty_id_raises_error(self, tmp_path):
        """Verify passing empty string raises PDFDownloadError."""
        with pytest.raises(PDFDownloadError, match="Empty arXiv ID"):
            download_pdf("", cache_dir=tmp_path)

    def test_invalid_pdf_content_raises_error(self, tmp_path):
        """Verify downloading non-PDF HTML error content raises PDFDownloadError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_bytes.return_value = [b"<!DOCTYPE html><html><body>Error</body></html>" * 50]

        with patch("httpx.Client.stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response

            with pytest.raises(PDFDownloadError, match="does not appear to be a valid PDF"):
                download_pdf("2401.12345", cache_dir=tmp_path, max_retries=1)

    def test_http_404_raises_error(self, tmp_path):
        """Verify 404 response raises PDFDownloadError."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.Client.stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response

            with pytest.raises(PDFDownloadError, match="HTTP 404"):
                download_pdf("0000.00000", cache_dir=tmp_path, max_retries=1)

    def test_temp_file_cleanup_on_failure(self, tmp_path):
        """Verify leftover .tmp files are cleaned up on failure."""
        with patch("httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = Exception("Simulated network drop")

            with pytest.raises(PDFDownloadError):
                download_pdf("1706.03762", cache_dir=tmp_path, max_retries=1)

            # Ensure no .tmp files remain in cache dir
            temp_files = list(tmp_path.glob("*.tmp*"))
            assert len(temp_files) == 0


class TestPdfDownloadIntegration:
    """Integration test connecting to live arXiv to download and verify caching."""

    def test_live_download_and_subsequent_cache_hit(self, tmp_path):
        """Download 'Attention Is All You Need' (1706.03762), verify file, then verify cache hit."""
        # First download: should hit network
        pdf_path = download_pdf("1706.03762", cache_dir=tmp_path, timeout=40.0)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 100_000  # Should be substantial (standard paper is >1MB)
        
        # Verify valid PDF magic bytes
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            assert header.startswith(b"%PDF")

        # Second retrieval: must hit cache immediately
        with patch("httpx.Client") as mock_client:
            cached_path = download_pdf("1706.03762", cache_dir=tmp_path)
            assert cached_path == pdf_path
            mock_client.assert_not_called()
