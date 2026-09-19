"""PDF downloader, caching engine, and structural document parser.

Handles reliable streaming PDF retrieval from arXiv with local disk caching,
custom academic User-Agent headers, retry resilience, atomic file writes,
and structural PyMuPDF document extraction.
"""

import os
from pathlib import Path
import re
import statistics
import time
from typing import Dict, List, Optional, Tuple, TypedDict, Union
import uuid

import httpx
import pymupdf

from src.config import settings
from src.logger import get_logger
from src.tools.arxiv_client import parse_query, strip_arxiv_version

logger = get_logger(__name__)

USER_AGENT = "arXiv-Paper-Digest-Agent/1.0 (academic research tool)"
MIN_PDF_SIZE_BYTES = 1024
MIN_BODY_WORD_COUNT = 200


class PDFDownloadError(Exception):
    """Raised when PDF retrieval fails due to network, 404, or corrupt content."""

    pass


class ParsedPaper(TypedDict):
    """Structured extraction output from an academic PDF document."""

    title: str
    abstract: str
    sections: Dict[str, str]  # e.g. {"Introduction": "...", "Methodology": "..."}
    references: List[str]
    full_text: str
    parsing_status: str  # "success" | "partial" | "metadata_only"


# Section heading detection patterns
_SECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Abstract", re.compile(r"^(?:[0-9IVXLCDM]+\.?\s*)?abstract\b", re.IGNORECASE)),
    ("Introduction", re.compile(r"^(?:[0-9IVXLCDM]+\.?\s*)?introduction\b", re.IGNORECASE)),
    (
        "Related Work",
        re.compile(r"^(?:[0-9IVXLCDM]+\.?\s*)?(?:related\s+work|background|prior\s+work)\b", re.IGNORECASE),
    ),
    (
        "Methodology",
        re.compile(
            r"^(?:[0-9IVXLCDM]+\.?\s*)?(?:method(?:ology|s)?|proposed\s+method|model\s+architecture|approach|framework|system\s+design)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Experiments",
        re.compile(
            r"^(?:[0-9IVXLCDM]+\.?\s*)?(?:experiments?|experimental\s+setup|implementation\s+details?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Results",
        re.compile(
            r"^(?:[0-9IVXLCDM]+\.?\s*)?(?:results?|evaluation|findings|empirical\s+results?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Discussion",
        re.compile(
            r"^(?:[0-9IVXLCDM]+\.?\s*)?(?:discussion|limitations?|ethical\s+considerations?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Conclusion",
        re.compile(
            r"^(?:[0-9IVXLCDM]+\.?\s*)?(?:conclusions?|concluding\s+remarks?|summary|future\s+work)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "References",
        re.compile(r"^(?:[0-9IVXLCDM]+\.?\s*)?(?:references|bibliography)\b", re.IGNORECASE),
    ),
]

_GENERIC_NUMBERED_HEADING = re.compile(
    r"^[0-9IVXLCDM]+\.?\s+([A-Z][A-Za-z0-9\s\-_]{2,50})$"
)


def sanitize_arxiv_id(arxiv_id: str) -> str:
    """Sanitize an arXiv ID into a filesystem-safe identifier without version suffix.

    Replaces slashes, colons, and illegal filesystem characters with underscores.
    e.g. 'quant-ph/0201082v1' -> 'quant-ph_0201082'
         '1706.03762v2' -> '1706.03762'

    Args:
        arxiv_id: Raw arXiv identifier or URL.

    Returns:
        str: Filesystem-safe canonical paper identifier.
    """
    if not arxiv_id or not arxiv_id.strip():
        return ""

    # Parse through query parser if it's a full URL or prefixed
    parsed = parse_query(arxiv_id)
    raw_id = parsed.clean_id if parsed.query_type == "arxiv_id" and parsed.clean_id else arxiv_id.strip()

    # Strip version suffix
    clean = strip_arxiv_version(raw_id)

    # Replace slashes and other unsafe characters
    sanitized = re.sub(r"[/\\?%*:|\"<>\s]", "_", clean)
    return sanitized


def get_pdf_cache_path(
    arxiv_id: str,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the local filesystem path for a cached paper PDF.

    Args:
        arxiv_id: Canonical or raw arXiv identifier.
        cache_dir: Optional directory override. Defaults to settings.pdf_cache_dir.

    Returns:
        Path: Target filepath for the cached PDF.
    """
    target_dir = Path(cache_dir or settings.pdf_cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_arxiv_id(arxiv_id)
    return target_dir / f"{sanitized}.pdf"


def download_pdf(
    arxiv_id: str,
    pdf_url: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> Path:
    """Retrieve an arXiv paper PDF, utilizing local disk cache when available.

    If a non-corrupt PDF exists in the cache directory (> 1024 bytes), it is
    loaded directly from disk without initiating network requests. Otherwise,
    the PDF is streamed from arXiv, written to a temporary file, validated,
    and atomically renamed.

    Args:
        arxiv_id: arXiv identifier (e.g. '1706.03762' or 'quant-ph/0201082').
        pdf_url: Optional direct PDF URL. If not provided, constructed from arxiv_id.
        cache_dir: Optional cache directory path override.
        force_download: If True, bypasses local cache and re-downloads.
        timeout: Network timeout in seconds for streaming download.
        max_retries: Number of retry attempts on transient network failures.

    Returns:
        Path: Path to the validated local PDF file.

    Raises:
        PDFDownloadError: If download fails after retries, returns HTTP error, or is corrupt.
    """
    if not arxiv_id or not arxiv_id.strip():
        raise PDFDownloadError("Cannot download PDF: Empty arXiv ID provided.")

    sanitized_id = sanitize_arxiv_id(arxiv_id)
    target_path = get_pdf_cache_path(sanitized_id, cache_dir=cache_dir)

    # 1. Local Cache Check
    if not force_download and target_path.exists():
        file_size = target_path.stat().st_size
        if file_size > MIN_PDF_SIZE_BYTES:
            logger.info(f"Cache hit for paper {sanitized_id}: '{target_path}' ({file_size} bytes)")
            return target_path
        else:
            logger.warning(
                f"Cached PDF for {sanitized_id} is suspiciously small ({file_size} bytes). Re-downloading."
            )

    # 2. Resolve target download URL
    clean_id = strip_arxiv_version(arxiv_id.strip())
    url = pdf_url or f"https://arxiv.org/pdf/{clean_id}.pdf"

    logger.info(f"Initiating streaming download for paper {sanitized_id} from {url}")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,*/*",
    }

    temp_filename = f"{sanitized_id}.tmp.{uuid.uuid4().hex}"
    temp_path = target_path.parent / temp_filename

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Download attempt {attempt}/{max_retries} for {sanitized_id}")
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 404:
                        raise PDFDownloadError(f"arXiv paper not found (HTTP 404): {url}")
                    response.raise_for_status()

                    bytes_written = 0
                    with open(temp_path, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=16384):
                            if chunk:
                                f.write(chunk)
                                bytes_written += len(chunk)

            if bytes_written < MIN_PDF_SIZE_BYTES:
                raise PDFDownloadError(
                    f"Downloaded file for {sanitized_id} is incomplete ({bytes_written} bytes)."
                )

            with open(temp_path, "rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF"):
                    raise PDFDownloadError(
                        f"Downloaded content for {sanitized_id} does not appear to be a valid PDF (header: {header!r})."
                    )

            os.replace(temp_path, target_path)
            logger.info(
                f"Successfully downloaded and cached paper {sanitized_id} to '{target_path}' ({bytes_written} bytes)"
            )
            return target_path

        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_error = exc
            logger.warning(
                f"Attempt {attempt}/{max_retries} failed for {sanitized_id} ({exc})."
            )
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

            if attempt < max_retries:
                sleep_seconds = attempt * 1.5
                logger.info(f"Retrying download in {sleep_seconds:.1f}s...")
                time.sleep(sleep_seconds)

        except PDFDownloadError:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

        except Exception as exc:
            last_error = exc
            logger.error(f"Unexpected error downloading {sanitized_id}: {exc}", exc_info=True)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            break

    if temp_path.exists():
        try:
            temp_path.unlink()
        except OSError:
            pass

    raise PDFDownloadError(
        f"Failed to download PDF for arXiv ID '{arxiv_id}' after {max_retries} attempts. Cause: {last_error}"
    )


def _is_noise_line(text: str, bbox: Tuple[float, float, float, float], page_height: float) -> bool:
    """Detect whether a line is header/footer noise, page number, or watermark.

    Args:
        text: Extracted text of the line.
        bbox: Bounding box tuple (x0, y0, x1, y1).
        page_height: Total height of the page in points.

    Returns:
        bool: True if the line should be filtered out.
    """
    clean = text.strip()
    if not clean:
        return True

    # arXiv stamp headers (e.g. arXiv:2401.12345v1 [cs.CL] 22 Jan 2024)
    if re.search(r"arxiv:\s*(?:\d{4}\.\d{4,5}|[a-z\-]+/\d{7})", clean, re.IGNORECASE):
        return True

    # Standalone page numbers or margin text
    is_top_margin = bbox[1] < page_height * 0.06
    is_bottom_margin = bbox[3] > page_height * 0.94

    if (is_top_margin or is_bottom_margin) and re.match(r"^\d{1,4}$", clean):
        return True

    # Common watermark/preprint notices in margins
    if (is_top_margin or is_bottom_margin) and re.search(
        r"under\s+review|preprint|proceedings\s+of|conference\s+on|published\s+as",
        clean,
        re.IGNORECASE,
    ):
        return True

    return False


def _match_section_heading(
    text: str,
    font_size: float,
    is_bold: bool,
    body_font_size: float,
) -> Optional[str]:
    """Determine if a line qualifies as a section heading.

    Args:
        text: Line text.
        font_size: Maximum font size in the line.
        is_bold: Whether the line contains bold weight text.
        body_font_size: Median body text font size of the document.

    Returns:
        Optional[str]: Canonical heading name if recognized, or None.
    """
    clean = text.strip()
    if not clean or len(clean) > 80:
        return None

    # Skip lines ending with sentence-ending punctuation (except numbered headings like '1.')
    if clean.endswith((".", ":", ";", "?", "!")) and not re.match(r"^[0-9IVXLCDM]+\.$", clean):
        return None

    is_prominent = font_size >= (body_font_size * 1.10) or (is_bold and font_size >= body_font_size)

    # 1. Match against known canonical section patterns
    for canonical_name, pattern in _SECTION_PATTERNS:
        if pattern.search(clean):
            # Abstract and References often have regular font size or bold
            if canonical_name in ["Abstract", "References"] or is_prominent:
                return canonical_name

    # 2. Match generic numbered heading if prominent
    if is_prominent:
        generic_match = _GENERIC_NUMBERED_HEADING.match(clean)
        if generic_match:
            heading_title = generic_match.group(1).strip()
            # Double check against canonical names
            for canonical_name, pattern in _SECTION_PATTERNS:
                if pattern.search(heading_title):
                    return canonical_name
            return heading_title

    return None


def parse_pdf(
    pdf_path: Union[str, Path],
    fallback_title: str = "",
    fallback_abstract: str = "",
) -> ParsedPaper:
    """Extract structured textual sections, abstract, and references from an academic PDF.

    Uses PyMuPDF to extract text spans and font metadata, calculates body font size,
    detects section headings dynamically, cleans header/footer artifacts, and
    partitions narrative content.

    Args:
        pdf_path: Absolute or relative path to the local PDF file.
        fallback_title: Optional title to use if unparsed or for scanned PDFs.
        fallback_abstract: Optional abstract to use if unparsed or for scanned PDFs.

    Returns:
        ParsedPaper: Structured document extraction containing:
            - title: Extracted or fallback title.
            - abstract: Extracted or fallback abstract.
            - sections: Dictionary mapping section headings to section text.
            - references: List of individual bibliographic reference entries.
            - full_text: Unified body text across all content sections.
            - parsing_status: "success", "partial", or "metadata_only".
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists() or path_obj.stat().st_size == 0:
        logger.warning(f"PDF does not exist or is empty: {pdf_path}")
        return ParsedPaper(
            title=fallback_title,
            abstract=fallback_abstract,
            sections={},
            references=[],
            full_text="",
            parsing_status="metadata_only",
        )

    try:
        doc = pymupdf.open(str(path_obj))
    except Exception as exc:
        logger.error(f"Failed to open PDF '{pdf_path}' with PyMuPDF: {exc}", exc_info=True)
        return ParsedPaper(
            title=fallback_title,
            abstract=fallback_abstract,
            sections={},
            references=[],
            full_text="",
            parsing_status="metadata_only",
        )

    if doc.page_count == 0:
        doc.close()
        return ParsedPaper(
            title=fallback_title,
            abstract=fallback_abstract,
            sections={},
            references=[],
            full_text="",
            parsing_status="metadata_only",
        )

    # 1. First pass: Collect all text spans to calculate document body font size & total words
    all_font_sizes: List[float] = []
    total_raw_words = 0

    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) == 0:  # Text block
                for line in block.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    words = line_text.split()
                    total_raw_words += len(words)
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if len(span_text) > 1:
                            all_font_sizes.append(round(span.get("size", 10.0), 1))

    # Defensive fallback for scanned / image-based PDFs
    if total_raw_words < MIN_BODY_WORD_COUNT or not all_font_sizes:
        logger.warning(
            f"PDF '{pdf_path}' contains only {total_raw_words} words (< {MIN_BODY_WORD_COUNT}). Marking as metadata_only."
        )
        doc.close()
        return ParsedPaper(
            title=fallback_title,
            abstract=fallback_abstract,
            sections={},
            references=[],
            full_text="",
            parsing_status="metadata_only",
        )

    body_font_size = statistics.median(all_font_sizes)
    logger.debug(f"Calculated body font size for '{path_obj.name}': {body_font_size} pt")

    # 2. Second pass: Structural section extraction
    extracted_title = ""
    candidate_title_lines: List[str] = []
    sections: Dict[str, List[str]] = {}
    current_section: str = "Header"
    sections[current_section] = []

    reference_lines: List[str] = []

    for page_idx, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_height = page.rect.height

        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                line_text = "".join(s.get("text", "") for s in spans).strip()
                if not line_text:
                    continue

                bbox = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
                if _is_noise_line(line_text, bbox, page_height):
                    continue

                max_size = max(s.get("size", body_font_size) for s in spans)
                is_bold = any((s.get("flags", 0) & 2 != 0) or ("bold" in s.get("font", "").lower()) for s in spans)

                # Capture title from page 1 before any section heading
                if page_idx == 0 and current_section == "Header" and max_size >= (body_font_size * 1.25):
                    candidate_title_lines.append(line_text)

                # Check if this line is a section heading
                detected_heading = _match_section_heading(line_text, max_size, is_bold, body_font_size)

                if detected_heading:
                    current_section = detected_heading
                    if current_section not in sections:
                        sections[current_section] = []
                    logger.debug(f"Detected heading '{current_section}' (from text: '{line_text}')")
                    continue

                # Accumulate content lines into active section
                if current_section == "References":
                    reference_lines.append(line_text)
                else:
                    sections[current_section].append(line_text)

    doc.close()

    # 3. Post-process Title
    if candidate_title_lines:
        extracted_title = " ".join(candidate_title_lines).strip()
    title = extracted_title if extracted_title else fallback_title

    # 4. Post-process Abstract
    abstract = ""
    if "Abstract" in sections and sections["Abstract"]:
        abstract = " ".join(sections["Abstract"]).strip()
    elif fallback_abstract:
        abstract = fallback_abstract
    elif "Header" in sections and sections["Header"]:
        # Try to find abstract text inside header block
        header_text = " ".join(sections["Header"])
        abs_match = re.search(r"abstract[:\s]*(.*?)(?:1\.?\s+introduction|introduction|$)", header_text, re.IGNORECASE)
        if abs_match:
            abstract = abs_match.group(1).strip()

    # Remove "Header" from standard sections dictionary
    if "Header" in sections:
        del sections["Header"]

    # Format sections as clean concatenated paragraphs
    formatted_sections: Dict[str, str] = {}
    for sec_name, lines in sections.items():
        if sec_name in ["Abstract", "References"]:
            continue
        sec_text = " ".join(lines).strip()
        # Normalize whitespace
        sec_text = re.sub(r"\s+", " ", sec_text)
        if sec_text:
            formatted_sections[sec_name] = sec_text

    # 5. Post-process References
    references: List[str] = []
    current_ref: List[str] = []

    for ref_line in reference_lines:
        # Check if line starts a new reference entry (e.g. [1], 1., or Author)
        if re.match(r"^\[\d+\]", ref_line) or re.match(r"^\d+\.\s+[A-Z]", ref_line):
            if current_ref:
                entry = " ".join(current_ref).strip()
                if len(entry) >= 15:
                    references.append(entry)
            current_ref = [ref_line]
        else:
            current_ref.append(ref_line)

    if current_ref:
        entry = " ".join(current_ref).strip()
        if len(entry) >= 15:
            references.append(entry)

    # 6. Build unified narrative full_text
    full_text_parts: List[str] = []
    if abstract:
        full_text_parts.append(f"Abstract\n{abstract}")
    for sec_name, sec_text in formatted_sections.items():
        full_text_parts.append(f"{sec_name}\n{sec_text}")

    full_text = "\n\n".join(full_text_parts)

    # 7. Determine parsing status
    # Success requires both substantive text and key identified sections
    if len(formatted_sections) >= 2 and len(full_text.split()) >= MIN_BODY_WORD_COUNT:
        parsing_status = "success"
    elif len(full_text.split()) >= MIN_BODY_WORD_COUNT:
        parsing_status = "partial"
    else:
        parsing_status = "metadata_only"

    logger.info(
        f"Parsed PDF '{path_obj.name}': status={parsing_status}, "
        f"sections={list(formatted_sections.keys())}, references={len(references)}, words={len(full_text.split())}"
    )

    return ParsedPaper(
        title=title,
        abstract=abstract,
        sections=formatted_sections,
        references=references,
        full_text=full_text,
        parsing_status=parsing_status,
    )
