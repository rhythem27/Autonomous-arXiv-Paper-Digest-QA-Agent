"""Section-aware document chunker for academic research papers.

Preserves structural section boundaries, maintains sentence integrity, injects
contextual metadata headers, and produces rich payloads for vector search.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.parsers.pdf_parser import ParsedPaper

logger = get_logger(__name__)

# Sentence boundary splitting pattern (splits after period, exclamation mark, or question mark followed by whitespace)
_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


@dataclass
class DocumentChunk:
    """Represents a discrete text chunk tagged with structural section metadata."""

    chunk_id: str
    arxiv_id: str
    section_name: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _split_text_into_sentences(text: str) -> List[str]:
    """Break a section's text into clean sentence units.

    Args:
        text: Raw section text string.

    Returns:
        List[str]: List of sentence strings.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    raw_sentences = _SENTENCE_SPLIT_REGEX.split(clean_text)
    sentences: List[str] = []

    for s in raw_sentences:
        s_stripped = s.strip()
        if s_stripped:
            sentences.append(s_stripped)

    return sentences


def _chunk_sentence_list(
    sentences: List[str],
    max_chars: int,
    overlap_chars: int,
) -> List[str]:
    """Assemble sentences into overlapping text chunks adhering to character limits.

    Args:
        sentences: List of discrete sentence strings.
        max_chars: Maximum characters per chunk body.
        overlap_chars: Target character overlap between consecutive chunks.

    Returns:
        List[str]: List of chunk body text strings.
    """
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_length = 0

    for sentence in sentences:
        # If an individual sentence is extraordinarily long, break by words
        if len(sentence) > max_chars:
            words = sentence.split()
            word_subchunk: List[str] = []
            word_len = 0
            for word in words:
                if word_len + len(word) + 1 > max_chars and word_subchunk:
                    sub_str = " ".join(word_subchunk)
                    if current_sentences:
                        chunks.append(" ".join(current_sentences))
                        current_sentences = []
                        current_length = 0
                    chunks.append(sub_str)
                    word_subchunk = [word]
                    word_len = len(word)
                else:
                    word_subchunk.append(word)
                    word_len += len(word) + 1
            if word_subchunk:
                sentence = " ".join(word_subchunk)

        sent_len = len(sentence) + 1  # include space

        if current_length + sent_len > max_chars and current_sentences:
            # Emit current chunk
            chunk_body = " ".join(current_sentences).strip()
            chunks.append(chunk_body)

            # Compute overlap sentences to retain for the next chunk
            overlap_sentences: List[str] = []
            overlap_acc = 0
            for s in reversed(current_sentences):
                if overlap_acc + len(s) + 1 <= overlap_chars:
                    overlap_sentences.insert(0, s)
                    overlap_acc += len(s) + 1
                else:
                    break

            current_sentences = overlap_sentences + [sentence]
            current_length = sum(len(s) + 1 for s in current_sentences)
        else:
            current_sentences.append(sentence)
            current_length += sent_len

    if current_sentences:
        final_chunk = " ".join(current_sentences).strip()
        # Avoid duplicate chunk if it's identical to the previous one
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return chunks


def chunk_paper(
    parsed_paper: ParsedPaper,
    arxiv_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[DocumentChunk]:
    """Divide a parsed paper into section-aware chunks with contextual headers.

    Strictly preserves section boundaries (never merging across sections).
    Injects contextual header `[Paper: {title} | Section: {section_name}]\\n\\n`
    at the start of every chunk to enrich embedding and RAG retrieval semantics.

    Args:
        parsed_paper: Extracted paper content dictionary from PyMuPDF parser.
        arxiv_id: Canonical arXiv identifier.
        chunk_size: Optional character limit override (defaults to settings.chunk_size * 4).
        chunk_overlap: Optional character overlap override (defaults to settings.chunk_overlap * 4).

    Returns:
        List[DocumentChunk]: Ordered list of structured document chunks.
    """
    title = parsed_paper.get("title") or "Untitled Paper"
    status = parsed_paper.get("parsing_status", "success")

    # Character-based approximation derived from token settings (~4 chars per token)
    target_chunk_size = chunk_size or (settings.chunk_size * 4)
    target_overlap = chunk_overlap or (settings.chunk_overlap * 4)

    # Defensive handling for scanned / image-only PDFs
    if status == "metadata_only" or (not parsed_paper.get("sections") and not parsed_paper.get("abstract")):
        abstract_text = parsed_paper.get("abstract") or "Full text not available (scanned or image-based PDF)."
        header = f"[Paper: {title} | Section: Abstract]\n\n"
        chunk_text = header + abstract_text
        metadata = {
            "arxiv_id": arxiv_id,
            "title": title,
            "section_name": "Abstract",
            "chunk_index": 0,
            "section_chunk_index": 0,
            "metadata_only": True,
            "word_count": len(abstract_text.split()),
            "char_count": len(abstract_text),
        }
        return [
            DocumentChunk(
                chunk_id=f"{arxiv_id}_abstract_0",
                arxiv_id=arxiv_id,
                section_name="Abstract",
                chunk_index=0,
                text=chunk_text,
                metadata=metadata,
            )
        ]

    # Collect sections to chunk in logical academic order
    ordered_sections: List[Tuple[str, str]] = []

    # Include Abstract if available and not already inside sections dict
    abstract = parsed_paper.get("abstract", "").strip()
    if abstract and "Abstract" not in parsed_paper.get("sections", {}):
        ordered_sections.append(("Abstract", abstract))

    for sec_name, sec_text in parsed_paper.get("sections", {}).items():
        if sec_text and sec_text.strip():
            ordered_sections.append((sec_name, sec_text.strip()))

    chunks: List[DocumentChunk] = []
    global_chunk_index = 0

    for section_name, section_text in ordered_sections:
        sentences = _split_text_into_sentences(section_text)
        if not sentences:
            continue

        raw_chunks = _chunk_sentence_list(
            sentences=sentences,
            max_chars=target_chunk_size,
            overlap_chars=target_overlap,
        )

        section_slug = re.sub(r"[^a-zA-Z0-9]+", "_", section_name.lower()).strip("_")

        for sec_chunk_idx, body in enumerate(raw_chunks):
            # Injected contextual header
            header = f"[Paper: {title} | Section: {section_name}]\n\n"
            full_text = header + body

            metadata = {
                "arxiv_id": arxiv_id,
                "title": title,
                "section_name": section_name,
                "chunk_index": global_chunk_index,
                "section_chunk_index": sec_chunk_idx,
                "word_count": len(body.split()),
                "char_count": len(body),
                "metadata_only": False,
            }

            chunk_obj = DocumentChunk(
                chunk_id=f"{arxiv_id}_{section_slug}_{sec_chunk_idx}",
                arxiv_id=arxiv_id,
                section_name=section_name,
                chunk_index=global_chunk_index,
                text=full_text,
                metadata=metadata,
            )

            chunks.append(chunk_obj)
            global_chunk_index += 1

    logger.info(
        f"Chunked paper '{arxiv_id}' into {len(chunks)} chunks across {len(ordered_sections)} sections."
    )
    return chunks
