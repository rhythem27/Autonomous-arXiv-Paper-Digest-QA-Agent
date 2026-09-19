"""arXiv client, paper metadata extraction, and query normalization tools.

Provides official arXiv Atom API integration with rate-limiting, retries,
defensive error handling, and query intent classification.
"""

import re
from typing import List, Literal, NamedTuple, Optional, TypedDict

import arxiv

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# Regular expression patterns for arXiv IDs and URLs
# Modern ID pattern: YYMM.number (4 or 5 digits), optional version suffix (v1, v2)
# e.g., 2401.12345, 1706.03762v2
_MODERN_ID_PATTERN = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$")

# Legacy ID pattern: category/YYMMnumber or category.sub/YYMMnumber, optional version suffix
# e.g., quant-ph/0201082v1, math.PR/0501001, cs/0112017
_LEGACY_ID_PATTERN = re.compile(r"^([a-zA-Z\-]+(?:\.[a-zA-Z]{2})?/\d{7})(?:v\d+)?$")

# arXiv URL pattern: https://arxiv.org/abs/... or /pdf/..., with optional .pdf and fragments
_ARXIV_URL_PATTERN = re.compile(
    r"^https?://arxiv\.org/(?:abs|pdf)/([a-zA-Z0-9.\-_/]+?)(?:\.pdf)?(?:[?#].*)?$",
    re.IGNORECASE,
)

# Optional "arxiv:" prefix pattern (e.g., "arXiv: 2401.12345" or "arxiv:1706.03762")
_ARXIV_PREFIX_PATTERN = re.compile(r"^arxiv:\s*", re.IGNORECASE)

# Version suffix removal pattern
_VERSION_SUFFIX_PATTERN = re.compile(r"v\d+$", re.IGNORECASE)


class ParsedQuery(NamedTuple):
    """Normalized query result representing intent and extracted data.

    Subclasses tuple to seamlessly support tuple unpacking:
    `query_type, clean_id, search_query = parse_query(...)`
    as well as attribute access and direct tuple comparison:
    `parse_query("...") == ("arxiv_id", "2401.12345", None)`
    """

    query_type: Literal["arxiv_id", "topic_search"]
    clean_id: Optional[str]
    search_query: Optional[str]


class PaperMetadata(TypedDict):
    """Structured metadata record for an arXiv research paper."""

    arxiv_id: str
    title: str
    authors: List[str]
    published: str
    updated: str
    abstract: str
    pdf_url: str
    primary_category: str
    categories: List[str]
    entry_id: str


def strip_arxiv_version(arxiv_id: str) -> str:
    """Strip the version suffix from an arXiv identifier (e.g., '1706.03762v2' -> '1706.03762').

    Args:
        arxiv_id: Raw arXiv ID string.

    Returns:
        str: Canonical arXiv ID without version suffix.
    """
    return _VERSION_SUFFIX_PATTERN.sub("", arxiv_id.strip())


def parse_query(raw_query: str) -> ParsedQuery:
    """Parse and classify an incoming user query into an arXiv ID lookup or topic search.

    Recognizes:
    1. Direct modern arXiv IDs (e.g. '2401.12345', '1706.03762v2').
    2. Legacy arXiv IDs (e.g. 'quant-ph/0201082v1', 'cs/0112017').
    3. arXiv URLs (e.g. 'https://arxiv.org/abs/1706.03762v1', 'http://arxiv.org/pdf/2401.12345.pdf').
    4. Strings with 'arxiv:' prefix (e.g. 'arXiv: 2401.12345').
    5. Natural language research queries (e.g. 'flash attention algorithms').

    Args:
        raw_query: Raw user input string.

    Returns:
        ParsedQuery: NamedTuple containing:
            - query_type: 'arxiv_id' or 'topic_search'
            - clean_id: Canonical paper ID without version suffix if 'arxiv_id', else None
            - search_query: Cleaned topic keywords if 'topic_search', else None
    """
    if not raw_query or not raw_query.strip():
        logger.debug("Received empty query, default to empty topic search")
        return ParsedQuery(query_type="topic_search", clean_id=None, search_query="")

    query = raw_query.strip()

    # 1. Check if query is an arXiv URL
    url_match = _ARXIV_URL_PATTERN.match(query)
    if url_match:
        extracted_id = url_match.group(1).strip()
        clean_id = strip_arxiv_version(extracted_id)
        logger.info(f"Classified query as arXiv URL, extracted ID: {clean_id}")
        return ParsedQuery(query_type="arxiv_id", clean_id=clean_id, search_query=None)

    # 2. Check and strip optional 'arxiv:' prefix if present
    prefixed = False
    cleaned_query = query
    if _ARXIV_PREFIX_PATTERN.match(query):
        cleaned_query = _ARXIV_PREFIX_PATTERN.sub("", query).strip()
        prefixed = True

    # 3. Check for modern arXiv ID
    modern_match = _MODERN_ID_PATTERN.match(cleaned_query)
    if modern_match:
        clean_id = modern_match.group(1)
        logger.info(f"Classified query as modern arXiv ID: {clean_id}")
        return ParsedQuery(query_type="arxiv_id", clean_id=clean_id, search_query=None)

    # 4. Check for legacy arXiv ID
    legacy_match = _LEGACY_ID_PATTERN.match(cleaned_query)
    if legacy_match:
        clean_id = legacy_match.group(1)
        logger.info(f"Classified query as legacy arXiv ID: {clean_id}")
        return ParsedQuery(query_type="arxiv_id", clean_id=clean_id, search_query=None)

    # 5. If it had an explicit 'arxiv:' prefix but didn't match patterns, try stripping version anyway
    if prefixed and "/" in cleaned_query:
        clean_id = strip_arxiv_version(cleaned_query)
        logger.info(f"Classified prefixed query as arXiv ID: {clean_id}")
        return ParsedQuery(query_type="arxiv_id", clean_id=clean_id, search_query=None)

    # 6. Otherwise, treat as natural language topic search
    normalized_topic = re.sub(r"\s+", " ", query)
    logger.info(f"Classified query as topic search: '{normalized_topic}'")
    return ParsedQuery(query_type="topic_search", clean_id=None, search_query=normalized_topic)


def get_arxiv_client(
    page_size: int = 10,
    delay_seconds: float = 3.0,
    num_retries: int = 3,
) -> arxiv.Client:
    """Create a resilient, rate-limited official arXiv API client.

    Adheres to official arXiv policy (<= 1 request per 3 seconds) and automatic
    exponential backoff retry behavior on transient HTTP errors.

    Args:
        page_size: Number of results per API pagination page.
        delay_seconds: Delay between successive API calls (default 3.0s).
        num_retries: Number of retries on network/HTTP failures.

    Returns:
        arxiv.Client: Configured client instance.
    """
    return arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_seconds,
        num_retries=num_retries,
    )


def _result_to_metadata(result: arxiv.Result) -> PaperMetadata:
    """Convert an arxiv.Result object into a standardized PaperMetadata dictionary.

    Args:
        result: Result instance from official arxiv client.

    Returns:
        PaperMetadata: Cleaned, structured paper metadata dictionary.
    """
    clean_id = strip_arxiv_version(result.get_short_id())
    clean_title = re.sub(r"\s+", " ", result.title).strip()
    clean_abstract = re.sub(r"\s+", " ", result.summary).strip()
    author_names = [author.name for author in result.authors]

    published_str = (
        result.published.isoformat()
        if hasattr(result.published, "isoformat")
        else str(result.published)
    )
    updated_str = (
        result.updated.isoformat()
        if hasattr(result.updated, "isoformat")
        else str(result.updated)
    )

    return PaperMetadata(
        arxiv_id=clean_id,
        title=clean_title,
        authors=author_names,
        published=published_str,
        updated=updated_str,
        abstract=clean_abstract,
        pdf_url=result.pdf_url or f"https://arxiv.org/pdf/{clean_id}.pdf",
        primary_category=result.primary_category or (result.categories[0] if result.categories else ""),
        categories=list(result.categories) if result.categories else [],
        entry_id=result.entry_id or f"http://arxiv.org/abs/{clean_id}",
    )


def fetch_paper_by_id(
    arxiv_id: str,
    client: Optional[arxiv.Client] = None,
) -> Optional[PaperMetadata]:
    """Retrieve metadata for a specific paper by its arXiv ID or URL.

    Args:
        arxiv_id: arXiv identifier (e.g. '1706.03762', '2401.12345v1', or full arXiv URL).
        client: Optional pre-configured arxiv.Client instance.

    Returns:
        Optional[PaperMetadata]: Metadata dictionary if found, None otherwise.
    """
    if not arxiv_id or not arxiv_id.strip():
        logger.warning("Empty arXiv ID provided to fetch_paper_by_id")
        return None

    # Resolve canonical clean ID via query parser or version stripper
    parsed = parse_query(arxiv_id)
    target_id = parsed.clean_id if parsed.query_type == "arxiv_id" and parsed.clean_id else strip_arxiv_version(arxiv_id)

    logger.info(f"Fetching paper metadata for arXiv ID: {target_id}")
    search = arxiv.Search(id_list=[target_id], max_results=1)
    active_client = client or get_arxiv_client()

    try:
        results = list(active_client.results(search))
        if not results:
            logger.warning(f"No paper found on arXiv for ID: {target_id}")
            return None

        metadata = _result_to_metadata(results[0])
        logger.info(f"Successfully retrieved paper: '{metadata['title']}' ({metadata['arxiv_id']})")
        return metadata

    except Exception as exc:
        logger.error(f"Error querying arXiv API for ID '{target_id}': {exc}", exc_info=True)
        return None


def search_papers_by_topic(
    topic: str,
    max_results: Optional[int] = None,
    client: Optional[arxiv.Client] = None,
) -> List[PaperMetadata]:
    """Search arXiv for papers relevant to a natural-language topic query.

    Args:
        topic: Natural language search string (e.g. 'KV-cache compression for LLMs').
        max_results: Maximum number of papers to retrieve (defaults to settings.arxiv_max_results).
        client: Optional pre-configured arxiv.Client instance.

    Returns:
        List[PaperMetadata]: Ranked list of paper metadata records matching the query.
    """
    if not topic or not topic.strip():
        logger.warning("Empty topic query provided to search_papers_by_topic")
        return []

    limit = max_results or settings.arxiv_max_results
    cleaned_topic = topic.strip()
    logger.info(f"Searching arXiv for topic: '{cleaned_topic}' (max_results={limit})")

    search = arxiv.Search(
        query=cleaned_topic,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    active_client = client or get_arxiv_client()

    try:
        results = list(active_client.results(search))
        papers = [_result_to_metadata(r) for r in results]
        logger.info(f"Retrieved {len(papers)} candidate papers for topic '{cleaned_topic}'")
        return papers

    except Exception as exc:
        logger.error(f"Error searching arXiv for topic '{cleaned_topic}': {exc}", exc_info=True)
        return []
