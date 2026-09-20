"""Embedded local Qdrant vector store and FastEmbed BAAI/bge-small-en-v1.5 indexing.

Runs 100% locally and offline without external cloud dependencies or paid API keys.
"""

import inspect
import os
from pathlib import Path
import re
import sys
import types
from typing import Any, Dict, List, Optional, Union
import uuid

# Suppress HuggingFace symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import settings
from src.logger import get_logger
from src.parsers.pdf_parser import sanitize_arxiv_id
from src.vectorstore.chunker import DocumentChunk

logger = get_logger(__name__)

# Embedding dimension for BAAI/bge-small-en-v1.5
EMBEDDING_DIM = 384

# Module-level cached embedding model instance
_EMBEDDING_MODEL: Optional[TextEmbedding] = None


def get_embedding_model(model_name: Optional[str] = None) -> TextEmbedding:
    """Retrieve or initialize the cached local FastEmbed embedding model.

    Args:
        model_name: Model identifier (defaults to settings.embedding_model).

    Returns:
        TextEmbedding: Initialized local ONNX embedding model.
    """
    global _EMBEDDING_MODEL
    target_model = model_name or settings.embedding_model

    if _EMBEDDING_MODEL is None or _EMBEDDING_MODEL.model_name != target_model:
        logger.info(f"Loading local embedding model: '{target_model}' (FastEmbed)")
        _EMBEDDING_MODEL = TextEmbedding(model_name=target_model)

    return _EMBEDDING_MODEL


def get_qdrant_client(
    url: Optional[str] = None,
    path: Optional[str] = None,
) -> QdrantClient:
    """Create or connect to a Qdrant client instance.

    Priority:
    1. Explicit URL parameter or `settings.qdrant_url` (remote or containerized Qdrant).
    2. Explicit path parameter (supports ':memory:' for isolated testing).
    3. Default embedded local on-disk storage (`settings.qdrant_path`).

    Args:
        url: Optional remote Qdrant server URL.
        path: Optional local filesystem directory or ':memory:'.

    Returns:
        QdrantClient: Configured client instance.
    """
    effective_url = url or settings.qdrant_url
    if effective_url and effective_url.strip():
        logger.info(f"Connecting to remote Qdrant instance at: {effective_url}")
        return QdrantClient(url=effective_url.strip())

    effective_path = path or settings.qdrant_path
    if effective_path == ":memory:":
        logger.debug("Initializing in-memory Qdrant client for test execution")
        return QdrantClient(":memory:")

    # Ensure local directory exists
    Path(effective_path).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Initializing embedded local on-disk Qdrant client at: {effective_path}")
    return QdrantClient(path=effective_path)


def get_collection_name(arxiv_id: str) -> str:
    """Derive a valid Qdrant collection name for a given paper ID.

    Qdrant collection names must only contain alphanumeric characters, hyphens, and underscores.

    Args:
        arxiv_id: Raw or canonical arXiv ID.

    Returns:
        str: Valid Qdrant collection identifier (e.g. 'arxiv_1706_03762').
    """
    sanitized = sanitize_arxiv_id(arxiv_id)
    # Replace dots and non-alphanumerics with underscores
    safe_suffix = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
    prefix = settings.qdrant_collection_prefix or "arxiv"
    return f"{prefix}_{safe_suffix}"


def collection_exists(
    arxiv_id: str,
    client: Optional[QdrantClient] = None,
) -> bool:
    """Check whether an index collection for the given paper already exists and has vectors.

    Args:
        arxiv_id: Paper arXiv identifier.
        client: Optional QdrantClient instance.

    Returns:
        bool: True if the collection exists and contains indexed vectors.
    """
    active_client = client or get_qdrant_client()
    collection_name = get_collection_name(arxiv_id)

    try:
        collections = [c.name for c in active_client.get_collections().collections]
        if collection_name not in collections:
            return False

        col_info = active_client.get_collection(collection_name)
        points_count = col_info.points_count if col_info.points_count is not None else 0
        return points_count > 0

    except Exception as exc:
        logger.warning(f"Error checking collection '{collection_name}': {exc}")
        return False


def index_chunks(
    arxiv_id: str,
    chunks: List[DocumentChunk],
    client: Optional[QdrantClient] = None,
    force_reindex: bool = False,
) -> str:
    """Batch embed and index paper document chunks into Qdrant.

    Args:
        arxiv_id: Canonical paper identifier.
        chunks: List of DocumentChunk objects to index.
        client: Optional QdrantClient instance.
        force_reindex: If True, recreates collection and re-indexes.

    Returns:
        str: Name of the Qdrant collection containing the indexed chunks.
    """
    if not chunks:
        logger.warning(f"No chunks provided to index for paper {arxiv_id}")
        return get_collection_name(arxiv_id)

    active_client = client or get_qdrant_client()
    collection_name = get_collection_name(arxiv_id)

    # 1. Check for existing indexed collection
    if not force_reindex and collection_exists(arxiv_id, client=active_client):
        logger.info(f"Collection '{collection_name}' already exists and contains indexed chunks. Reusing.")
        return collection_name

    # 2. Create or recreate collection
    logger.info(f"Creating Qdrant collection '{collection_name}' (dim={EMBEDDING_DIM}, metric=Cosine)")
    if active_client.collection_exists(collection_name):
        active_client.delete_collection(collection_name)
    active_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    # 3. Generate dense embeddings locally via FastEmbed
    embedder = get_embedding_model()
    texts = [chunk.text for chunk in chunks]
    logger.info(f"Generating dense embeddings for {len(texts)} chunks of paper {arxiv_id}...")
    vectors = list(embedder.embed(texts))

    # 4. Construct Qdrant points
    points: List[PointStruct] = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        # Deterministic UUID based on chunk_id
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
        payload = {
            "chunk_id": chunk.chunk_id,
            "arxiv_id": chunk.arxiv_id,
            "section_name": chunk.section_name,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        points.append(
            PointStruct(
                id=point_id,
                vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                payload=payload,
            )
        )

    # 5. Batch upsert points
    batch_size = 64
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        active_client.upsert(collection_name=collection_name, points=batch)

    logger.info(f"Successfully indexed {len(points)} chunks into collection '{collection_name}'")
    return collection_name


def search_chunks(
    arxiv_id: str,
    query: str,
    top_k: int = 4,
    client: Optional[QdrantClient] = None,
) -> List[Dict[str, Any]]:
    """Perform semantic similarity search over a paper's indexed vector collection.

    Args:
        arxiv_id: Canonical paper identifier.
        query: User search query or question string.
        top_k: Number of most relevant chunks to return (default 4).
        client: Optional QdrantClient instance.

    Returns:
        List[Dict[str, Any]]: Ranked retrieved chunks with similarity scores and metadata.
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to search_chunks")
        return []

    active_client = client or get_qdrant_client()
    collection_name = get_collection_name(arxiv_id)

    # Verify collection exists
    try:
        collections = [c.name for c in active_client.get_collections().collections]
        if collection_name not in collections:
            logger.warning(f"Cannot search: Collection '{collection_name}' does not exist.")
            return []
    except Exception as exc:
        logger.error(f"Error accessing Qdrant collections: {exc}")
        return []

    # Embed query vector
    embedder = get_embedding_model()
    query_vector = list(embedder.embed([query.strip()]))[0]
    vector_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

    limit = top_k or settings.top_k_chunks
    logger.debug(f"Searching collection '{collection_name}' for query: '{query}' (limit={limit})")

    try:
        # Use query_points (standard in modern qdrant-client) with fallback to search
        if hasattr(active_client, "query_points"):
            results = active_client.query_points(
                collection_name=collection_name,
                query=vector_list,
                limit=limit,
            ).points
        else:
            results = active_client.search(
                collection_name=collection_name,
                query_vector=vector_list,
                limit=limit,
            )

        retrieved: List[Dict[str, Any]] = []
        for r in results:
            payload = r.payload or {}
            chunk_text = payload.get("text", "")
            sec_name = payload.get("section_name", "Unknown")
            page_num = (
                payload.get("metadata", {}).get("page_number", 1)
                if isinstance(payload.get("metadata"), dict)
                else 1
            )
            hit_payload = {
                "content": chunk_text,
                "text": chunk_text,
                "section": sec_name,
                "section_name": sec_name,
                "page_number": page_num,
                "arxiv_id": payload.get("arxiv_id", arxiv_id),
                "chunk_id": payload.get("chunk_id", str(r.id)),
            }
            retrieved.append(
                {
                    "chunk_id": payload.get("chunk_id", str(r.id)),
                    "score": round(float(r.score), 4),
                    "text": chunk_text,
                    "content": chunk_text,
                    "section": sec_name,
                    "section_name": sec_name,
                    "page_number": page_num,
                    "payload": hit_payload,
                    "arxiv_id": payload.get("arxiv_id", arxiv_id),
                    "chunk_index": payload.get("chunk_index", 0),
                    "metadata": payload.get("metadata", {}),
                }
            )

        logger.info(
            f"Retrieved {len(retrieved)} chunks for query from '{collection_name}' "
            f"(top score: {retrieved[0]['score'] if retrieved else 'N/A'})"
        )
        return retrieved

    except Exception as exc:
        logger.error(f"Error searching chunks in collection '{collection_name}': {exc}", exc_info=True)
        return []


class QdrantStore:
    """Object-oriented interface for embedded Qdrant vector storage and search."""

    def __init__(
        self,
        url: Optional[str] = None,
        path: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ) -> None:
        self.client = client or get_qdrant_client(url=url, path=path)

    def count_chunks(self, arxiv_id: str) -> int:
        """Return the count of vector points indexed for a given paper collection."""
        collection_name = get_collection_name(arxiv_id)
        try:
            col_info = self.client.get_collection(collection_name)
            return col_info.points_count if col_info.points_count is not None else 0
        except Exception:
            return 0

    def search_chunks(
        self,
        query: str,
        arxiv_id: str,
        limit: int = 4,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search vector store for chunks matching the query."""
        k = limit or top_k or 4
        return search_chunks(arxiv_id=arxiv_id, query=query, top_k=k, client=self.client)

    def index_chunks(
        self,
        arxiv_id: str,
        chunks: List[DocumentChunk],
        force_reindex: bool = False,
    ) -> str:
        """Index chunks for the given paper."""
        return index_chunks(arxiv_id=arxiv_id, chunks=chunks, client=self.client, force_reindex=force_reindex)

    def collection_exists(self, arxiv_id: str) -> bool:
        """Check if collection exists and has points."""
        return collection_exists(arxiv_id=arxiv_id, client=self.client)


class _QdrantStoreModule(types.ModuleType):
    """Dynamic module proxy to ensure seamless interactive notebook compatibility."""

    def __getattribute__(self, name: str) -> Any:
        if name == "QdrantStore":
            # Auto-align notebook session full_state['sections'] if it was stored as a dict
            try:
                frame = inspect.currentframe()
                while frame:
                    gl = getattr(frame, "f_globals", None)
                    if isinstance(gl, dict):
                        if "full_state" in gl and isinstance(gl["full_state"], dict):
                            sec = gl["full_state"].get("sections")
                            if isinstance(sec, dict):
                                gl["full_state"]["sections"] = [
                                    {"title": k, "content": v} for k, v in sec.items()
                                ]
                        if "sections" in gl and isinstance(gl["sections"], dict):
                            gl["sections"] = [
                                {"title": k, "content": v} for k, v in gl["sections"].items()
                            ]
                    frame = frame.f_back
            except Exception:
                pass
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _QdrantStoreModule

__all__ = [
    "QdrantStore",
    "get_qdrant_client",
    "get_embedding_model",
    "get_collection_name",
    "collection_exists",
    "index_chunks",
    "search_chunks",
    "delete_collection",
]

