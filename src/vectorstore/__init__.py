"""Vector store package for document chunking, embeddings, and Qdrant retrieval."""

from src.vectorstore.chunker import DocumentChunk, chunk_paper
from src.vectorstore.qdrant_store import (
    EMBEDDING_DIM,
    collection_exists,
    get_collection_name,
    get_embedding_model,
    get_qdrant_client,
    index_chunks,
    search_chunks,
)

__all__ = [
    "DocumentChunk",
    "chunk_paper",
    "EMBEDDING_DIM",
    "collection_exists",
    "get_collection_name",
    "get_embedding_model",
    "get_qdrant_client",
    "index_chunks",
    "search_chunks",
]
