"""Unit and integration tests for embedded Qdrant local storage and FastEmbed indexing."""

import pytest

from src.vectorstore.chunker import DocumentChunk
from src.vectorstore.qdrant_store import (
    EMBEDDING_DIM,
    collection_exists,
    get_collection_name,
    get_embedding_model,
    get_qdrant_client,
    index_chunks,
    search_chunks,
)


@pytest.fixture
def in_memory_qdrant():
    """Provide an isolated in-memory Qdrant client for fast unit tests."""
    return get_qdrant_client(path=":memory:")


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Provide sample DocumentChunks across multiple academic sections."""
    return [
        DocumentChunk(
            chunk_id="1706.03762_intro_0",
            arxiv_id="1706.03762",
            section_name="Introduction",
            chunk_index=0,
            text=(
                "[Paper: Attention Is All You Need | Section: Introduction]\n\n"
                "Recurrent neural networks and LSTMs have been firmly established as state of the art "
                "approaches in sequence modeling. Sequential computation inhibits parallelization during training."
            ),
            metadata={"arxiv_id": "1706.03762", "section_name": "Introduction", "title": "Attention Is All You Need"},
        ),
        DocumentChunk(
            chunk_id="1706.03762_method_0",
            arxiv_id="1706.03762",
            section_name="Methodology",
            chunk_index=1,
            text=(
                "[Paper: Attention Is All You Need | Section: Methodology]\n\n"
                "The Transformer architecture uses stacked self-attention and point-wise fully connected layers. "
                "Multi-head attention maps queries, keys, and values to d_k dimensional representation subspaces."
            ),
            metadata={"arxiv_id": "1706.03762", "section_name": "Methodology", "title": "Attention Is All You Need"},
        ),
        DocumentChunk(
            chunk_id="1706.03762_results_0",
            arxiv_id="1706.03762",
            section_name="Results",
            chunk_index=2,
            text=(
                "[Paper: Attention Is All You Need | Section: Results]\n\n"
                "On the WMT 2014 English-to-German translation task, the big transformer model achieves a state of the art "
                "BLEU score of 28.4, outperforming the best previously reported ensembles by over 2.0 BLEU."
            ),
            metadata={"arxiv_id": "1706.03762", "section_name": "Results", "title": "Attention Is All You Need"},
        ),
    ]


class TestQdrantStoreUnit:
    """Unit tests for configuration, collection naming, and embedding dimension."""

    def test_get_collection_name(self):
        """Verify collection naming adheres to Qdrant alphanumeric rules."""
        assert get_collection_name("1706.03762") == "arxiv_1706_03762"
        assert get_collection_name("quant-ph/0201082") == "arxiv_quant-ph_0201082"
        assert get_collection_name("2401.12345v2") == "arxiv_2401_12345"

    def test_embedding_model_properties(self):
        """Verify FastEmbed model produces 384-dimensional dense embeddings."""
        embedder = get_embedding_model()
        vectors = list(embedder.embed(["Test text for embedding verification."]))
        assert len(vectors) == 1
        assert len(vectors[0]) == EMBEDDING_DIM
        assert EMBEDDING_DIM == 384

    def test_in_memory_client_creation(self, in_memory_qdrant):
        """Verify in-memory client creates without error."""
        collections = in_memory_qdrant.get_collections().collections
        assert isinstance(collections, list)


class TestQdrantIndexingAndSearch:
    """Integration tests verifying end-to-end embedding, indexing, and retrieval."""

    def test_index_and_semantic_retrieval(self, in_memory_qdrant, sample_chunks):
        """Verify chunks are indexed and semantically retrieved with high relevance."""
        arxiv_id = "1706.03762"

        # 1. Index chunks
        col_name = index_chunks(arxiv_id, sample_chunks, client=in_memory_qdrant)
        assert col_name == "arxiv_1706_03762"

        # 2. Check collection existence
        assert collection_exists(arxiv_id, client=in_memory_qdrant) is True

        # 3. Query methodology / multi-head attention
        results = search_chunks(
            arxiv_id=arxiv_id,
            query="How does multi-head attention map queries, keys, and values?",
            top_k=2,
            client=in_memory_qdrant,
        )

        assert len(results) == 2
        # Top result must be Methodology
        top_result = results[0]
        assert top_result["section_name"] == "Methodology"
        assert "Multi-head attention" in top_result["text"]
        assert top_result["score"] > 0.5  # High cosine similarity
        assert top_result["arxiv_id"] == "1706.03762"

        # 4. Query translation results / BLEU score
        bleu_results = search_chunks(
            arxiv_id=arxiv_id,
            query="What is the BLEU score on WMT 2014 translation benchmark?",
            top_k=1,
            client=in_memory_qdrant,
        )
        assert len(bleu_results) == 1
        assert bleu_results[0]["section_name"] == "Results"
        assert "BLEU score of 28.4" in bleu_results[0]["text"]

    def test_collection_reuse_avoids_reindexing(self, in_memory_qdrant, sample_chunks):
        """Verify index_chunks returns existing collection name without re-indexing."""
        arxiv_id = "1706.03762"
        col_1 = index_chunks(arxiv_id, sample_chunks, client=in_memory_qdrant)
        assert col_1 == "arxiv_1706_03762"

        # Re-run index_chunks without force_reindex
        col_2 = index_chunks(arxiv_id, sample_chunks, client=in_memory_qdrant, force_reindex=False)
        assert col_2 == col_1

    def test_search_empty_query_or_missing_collection(self, in_memory_qdrant):
        """Verify searching empty query or missing collection handles defensively."""
        assert search_chunks("1706.03762", "", client=in_memory_qdrant) == []
        assert search_chunks("9999.99999", "Some query", client=in_memory_qdrant) == []
