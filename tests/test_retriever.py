import pytest
from unittest.mock import MagicMock
from rag.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_routing(monkeypatch):
    # Mock dependencies
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]

    mock_faiss = MagicMock()
    mock_faiss.search.return_value = [
        {"chunk_id": "c1", "text": "artificial intelligence", "filename": "ai.pdf", "page": 1, "score": 0.8},
        {"chunk_id": "c2", "text": "deep learning models", "filename": "dl.pdf", "page": 2, "score": 0.7},
    ]

    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        {"chunk_id": "c2", "text": "deep learning models", "filename": "dl.pdf", "page": 2, "score": 1.2},
        {"chunk_id": "c3", "text": "natural language processing", "filename": "nlp.pdf", "page": 3, "score": 1.1},
    ]

    # Patch inside HybridRetriever
    monkeypatch.setattr("rag.hybrid_retriever.EmbeddingService", lambda: mock_embedder)
    monkeypatch.setattr("rag.hybrid_retriever.FaissStore", lambda: mock_faiss)
    monkeypatch.setattr("rag.hybrid_retriever.BM25Store", lambda: mock_bm25)

    retriever = HybridRetriever()
    results = retriever.retrieve("deep learning", top_k=2)

    # Check that search functions were called
    assert mock_embedder.embed_query.called
    assert mock_faiss.search.called
    assert mock_bm25.search.called

    # Assert correct fusion sorting (c2 should be first since it appears in both semantic and lexical)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c2"

    # Test filtering by active filename
    filtered_results = retriever.retrieve("deep learning", top_k=2, active_filename="ai.pdf")
    assert len(filtered_results) == 1
    assert filtered_results[0]["filename"] == "ai.pdf"
