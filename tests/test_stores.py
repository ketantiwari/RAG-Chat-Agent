import numpy as np
import pytest
from vectorstore.faiss_store import FaissStore
from vectorstore.bm25_store import BM25Store, tokenize


def test_tokenize():
    assert tokenize("Hello, World 123!") == ["hello", "world", "123"]


def test_faiss_store_lifecycle():
    # Dimension 4
    store = FaissStore(dim=4)
    assert store.index.ntotal == 0
    assert len(store.metadata) == 0

    # Add mock vectors
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype="float32")
    metadata = [
        {"chunk_id": "c1", "text": "alpha", "page": 1, "filename": "doc.pdf"},
        {"chunk_id": "c2", "text": "beta", "page": 1, "filename": "doc.pdf"},
        {"chunk_id": "c3", "text": "gamma", "page": 2, "filename": "doc.pdf"},
    ]
    store.add(vectors, metadata)
    assert store.index.ntotal == 3
    assert len(store.metadata) == 3

    # Search (query vector close to alpha)
    query = np.array([1.0, 0.1, 0.0, 0.0], dtype="float32")
    hits = store.search(query, top_k=2)
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "c1"

    # Reload store from file to test persistence
    store2 = FaissStore(dim=4)
    assert store2.index.ntotal == 3
    assert len(store2.metadata) == 3

    # Reset
    store2.reset()
    assert store2.index.ntotal == 0
    assert len(store2.metadata) == 0


def test_bm25_store_lifecycle():
    store = BM25Store()
    assert store.bm25 is None
    assert len(store.texts) == 0

    # Add text documents
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence and machine learning",
        "hybrid retrieval combines lexical and semantic search",
    ]
    metadata = [
        {"chunk_id": "c1", "page": 1, "filename": "doc.pdf"},
        {"chunk_id": "c2", "page": 2, "filename": "doc.pdf"},
        {"chunk_id": "c3", "page": 3, "filename": "doc.pdf"},
    ]
    store.add(texts, metadata)
    assert len(store.texts) == 3
    assert store.bm25 is not None

    # Search
    hits = store.search("lexical semantic search", top_k=2)
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "c3"

    # Reload persistence test
    store2 = BM25Store()
    assert len(store2.texts) == 3

    # Reset
    store2.reset()
    assert len(store2.texts) == 0
    assert store2.bm25 is None
