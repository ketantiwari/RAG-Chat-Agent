from __future__ import annotations

import time
from collections import defaultdict
from embeddings.embedder import EmbeddingService
from utils.config import settings
from utils.logger import get_logger
from vectorstore.faiss_store import FaissStore
from vectorstore.bm25_store import BM25Store, tokenize

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self) -> None:
        self.embedder = EmbeddingService()
        self.faiss = FaissStore()
        self.bm25 = BM25Store()

    def retrieve(self, query: str, top_k: int = settings.top_k, active_filename: str | None = None) -> list[dict]:
        start = time.perf_counter()
        qv = self.embedder.embed_query(query)
        sem = self.faiss.search(qv, top_k=top_k * 2)
        lex = self.bm25.search(query, top_k=top_k * 2)
        if active_filename:
            target = active_filename.strip().lower()
            sem = [x for x in sem if x.get("filename", "").strip().lower() == target]
            lex = [x for x in lex if x.get("filename", "").strip().lower() == target]
        fused = self._fuse_and_rerank(query, sem, lex, top_k)
        logger.info(
            "Retrieved context",
            extra={
                "event": "retrieval_latency",
                "meta": {"ms": (time.perf_counter() - start) * 1000, "semantic_hits": len(sem), "bm25_hits": len(lex), "final_hits": len(fused)},
            },
        )
        return fused

    def _fuse_and_rerank(self, query: str, sem: list[dict], lex: list[dict], top_k: int) -> list[dict]:
        fused: dict[str, dict] = defaultdict(dict)
        query_tokens = set(tokenize(query))
        for rank, item in enumerate(sem, start=1):
            cid = item["chunk_id"]
            fused[cid] = dict(item)
            fused[cid]["semantic_score"] = item["score"]
            fused[cid]["rrf"] = fused[cid].get("rrf", 0.0) + 1.0 / (60 + rank)
            fused[cid]["source"] = "semantic"
        for rank, item in enumerate(lex, start=1):
            cid = item["chunk_id"]
            if cid not in fused:
                fused[cid] = dict(item)
            fused[cid]["bm25_score"] = item["score"]
            fused[cid]["rrf"] = fused[cid].get("rrf", 0.0) + 1.0 / (60 + rank)
            fused[cid]["source"] = "hybrid"

        for chunk in fused.values():
            s = chunk.get("semantic_score", 0.0)
            b = chunk.get("bm25_score", 0.0)
            overlap = len(query_tokens.intersection(set(tokenize(chunk["text"])))) / max(1, len(query_tokens))
            chunk["score"] = (settings.semantic_weight * s) + (settings.bm25_weight * b) + 0.1 * overlap + 0.15 * chunk.get("rrf", 0.0)
            chunk["retrieval_source"] = chunk.get("source", "hybrid")

        return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
