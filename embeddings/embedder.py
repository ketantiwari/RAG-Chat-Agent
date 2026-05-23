from __future__ import annotations

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from cache.file_cache import FileCache
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.embedding_model)
        self.cache = FileCache("embeddings")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            key = self.cache.stable_hash(text)
            cached = self.cache.get(key)
            if cached is not None:
                vectors.append(cached)
                continue
            start = time.perf_counter()
            vector = self.model.encode([text], normalize_embeddings=True)[0]
            self.cache.set(key, vector)
            logger.info("Embedding generated", extra={"event": "embedding_latency", "meta": {"ms": (time.perf_counter() - start) * 1000}})
            vectors.append(vector)
        return np.array(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
