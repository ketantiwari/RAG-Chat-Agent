from __future__ import annotations

import time
import numpy as np
import google.generativeai as genai
from cache.file_cache import FileCache
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
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
            try:
                genai.configure(api_key=settings.gemini_api_key)
                response = genai.embed_content(
                    model=settings.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                vector = np.array(response['embedding'], dtype="float32")
                self.cache.set(key, vector)
                logger.info(
                    "Embedding generated via Gemini API", 
                    extra={"event": "embedding_latency", "meta": {"ms": (time.perf_counter() - start) * 1000}}
                )
                vectors.append(vector)
            except Exception as e:
                logger.error(f"Failed to generate embedding via Gemini API: {e}", exc_info=True)
                # Fallback vector of zeros matching FAISS dimension
                vectors.append(np.zeros(768, dtype="float32"))
                
        return np.array(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        key = self.cache.stable_hash(f"query||{query}")
        cached = self.cache.get(key)
        if cached is not None:
            return cached
            
        start = time.perf_counter()
        try:
            genai.configure(api_key=settings.gemini_api_key)
            response = genai.embed_content(
                model=settings.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            vector = np.array(response['embedding'], dtype="float32")
            self.cache.set(key, vector)
            logger.info(
                "Query embedding generated via Gemini API", 
                extra={"event": "embedding_latency", "meta": {"ms": (time.perf_counter() - start) * 1000}}
            )
            return vector
        except Exception as e:
            logger.error(f"Failed to generate query embedding via Gemini API: {e}", exc_info=True)
            return np.zeros(768, dtype="float32")
