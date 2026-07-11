from __future__ import annotations

import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Store:
    def __init__(self) -> None:
        self.path = Path(settings.faiss_dir) / "bm25.pkl"
        self.texts: list[str] = []
        self.metadata: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not self.path.exists():
            logger.info("No existing BM25 index found. Initialized clean BM25 store.")
            return
        try:
            with self.path.open("rb") as f:
                payload = pickle.load(f)
            self.texts = payload["texts"]
            self.metadata = payload["metadata"]
            self.bm25 = BM25Okapi([tokenize(t) for t in self.texts]) if self.texts else None
            logger.info(f"Loaded existing BM25 store from {self.path}. Total documents: {len(self.texts)}")
        except Exception as e:
            logger.error(f"Failed to load existing BM25 store: {e}", exc_info=True)

    def add(self, texts: list[str], metadata: list[dict]) -> None:
        if not texts:
            logger.warning("Attempted to add empty texts list to BM25 store.")
            return
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        logger.info(f"Added {len(texts)} documents to BM25 store. New total: {len(self.texts)}")
        self.save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.bm25 is None or not self.texts:
            logger.info("BM25 search query ignored: store is empty.")
            return []
        logger.debug(f"Searching BM25 store for query: '{query}' (top_k={top_k})")
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        hits = [{**self.metadata[i], "score": float(s)} for i, s in ranked]
        logger.info(f"BM25 search completed. Found {len(hits)} hits.")
        return hits

    def save(self) -> None:
        try:
            with self.path.open("wb") as f:
                pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)
            logger.info(f"Saved BM25 store. Path: {self.path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 store: {e}", exc_info=True)

    def reset(self) -> None:
        self.texts = []
        self.metadata = []
        self.bm25 = None
        logger.info("BM25 store reset to clean state.")
        self.save()
