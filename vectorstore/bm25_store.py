from __future__ import annotations

import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from utils.config import settings


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
            return
        with self.path.open("rb") as f:
            payload = pickle.load(f)
        self.texts = payload["texts"]
        self.metadata = payload["metadata"]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts]) if self.texts else None

    def add(self, texts: list[str], metadata: list[dict]) -> None:
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.bm25 is None or not self.texts:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{**self.metadata[i], "score": float(s)} for i, s in ranked]

    def save(self) -> None:
        with self.path.open("wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)

    def reset(self) -> None:
        self.texts = []
        self.metadata = []
        self.bm25 = None
        self.save()
