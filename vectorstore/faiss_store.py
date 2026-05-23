from __future__ import annotations

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import Any

from utils.config import settings


class FaissStore:
    def __init__(self, dim: int = 384) -> None:
        self.index_path = Path(settings.faiss_dir) / "docs.index"
        self.meta_path = Path(settings.faiss_dir) / "docs_meta.pkl"
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: list[dict[str, Any]] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.meta_path.exists():
            with self.meta_path.open("rb") as f:
                self.metadata = pickle.load(f)

    def add(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if embeddings.size == 0:
            return
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadata)
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(np.array([query_vector], dtype="float32"), top_k)
        hits: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(1.0 / (1.0 + dist))
            hits.append(item)
        return hits

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("wb") as f:
            pickle.dump(self.metadata, f)

    def reset(self) -> None:
        dim = self.index.d
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        self.save()
