from __future__ import annotations

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import Any

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class FaissStore:
    def __init__(self, dim: int = 768) -> None:
        self.index_path = Path(settings.faiss_dir) / "docs.index"
        self.meta_path = Path(settings.faiss_dir) / "docs_meta.pkl"
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: list[dict[str, Any]] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with self.meta_path.open("rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index from {self.index_path}. Total vectors: {self.index.ntotal}")
            except Exception as e:
                logger.error(f"Failed to load existing FAISS index: {e}", exc_info=True)
        else:
            logger.info("No existing FAISS index found. Initialized clean index.")

    def add(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if embeddings.size == 0:
            logger.warning("Attempted to add empty embeddings array to FAISS store.")
            return
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadata)
        logger.info(f"Added {len(embeddings)} vectors to FAISS index. New total: {self.index.ntotal}")
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            logger.info("FAISS search query ignored: index is empty.")
            return []
        logger.debug(f"Searching FAISS index for top_k={top_k}")
        distances, indices = self.index.search(np.array([query_vector], dtype="float32"), top_k)
        hits: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(1.0 / (1.0 + dist))
            hits.append(item)
        logger.info(f"FAISS search completed. Found {len(hits)} hits.")
        return hits

    def save(self) -> None:
        try:
            faiss.write_index(self.index, str(self.index_path))
            with self.meta_path.open("wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved FAISS index and metadata. Index path: {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index/metadata: {e}", exc_info=True)

    def reset(self) -> None:
        dim = self.index.d
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        logger.info("FAISS index reset to clean state.")
        self.save()
