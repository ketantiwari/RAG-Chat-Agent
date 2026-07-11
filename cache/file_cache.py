from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class FileCache:
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.base_dir = Path(settings.cache_dir) / namespace
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized FileCache directory for namespace '{namespace}' at {self.base_dir}")

    @staticmethod
    def stable_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.base_dir / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            logger.debug(f"Cache miss in namespace '{self.namespace}' for key: {key}")
            return None
        try:
            with path.open("rb") as f:
                data = pickle.load(f)
            logger.info(f"Cache hit in namespace '{self.namespace}' for key: {key}")
            return data
        except Exception as e:
            logger.error(f"Failed to read cache file {path}: {e}", exc_info=True)
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        try:
            with path.open("wb") as f:
                pickle.dump(value, f)
            logger.info(f"Cache set in namespace '{self.namespace}' for key: {key}")
        except Exception as e:
            logger.error(f"Failed to write cache file {path}: {e}", exc_info=True)

    def clear(self) -> None:
        logger.info(f"Clearing all cache entries in namespace '{self.namespace}' at {self.base_dir}")
        count = 0
        for p in self.base_dir.glob("*.pkl"):
            try:
                p.unlink(missing_ok=True)
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete cache file {p}: {e}", exc_info=True)
        logger.info(f"Cleared {count} cache entries in namespace '{self.namespace}'.")
