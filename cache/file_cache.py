from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

from utils.config import settings


class FileCache:
    def __init__(self, namespace: str) -> None:
        self.base_dir = Path(settings.cache_dir) / namespace
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def stable_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.base_dir / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    def set(self, key: str, value: Any) -> None:
        with self._path(key).open("wb") as f:
            pickle.dump(value, f)

    def clear(self) -> None:
        for p in self.base_dir.glob("*.pkl"):
            p.unlink(missing_ok=True)
