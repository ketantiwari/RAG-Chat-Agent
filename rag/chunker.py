from __future__ import annotations

from utils.config import settings


def chunk_text(text: str, chunk_size: int = settings.chunk_size, overlap: int = settings.chunk_overlap) -> list[str]:
    if not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, 0)
    return chunks
