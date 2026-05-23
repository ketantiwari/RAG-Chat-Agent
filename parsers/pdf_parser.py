from __future__ import annotations

import fitz
from pathlib import Path
from cache.file_cache import FileCache


class PDFParser:
    def __init__(self) -> None:
        self.cache = FileCache("parsed_pdf")

    def parse(self, file_path: Path) -> list[dict]:
        raw = file_path.read_bytes()
        cache_key = self.cache.stable_hash(raw.hex())
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        pages: list[dict] = []
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i + 1, "text": text})
        doc.close()
        self.cache.set(cache_key, pages)
        return pages
