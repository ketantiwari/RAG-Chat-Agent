from __future__ import annotations

from pathlib import Path
from embeddings.embedder import EmbeddingService
from parsers.pdf_parser import PDFParser
from rag.chunker import chunk_text
from vectorstore.faiss_store import FaissStore
from vectorstore.bm25_store import BM25Store


class IngestionService:
    def __init__(self) -> None:
        self.parser = PDFParser()
        self.embedder = EmbeddingService()
        self.faiss = FaissStore()
        self.bm25 = BM25Store()

    def ingest_pdf(self, file_path: Path) -> int:
        pages = self.parser.parse(file_path)
        all_texts: list[str] = []
        all_meta: list[dict] = []
        for page in pages:
            chunks = chunk_text(page["text"])
            for i, ch in enumerate(chunks):
                chunk_id = f"{file_path.name}-p{page['page']}-c{i}"
                meta = {
                    "doc_id": file_path.stem,
                    "filename": file_path.name,
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    "text": ch,
                }
                all_texts.append(ch)
                all_meta.append(meta)

        if not all_texts:
            return 0
        vectors = self.embedder.embed_texts(all_texts)
        self.faiss.add(vectors, all_meta)
        self.bm25.add(all_texts, all_meta)
        return len(all_texts)

    def reset_indexes(self) -> None:
        self.faiss.reset()
        self.bm25.reset()
