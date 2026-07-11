from __future__ import annotations

from pathlib import Path
from embeddings.embedder import EmbeddingService
from parsers.pdf_parser import PDFParser
from rag.chunker import chunk_text
from vectorstore.faiss_store import FaissStore
from vectorstore.bm25_store import BM25Store
from utils.logger import get_logger

logger = get_logger(__name__)


class IngestionService:
    def __init__(self) -> None:
        self.parser = PDFParser()
        self.embedder = EmbeddingService()
        self.faiss = FaissStore()
        self.bm25 = BM25Store()

    def ingest_pdf(self, file_path: Path) -> int:
        logger.info(f"Ingestion service starting parse for file: {file_path}")
        pages = self.parser.parse(file_path)
        logger.info(f"PDF parsed: {len(pages)} pages extracted from {file_path.name}")
        
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
            logger.warning(f"No text extracted from PDF {file_path.name}, skipping vector store indices update.")
            return 0
            
        logger.info(f"Chunking finished. Total chunks: {len(all_texts)}. Generating embeddings...")
        vectors = self.embedder.embed_texts(all_texts)
        
        logger.info("Adding vectors to FAISS index store...")
        self.faiss.add(vectors, all_meta)
        
        logger.info("Adding documents to BM25 index store...")
        self.bm25.add(all_texts, all_meta)
        
        logger.info(f"Ingestion completed successfully for {file_path.name}.")
        return len(all_texts)

    def reset_indexes(self) -> None:
        logger.info("Resetting FAISS and BM25 store indices.")
        self.faiss.reset()
        self.bm25.reset()
