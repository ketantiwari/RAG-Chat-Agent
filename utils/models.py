from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class ChunkMetadata(BaseModel):
    doc_id: str
    filename: str
    page: int
    chunk_id: str


class RetrievedChunk(BaseModel):
    text: str
    metadata: ChunkMetadata
    score: float
    source: Literal["semantic", "bm25", "hybrid", "web"]


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    use_web_search: bool = False
    provider: Literal["gemini", "groq"] = "gemini"
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    tavily_api_key: str | None = None
    top_k: int = 5


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    retrieved_context: list[dict]
    retrieval_mode: str
    metrics: dict = Field(default_factory=dict)
