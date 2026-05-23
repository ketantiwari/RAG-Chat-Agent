from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.rag_routes import router as rag_router

app = FastAPI(title="Agentic Hybrid RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
