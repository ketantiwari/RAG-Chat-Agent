from __future__ import annotations

import json
import time
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from api.dependencies import ingestion, workflow, memory, active_document
from cache.file_cache import FileCache
from evaluation.prompt_eval import evaluate_response
from rag.hybrid_retriever import HybridRetriever
from utils.config import settings
from utils.models import ChatRequest, ChatResponse, IngestResponse

router = APIRouter(prefix="/api", tags=["rag"])
response_cache = FileCache("responses")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    destination = Path(settings.upload_dir) / file.filename
    destination.write_bytes(await file.read())
    try:
        # MVP single-document mode: each new upload replaces the active retrieval corpus.
        ingestion.reset_indexes()
        memory.sessions.clear()
        response_cache.clear()
        count = ingestion.ingest_pdf(destination)
        active_document["filename"] = file.filename
        # Rebuild retriever in-memory state so stale prior indexes are not used.
        workflow.retriever = HybridRetriever()
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {ex}") from ex
    return IngestResponse(filename=file.filename, chunks_indexed=count, message="Ingestion completed.")


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    cache_key = response_cache.stable_hash(f"{payload.session_id}|{payload.query}|{payload.use_web_search}|{payload.provider}")
    cached = response_cache.get(cache_key)
    if cached is not None:
        return ChatResponse(**cached)

    mem = memory.get_context(payload.session_id)
    turns = "\n".join(
        [f"User: {t['user_query']}\nAssistant: {t['assistant_response']}" for t in mem["recent_turns"]]
    )
    start = time.perf_counter()
    result = workflow.run(
        {
            "query": payload.query,
            "use_web_search": payload.use_web_search,
            "provider": payload.provider,
            "gemini_api_key": payload.gemini_api_key,
            "groq_api_key": payload.groq_api_key,
            "tavily_api_key": payload.tavily_api_key,
            "top_k": payload.top_k,
            "memory_summary": mem.get("summary", ""),
            "memory_turns": turns,
            "active_filename": active_document.get("filename"),
        }
    )

    answer = result.get("answer", "No response generated.")
    citations = result.get("citations", [])
    retrieved_context = result.get("retrieved_chunks", [])
    metrics = evaluate_response(payload.query, answer, retrieved_context)
    metrics["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)

    memory.add_turn(
        payload.session_id,
        payload.query,
        answer,
        retrieved_context,
        citations,
    )

    output = ChatResponse(
        answer=answer,
        citations=citations,
        retrieved_context=retrieved_context,
        retrieval_mode=result.get("retrieval_mode", "hybrid-docs-only"),
        metrics=metrics,
    )
    response_cache.set(cache_key, output.model_dump())
    return output


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    mem = memory.get_context(payload.session_id)
    turns = "\n".join(
        [f"User: {t['user_query']}\nAssistant: {t['assistant_response']}" for t in mem["recent_turns"]]
    )
    start = time.perf_counter()
    state = workflow.prepare_state_for_stream(
        {
            "query": payload.query,
            "use_web_search": payload.use_web_search,
            "provider": payload.provider,
            "gemini_api_key": payload.gemini_api_key,
            "groq_api_key": payload.groq_api_key,
            "tavily_api_key": payload.tavily_api_key,
            "top_k": payload.top_k,
            "memory_summary": mem.get("summary", ""),
            "memory_turns": turns,
            "active_filename": active_document.get("filename"),
        }
    )
    prompt = workflow.build_prompt(state)
    citations = state.get("citations", [])
    retrieved_context = state.get("retrieved_chunks", [])

    def gen():
        meta = {
            "retrieval_mode": state.get("retrieval_mode", "hybrid-docs-only"),
            "citations": citations,
            "retrieved_context": retrieved_context,
        }
        yield json.dumps({"type": "meta", "data": meta}) + "\n"
        built = ""
        for chunk in workflow.llm.generate_stream(
            prompt=prompt,
            provider=payload.provider,
            gemini_api_key=payload.gemini_api_key,
            groq_api_key=payload.groq_api_key,
        ):
            built += chunk
            yield json.dumps({"type": "chunk", "data": chunk}) + "\n"

        metrics = evaluate_response(payload.query, built, retrieved_context)
        metrics["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        memory.add_turn(payload.session_id, payload.query, built, retrieved_context, citations)
        yield json.dumps({"type": "done", "data": {"metrics": metrics, "answer": built}}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
