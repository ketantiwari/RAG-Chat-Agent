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
from utils.logger import get_logger

router = APIRouter(prefix="/api", tags=["rag"])
response_cache = FileCache("responses")
logger = get_logger(__name__)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        logger.warning(f"Rejected ingest file: {file.filename} is not a PDF.")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    logger.info(f"Starting PDF ingestion for file: {file.filename}")
    destination = Path(settings.upload_dir) / file.filename
    try:
        destination.write_bytes(await file.read())
        # MVP single-document mode: each new upload replaces the active retrieval corpus.
        ingestion.reset_indexes()
        memory.sessions.clear()
        response_cache.clear()
        logger.info(f"Cleared cache and active indices for single-document mode ingest of {file.filename}")
        
        count = ingestion.ingest_pdf(destination)
        active_document["filename"] = file.filename
        # Rebuild retriever in-memory state so stale prior indexes are not used.
        workflow.retriever = HybridRetriever()
        logger.info(f"Successfully completed PDF ingestion: {file.filename}. Indexed {count} chunks.")
    except Exception as ex:
        logger.error(f"Failed to ingest PDF {file.filename}: {ex}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {ex}") from ex
        
    return IngestResponse(filename=file.filename, chunks_indexed=count, message="Ingestion completed.")


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    cache_key = response_cache.stable_hash(f"{payload.session_id}|{payload.query}|{payload.use_web_search}|{payload.provider}")
    cached = response_cache.get(cache_key)
    if cached is not None:
        logger.info(f"Cache hit for query in session {payload.session_id}")
        return ChatResponse(**cached)

    logger.info(f"Cache miss for query in session {payload.session_id}. Running workflow.")
    mem = memory.get_context(payload.session_id)
    turns = "\n".join(
        [f"User: {t['user_query']}\nAssistant: {t['assistant_response']}" for t in mem["recent_turns"]]
    )
    start = time.perf_counter()
    try:
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
    except Exception as ex:
        logger.error(f"Workflow execution failed: {ex}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {ex}") from ex

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
    logger.info(
        f"Workflow completed successfully. Latency: {metrics['total_latency_ms']}ms. "
        f"Answer Relevance: {metrics.get('answer_relevance')}, Hallucination Risk: {metrics.get('hallucination_risk')}"
    )
    return output


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    logger.info(f"Initiated chat stream for session: {payload.session_id}")
    mem = memory.get_context(payload.session_id)
    turns = "\n".join(
        [f"User: {t['user_query']}\nAssistant: {t['assistant_response']}" for t in mem["recent_turns"]]
    )
    start = time.perf_counter()
    try:
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
    except Exception as ex:
        logger.error(f"Failed to prepare state for chat stream: {ex}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to prepare state for stream: {ex}") from ex

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
        try:
            for chunk in workflow.llm.generate_stream(
                prompt=prompt,
                provider=payload.provider,
                gemini_api_key=payload.gemini_api_key,
                groq_api_key=payload.groq_api_key,
            ):
                built += chunk
                yield json.dumps({"type": "chunk", "data": chunk}) + "\n"
        except Exception as ex:
            logger.error(f"Error during LLM stream generation: {ex}", exc_info=True)
            yield json.dumps({"type": "error", "data": str(ex)}) + "\n"
            return

        metrics = evaluate_response(payload.query, built, retrieved_context)
        metrics["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        memory.add_turn(payload.session_id, payload.query, built, retrieved_context, citations)
        logger.info(
            f"Streaming completed. Total Latency: {metrics['total_latency_ms']}ms. "
            f"Answer Relevance: {metrics.get('answer_relevance')}, Hallucination Risk: {metrics.get('hallucination_risk')}"
        )
        yield json.dumps({"type": "done", "data": {"metrics": metrics, "answer": built}}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
