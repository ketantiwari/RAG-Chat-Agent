from __future__ import annotations

import uuid
import json
import requests
import streamlit as st
from requests.exceptions import RequestException

API_BASE = "http://localhost:8000/api"


def render_answer_chips(retrieval_mode: str, citations: list[dict], retrieved_context: list[dict], metrics: dict) -> None:
    latency = metrics.get("total_latency_ms", "-")
    hall = metrics.get("hallucination_risk", "-")
    html = f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
      <span style="background:#1f2937;color:#e5e7eb;padding:2px 10px;border-radius:999px;font-size:12px;">Mode: {retrieval_mode}</span>
      <span style="background:#1f2937;color:#e5e7eb;padding:2px 10px;border-radius:999px;font-size:12px;">Citations: {len(citations)}</span>
      <span style="background:#1f2937;color:#e5e7eb;padding:2px 10px;border-radius:999px;font-size:12px;">Chunks: {len(retrieved_context)}</span>
      <span style="background:#1f2937;color:#e5e7eb;padding:2px 10px;border-radius:999px;font-size:12px;">Latency: {latency} ms</span>
      <span style="background:#1f2937;color:#e5e7eb;padding:2px 10px;border-radius:999px;font-size:12px;">Hallucination Risk: {hall}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


st.set_page_config(page_title="Agentic Hybrid RAG Assistant", page_icon=":brain:", layout="wide")
st.title("Agentic AI + Hybrid RAG Assistant")
st.caption("FastAPI + Streamlit + LangGraph + FAISS + BM25 + Tavily")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox("LLM Provider", ["gemini", "groq"])
    gemini_key = st.text_input("Gemini API Key", type="password")
    groq_key = st.text_input("Groq API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")
    use_web_search = st.toggle("Enable Web Search", value=False)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Ingest PDF", use_container_width=True):
        if not uploaded:
            st.warning("Please upload a PDF first.")
        else:
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            try:
                resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=120)
                if resp.ok:
                    st.success(resp.json().get("message", "Ingested"))
                    st.json(resp.json())
                else:
                    st.error(resp.text)
            except RequestException:
                st.error("Backend is not reachable at http://localhost:8000. Start FastAPI first.")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_answer_chips(
                retrieval_mode=msg.get("retrieval_mode", "unknown"),
                citations=msg.get("citations", []),
                retrieved_context=msg.get("retrieved_context", []),
                metrics=msg.get("metrics", {}),
            )
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander("Details"):
                st.write("Citations")
                st.json(msg.get("citations", []))
                st.write("Retrieved Context")
                st.json(msg.get("retrieved_context", []))
                st.write("Metrics")
                st.json(msg.get("metrics", {}))

query = st.chat_input("Ask a question about your uploaded PDFs...")
if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    payload = {
        "query": query,
        "session_id": st.session_state.session_id,
        "use_web_search": use_web_search,
        "provider": provider,
        "gemini_api_key": gemini_key or None,
        "groq_api_key": groq_key or None,
        "tavily_api_key": tavily_key or None,
        "top_k": 3,
    }
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(f"{API_BASE}/chat/stream", json=payload, timeout=300, stream=True)
        except RequestException:
            resp = None
    with st.chat_message("assistant"):
        if resp is None:
            st.error("Backend is not reachable at http://localhost:8000. Start FastAPI first.")
        elif not resp.ok:
            st.error(resp.text)
        else:
            stream_box = st.empty()
            built = ""
            retrieval_mode = "unknown"
            citations: list[dict] = []
            retrieved_context: list[dict] = []
            metrics: dict = {}
            for line in resp.iter_lines(chunk_size=1, decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                etype = event.get("type")
                data = event.get("data", {})
                if etype == "meta":
                    retrieval_mode = data.get("retrieval_mode", "unknown")
                    citations = data.get("citations", [])
                    retrieved_context = data.get("retrieved_context", [])
                    render_answer_chips(
                        retrieval_mode=retrieval_mode,
                        citations=citations,
                        retrieved_context=retrieved_context,
                        metrics=metrics,
                    )
                elif etype == "chunk":
                    text_piece = data if isinstance(data, str) else str(data)
                    for token in text_piece.split(" "):
                        if token:
                            built += token + " "
                            stream_box.markdown(built)
                elif etype == "done":
                    metrics = data.get("metrics", {})
                    render_answer_chips(
                        retrieval_mode=retrieval_mode,
                        citations=citations,
                        retrieved_context=retrieved_context,
                        metrics=metrics,
                    )
            with st.expander("Details"):
                st.write("Citations")
                st.json(citations)
                st.write("Retrieved Context")
                st.json(retrieved_context)
                st.write("Metrics")
                st.json(metrics)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": built,
                    "retrieved_context": retrieved_context,
                    "citations": citations,
                    "metrics": metrics,
                    "retrieval_mode": retrieval_mode,
                }
            )
