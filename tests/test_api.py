import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from main import app
from api.dependencies import active_document


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("api.routes.rag_routes.ingestion")
@patch("api.routes.rag_routes.workflow")
def test_ingest_endpoint(mock_workflow, mock_ingestion, client):
    # Mock PDF ingestion count
    mock_ingestion.ingest_pdf.return_value = 10
    
    # Upload mock PDF file
    file_content = b"%PDF-1.4 mock content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    response = client.post("/api/ingest", files=files)
    
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["filename"] == "test.pdf"
    assert res_data["chunks_indexed"] == 10
    assert "Ingestion completed" in res_data["message"]
    
    assert active_document["filename"] == "test.pdf"


@patch("api.routes.rag_routes.workflow")
def test_chat_endpoint(mock_workflow, client):
    # Mock workflow response
    mock_workflow.run.return_value = {
        "answer": "This is a mock answer",
        "citations": [{"label": "test.pdf p.1", "chunk_id": "test-c1", "score": 0.95}],
        "retrieved_chunks": [{"text": "context chunk text", "filename": "test.pdf", "page": 1, "chunk_id": "test-c1", "score": 0.95}],
        "retrieval_mode": "hybrid-docs-only"
    }

    payload = {
        "query": "What is hybrid retrieval?",
        "session_id": "test_session",
        "use_web_search": False,
        "provider": "gemini",
        "gemini_api_key": "test_key",
        "top_k": 3
    }
    
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["answer"] == "This is a mock answer"
    assert len(res_data["citations"]) == 1
    assert res_data["retrieval_mode"] == "hybrid-docs-only"
    assert "metrics" in res_data


@patch("api.routes.rag_routes.workflow")
def test_chat_stream_endpoint(mock_workflow, client):
    # Mock stream preparation
    mock_workflow.prepare_state_for_stream.return_value = {
        "citations": [{"label": "test.pdf p.1"}],
        "retrieved_chunks": [{"text": "context chunk text"}],
        "retrieval_mode": "hybrid-docs-only"
    }
    mock_workflow.build_prompt.return_value = "Mocked System Prompt"
    
    # Mock generator stream
    mock_workflow.llm.generate_stream.return_value = ["Hello ", "stream ", "world!"]

    payload = {
        "query": "Stream this please",
        "session_id": "stream_session",
        "use_web_search": False,
        "provider": "gemini",
        "gemini_api_key": "test_key",
        "top_k": 3
    }
    
    response = client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200
    
    # Parse NDJSON response lines
    lines = [json.loads(line) for line in response.iter_lines() if line]
    assert len(lines) >= 5 # meta, 3 chunks, done
    
    # Verify meta event
    assert lines[0]["type"] == "meta"
    assert lines[0]["data"]["retrieval_mode"] == "hybrid-docs-only"
    
    # Verify chunk events
    assert lines[1]["type"] == "chunk"
    assert lines[1]["data"] == "Hello "
    
    # Verify done event
    assert lines[-1]["type"] == "done"
    assert lines[-1]["data"]["answer"] == "Hello stream world!"
