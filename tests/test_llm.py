import pytest
from unittest.mock import MagicMock, patch
from agents.llm_client import LLMService
from utils.config import settings


@patch("agents.llm_client.Groq")
@patch("agents.llm_client.genai")
def test_llm_service_gemini(mock_genai, mock_groq):
    # Set up mock Gemini response
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Gemini Response Text"
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    service = LLMService()
    
    # Assert Gemini provider works with key
    res = service.generate("hello", "gemini", "mock-key", None)
    assert res == "Gemini Response Text"
    mock_genai.configure.assert_called_with(api_key="mock-key")
    mock_genai.GenerativeModel.assert_called_with(settings.gemini_model)


@patch("agents.llm_client.Groq")
@patch("agents.llm_client.genai")
def test_llm_service_groq(mock_genai, mock_groq):
    # Set up mock Groq response
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Groq Response Text"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_groq.return_value = mock_client

    service = LLMService()
    
    # Assert Groq provider works with key
    res = service.generate("hello", "groq", None, "mock-key")
    assert res == "Groq Response Text"
    mock_groq.assert_called_with(api_key="mock-key")
    mock_client.chat.completions.create.assert_called_once()


def test_llm_service_missing_keys(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "groq_api_key", "")
    service = LLMService()
    
    # With no API keys, expect error messages
    res_gemini = service.generate("hello", "gemini", "", "")
    assert "Gemini API key missing" in res_gemini

    res_groq = service.generate("hello", "groq", "", "")
    assert "Groq API key missing" in res_groq


@patch("agents.llm_client.Groq")
@patch("agents.llm_client.genai")
def test_llm_service_streaming(mock_genai, mock_groq):
    # Set up mock streaming responses
    mock_model = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Hello "
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "world!"
    mock_model.generate_content.return_value = [mock_chunk1, mock_chunk2]
    mock_genai.GenerativeModel.return_value = mock_model

    service = LLMService()
    chunks = list(service.generate_stream("hello", "gemini", "mock-key", ""))
    assert chunks == ["Hello ", "world!"]
