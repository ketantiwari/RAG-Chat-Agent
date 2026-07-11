from utils.config import settings


def test_settings_defaults():
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.llm_provider in ["gemini", "groq"]
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.top_k == 5
    assert settings.semantic_weight == 0.55
    assert settings.bm25_weight == 0.45
    assert settings.memory_window == 6
