from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = Path("data")
    upload_dir: Path = Path("uploaded_files")
    cache_dir: Path = Path("cache_data")
    faiss_dir: Path = Path("faiss_index")
    logs_dir: Path = Path("logs")
    prompts_dir: Path = Path("prompts")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    semantic_weight: float = 0.55
    bm25_weight: float = 0.45
    memory_window: int = 6


settings = Settings()

for folder in [
    settings.data_dir,
    settings.upload_dir,
    settings.cache_dir,
    settings.faiss_dir,
    settings.logs_dir,
    settings.prompts_dir,
]:
    folder.mkdir(parents=True, exist_ok=True)
