import pytest
from pathlib import Path
from utils.config import settings


@pytest.fixture(autouse=True)
def mock_settings_paths(tmp_path):
    # Store original paths
    original_faiss = settings.faiss_dir
    original_cache = settings.cache_dir
    original_logs = settings.logs_dir
    original_upload = settings.upload_dir
    original_data = settings.data_dir

    # Override with temp paths
    settings.faiss_dir = tmp_path / "faiss_index"
    settings.cache_dir = tmp_path / "cache_data"
    settings.logs_dir = tmp_path / "logs"
    settings.upload_dir = tmp_path / "uploaded_files"
    settings.data_dir = tmp_path / "data"

    # Create directories
    for folder in [
        settings.faiss_dir,
        settings.cache_dir,
        settings.logs_dir,
        settings.upload_dir,
        settings.data_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    yield

    # Restore original paths
    settings.faiss_dir = original_faiss
    settings.cache_dir = original_cache
    settings.logs_dir = original_logs
    settings.upload_dir = original_upload
    settings.data_dir = original_data
