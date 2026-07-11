# ==============================================================================
# RAG Chat Agent DevOps Makefile
# ==============================================================================
# Useful targets for local development, testing, and Docker operations.
# Note: For Windows environments without make installed, please use Git Bash or WSL.

.PHONY: install run-backend run-frontend test docker-build docker-up docker-down clean help

# Default target
help:
	@echo "Available Makefile commands:"
	@echo "  install        - Install dependencies and developer test libraries"
	@echo "  run-backend    - Start the FastAPI backend server on port 8000"
	@echo "  run-frontend   - Start the Streamlit user interface on port 8501"
	@echo "  test           - Run pytest unit tests with coverage reporting"
	@echo "  docker-build   - Build the Docker image locally"
	@echo "  docker-up      - Build and spin up the Docker Compose container services"
	@echo "  docker-down    - Bring down all active container services and networks"
	@echo "  clean          - Remove all logs, caches, indices, and pycache outputs"

# Install project and dev/test requirements
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock httpx

# Run FastAPI server
run-backend:
	uvicorn main:app --reload --port 8000

# Run Streamlit interface
run-frontend:
	streamlit run app.py --server.port 8501

# Run the test suite with coverage
test:
	pytest --cov=. --cov-report=term-missing tests/

# Build standalone docker image
docker-build:
	docker build -t rag-chat-agent:latest .

# Run Compose services
docker-up:
	docker compose up --build -d

# Stop Compose services
docker-down:
	docker compose down

# Clean all temporary execution artifacts
clean:
	@echo "Cleaning up temporary files..."
	rm -rf .pytest_cache .coverage htmlcov coverage.xml
	rm -rf cache_data/*
	rm -rf faiss_index/*
	rm -rf logs/*
	rm -rf uploaded_files/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean completed."
