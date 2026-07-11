# Agentic Hybrid RAG Assistant

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ketantiwari/RAG-Chat-Agent/ci.yml?branch=main&label=CI%2FCD)
![License](https://img.shields.io/github/license/ketantiwari/RAG-Chat-Agent?label=License)
![Python Version](https://img.shields.io/badge/python-3.11-blue?logo=python)

A production-ready Streamlit and FastAPI application that utilizes an Agentic Hybrid RAG workflow (FAISS semantic + BM25 lexical) orchestrated by LangGraph, with optional web-search grounding, response caching, and structured logging telemetry to parse documents and provide context-grounded answers.

## 🏗️ System Architecture

The application is structured into a containerized frontend (Streamlit) and backend (FastAPI) communicating via a structured JSON API. The graph routing and agent logic are orchestrated using **LangGraph**.

```mermaid
graph TD
    User([User])
    subgraph Frontend [Streamlit UI]
        UI[app.py]
    end
    subgraph Backend [FastAPI Server]
        API[main.py / rag_routes.py]
        Workflow[agents/workflow.py]
        LLM[agents/llm_client.py]
        Retriever[rag/hybrid_retriever.py]
        FAISS[(FAISS Vector Store)]
        BM25[(BM25 Lexical Store)]
        FileCache[(File Cache)]
        Memory[memory/conversation_memory.py]
    end
    
    User -->|Chat / Upload| UI
    UI -->|HTTP Requests| API
    API -->|Run State Graph| Workflow
    Workflow -->|Retrieve Context| Retriever
    Retriever -->|Vector Search| FAISS
    Retriever -->|Keyword Search| BM25
    Workflow -->|Generate Answer| LLM
    Workflow -->|Update Memory| Memory
    API -->|Evaluate Heuristics| Eval[evaluation/prompt_eval.py]
    API -->|Read/Write Cache| FileCache
    LLM -->|API Call| Gemini[Gemini API]
    LLM -->|API Call| Groq[Groq API]
    Workflow -->|Web Search| Tavily[Tavily API]
```

### Routing & Node Execution Flow
1. **`router` Node**: Sets retrieval mode depending on user web-search requests.
2. **`rag` Node**: Performs dual-retrieval (FAISS L2 distance + BM25 scores) and executes Reciprocal Rank Fusion (RRF) with token overlap penalties to output high-quality, relevant context.
3. **`web` Node**: Interacts with Tavily Web Search if additional external grounding is requested.
4. **`synth` Node**: Gathers the context blocks, loads prompt versions from `prompts/`, and queries LLM models (Gemini/Groq) for responses.

---

## 🌟 Core Features
* **Hybrid Retrieval**: Combines FAISS vector search and BM25 lexical search with Reciprocal Rank Fusion (RRF) and token overlap penalties.
* **Agentic State-Routing**: Orchestrates workflow nodes (router, rag, web, synth) using LangGraph.
* **Optional Web Grounding**: Integrates Tavily search on-demand via a UI toggle.
* **Local Caching**: Utilizes a FileCache namespace structure to store embeddings, parsed PDFs, and LLM responses locally on disk to reduce API consumption.
* **Structured Logging & Metrics**: Registers Rotating File and stdout handlers using JSON logging format, tracking token consumption, latencies, and response evaluation scores (grounding, relevance, risk).

---

## 📂 Project Structure
```text
.
├── Dockerfile                  # Multi-stage optimized Docker build configuration
├── docker-compose.yml          # Container service orchestrations
├── Makefile                    # Shortcuts for developer commands
├── LICENSE                     # MIT License
├── README.md                   # System documentation
├── requirements.txt            # Main dependencies
├── app.py                      # Streamlit frontend application
├── main.py                     # FastAPI entry point
├── .env.example                # Environmental template variables
├── .github/                    # CI/CD pipelines
│   └── workflows/ci.yml        # GitHub Actions unit testing pipeline
├── tests/                      # Unit testing suite
│   ├── conftest.py             # Isolated configurations and mocks
│   ├── test_api.py             # FastAPI client router tests
│   ├── test_cache.py           # File cache unit tests
│   ├── test_chunker.py         # Text chunker tests
│   ├── test_config.py          # Configurations checks
│   ├── test_llm.py             # LLM mock client tests
│   ├── test_retriever.py       # Hybrid ranking logic assertions
│   └── test_stores.py          # Index persistency test scripts
├── api/                        # API router definitions
├── agents/                     # LangGraph workflow and LLM call clients
├── rag/                        # Ingest services and rank fusions
├── embeddings/                 # SentenceTransformers local embeds
├── vectorstore/                # FAISS vector store and BM25 store definitions
├── parsers/                    # PyMuPDF PDF extractors
├── websearch/                  # Web search clients
├── memory/                     # Conversation context and summarizers
├── evaluation/                 # Retrieval metrics and risk calculations
├── cache/                      # File cache handlers
└── prompts/                    # Versioned system prompt templates
```

---

## 🛠️ Local Development

### Option 1: Native Installation
1. **Install python packages**: Make sure you have python 3.11 installed, then run:
   ```bash
   make install
   ```
2. **Configure environment settings**: Copy the example env file and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
3. **Launch the application backend**:
   ```bash
   make run-backend
   ```
4. **Launch the application frontend**:
   ```bash
   make run-frontend
   ```

### Option 2: Running with Docker (Containerized)
To build and run the application stack inside Docker (isolated from local Python versions):

* **Build the Docker images**:
  ```bash
  make docker-build
  ```
* **Spin up the container services on ports 8000 and 8501**:
  ```bash
  make docker-up
  ```
* **Spin down the container services**:
  ```bash
  make docker-down
  ```

---

## 🧪 Testing
To run the unit tests locally (which mock external model downloads and API calls to run in under 15 seconds):

```bash
make test
```
Unit tests run automatically in GitHub Actions on every push or pull request to the main or master branches, as defined in `.github/workflows/ci.yml`.

---

## 🚀 Deployment Instructions

### 1. Deploying to Streamlit Community Cloud (Recommended)
Streamlit Community Cloud is the fastest way to launch a live demo:

1. Push your code to a public GitHub repository.
2. Go to Streamlit Community Cloud and log in with your GitHub account.
3. Click **New app**, then select your repository, branch (`main`), and set the main file path to: `app.py`
4. Click **Advanced settings** and paste your environment variables from `.env` into the Secrets text area (using TOML format), for example:
   ```toml
   LLM_PROVIDER = "gemini"
   GEMINI_API_KEY = "AIzaSy..."
   GROQ_API_KEY = "gsk_..."
   TAVILY_API_KEY = "tvly-..."
   LOG_LEVEL = "INFO"
   ```
5. Click **Deploy!** Your app will be live and automatically rebuilds on every git push.

### 2. Deploying to Render (Docker Containerized)
To deploy the Docker image on Render for container-based operations:

1. Sign up on Render and click **New > Web Service**.
2. Connect your GitHub repository.
3. Choose **Docker** as the runtime (Render will automatically detect the `Dockerfile` in the root).
4. Add your **Environment Variables** in the Render Dashboard (using standard `KEY=VALUE` format).
5. Specify the port mapping by adding `PORT=8501` to Render's environment config.
6. Click **Deploy Web Service**.

---

## 🏷️ Setting up GitHub Releases (v1.0.0)
To create a tagged release for your project `v1.0.0`:

1. Tag the commit in your local git environment:
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0 - Production-ready MVP"
   ```
2. Push the tag to GitHub:
   ```bash
   git push origin v1.0.0
   ```
3. Go to the **Releases** section on your GitHub repository page, click **Draft a new release**, select `v1.0.0` as the tag, write a brief title (e.g. `v1.0.0 Production MVP`), summarize the feature checklist, and publish!
