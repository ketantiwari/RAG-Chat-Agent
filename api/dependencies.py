from __future__ import annotations

from agents.workflow import AgenticWorkflow
from memory.conversation_memory import ConversationMemory
from rag.ingest import IngestionService

workflow = AgenticWorkflow()
memory = ConversationMemory()
ingestion = IngestionService()
active_document: dict[str, str | None] = {"filename": None}
