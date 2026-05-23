from __future__ import annotations

from pathlib import Path
from utils.config import settings


class PromptManager:
    def __init__(self) -> None:
        self.prompt_dir = Path(settings.prompts_dir)

    def load(self, version: str = "rag_v1.txt") -> str:
        path = self.prompt_dir / version
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def render(self, *, query: str, context_blocks: str, conversation_summary: str, recent_turns: str, version: str = "rag_v1.txt") -> str:
        template = self.load(version)
        return template.format(
            query=query,
            context_blocks=context_blocks,
            conversation_summary=conversation_summary or "None",
            recent_turns=recent_turns or "None",
        )
