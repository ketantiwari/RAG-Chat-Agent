from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class Turn:
    user_query: str
    assistant_response: str
    retrieved_chunks: list[dict]
    citations: list[dict]
    timestamp: str


class ConversationMemory:
    def __init__(self, window_size: int = 6) -> None:
        self.window_size = window_size
        self.sessions: dict[str, dict] = {}

    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = {"turns": [], "summary": ""}

    def add_turn(self, session_id: str, user_query: str, assistant_response: str, retrieved_chunks: list[dict], citations: list[dict]) -> None:
        self._ensure_session(session_id)
        turn = Turn(
            user_query=user_query,
            assistant_response=assistant_response,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.sessions[session_id]["turns"].append(asdict(turn))
        if len(self.sessions[session_id]["turns"]) > (self.window_size * 2):
            self._summarize_session(session_id)

    def get_context(self, session_id: str) -> dict:
        self._ensure_session(session_id)
        turns = self.sessions[session_id]["turns"]
        return {"summary": self.sessions[session_id]["summary"], "recent_turns": turns[-self.window_size :]}

    def _summarize_session(self, session_id: str) -> None:
        turns = self.sessions[session_id]["turns"]
        old_turns = turns[:-self.window_size]
        compressed = []
        for t in old_turns[-6:]:
            compressed.append(f"Q: {t['user_query'][:180]} | A: {t['assistant_response'][:220]}")
        previous_summary = self.sessions[session_id]["summary"]
        merged = (previous_summary + "\n" + "\n".join(compressed)).strip()
        self.sessions[session_id]["summary"] = merged[-4000:]
        self.sessions[session_id]["turns"] = turns[-self.window_size :]
