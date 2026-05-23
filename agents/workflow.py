from __future__ import annotations

from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from agents.llm_client import LLMService
from rag.hybrid_retriever import HybridRetriever
from utils.prompt_manager import PromptManager
from websearch.tavily_client import TavilySearchService


class AgentState(TypedDict, total=False):
    query: str
    use_web_search: bool
    provider: str
    gemini_api_key: str | None
    groq_api_key: str | None
    tavily_api_key: str | None
    top_k: int
    memory_summary: str
    memory_turns: str
    retrieved_chunks: list[dict]
    web_chunks: list[dict]
    answer: str
    citations: list[dict]
    retrieval_mode: str
    active_filename: str | None


class AgenticWorkflow:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.llm = LLMService()
        self.prompts = PromptManager()
        self.websearch = TavilySearchService()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self.router_node)
        graph.add_node("rag", self.rag_node)
        graph.add_node("web", self.web_node)
        graph.add_node("synth", self.synth_node)

        graph.set_entry_point("router")
        graph.add_edge("router", "rag")
        graph.add_conditional_edges("rag", self.route_after_rag, {"web": "web", "synth": "synth"})
        graph.add_edge("web", "synth")
        graph.add_edge("synth", END)
        return graph.compile()

    def router_node(self, state: AgentState) -> AgentState:
        state["retrieval_mode"] = "hybrid+web" if state.get("use_web_search") else "hybrid-docs-only"
        return state

    def rag_node(self, state: AgentState) -> AgentState:
        chunks = self.retriever.retrieve(
            state["query"],
            top_k=state.get("top_k", 5),
            active_filename=state.get("active_filename"),
        )
        state["retrieved_chunks"] = chunks
        citations = []
        seen = set()
        for c in chunks:
            label = f"{c['filename']} p.{c['page']}"
            if label in seen:
                continue
            seen.add(label)
            citations.append({"label": label, "chunk_id": c["chunk_id"], "score": c["score"]})
            if len(citations) >= 3:
                break
        state["citations"] = citations
        return state

    def route_after_rag(self, state: AgentState) -> str:
        return "web" if state.get("use_web_search") else "synth"

    def web_node(self, state: AgentState) -> AgentState:
        state["web_chunks"] = self.websearch.search(
            query=state["query"],
            api_key=state.get("tavily_api_key", "") or "",
            max_results=3,
        )
        return state

    def synth_node(self, state: AgentState) -> AgentState:
        prompt = self.build_prompt(state)
        state["answer"] = self.llm.generate(
            prompt=prompt,
            provider=state["provider"],
            gemini_api_key=state.get("gemini_api_key"),
            groq_api_key=state.get("groq_api_key"),
        )
        return state

    def run(self, payload: dict[str, Any]) -> AgentState:
        return self.graph.invoke(payload)

    def build_prompt(self, state: AgentState) -> str:
        context = []
        for c in state.get("retrieved_chunks", []):
            context.append(f"Source: {c['filename']} p.{c['page']}\nText: {c['text']}")
        for i, c in enumerate(state.get("web_chunks", []), start=1):
            context.append(f"[WEB {i}] {c['title']} ({c['url']}) :: {c['content']}")
        return self.prompts.render(
            query=state["query"],
            context_blocks="\n\n".join(context) if context else "No context found.",
            conversation_summary=state.get("memory_summary", ""),
            recent_turns=state.get("memory_turns", ""),
        )

    def prepare_state_for_stream(self, payload: dict[str, Any]) -> AgentState:
        state: AgentState = dict(payload)
        state = self.router_node(state)
        state = self.rag_node(state)
        if self.route_after_rag(state) == "web":
            state = self.web_node(state)
        return state
