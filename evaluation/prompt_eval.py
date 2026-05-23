from __future__ import annotations

import re


def evaluate_response(query: str, answer: str, retrieved_chunks: list[dict]) -> dict:
    query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    answer_terms = set(re.findall(r"[a-zA-Z0-9]+", answer.lower()))
    overlap = len(query_terms.intersection(answer_terms)) / max(1, len(query_terms))

    ctx_text = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
    grounded_hits = sum(1 for t in answer_terms if t in ctx_text)
    grounding = grounded_hits / max(1, len(answer_terms))

    retrieval_quality = min(1.0, len(retrieved_chunks) / 5)
    hallucination_risk = max(0.0, 1.0 - grounding)
    return {
        "answer_relevance": round(overlap, 3),
        "context_relevance": round(grounding, 3),
        "hallucination_risk": round(hallucination_risk, 3),
        "retrieval_quality": round(retrieval_quality, 3),
    }
