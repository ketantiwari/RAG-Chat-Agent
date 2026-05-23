from __future__ import annotations

from tavily import TavilyClient
from utils.config import settings


class TavilySearchService:
    def search(self, query: str, api_key: str, max_results: int = 5) -> list[dict]:
        api_key = api_key or settings.tavily_api_key
        if not api_key:
            return []
        try:
            client = TavilyClient(api_key=api_key)
            resp = client.search(query=query, max_results=max_results)
            results = resp.get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": float(r.get("score", 0.0)),
                    "source": "web",
                }
                for r in results
            ]
        except Exception:
            return []
