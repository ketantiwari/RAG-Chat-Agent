from __future__ import annotations

import time
from typing import Iterator
import google.generativeai as genai
from groq import Groq
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:
    def generate(self, prompt: str, provider: str, gemini_api_key: str | None, groq_api_key: str | None) -> str:
        start = time.perf_counter()
        try:
            gemini_key = gemini_api_key or settings.gemini_api_key
            groq_key = groq_api_key or settings.groq_api_key
            if provider == "groq":
                if not groq_key:
                    return "Groq API key missing. Add it in sidebar or .env (GROQ_API_KEY)."
                client = Groq(api_key=groq_key)
                resp = client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                text = resp.choices[0].message.content or ""
            else:
                if not gemini_key:
                    return "Gemini API key missing. Add it in sidebar or .env (GEMINI_API_KEY)."
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel(settings.gemini_model)
                resp = model.generate_content(prompt)
                text = resp.text if getattr(resp, "text", None) else "No response generated."
            return text
        finally:
            logger.info("LLM call completed", extra={"event": "llm_latency", "meta": {"ms": (time.perf_counter() - start) * 1000, "provider": provider}})

    def generate_stream(self, prompt: str, provider: str, gemini_api_key: str | None, groq_api_key: str | None) -> Iterator[str]:
        gemini_key = gemini_api_key or settings.gemini_api_key
        groq_key = groq_api_key or settings.groq_api_key
        if provider == "groq":
            if not groq_key:
                yield "Groq API key missing. Add it in sidebar or .env (GROQ_API_KEY)."
                return
            client = Groq(api_key=groq_key)
            stream = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
            return

        if not gemini_key:
            yield "Gemini API key missing. Add it in sidebar or .env (GEMINI_API_KEY)."
            return
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(settings.gemini_model)
        stream = model.generate_content(prompt, stream=True)
        try:
            for chunk in stream:
                text = self._safe_gemini_chunk_text(chunk)
                if text:
                    yield text
        except Exception as ex:
            logger.info("Gemini stream ended with non-fatal exception", extra={"event": "gemini_stream_exception", "meta": {"error": str(ex)}})
            if "response.text" in str(ex) or "finish_reason" in str(ex):
                return
            yield "\n\n[Streaming interrupted due to an upstream response issue.]"

    @staticmethod
    def _safe_gemini_chunk_text(chunk: object) -> str:
        text = ""
        try:
            text = getattr(chunk, "text", "") or ""
            if text:
                return text
        except Exception:
            pass

        candidates = getattr(chunk, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", "") or ""
                if part_text:
                    text += part_text
        return text
