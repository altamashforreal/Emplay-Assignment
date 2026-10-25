"""
LLM client – wraps Ollama for the agents.
All agents use this to call the LLM with structured output support.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


class OllamaLLM:
    """Simple Ollama LLM client for agent use."""

    def __init__(self):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.llm_model
        self._client = httpx.Client(timeout=300.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat message and return the response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        resp = self._client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        latency = time.time() - start
        content = resp.json()["message"]["content"]
        logger.debug(f"LLM response in {latency:.2f}s ({len(content)} chars)")
        return content

    def chat_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Call LLM and parse JSON from the response.
        Tries strict JSON parse first, then extracts JSON block.
        """
        response = self.chat(prompt, system, temperature)
        return _extract_json(response)

    def close(self):
        self._client.close()


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code fences."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting first { ... } block
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
    return {}


# Singleton
_llm: OllamaLLM | None = None


def get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM()
    return _llm
