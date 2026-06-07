"""Azure OpenAI wrapper used by LLM repair agents."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


DEFAULT_API_VERSION = "2024-12-01-preview"

AVAILABLE_MODELS = ("gpt-4o", "gpt-4.1", "gpt-4.1-mini")


def make_client() -> AzureOpenAI:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_BASE_URL")
    if not api_key or not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_BASE_URL must be set as "
            "environment variables. See .env.example for a template."
        )
    return AzureOpenAI(
        api_key=api_key,
        api_version=os.environ.get("AZURE_API_VERSION", DEFAULT_API_VERSION),
        azure_endpoint=endpoint,
    )


_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = make_client()
    return _client


def message(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": content}


@retry(wait=wait_random_exponential(multiplier=1, max=30),
       stop=stop_after_attempt(4))
def chat(
    messages: List[Dict[str, str]],
    *,
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    json_mode: bool = False,
    max_tokens: int = 1024,
) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 90,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def chat_json(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    raw = chat(messages, json_mode=True, **kwargs)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise
