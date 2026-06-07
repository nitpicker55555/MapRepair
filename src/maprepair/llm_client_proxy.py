"""OpenAI-compatible proxy client for cross-vendor experiments (exp25+).

Wraps https://api.openai-hub.com/v1/chat/completions which routes to:
  - OpenAI: gpt-5, gpt-5-mini, gpt-5.5, gpt-4.x, o3, o4-mini, ...
  - Anthropic: claude-opus-4-*, claude-sonnet-4-*, claude-haiku-4-*
  - Google: gemini-2.5-pro, gemini-2.5-flash, gemini-3.x-*

Handles per-family quirks:
  - Reasoning models (gpt-5*, o3*, o4*): need `max_completion_tokens`
    (NOT `max_tokens`) and do not accept `temperature`.
  - Anthropic / Gemini sometimes wrap JSON in ```json ... ``` fences;
    we strip these in `chat_json`.
  - Default timeout is 90s; opus models often need 150s+.

The interface mirrors `maprepair.llm_client`:
  message(role, content) -> dict
  chat(messages, ...) -> str
  chat_json(messages, ...) -> dict
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential


DEFAULT_BASE_URL = "https://api.openai-hub.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_HUB_API_KEY"

# Models that use the o-series / gpt-5 reasoning param convention
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-5-mini", "gpt-5-nano")
# Models that reject `temperature` (reasoning ones, plus some thinking variants)
_NO_TEMPERATURE_MODELS = set(_REASONING_PREFIXES)


def is_reasoning_model(model: str) -> bool:
    return any(model == p or model.startswith(p + "-") or model.startswith(p)
               for p in _REASONING_PREFIXES)


def message(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": content}


@retry(wait=wait_random_exponential(multiplier=1, max=20),
       stop=stop_after_attempt(4))
def chat(
    messages: List[Dict[str, str]],
    *,
    model: str = "gpt-5.5",
    temperature: float = 0.0,
    json_mode: bool = False,
    max_tokens: int = 1024,
    timeout: float = 120,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Call chat-completions; returns the raw assistant string."""
    key = api_key or os.environ.get(DEFAULT_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{DEFAULT_API_KEY_ENV} env var not set and no api_key provided"
        )

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    # Reasoning models use `max_completion_tokens`, others use `max_tokens`.
    if is_reasoning_model(model):
        # bump budget — reasoning tokens consume the limit silently
        body["max_completion_tokens"] = max(max_tokens, 4000)
    else:
        body["max_tokens"] = max_tokens
        # Only non-reasoning models accept `temperature` on this proxy
        body["temperature"] = temperature
    if json_mode and not is_reasoning_model(model):
        # JSON mode is OpenAI-specific; safe on gpt-* non-reasoning + may
        # be ignored on Anthropic/Gemini through the proxy. We strip
        # fences in chat_json regardless.
        body["response_format"] = {"type": "json_object"}

    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"proxy {r.status_code}: {r.text[:400]}")
    body_json = r.json()
    return body_json["choices"][0]["message"]["content"]


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def chat_json(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Call chat with json_mode=True and parse the returned JSON.

    Handles:
      * markdown ```json ... ``` fences (Claude/Gemini)
      * leading prose before the JSON object
      * trailing text after the JSON object
    """
    raw = chat(messages, json_mode=True, **kwargs)
    if not raw:
        raise ValueError("empty response from proxy")
    raw = raw.strip()
    # Try raw parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fence
    m = _JSON_FENCE_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Extract first {...} block
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"could not parse JSON from {raw[:200]!r}")
