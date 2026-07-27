#!/usr/bin/env python3
"""
Shared LLM client wrapper for PedDialog-CoT scripts.

Provides a thin wrapper around the OpenAI-compatible API with:
- JSON response parsing with fallback
- Retry with exponential backoff
"""

import json
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        max_tokens: int = 8192,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, system: str, user: str, temperature: float = 0.1, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def chat_json(self, system: str, user: str, temperature: float = 0.1) -> Optional[dict]:
        raw = self.chat(system, user, temperature)
        return parse_json(raw)


def parse_json(text: str) -> Optional[dict]:
    """Parse JSON from LLM output with fallback cleaning."""
    text = text.strip()
    # Remove markdown fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
