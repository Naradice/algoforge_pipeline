"""Real Google GenAI SDK (v1.67+) Transport for Gemini API calls.

Supports:
- Gemini 2.5 Flash / Gemini 2.5 Pro
- Pre-call accurate token counting via client.models.count_tokens
- Structured JSON output with response_schema
- Usage metadata extraction for Post-call settlement
- Strict privacy: Never logs or prints API keys
"""

import os
from typing import Dict, Any, Optional, Tuple, Type
from pydantic import BaseModel

from research_pipeline.config import load_project_env

load_project_env()

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GeminiTransport:
    """Production transport interacting with Google GenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        if not GENAI_AVAILABLE:
            raise RuntimeError("google-genai package is not installed.")

        # Priority: explicit api_key -> env GEMINI_API_KEY -> env GOOGLE_API_KEY
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured in environment. "
                "Please set $env:GEMINI_API_KEY before running real API experiments."
            )

        self.client = genai.Client(api_key=resolved_key)

    def count_tokens(
        self,
        model: str,
        contents: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> int:
        """Call countTokens API including system instructions, schema, and contents."""
        config = types.GenerateContentConfig()
        if system_instruction:
            config.system_instruction = system_instruction
        if response_schema:
            config.response_schema = response_schema
            config.response_mime_type = "application/json"

        res = self.client.models.count_tokens(
            model=model,
            contents=contents,
            config=config,
        )
        base_tokens = res.total_tokens
        return int(base_tokens * 1.15) + 100

    def generate(
        self,
        model: str,
        prompt: str,
        system_instruction: str,
        condition_id: str,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> Tuple[str, int, Dict[str, int]]:
        """Call generateContent and return (raw_text, http_status, token_usage)."""
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            seed=seed,
        )

        if condition_id in ("C2", "C3", "C4"):
            config.response_mime_type = "application/json"
            if response_schema is not None:
                config.response_schema = response_schema

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        raw_text = response.text or ""
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            token_usage["prompt_tokens"] = response.usage_metadata.prompt_token_count or 0
            token_usage["completion_tokens"] = response.usage_metadata.candidates_token_count or 0
            token_usage["total_tokens"] = response.usage_metadata.total_token_count or 0

        return raw_text, 200, token_usage
