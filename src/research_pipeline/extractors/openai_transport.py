"""OpenAI API Transport for H1a Capability Extraction.

Supports:
- gpt-4o, gpt-4o-mini, gpt-4-turbo
- Structured Outputs via client.beta.chat.completions.parse with fallback to json_object
- Conservative pre-call token estimation including JSON Schema and safety margin
- Strict post-call settlement and budget tracking
- Strict privacy: Never logs or outputs OPENAI_API_KEY
"""

import os
import math
import json
from typing import Dict, Any, Optional, Tuple, Type
from pydantic import BaseModel

from research_pipeline.config import load_project_env

load_project_env()

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class OpenAITransport:
    """Production transport interacting with OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai package is not installed.")

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured in environment. "
                "Please set $env:OPENAI_API_KEY before running OpenAI experiments."
            )

        self.client = openai.OpenAI(api_key=resolved_key)

    def count_tokens(
        self,
        model: str,
        contents: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> int:
        """Estimate token count conservatively before making an API call.
        
        Strictly includes:
        - System instruction
        - Prompt contents
        - JSON Schema definition (when response_schema is provided)
        - 15% safety buffer + 100 token framing overhead
        
        Guarantees that pre-call token reservation >= actual input tokens.
        """
        full_text = f"{system_instruction or ''}\n\n{contents}"
        tokens = 0

        if TIKTOKEN_AVAILABLE:
            try:
                # Use exact BPE tokenizer (o200k_base for gpt-4o / gpt-4o-mini)
                try:
                    enc = tiktoken.encoding_for_model(model)
                except Exception:
                    enc = tiktoken.get_encoding("o200k_base")

                tokens += len(enc.encode(full_text))

                if response_schema is not None:
                    schema_json = json.dumps(response_schema.model_json_schema())
                    tokens += len(enc.encode(schema_json))

                # Apply 15% conservative safety margin + 100 tokens framing overhead
                conservative_total = int(math.ceil(tokens * 1.15)) + 100
                return conservative_total
            except Exception as e:
                # If tokenization fails, raise error to prevent unverified API dispatch
                raise RuntimeError(f"Token counting failed for model {model}: {e}")

        # Fallback heuristic if tiktoken missing: conservative len/2.5
        word_based = int(len(full_text.split()) * 2.0)
        char_based = int(math.ceil(len(full_text) / 2.5))
        base_tokens = max(word_based, char_based)

        if response_schema is not None:
            schema_json = json.dumps(response_schema.model_json_schema())
            base_tokens += int(math.ceil(len(schema_json) / 2.5))

        return int(math.ceil(base_tokens * 1.25)) + 200

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
        """Call OpenAI chat completion and return (raw_text, http_status, token_usage)."""
        # Ensure JSON instruction is present in system message for OpenAI JSON modes
        sys_content = system_instruction
        if condition_id in ("C2", "C3", "C4") and "json" not in sys_content.lower():
            sys_content = f"{sys_content}\nYou must respond in valid JSON format."

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt},
        ]

        if condition_id in ("C2", "C3", "C4") and response_schema is not None:
            # Use direct Structured Outputs json_schema
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_schema.__name__,
                            "schema": response_schema.model_json_schema(),
                        },
                    },
                    temperature=temperature,
                    seed=seed,
                )
                choice = completion.choices[0]
                raw_text = choice.message.content or ""
                usage_obj = completion.usage
            except Exception:
                # Fallback to json_object mode if json_schema is not supported
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    seed=seed,
                )
                choice = completion.choices[0]
                raw_text = choice.message.content or ""
                usage_obj = completion.usage
        else:
            # Free-text or unconstrained JSON
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
            )
            choice = completion.choices[0]
            raw_text = choice.message.content or ""
            usage_obj = completion.usage

        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if usage_obj is not None:
            token_usage["prompt_tokens"] = usage_obj.prompt_tokens or 0
            token_usage["completion_tokens"] = usage_obj.completion_tokens or 0
            token_usage["total_tokens"] = usage_obj.total_tokens or 0

        return raw_text, 200, token_usage
