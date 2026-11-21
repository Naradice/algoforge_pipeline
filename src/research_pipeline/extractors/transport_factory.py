"""Transport Factory providing unified interface for Google GenAI and OpenAI APIs."""

import os
from typing import Optional, Any, Tuple

from research_pipeline.config import load_project_env
from research_pipeline.extractors.gemini_transport import GeminiTransport
from research_pipeline.extractors.openai_transport import OpenAITransport
from research_pipeline.extractors.llm_extractor import MockLLMTransport

# Ensure .env is loaded from project paths
load_project_env()


def detect_configured_provider() -> Tuple[Optional[str], Optional[str]]:
    """Detect available provider from environment variables.
    
    Returns: (provider_name, default_model)
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini", "gemini-2.5-flash"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    return None, None


def create_transport(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """Instantiate appropriate transport client.
    
    provider: 'gemini', 'openai', or 'mock'. If None, auto-detects.
    """
    if provider is None:
        detected, _ = detect_configured_provider()
        provider = detected

    if provider == "gemini":
        return GeminiTransport(api_key=api_key)
    elif provider == "openai":
        return OpenAITransport(api_key=api_key)
    elif provider == "mock":
        return MockLLMTransport()
    else:
        raise ValueError(
            "No LLM provider configured. Please set either:\n"
            "  $env:GEMINI_API_KEY='<your-gemini-key>' (for Gemini)\n"
            "or\n"
            "  $env:OPENAI_API_KEY='<your-openai-key>' (for OpenAI)"
        )
