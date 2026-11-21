"""Unit tests for Multi-Provider Transports and Factory (Gemini and OpenAI)."""

import pytest
from unittest.mock import MagicMock, patch

from research_pipeline.extractors.transport_factory import (
    create_transport,
    detect_configured_provider,
)
from research_pipeline.extractors.openai_transport import OpenAITransport
from research_pipeline.extractors.models import ExtractedKnowledgeRecord


def test_detect_configured_provider(monkeypatch):
    # Test neither configured
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prov, model = detect_configured_provider()
    assert prov is None

    # Test Gemini configured
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")
    prov, model = detect_configured_provider()
    assert prov == "gemini"
    assert model == "gemini-2.5-flash"

    # Test OpenAI configured
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    prov, model = detect_configured_provider()
    assert prov == "openai"
    assert model == "gpt-4o-mini"


def test_create_transport_factory(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    
    # Factory auto detection
    transport = create_transport(provider="mock")
    assert transport is not None

    # Factory explicit openai
    transport_openai = create_transport(provider="openai", api_key="sk-dummy")
    assert isinstance(transport_openai, OpenAITransport)


def test_openai_transport_token_count_conservative():
    transport = OpenAITransport(api_key="sk-test-dummy")
    count = transport.count_tokens("gpt-4o", "This is a test prompt with some words.", "System prompt")
    assert count >= 100  # Conservative estimate with safety margin


def test_openai_transport_generate_mocked(monkeypatch):
    transport = OpenAITransport(api_key="sk-test-dummy")
    
    mock_choice = MagicMock()
    mock_choice.message.content = '{"phenomenas": [], "models": [], "methods": [], "capabilities": [], "assumptions": [], "limitations": [], "local_relations": []}'
    
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 150
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 200

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = mock_usage

    with patch.object(transport.client.chat.completions, "create", return_value=mock_completion):
        raw_text, status, usage = transport.generate(
            model="gpt-4o-mini",
            prompt="Test prompt",
            system_instruction="System",
            condition_id="C4",
            response_schema=ExtractedKnowledgeRecord,
        )
        assert status == 200
        assert usage["prompt_tokens"] == 150
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 200
        assert "phenomenas" in raw_text
