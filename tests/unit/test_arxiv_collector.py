"""Unit tests for ArxivCollector with mocks."""

import pytest
import httpx
import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock

from research_pipeline.config import Config
from research_pipeline.collectors.arxiv import ArxivCollector
from research_pipeline.collectors.models import (
    SearchQuery,
    CandidateRef,
    VersionMismatchError,
)
from research_pipeline.collectors.rate_limiter import RateLimiter


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, xml_response: str):
        self.xml_response = xml_response
        self._root = ET.fromstring(xml_response)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        from urllib.parse import parse_qs
        qs = parse_qs(request.url.query.decode("utf-8") if isinstance(request.url.query, bytes) else str(request.url.query))
        id_list = qs.get("id_list", [None])[0]

        if not id_list:
            # Return full feed
            return httpx.Response(200, text=self.xml_response, request=request)

        # Filter entry matching id_list
        matched_entry = None
        for entry in self._root.findall("atom:entry", {"atom": "http://www.w3.org/2005/Atom"}):
            id_elem = entry.find("atom:id", {"atom": "http://www.w3.org/2005/Atom"})
            if id_elem is not None and id_elem.text and id_list in id_elem.text:
                matched_entry = entry
                break

        if matched_entry is not None:
            feed = ET.Element("feed", xmlns="http://www.w3.org/2005/Atom")
            feed.append(matched_entry)
            single_xml = ET.tostring(feed, encoding="utf-8").decode("utf-8")
            return httpx.Response(200, text=single_xml, request=request)

        return httpx.Response(200, text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>', request=request)


@pytest.mark.asyncio
async def test_arxiv_search(test_config: Config, arxiv_atom_xml: str):
    client = httpx.AsyncClient(transport=MockTransport(arxiv_atom_xml))
    collector = ArxivCollector(test_config, RateLimiter(), client=client)

    candidates = await collector.search(SearchQuery(query_text="market simulation", max_results=10))
    assert len(candidates) == 3

    c1 = candidates[0]
    assert c1.canonical_identifier == "2301.12345"
    assert c1.requested_version == "v3"
    assert "Adaptive Learning" in c1.title

    c3 = candidates[2]
    assert c3.canonical_identifier == "math/0309099"
    assert c3.requested_version == "v2"


@pytest.mark.asyncio
async def test_arxiv_fetch_success(test_config: Config, arxiv_atom_xml: str):
    client = httpx.AsyncClient(transport=MockTransport(arxiv_atom_xml))
    collector = ArxivCollector(test_config, RateLimiter(), client=client)

    candidate = CandidateRef(
        source_type="paper",
        canonical_identifier="2301.12345",
        title="Sample",
        primary_url="https://arxiv.org/abs/2301.12345",
        requested_version=None,  # Requesting latest
    )
    content = await collector.fetch(candidate)
    assert content.source_id == "arxiv:2301.12345"
    assert content.upstream_version == "v3"
    assert "agent-based simulation" in content.content_text
    assert content.content_scope == "abstract_only"
    assert "metadata.json" in content.files_to_store
    assert "abstract.txt" in content.files_to_store
    assert "raw_entry.xml" in content.files_to_store


@pytest.mark.asyncio
async def test_arxiv_fetch_version_mismatch(test_config: Config, arxiv_atom_xml: str):
    client = httpx.AsyncClient(transport=MockTransport(arxiv_atom_xml))
    collector = ArxivCollector(test_config, RateLimiter(), client=client)

    # Candidate explicitly requested v1, but upstream XML contains v3
    candidate = CandidateRef(
        source_type="paper",
        canonical_identifier="2301.12345",
        title="Sample",
        primary_url="https://arxiv.org/abs/2301.12345v1",
        requested_version="v1",
    )
    with pytest.raises(VersionMismatchError):
        await collector.fetch(candidate)
