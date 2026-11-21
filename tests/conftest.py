"""Pytest fixtures for HDE test suite."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import aiosqlite

from research_pipeline.config import Config, load_config
from research_pipeline.storage.db import init_db

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Provide isolated configuration with temporary directories."""
    data_dir = tmp_path / "data"
    cfg = load_config(
        data_dir_override=data_dir,
        contact_email_override="test-user@example.com",
    )
    return cfg


@pytest.fixture
async def test_db(test_config: Config) -> aiosqlite.Connection:
    """Initialize test database with schema and provide connection."""
    conn = await init_db(test_config.db_path)
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def arxiv_atom_xml() -> str:
    xml_path = FIXTURES_DIR / "arxiv_atom_samples.xml"
    return xml_path.read_text(encoding="utf-8")


@pytest.fixture
def github_samples() -> dict:
    json_path = FIXTURES_DIR / "github_api_samples.json"
    return json.loads(json_path.read_text(encoding="utf-8"))
