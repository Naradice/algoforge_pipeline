"""Test .env file loading hierarchy in research_pipeline."""

from pathlib import Path
from research_pipeline.config import load_project_env


def test_load_project_env_finds_dotenv():
    loaded = load_project_env()
    assert len(loaded) > 0, "load_project_env should find at least one .env file in the hierarchy"
    names = [p.name for p in loaded]
    assert ".env" in names
