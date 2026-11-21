"""Configuration management for HDE."""

from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

def load_project_env() -> list[Path]:
    """Find and load .env files from project hierarchy.
    
    Search order:
    1. research_pipeline root
    2. finance workspace root
    3. current working directory
    """
    candidates = [
        Path(__file__).resolve().parent.parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    loaded = []
    for p in candidates:
        if p.exists() and p.is_file():
            load_dotenv(dotenv_path=p, override=False)
            if p not in loaded:
                loaded.append(p)
    return loaded


# Automatically load on import
load_project_env()


class ConfigurationError(ValueError):
    """Raised when configuration values are missing or invalid."""


EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")


def validate_email(email: str | None) -> str:
    if not email or not EMAIL_REGEX.match(email.strip()):
        raise ConfigurationError(
            f"Invalid or missing HDE_CONTACT_EMAIL: {email!r}. "
            "A valid contact email is required for academic API usage etiquette."
        )
    return email.strip()


@dataclass(frozen=True)
class Config:
    contact_email: str
    github_token: str | None
    data_dir: Path
    db_path: Path
    lock_file_path: Path
    storage_dir: Path
    tmp_dir: Path
    orphan_dir: Path
    papers_dir: Path
    oss_dir: Path
    max_rate_limit_wait_seconds: float = 120.0
    max_storage_bytes: int = 5 * 1024 * 1024 * 1024  # 5 GB
    arxiv_min_interval_seconds: float = 3.5
    github_auth_min_interval_seconds: float = 1.5
    github_unauth_min_interval_seconds: float = 60.0
    collector_version: str = "0.1.0"
    arxiv_extractor_version: str = "arxiv:1.0.0"
    github_extractor_version: str = "github_readme:1.0.0"

    @property
    def user_agent(self) -> str:
        return f"research_pipeline/{self.collector_version} ({self.contact_email})"


def load_config(
    data_dir_override: Path | None = None,
    contact_email_override: str | None = None,
) -> Config:
    email = contact_email_override or os.getenv("HDE_CONTACT_EMAIL", "admin@example.com")
    validated_email = validate_email(email)

    base_data_dir = data_dir_override or Path(os.getenv("HDE_DATA_DIR", "data"))
    if not base_data_dir.is_absolute():
        # Relative to research_pipeline root
        pipeline_root = Path(__file__).resolve().parent.parent.parent
        base_data_dir = pipeline_root / base_data_dir

    storage_dir = base_data_dir / "storage"

    cfg = Config(
        contact_email=validated_email,
        github_token=os.getenv("GITHUB_TOKEN") or None,
        data_dir=base_data_dir,
        db_path=base_data_dir / "hde.db",
        lock_file_path=base_data_dir / "hde.lock",
        storage_dir=storage_dir,
        tmp_dir=storage_dir / ".tmp",
        orphan_dir=storage_dir / ".orphan",
        papers_dir=storage_dir / "papers",
        oss_dir=storage_dir / "oss",
        max_rate_limit_wait_seconds=float(os.getenv("MAX_RATE_LIMIT_WAIT_SECONDS", "120.0")),
        max_storage_bytes=int(os.getenv("MAX_STORAGE_BYTES", str(5 * 1024 * 1024 * 1024))),
    )

    # Ensure required directories exist
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg.orphan_dir.mkdir(parents=True, exist_ok=True)
    cfg.papers_dir.mkdir(parents=True, exist_ok=True)
    cfg.oss_dir.mkdir(parents=True, exist_ok=True)

    return cfg
