"""Local storage and document-indexing defaults."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_tokens: int = 220
    overlap_tokens: int = 32
    max_upload_bytes: int = 20 * 1024 * 1024
    max_pages: int = 300

    def __post_init__(self):
        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("Chunk overlap must be smaller than the chunk size.")


def get_settings() -> Settings:
    values = dotenv_values(ROOT / ".env")
    path = Path(os.getenv("RAG_DATA_DIR") or values.get("RAG_DATA_DIR") or "data").expanduser()
    return Settings(data_dir=(ROOT / path).resolve())
