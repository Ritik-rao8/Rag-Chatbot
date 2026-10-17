"""Adapt a local Sentence Transformer to LangChain's embedding interface."""

import os
from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: Path):
        # Keep model caches in the project's ignored data directory.
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name, cache_folder=str(cache_dir), device="cpu", trust_remote_code=False
        )
        self.tokenizer = self.model.tokenizer
        self.max_tokens = self.model.max_seq_length
        self._cached_query = lru_cache(maxsize=128)(self._query_vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        ).tolist()

    def _query_vector(self, text: str) -> tuple[float, ...]:
        return tuple(self.embed_documents([text])[0])

    def embed_query(self, text: str) -> list[float]:
        return list(self._cached_query(text))


@lru_cache(maxsize=2)
def load_embeddings(model_name: str, cache_dir: Path) -> LocalEmbeddings:
    return LocalEmbeddings(model_name, cache_dir)
