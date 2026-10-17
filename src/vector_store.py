"""Persistent Chroma search plus a small source manifest for the inspection UI."""

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from threading import RLock

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.chunking import chunk_pages
from src.config import Settings
from src.embeddings import load_embeddings
from src.ingestion import extract_pdf
from src.retrieval import KeywordIndex, fuse


class DocumentLibrary:
    def __init__(self, settings: Settings, embeddings=None):
        self.settings = settings
        self._embeddings = embeddings
        self._store = None
        self._lock = RLock()
        self._keyword_signature = None
        self._keyword_index = None
        profile = {
            "schema": 1,
            "model": settings.model_name,
            "chunk_tokens": settings.chunk_tokens,
            "overlap_tokens": settings.overlap_tokens,
        }
        fingerprint = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:12]
        self.collection_name = f"pdf-{fingerprint}"
        self.sources_dir = settings.data_dir / self.collection_name / "sources"
        self.sources_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = load_embeddings(
                self.settings.model_name, self.settings.data_dir / "models"
            )
        return self._embeddings

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.settings.data_dir / "chroma"),
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"},
                client_settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._store

    def list_sources(self) -> list[dict]:
        return [
            json.loads(path.read_text())
            for path in sorted(self.sources_dir.glob("*/manifest.json"))
        ]

    def _source_dir(self, document_id: str) -> Path:
        if len(document_id) != 64 or any(c not in "0123456789abcdef" for c in document_id):
            raise ValueError("Invalid document ID.")
        return self.sources_dir / document_id

    def read_pages(self, document_id: str) -> list[Document]:
        entries = json.loads((self._source_dir(document_id) / "pages.json").read_text())
        return [Document(**entry) for entry in entries]

    def read_chunks(self, document_id: str) -> list[Document]:
        entries = json.loads((self._source_dir(document_id) / "chunks.json").read_text())
        return [Document(**entry) for entry in entries]

    def pdf_path(self, document_id: str) -> Path:
        return self._source_dir(document_id) / "document.pdf"

    def index_pdf(self, content: bytes, filename: str) -> tuple[dict, bool]:
        """Return (manifest, added); identical bytes are indexed only once."""
        with self._lock:
            extracted = extract_pdf(
                content,
                filename,
                max_bytes=self.settings.max_upload_bytes,
                max_pages=self.settings.max_pages,
            )
            return self._index_source(extracted, content)

    def index_url(self, url: str) -> tuple[dict, bool]:
        from src.web_ingestion import load_webpage

        extracted = load_webpage(url)
        with self._lock:
            return self._index_source(extracted, extracted.pages[0].page_content.encode())

    def _index_source(self, extracted, content: bytes) -> tuple[dict, bool]:
        source_dir = self._source_dir(extracted.document_id)
        manifest_path = source_dir / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text()), False

        embeddings = self.embeddings
        chunks = chunk_pages(
            extracted.pages,
            embeddings.tokenizer,
            chunk_tokens=self.settings.chunk_tokens,
            overlap_tokens=self.settings.overlap_tokens,
        )
        if not chunks:
            raise ValueError("No searchable chunks could be extracted from this source.")
        if any(
            len(embeddings.tokenizer.encode(chunk.page_content)) > embeddings.max_tokens
            for chunk in chunks
        ):
            raise ValueError("A chunk exceeds the embedding model's limit. Reduce chunk_tokens.")

        source_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "document_id": extracted.document_id,
            "filename": extracted.filename,
            "total_pages": extracted.total_pages,
            "empty_pages": extracted.empty_pages,
            "chunk_count": len(chunks),
            "size_bytes": len(content),
            "source_type": extracted.source_type,
            "url": extracted.url,
            "fetched_at": extracted.fetched_at,
        }
        # Clear any unfinished previous attempt, then commit the manifest last.
        self.store.delete(where={"document_id": extracted.document_id})
        try:
            for start in range(0, len(chunks), 64):
                batch = chunks[start : start + 64]
                self.store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
            (
                source_dir / ("document.pdf" if extracted.source_type == "pdf" else "snapshot.txt")
            ).write_bytes(content)
            for name, documents in [("pages", extracted.pages), ("chunks", chunks)]:
                entries = [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in documents
                ]
                (source_dir / f"{name}.json").write_text(json.dumps(entries, indent=2))
            temporary = source_dir / "manifest.tmp"
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(manifest_path)
        except Exception:
            self.store.delete(where={"document_id": extracted.document_id})
            raise
        return manifest, True

    def delete_source(self, document_id: str) -> None:
        """Remove the source from retrieval and delete its local stored files."""
        with self._lock:
            source_dir = self._source_dir(document_id)
            if source_dir.is_symlink() or source_dir.resolve().parent != self.sources_dir.resolve():
                raise ValueError("Invalid source directory.")
            manifest = source_dir / "manifest.json"
            if not manifest.exists():
                raise ValueError("This source is no longer in the library.")
            client = chromadb.PersistentClient(
                path=str(self.settings.data_dir / "chroma"),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            # No embedding model is needed to delete vectors.
            collection = client.get_collection(self.collection_name)
            hidden = source_dir / "deleted.json"
            manifest.rename(hidden)
            try:
                collection.delete(where={"document_id": document_id})
            except Exception:
                hidden.rename(manifest)
                raise
            shutil.rmtree(source_dir)

    def search(self, query: str, *, k: int = 4, document_id: str | None = None, hybrid=False):
        query = query.strip()
        if not query:
            raise ValueError("Enter a question or search phrase.")
        if not 1 <= k <= 10:
            raise ValueError("Choose between 1 and 10 results.")
        with self._lock:
            # Only committed sources may be searched, including after interrupted indexing.
            source_ids = [source["document_id"] for source in self.list_sources()]
            if document_id:
                if document_id not in source_ids:
                    raise ValueError("The selected source is not in this library.")
                source_ids = [document_id]
            if not source_ids:
                return []
            if len(self.embeddings.tokenizer.encode(query)) > self.embeddings.max_tokens:
                raise ValueError("This search is too long. Use a shorter question or phrase.")
            where = {"document_id": {"$in": source_ids}}
            semantic = self.store.similarity_search_with_score(
                query, k=max(k, 12) if hybrid else k, filter=where
            )
            if not hybrid:
                return semantic
            signature = tuple(sorted(source["document_id"] for source in self.list_sources()))
            if self._keyword_signature != signature:
                self._keyword_index = KeywordIndex(
                    [chunk for source_id in signature for chunk in self.read_chunks(source_id)]
                )
                self._keyword_signature = signature
            lexical = self._keyword_index.search(query, set(source_ids))
            selected = fuse([doc for doc, _ in semantic], lexical, k)
            distances = {doc.metadata["chunk_id"]: distance for doc, distance in semantic}
            missing = [
                doc.metadata["chunk_id"]
                for doc in selected
                if doc.metadata["chunk_id"] not in distances
            ]
            if missing:
                vectors = self.store.get(ids=missing, include=["embeddings"])
                query_vector = self.embeddings.embed_query(query)
                import numpy as np

                for chunk_id, vector in zip(vectors["ids"], vectors["embeddings"]):
                    denominator = np.linalg.norm(query_vector) * np.linalg.norm(vector)
                    distances[chunk_id] = (
                        float(1 - np.dot(query_vector, vector) / denominator)
                        if denominator
                        else 1.0
                    )
            return [(doc, float(distances[doc.metadata["chunk_id"]])) for doc in selected]

    def describe(self) -> dict:
        return {**asdict(self.settings), "collection": self.collection_name, "distance": "cosine"}
