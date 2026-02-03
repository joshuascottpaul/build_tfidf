"""Index metadata and signature validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class IndexMetadata:
    schema_version: int
    created_at: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    cleaning_rules: str
    vector_backend: str
    weight_semantic: float
    weight_lexical: float

    def signature(self) -> str:
        raw = (
            f"{self.schema_version}|{self.embedding_model}|{self.embedding_dimensions}|"
            f"{self.chunk_size}|{self.chunk_overlap}|{self.cleaning_rules}|"
            f"{self.vector_backend}|{self.weight_semantic}|{self.weight_lexical}"
        )
        return sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["index_signature"] = self.signature()
        return data


def validate_signature(meta: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "embedding_model",
        "embedding_dimensions",
        "chunk_size",
        "chunk_overlap",
        "cleaning_rules",
        "vector_backend",
        "weight_semantic",
        "weight_lexical",
        "index_signature",
    }
    missing = required - set(meta)
    if missing:
        raise ValueError(f"Missing metadata keys: {sorted(missing)}")

    derived = IndexMetadata(
        schema_version=int(meta["schema_version"]),
        created_at=str(meta.get("created_at", "")),
        embedding_model=str(meta["embedding_model"]),
        embedding_dimensions=int(meta["embedding_dimensions"]),
        chunk_size=int(meta["chunk_size"]),
        chunk_overlap=int(meta["chunk_overlap"]),
        cleaning_rules=str(meta["cleaning_rules"]),
        vector_backend=str(meta["vector_backend"]),
        weight_semantic=float(meta["weight_semantic"]),
        weight_lexical=float(meta["weight_lexical"]),
    )
    if derived.signature() != meta["index_signature"]:
        raise ValueError("Index signature mismatch. Rebuild required.")
