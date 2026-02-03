"""Manifest helpers for incremental updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    sha256: str
    mtime: float
    chunk_indices: list[int]


def build_manifest(entries: Iterable[ManifestEntry]) -> dict:
    return {"entries": [entry.__dict__ for entry in entries]}


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"entries": []}
    return __import__("json").loads(path.read_text(encoding="utf-8"))
