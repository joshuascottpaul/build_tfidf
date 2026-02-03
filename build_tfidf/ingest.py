"""File discovery and ingest utilities."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".ruff_cache",
    ".eggs",
}


@dataclass(frozen=True)
class IngestedFile:
    path: Path
    text: str
    mtime: float
    sha256: str


def iter_markdown_files(root: Path, exclude_dirs: Iterable[str] | None = None) -> list[Path]:
    excludes = set(exclude_dirs or DEFAULT_EXCLUDE_DIRS)
    paths: list[Path] = []
    for path in root.rglob("*.md"):
        if any(part in excludes for part in path.parts):
            continue
        paths.append(path)
    return sorted(paths)


def read_text_strict(
    path: Path,
    max_bytes: int = 2_000_000,
    max_replacement_ratio: float = 0.25,
) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None

    if len(data) > max_bytes:
        return None

    text = data.decode("utf-8", errors="replace")
    if not text:
        return None

    replacement_count = text.count("\ufffd")
    if replacement_count / max(len(text), 1) > max_replacement_ratio:
        return None

    return text


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
