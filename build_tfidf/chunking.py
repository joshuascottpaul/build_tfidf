"""Heading-aware chunking with token limits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import tiktoken


@dataclass(frozen=True)
class Chunk:
    path: Path
    heading: str
    chunk_index: int
    text: str
    sha256: str


def _heading_path(lines: list[str], idx: int) -> str:
    current: list[str] = []
    for i in range(idx, -1, -1):
        line = lines[i].strip()
        if line.startswith("#"):
            current.append(line.lstrip("# ").strip())
            break
    return " / ".join(reversed(current)) if current else ""


def _token_chunks(tokens: list[int], max_tokens: int, overlap: int) -> list[list[int]]:
    if max_tokens <= 0:
        return []
    step = max(max_tokens - overlap, 1)
    return [tokens[i : i + max_tokens] for i in range(0, len(tokens), step)]


def chunk_text(
    path: Path,
    text: str,
    encoding_name: str = "cl100k_base",
    max_tokens: int = 800,
    overlap: int = 100,
    hard_cap: int = 1000,
) -> list[Chunk]:
    enc = tiktoken.get_encoding(encoding_name)
    lines = text.split("\n")
    tokens = enc.encode(text)
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1
    chunks: list[Chunk] = []
    for chunk_index, token_slice in enumerate(_token_chunks(tokens, max_tokens, overlap)):
        if len(token_slice) > hard_cap:
            token_slice = token_slice[:hard_cap]
        chunk_text = enc.decode(token_slice)
        char_pos = text.find(chunk_text[:50]) if chunk_text else 0
        line_idx = 0
        for i, start in enumerate(line_offsets):
            if start <= char_pos:
                line_idx = i
            else:
                break
        heading = _heading_path(lines, line_idx)
        full_text = f"{heading}\n\n{chunk_text}".strip()
        digest = sha256(f"{path}:{chunk_index}:{full_text}".encode("utf-8")).hexdigest()
        chunks.append(Chunk(path=path, heading=heading, chunk_index=chunk_index, text=full_text, sha256=digest))
    return chunks
