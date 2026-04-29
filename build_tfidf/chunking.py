"""Heading-aware chunking with token limits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable

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
        char_pos = text.find(chunk_text[:50]) if chunk_text else -1
        if char_pos < 0:
            char_pos = 0
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


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_paragraphs(text: str) -> list[str]:
    """Split on double newlines, drop empties."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_text_semantic(
    path: Path,
    text: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    threshold: float = 0.5,
    min_tokens: int = 100,
    max_tokens: int = 800,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split text at topic boundaries detected by embedding similarity.

    Falls back to token-window chunking if fewer than 3 paragraphs or
    if the embedding call fails.
    """
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) < 3:
        return chunk_text(path, text, encoding_name=encoding_name, max_tokens=max_tokens)

    try:
        vectors = embed_fn(paragraphs)
    except Exception:
        return chunk_text(path, text, encoding_name=encoding_name, max_tokens=max_tokens)

    enc = tiktoken.get_encoding(encoding_name)
    lines = text.split("\n")

    # Find split points where similarity drops below threshold
    segments: list[list[str]] = [[paragraphs[0]]]
    for i in range(1, len(paragraphs)):
        sim = _cosine_sim(vectors[i - 1], vectors[i])
        if sim < threshold:
            segments.append([paragraphs[i]])
        else:
            segments[-1].append(paragraphs[i])

    # Merge small segments with their neighbors
    merged: list[str] = []
    buf: list[str] = []
    for seg in segments:
        buf.extend(seg)
        joined = "\n\n".join(buf)
        if len(enc.encode(joined)) >= min_tokens:
            merged.append(joined)
            buf = []
    if buf:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + "\n\n".join(buf)
        else:
            merged.append("\n\n".join(buf))

    # Split oversized segments using token-window chunking
    chunks: list[Chunk] = []
    chunk_index = 0
    for segment_text in merged:
        if len(enc.encode(segment_text)) > max_tokens:
            sub_chunks = chunk_text(path, segment_text, encoding_name=encoding_name, max_tokens=max_tokens)
            for sc in sub_chunks:
                digest = sha256(f"{path}:{chunk_index}:{sc.text}".encode("utf-8")).hexdigest()
                chunks.append(Chunk(path=path, heading=sc.heading, chunk_index=chunk_index, text=sc.text, sha256=digest))
                chunk_index += 1
        else:
            char_pos = text.find(segment_text[:50]) if segment_text else 0
            if char_pos < 0:
                char_pos = 0
            line_idx = 0
            line_offsets = []
            offset = 0
            for line in lines:
                line_offsets.append(offset)
                offset += len(line) + 1
            for i, start in enumerate(line_offsets):
                if start <= char_pos:
                    line_idx = i
                else:
                    break
            heading = _heading_path(lines, line_idx)
            full_text = f"{heading}\n\n{segment_text}".strip()
            digest = sha256(f"{path}:{chunk_index}:{full_text}".encode("utf-8")).hexdigest()
            chunks.append(Chunk(path=path, heading=heading, chunk_index=chunk_index, text=full_text, sha256=digest))
            chunk_index += 1

    return chunks
