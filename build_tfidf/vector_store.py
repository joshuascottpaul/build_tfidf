"""FAISS vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


@dataclass(frozen=True)
class VectorIndex:
    index: faiss.Index
    dim: int


def build_index(vectors: Iterable[list[float]]) -> VectorIndex:
    arr = np.array(list(vectors), dtype="float32")
    if arr.ndim != 2:
        raise ValueError("Vectors must be 2D")
    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(arr)
    index.add(arr)
    return VectorIndex(index=index, dim=dim)


def search(index: VectorIndex, query_vec: list[float], top_k: int) -> list[tuple[int, float]]:
    vec = np.array([query_vec], dtype="float32")
    faiss.normalize_L2(vec)
    scores, ids = index.index.search(vec, top_k)
    return list(zip(ids[0].tolist(), scores[0].tolist()))


def save(index: VectorIndex, path: Path) -> None:
    faiss.write_index(index.index, str(path))


def load(path: Path) -> VectorIndex:
    idx = faiss.read_index(str(path))
    return VectorIndex(index=idx, dim=idx.d)
