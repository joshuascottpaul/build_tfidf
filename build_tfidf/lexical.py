"""BM25 lexical index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import re

from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class LexicalIndex:
    bm25: BM25Okapi
    tokens: list[list[str]]


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_index(texts: Iterable[str]) -> LexicalIndex:
    tokens = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokens)
    return LexicalIndex(bm25=bm25, tokens=tokens)


def search(index: LexicalIndex, query: str, top_k: int) -> list[tuple[int, float]]:
    q = _tokenize(query)
    scores = index.bm25.get_scores(q)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
