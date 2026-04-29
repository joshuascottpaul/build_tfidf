"""Hybrid scoring utilities."""

from __future__ import annotations


def minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _fuse_minmax(
    semantic: dict[int, float],
    lexical: dict[int, float],
    weight_semantic: float,
    weight_lexical: float,
) -> list[tuple[int, float]]:
    all_ids = sorted(set(semantic) | set(lexical))
    sem_scores = [semantic.get(i, 0.0) for i in all_ids]
    lex_scores = [lexical.get(i, 0.0) for i in all_ids]

    sem_norm = minmax_normalize(sem_scores)
    lex_norm = minmax_normalize(lex_scores)

    fused = []
    for idx, doc_id in enumerate(all_ids):
        score = weight_semantic * sem_norm[idx] + weight_lexical * lex_norm[idx]
        fused.append((doc_id, score))
    return sorted(fused, key=lambda x: x[1], reverse=True)


def _fuse_rrf(
    semantic: dict[int, float],
    lexical: dict[int, float],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Rank-based, ignores raw score magnitudes."""
    sem_ranked = sorted(semantic, key=lambda i: semantic[i], reverse=True)
    lex_ranked = sorted(lexical, key=lambda i: lexical[i], reverse=True)

    scores: dict[int, float] = {}
    for rank, doc_id in enumerate(sem_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(lex_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fuse_scores(
    semantic: dict[int, float],
    lexical: dict[int, float],
    weight_semantic: float = 0.7,
    weight_lexical: float = 0.3,
    method: str = "minmax",
) -> list[tuple[int, float]]:
    if method == "rrf":
        return _fuse_rrf(semantic, lexical)
    return _fuse_minmax(semantic, lexical, weight_semantic, weight_lexical)
