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


def fuse_scores(
    semantic: dict[int, float],
    lexical: dict[int, float],
    weight_semantic: float = 0.7,
    weight_lexical: float = 0.3,
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
