"""Cross-encoder re-ranking via flashrank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RerankConfig:
    model: str = "ms-marco-MiniLM-L-12-v2"
    top_n: int = 10


def rerank(query: str, candidates: Iterable[dict], config: RerankConfig) -> list[dict]:
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError as exc:
        raise ImportError("flashrank is not installed. Run: pip install flashrank") from exc

    candidates = list(candidates)
    ranker = Ranker(model_name=config.model)
    passages = [{"id": i, "text": c["text"]} for i, c in enumerate(candidates)]
    result = ranker.rerank(RerankRequest(query=query, passages=passages))
    ranked_ids = [r["id"] for r in result]
    ordered = [candidates[i] for i in ranked_ids]
    return ordered[: config.top_n]
