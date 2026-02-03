"""Optional LLM re-ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from openai import OpenAI


@dataclass(frozen=True)
class RerankConfig:
    model: str
    top_n: int


def rerank(query: str, candidates: Iterable[dict], config: RerankConfig) -> list[dict]:
    client = OpenAI()
    payload = []
    for c in candidates:
        payload.append({"id": c["sha256"], "text": c["text"]})

    prompt = (
        "You are a ranking engine. Rank the candidate snippets by relevance to the query. "
        "Return a JSON array of objects with fields id and score (0 to 1)."
    )

    resp = client.responses.create(
        model=config.model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Query: {query}"},
            {"role": "user", "content": f"Candidates: {payload}"},
        ],
    )

    text = resp.output_text
    # Expected JSON array. Caller should handle exceptions.
    import json

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return list(candidates)
        parsed = json.loads(text[start : end + 1])
    scores = {item["id"]: item["score"] for item in parsed}
    ranked = sorted(candidates, key=lambda c: scores.get(c["sha256"], 0), reverse=True)
    return ranked
