"""Optional LLM re-ranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from openai import OpenAI


@dataclass(frozen=True)
class RerankConfig:
    model: str
    top_n: int


def rerank(query: str, candidates: Iterable[dict], config: RerankConfig) -> list[dict]:
    candidates = list(candidates)
    client = OpenAI()
    payload = [{"id": c["sha256"], "text": c["text"]} for c in candidates]

    prompt = (
        "You are a ranking engine. Rank the candidate snippets by relevance to the query. "
        "Return a JSON array of objects with fields id and score (0 to 1)."
    )

    resp = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Query: {query}"},
            {"role": "user", "content": f"Candidates: {payload}"},
        ],
    )

    text = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return candidates
        parsed = json.loads(text[start : end + 1])
    scores = {item["id"]: item["score"] for item in parsed}
    return sorted(candidates, key=lambda c: scores.get(c["sha256"], 0), reverse=True)
