"""HyDE -- Hypothetical Document Embeddings.

Generates a hypothetical answer to a query using an LLM, then embeds
that answer for semantic search instead of the raw query.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request


_SYSTEM_PROMPT = (
    "Write a brief, informative passage (3-5 sentences) that directly "
    "answers the following question. Do not include preamble or hedging."
)


def generate_hypothetical(query: str, provider: str) -> str:
    """Generate a hypothetical document that answers the query.

    Returns the raw query unchanged if no LLM is available (fastembed).
    """
    provider = provider.lower()
    if provider == "ollama":
        return _generate_ollama(query)
    if provider == "openai":
        return _generate_openai(query)
    # fastembed has no LLM -- fall back to raw query
    print("[hyde] no LLM available for fastembed provider, using raw query", file=sys.stderr)
    return query


def _generate_ollama(query: str) -> str:
    model = os.getenv("HYDE_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2"))
    payload = json.dumps({
        "model": model,
        "prompt": f"{_SYSTEM_PROMPT}\n\nQuestion: {query}",
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["response"]


def _generate_openai(query: str) -> str:
    from openai import OpenAI

    model = os.getenv("HYDE_MODEL", "gpt-4o-mini")
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=256,
    )
    return resp.choices[0].message.content or query
