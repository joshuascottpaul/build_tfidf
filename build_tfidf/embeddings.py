"""Embedding providers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable

from openai import OpenAI


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimensions: int | None
    batch_size: int
    rpm_limit: int
    fallback_to_ollama: bool
    ollama_model: str


def _rate_limit_sleep(last_call: float, rpm_limit: int) -> float:
    if rpm_limit <= 0:
        return time.time()
    min_interval = 60.0 / rpm_limit
    now = time.time()
    elapsed = now - last_call
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    return time.time()


def embed_openai(texts: Iterable[str], config: EmbeddingConfig) -> list[list[float]]:
    client = OpenAI()
    batch = []
    out: list[list[float]] = []
    last_call = 0.0
    for text in texts:
        batch.append(text)
        if len(batch) >= config.batch_size:
            last_call = _rate_limit_sleep(last_call, config.rpm_limit)
            resp = client.embeddings.create(
                model=config.model,
                input=batch,
                dimensions=config.dimensions,
            )
            out.extend([row.embedding for row in resp.data])
            batch = []
    if batch:
        last_call = _rate_limit_sleep(last_call, config.rpm_limit)
        resp = client.embeddings.create(
            model=config.model,
            input=batch,
            dimensions=config.dimensions,
        )
        out.extend([row.embedding for row in resp.data])
    return out


def embed_ollama(texts: Iterable[str], config: EmbeddingConfig) -> list[list[float]]:
    import json
    import urllib.request

    out: list[list[float]] = []
    for text in texts:
        payload = json.dumps({"model": config.ollama_model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            out.append(data["embedding"])
    return out


def embed_texts(texts: Iterable[str], config: EmbeddingConfig) -> list[list[float]]:
    provider = config.provider.lower()
    if provider == "openai":
        try:
            return embed_openai(texts, config)
        except Exception:
            if config.fallback_to_ollama:
                return embed_ollama(texts, config)
            raise
    if provider == "ollama":
        return embed_ollama(texts, config)
    raise ValueError(f"Unknown embedding provider: {config.provider}")


def load_config_from_env() -> EmbeddingConfig:
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    model = os.getenv("OPENAI_MODEL", "text-embedding-3-large")
    dim_raw = os.getenv("DIMENSIONS", "")
    dimensions = int(dim_raw) if dim_raw else None
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    rpm_limit = int(os.getenv("RPM_LIMIT", "60"))
    fallback_to_ollama = os.getenv("FALLBACK_TO_OLLAMA", "false").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
    return EmbeddingConfig(
        provider=provider,
        model=model,
        dimensions=dimensions,
        batch_size=batch_size,
        rpm_limit=rpm_limit,
        fallback_to_ollama=fallback_to_ollama,
        ollama_model=ollama_model,
    )
