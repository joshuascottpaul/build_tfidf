from __future__ import annotations

import json
from unittest.mock import patch, MagicMock
from io import BytesIO

from build_tfidf.embeddings import embed_ollama, EmbeddingConfig


def _make_config(**overrides) -> EmbeddingConfig:
    defaults = dict(
        provider="ollama",
        model="nomic-embed-text",
        dimensions=None,
        batch_size=2,
        rpm_limit=0,
        fallback_to_ollama=False,
        ollama_model="nomic-embed-text",
    )
    defaults.update(overrides)
    return EmbeddingConfig(**defaults)


def _mock_response(data: dict) -> MagicMock:
    body = json.dumps(data).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_ollama_batch_endpoint():
    cfg = _make_config(batch_size=3)
    texts = ["hello", "world", "test"]
    batch_resp = _mock_response({
        "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    })

    with patch("urllib.request.urlopen", return_value=batch_resp) as mock_open:
        result = embed_ollama(texts, cfg)

    assert len(result) == 3
    assert result[0] == [0.1, 0.2]
    # Verify it called the batch endpoint
    call_args = mock_open.call_args[0][0]
    assert "/api/embed" in call_args.full_url


def test_ollama_fallback_to_single():
    cfg = _make_config(batch_size=2)
    texts = ["hello", "world"]

    single_responses = [
        _mock_response({"embedding": [0.1, 0.2]}),
        _mock_response({"embedding": [0.3, 0.4]}),
    ]

    call_count = [0]

    def side_effect(req, **kwargs):
        if req.full_url.endswith("/api/embed"):
            raise Exception("batch not supported")
        resp = single_responses[call_count[0]]
        call_count[0] += 1
        return resp

    with patch("urllib.request.urlopen", side_effect=side_effect):
        result = embed_ollama(texts, cfg)

    assert len(result) == 2
    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.3, 0.4]


def test_provider_default_weights():
    openai_cfg = _make_config(provider="openai")
    fastembed_cfg = _make_config(provider="fastembed")
    ollama_cfg = _make_config(provider="ollama")

    assert openai_cfg.default_weights() == (0.7, 0.3)
    assert fastembed_cfg.default_weights() == (0.6, 0.4)
    assert ollama_cfg.default_weights() == (0.6, 0.4)
