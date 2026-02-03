from __future__ import annotations

from pathlib import Path

import build_tfidf.index as index
from build_tfidf.embeddings import EmbeddingConfig


def test_update_incremental(monkeypatch, tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    f1 = corpus / "alpha.md"
    f2 = corpus / "beta.md"
    f1.write_text("# Alpha\n\nalpha note", encoding="utf-8")
    f2.write_text("# Beta\n\nbeta note", encoding="utf-8")

    calls = {"count": 0}

    def _fake_embed(texts, _cfg=None):
        calls["count"] += 1
        def vec(t: str) -> list[float]:
            t = t.lower()
            return [float(t.count("alpha")), float(t.count("beta")), float(t.count("gamma"))]
        return [vec(t) for t in texts]

    monkeypatch.setattr(index, "embed_texts", _fake_embed)
    monkeypatch.setattr(index, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(index, "VEC_PATH", index.DATA_DIR / "index.faiss")
    monkeypatch.setattr(index, "VECTORS_PATH", index.DATA_DIR / "vectors.npy")
    monkeypatch.setattr(index, "META_PATH", index.DATA_DIR / "metadata.json")
    monkeypatch.setattr(index, "MANIFEST_PATH", index.DATA_DIR / "manifest.json")
    monkeypatch.setattr(index, "LEX_PATH", index.DATA_DIR / "lexical.json")

    cfg = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-large",
        dimensions=None,
        batch_size=32,
        rpm_limit=60,
        fallback_to_ollama=False,
        ollama_model="nomic-embed-text",
    )

    index.build(corpus, cfg)
    first_calls = calls["count"]

    f1.write_text("# Alpha\n\nalpha note updated", encoding="utf-8")
    index.update(corpus, cfg)

    assert calls["count"] >= first_calls + 1
