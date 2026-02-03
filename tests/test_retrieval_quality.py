from __future__ import annotations

import json
from pathlib import Path

import build_tfidf.index as index
from build_tfidf.embeddings import EmbeddingConfig


def _fake_embed(texts, _cfg=None):
    def vec(t: str) -> list[float]:
        t = t.lower()
        return [
            float(t.count("alpha")),
            float(t.count("beta")),
            float(t.count("gamma")),
        ]

    return [vec(t) for t in texts]


def _patch_paths(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(index, "DATA_DIR", data_dir)
    monkeypatch.setattr(index, "VEC_PATH", data_dir / "index.faiss")
    monkeypatch.setattr(index, "META_PATH", data_dir / "metadata.json")
    monkeypatch.setattr(index, "MANIFEST_PATH", data_dir / "manifest.json")
    monkeypatch.setattr(index, "LEX_PATH", data_dir / "lexical.json")


def test_retrieval_quality(monkeypatch, tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.md").write_text("# Alpha\n\nalpha note", encoding="utf-8")
    (corpus / "beta.md").write_text("# Beta\n\nbeta note", encoding="utf-8")

    monkeypatch.setattr(index, "embed_texts", _fake_embed)
    _patch_paths(monkeypatch, tmp_path)

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

    gold_path = Path(__file__).parent / "data" / "gold_queries.jsonl"
    records = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines()]
    for rec in records:
        results = index.query(rec["query"], cfg, top_k=5)
        got_paths = [Path(r[0]["path"]).as_posix() for r in results]
        expected = rec["expected_paths"]
        assert any(any(p.endswith(e) for p in got_paths) for e in expected)
