from __future__ import annotations

from build_tfidf.metadata import IndexMetadata, validate_signature


def test_index_signature_mismatch():
    meta = IndexMetadata(
        schema_version=1,
        created_at="2025-01-01T00:00:00Z",
        embedding_model="text-embedding-3-large",
        embedding_dimensions=3,
        chunk_size=800,
        chunk_overlap=100,
        cleaning_rules="front_matter",
        vector_backend="faiss",
        weight_semantic=0.7,
        weight_lexical=0.3,
    ).to_dict()

    meta["embedding_dimensions"] = 5
    try:
        validate_signature(meta)
    except ValueError:
        assert True
    else:
        assert False
