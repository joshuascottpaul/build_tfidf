from __future__ import annotations

from pathlib import Path

from build_tfidf.chunking import chunk_text_semantic, chunk_text


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Return vectors where topic A texts cluster together and topic B apart."""
    vecs = []
    for t in texts:
        if "alpha" in t.lower():
            vecs.append([1.0, 0.0, 0.0])
        elif "beta" in t.lower():
            vecs.append([0.0, 1.0, 0.0])
        else:
            vecs.append([0.0, 0.0, 1.0])
    return vecs


def test_semantic_chunking_splits_at_topic_boundary():
    text = (
        "Alpha introduction paragraph about alpha topic.\n\n"
        "More alpha discussion and details about alpha.\n\n"
        "Beta introduction paragraph about beta topic.\n\n"
        "More beta discussion and details about beta."
    )
    chunks = chunk_text_semantic(
        Path("test.md"), text, embed_fn=_fake_embed, threshold=0.5, min_tokens=5,
    )
    # Should split into at least 2 chunks (alpha group and beta group)
    assert len(chunks) >= 2
    # First chunk should contain alpha content
    assert "alpha" in chunks[0].text.lower()
    # Last chunk should contain beta content
    assert "beta" in chunks[-1].text.lower()


def test_semantic_chunking_falls_back_for_short_text():
    text = "Short paragraph one.\n\nShort paragraph two."
    chunks = chunk_text_semantic(
        Path("test.md"), text, embed_fn=_fake_embed, threshold=0.5,
    )
    # Fewer than 3 paragraphs, should fall back to token chunking
    token_chunks = chunk_text(Path("test.md"), text)
    assert len(chunks) == len(token_chunks)


def test_semantic_chunking_falls_back_on_embed_error():
    def _failing_embed(texts):
        raise RuntimeError("embed failed")

    text = (
        "Para one.\n\n"
        "Para two.\n\n"
        "Para three.\n\n"
        "Para four."
    )
    chunks = chunk_text_semantic(
        Path("test.md"), text, embed_fn=_failing_embed, threshold=0.5,
    )
    # Should fall back to token chunking without error
    assert len(chunks) >= 1
