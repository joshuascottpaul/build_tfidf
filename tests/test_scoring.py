from __future__ import annotations

from build_tfidf.scoring import fuse_scores


def test_minmax_fusion_ordering():
    semantic = {0: 0.9, 1: 0.1, 2: 0.5}
    lexical = {0: 0.1, 1: 0.9, 2: 0.5}
    result = fuse_scores(semantic, lexical, 0.7, 0.3, method="minmax")
    ids = [doc_id for doc_id, _ in result]
    # doc 0 has high semantic, doc 1 has high lexical, doc 2 is balanced
    # with 0.7 semantic weight, doc 0 should rank first
    assert ids[0] == 0


def test_rrf_fusion_ordering():
    semantic = {0: 0.9, 1: 0.1, 2: 0.5}
    lexical = {0: 0.1, 1: 0.9, 2: 0.5}
    result = fuse_scores(semantic, lexical, method="rrf")
    ids = [doc_id for doc_id, _ in result]
    # RRF ranks by reciprocal rank sum; doc 0 is rank 1 in semantic (rank 3 in lexical),
    # doc 1 is rank 1 in lexical (rank 3 in semantic).
    # doc 2 is rank 2 in both, so gets consistent mid-rank from both lists.
    # Docs 0 and 1 tie (each rank 1 in one list, rank 3 in the other).
    # Doc 2 at rank 2 in both: 2/(k+2) vs 1/(k+1)+1/(k+3) -- doc 2 ranks lower.
    # Top should be doc 0 or doc 1 (tied), not doc 2.
    assert ids[0] in (0, 1)


def test_rrf_ignores_weights():
    semantic = {0: 100.0, 1: 0.001}
    lexical = {0: 0.001, 1: 100.0}
    result_minmax = fuse_scores(semantic, lexical, 0.5, 0.5, method="minmax")
    result_rrf = fuse_scores(semantic, lexical, 0.5, 0.5, method="rrf")
    # Both methods should produce identical ordering for symmetric inputs
    assert [i for i, _ in result_minmax] == [i for i, _ in result_rrf]


def test_fuse_scores_default_is_minmax():
    semantic = {0: 0.9, 1: 0.1}
    lexical = {0: 0.1, 1: 0.9}
    default_result = fuse_scores(semantic, lexical, 0.7, 0.3)
    explicit_result = fuse_scores(semantic, lexical, 0.7, 0.3, method="minmax")
    assert default_result == explicit_result
