"""Index build, update, and query orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .chunking import Chunk, chunk_text
from .cleaning import clean_text
from .embeddings import EmbeddingConfig, embed_texts
from .ingest import DEFAULT_EXCLUDE_DIRS, iter_markdown_files, read_text_strict, sha256_text
from .manifest import ManifestEntry, build_manifest, load_manifest
from .lexical import LexicalIndex, build_index as build_lexical, search as search_lexical
from .metadata import IndexMetadata, validate_signature
from .rerank import RerankConfig, rerank
from .scoring import fuse_scores
from .vector_store import VectorIndex, build_index as build_vector, load as load_vector, save as save_vector, search


DATA_DIR = Path("build_tfidf/data")
VEC_PATH = DATA_DIR / "index.faiss"
VECTORS_PATH = DATA_DIR / "vectors.npy"
META_PATH = DATA_DIR / "metadata.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"
LEX_PATH = DATA_DIR / "lexical.json"


SCHEMA_VERSION = 1
CLEANING_RULES = "front_matter,optional_code_fences,normalize_whitespace"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_chunks(
    paths: list[Path],
    remove_code: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in paths:
        text = read_text_strict(path)
        if text is None:
            continue
        cleaned = clean_text(text, remove_code=remove_code)
        all_chunks.extend(chunk_text(path, cleaned, max_tokens=chunk_size, overlap=chunk_overlap))
    return all_chunks


def build(
    root: Path,
    embed_config: EmbeddingConfig,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    weight_semantic: float = 0.7,
    weight_lexical: float = 0.3,
    remove_code: bool = False,
) -> None:
    _ensure_data_dir()
    paths = iter_markdown_files(root, DEFAULT_EXCLUDE_DIRS)
    all_chunks = _build_chunks(paths, remove_code, chunk_size, chunk_overlap)

    if not all_chunks:
        raise SystemExit("No chunks to index.")

    vectors = embed_texts([c.text for c in all_chunks], embed_config)
    if not vectors or not vectors[0]:
        raise SystemExit("Embedding provider returned no vectors.")
    vindex = build_vector(vectors)
    save_vector(vindex, VEC_PATH)
    _save_vectors(vectors)

    lex = build_lexical([c.text for c in all_chunks])
    _save_json(LEX_PATH, {"texts": [c.text for c in all_chunks]})

    meta = IndexMetadata(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        embedding_model=embed_config.model,
        embedding_dimensions=len(vectors[0]),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        cleaning_rules=CLEANING_RULES,
        vector_backend="faiss",
        weight_semantic=weight_semantic,
        weight_lexical=weight_lexical,
    )
    _save_json(META_PATH, meta.to_dict())

    manifest_entries = []
    chunk_map: dict[str, list[int]] = {}
    for idx, c in enumerate(all_chunks):
        chunk_map.setdefault(str(c.path), []).append(idx)
    for path in paths:
        text = read_text_strict(path)
        if text is None:
            continue
        digest = sha256_text(text)
        manifest_entries.append(
            ManifestEntry(
                path=str(path),
                sha256=digest,
                mtime=path.stat().st_mtime,
                chunk_indices=chunk_map.get(str(path), []),
            )
        )

    manifest = {
        "chunks": [{**asdict(c), "path": str(c.path)} for c in all_chunks],
        **build_manifest(manifest_entries),
    }
    _save_json(MANIFEST_PATH, manifest)


def _load_lexical() -> LexicalIndex:
    data = _load_json(LEX_PATH)
    texts = data["texts"]
    return build_lexical(texts)


def _save_vectors(vectors: list[list[float]]) -> None:
    import numpy as np

    arr = np.array(vectors, dtype="float32")
    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, arr)


def _load_vectors() -> list[list[float]]:
    import numpy as np

    arr = np.load(VECTORS_PATH)
    return arr.tolist()


def query(
    query_text: str,
    embed_config: EmbeddingConfig,
    top_k: int = 10,
    weight_semantic: float = 0.7,
    weight_lexical: float = 0.3,
    rerank_model: str | None = None,
    rerank_top_n: int = 30,
) -> list[tuple[dict, float]]:
    meta = _load_json(META_PATH)
    validate_signature(meta)

    vindex = load_vector(VEC_PATH)
    query_vec = embed_texts([query_text], embed_config)[0]
    sem_hits = search(vindex, query_vec, top_k=top_k * 5)
    sem_scores = {idx: score for idx, score in sem_hits if idx >= 0}

    lex_index = _load_lexical()
    lex_hits = search_lexical(lex_index, query_text, top_k=top_k * 5)
    lex_scores = {idx: score for idx, score in lex_hits}

    fused = fuse_scores(sem_scores, lex_scores, weight_semantic, weight_lexical)
    manifest = _load_json(MANIFEST_PATH)

    results = []
    for idx, score in fused[: max(top_k, rerank_top_n)]:
        chunk = manifest["chunks"][idx]
        results.append((chunk, score))

    if rerank_model:
        rerank_cfg = RerankConfig(model=rerank_model, top_n=rerank_top_n)
        reranked = rerank(query_text, [c for c, _ in results[:rerank_top_n]], rerank_cfg)
        reranked_ids = {c["sha256"] for c in reranked[:top_k]}
        reranked_set = [(c, s) for (c, s) in results if c["sha256"] in reranked_ids]
        if len(reranked_set) < top_k:
            for item in results:
                if item in reranked_set:
                    continue
                reranked_set.append(item)
                if len(reranked_set) >= top_k:
                    break
        results = reranked_set

    return results[:top_k]


def update(
    root: Path,
    embed_config: EmbeddingConfig,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    weight_semantic: float = 0.7,
    weight_lexical: float = 0.3,
    remove_code: bool = False,
) -> None:
    _ensure_data_dir()
    current_paths = iter_markdown_files(root, DEFAULT_EXCLUDE_DIRS)
    manifest = load_manifest(MANIFEST_PATH)
    entries = {e["path"]: e for e in manifest.get("entries", [])}

    current_set = {str(p) for p in current_paths}
    previous_set = set(entries.keys())

    changed_paths: list[Path] = []
    removed_paths = previous_set - current_set

    for path in current_paths:
        text = read_text_strict(path)
        if text is None:
            continue
        sha = sha256_text(text)
        mtime = path.stat().st_mtime
        prev = entries.get(str(path))
        if not prev or prev["sha256"] != sha or prev["mtime"] != mtime:
            changed_paths.append(path)

    if not entries:
        build(root, embed_config, chunk_size, chunk_overlap, weight_semantic, weight_lexical, remove_code)
        return

    if not changed_paths and not removed_paths:
        return

    # Load existing artifacts
    manifest = _load_json(MANIFEST_PATH)
    existing_chunks: list[dict] = manifest.get("chunks", [])
    existing_vectors = _load_vectors()
    existing_texts = _load_json(LEX_PATH).get("texts", [])

    # Filter out removed or changed paths
    remove_set = {str(p) for p in changed_paths} | set(removed_paths)
    kept_chunks = []
    kept_vectors = []
    kept_texts = []
    for idx, chunk in enumerate(existing_chunks):
        if chunk["path"] in remove_set:
            continue
        kept_chunks.append(chunk)
        kept_vectors.append(existing_vectors[idx])
        kept_texts.append(existing_texts[idx])

    # Rebuild chunks for changed paths
    new_chunks = _build_chunks(changed_paths, remove_code, chunk_size, chunk_overlap)
    if new_chunks:
        new_vectors = embed_texts([c.text for c in new_chunks], embed_config)
    else:
        new_vectors = []

    # Append new chunks and vectors
    for c in new_chunks:
        kept_chunks.append({**asdict(c), "path": str(c.path)})
        kept_texts.append(c.text)
    kept_vectors.extend(new_vectors)

    # Persist updated artifacts
    _save_vectors(kept_vectors)
    vindex = build_vector(kept_vectors)
    save_vector(vindex, VEC_PATH)
    _save_json(LEX_PATH, {"texts": kept_texts})

    # Rebuild manifest entries
    chunk_map: dict[str, list[int]] = {}
    for idx, c in enumerate(kept_chunks):
        chunk_map.setdefault(c["path"], []).append(idx)
    manifest_entries = []
    for path in current_paths:
        text = read_text_strict(path)
        if text is None:
            continue
        digest = sha256_text(text)
        manifest_entries.append(
            ManifestEntry(
                path=str(path),
                sha256=digest,
                mtime=path.stat().st_mtime,
                chunk_indices=chunk_map.get(str(path), []),
            )
        )
    new_manifest = {
        "chunks": kept_chunks,
        **build_manifest(manifest_entries),
    }
    _save_json(MANIFEST_PATH, new_manifest)
