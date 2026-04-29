"""Index build, update, and query orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from .chunking import Chunk, chunk_text, chunk_text_semantic
from .cleaning import clean_text
from .embeddings import EmbeddingConfig, embed_texts
from .hyde import generate_hypothetical
from .ingest import DEFAULT_EXCLUDE_DIRS, iter_files, read_text_strict, sha256_text
from .manifest import ManifestEntry, build_manifest
from .lexical import LexicalIndex, build_index as build_lexical, search as search_lexical
from .metadata import IndexMetadata, validate_signature
from .rerank import RerankConfig, rerank
from .scoring import fuse_scores
from .vector_store import build_index as build_vector, load as load_vector, save as save_vector, search


INDEX_DIR_NAME = ".tfidf-index"

SCHEMA_VERSION = 1
CLEANING_RULES = "front_matter,optional_code_fences,normalize_whitespace"


def _data_dir(root: Path) -> Path:
    return root / INDEX_DIR_NAME


def _vec_path(root: Path) -> Path:
    return _data_dir(root) / "index.faiss"


def _vectors_path(root: Path) -> Path:
    return _data_dir(root) / "vectors.npy"


def _meta_path(root: Path) -> Path:
    return _data_dir(root) / "metadata.json"


def _manifest_path(root: Path) -> Path:
    return _data_dir(root) / "manifest.json"


def _lex_path(root: Path) -> Path:
    return _data_dir(root) / "lexical.json"


def _ensure_data_dir(root: Path) -> None:
    _data_dir(root).mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_chunks(
    paths: list[Path],
    remove_code: bool,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str = "token",
    embed_config: EmbeddingConfig | None = None,
) -> tuple[list[Chunk], dict[str, str]]:
    """Returns (chunks, path->text mapping) to avoid re-reading files."""
    all_chunks: list[Chunk] = []
    texts: dict[str, str] = {}
    for path in paths:
        text = read_text_strict(path)
        if text is None:
            continue
        texts[str(path)] = text
        cleaned = clean_text(text, remove_code=remove_code)
        if chunking_strategy == "semantic" and embed_config is not None:
            embed_fn = lambda t: embed_texts(t, embed_config)
            all_chunks.extend(chunk_text_semantic(path, cleaned, embed_fn=embed_fn, max_tokens=chunk_size))
        else:
            all_chunks.extend(chunk_text(path, cleaned, max_tokens=chunk_size, overlap=chunk_overlap))
    return all_chunks, texts


def build(
    root: Path,
    embed_config: EmbeddingConfig,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    weight_semantic: float | None = None,
    weight_lexical: float | None = None,
    remove_code: bool = False,
    file_types: set[str] | None = None,
    chunking_strategy: str = "token",
) -> None:
    _ensure_data_dir(root)
    default_sem, default_lex = embed_config.default_weights()
    weight_semantic = weight_semantic if weight_semantic is not None else default_sem
    weight_lexical = weight_lexical if weight_lexical is not None else default_lex
    paths = iter_files(root, file_types=file_types, exclude_dirs=DEFAULT_EXCLUDE_DIRS)
    all_chunks, path_texts = _build_chunks(
        paths, remove_code, chunk_size, chunk_overlap,
        chunking_strategy=chunking_strategy, embed_config=embed_config,
    )

    if not all_chunks:
        raise ValueError("No chunks to index. Check that the corpus contains files matching the specified file types.")

    chunk_texts = [c.text for c in all_chunks]
    vectors = embed_texts(chunk_texts, embed_config)
    if not vectors or not vectors[0]:
        raise ValueError("Embedding provider returned no vectors. Check your API key or provider configuration.")
    vindex = build_vector(vectors)
    save_vector(vindex, _vec_path(root))
    _save_vectors(vectors, root)

    _save_json(_lex_path(root), {"texts": chunk_texts})

    meta = IndexMetadata(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        embedding_model=embed_config.model,
        embedding_dimensions=len(vectors[0]),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        cleaning_rules=f"{CLEANING_RULES}|remove_code={remove_code}",
        vector_backend="faiss",
        weight_semantic=weight_semantic,
        weight_lexical=weight_lexical,
        chunking_strategy=chunking_strategy,
        file_types=",".join(sorted(file_types)) if file_types else "md",
    )
    _save_json(_meta_path(root), meta.to_dict())

    chunk_map: dict[str, list[int]] = {}
    for idx, c in enumerate(all_chunks):
        chunk_map.setdefault(str(c.path), []).append(idx)

    manifest_entries = []
    for path in paths:
        text = path_texts.get(str(path))
        if text is None:
            continue
        manifest_entries.append(
            ManifestEntry(
                path=str(path),
                sha256=sha256_text(text),
                mtime=path.stat().st_mtime,
                chunk_indices=chunk_map.get(str(path), []),
            )
        )

    manifest = {
        "chunks": [{**asdict(c), "path": str(c.path)} for c in all_chunks],
        **build_manifest(manifest_entries),
    }
    _save_json(_manifest_path(root), manifest)


def _load_lexical(root: Path) -> LexicalIndex:
    data = _load_json(_lex_path(root))
    return build_lexical(data["texts"])


def _save_vectors(vectors: list[list[float]], root: Path) -> None:
    import numpy as np

    arr = np.array(vectors, dtype="float32")
    _data_dir(root).mkdir(parents=True, exist_ok=True)
    np.save(_vectors_path(root), arr)


def _load_vectors(root: Path) -> list[list[float]]:
    import numpy as np

    return np.load(_vectors_path(root)).tolist()


def query(
    query_text: str,
    embed_config: EmbeddingConfig,
    root: Path = Path("."),
    top_k: int = 10,
    weight_semantic: float | None = None,
    weight_lexical: float | None = None,
    fusion_method: str = "minmax",
    rerank_model: str | None = None,
    rerank_top_n: int = 30,
    dedupe_by_path: bool = True,
    hyde: bool = False,
) -> list[tuple[dict, float]]:
    meta = _load_json(_meta_path(root))
    validate_signature(meta)
    default_sem, default_lex = embed_config.default_weights()
    weight_semantic = weight_semantic if weight_semantic is not None else default_sem
    weight_lexical = weight_lexical if weight_lexical is not None else default_lex

    vindex = load_vector(_vec_path(root))
    embed_query = query_text
    if hyde:
        embed_query = generate_hypothetical(query_text, embed_config.provider)
    query_vec = embed_texts([embed_query], embed_config)[0]
    fetch_k = rerank_top_n if rerank_model else top_k
    sem_hits = search(vindex, query_vec, top_k=fetch_k * 5)
    sem_scores = {idx: score for idx, score in sem_hits if idx >= 0}

    lex_index = _load_lexical(root)
    lex_hits = search_lexical(lex_index, query_text, top_k=fetch_k * 5)
    lex_scores = {idx: score for idx, score in lex_hits}

    fused = fuse_scores(sem_scores, lex_scores, weight_semantic, weight_lexical, method=fusion_method)
    manifest = _load_json(_manifest_path(root))

    results = []
    for idx, score in fused[:fetch_k]:
        chunk = manifest["chunks"][idx]
        results.append((chunk, score))

    if rerank_model:
        rerank_cfg = RerankConfig(model=rerank_model, top_n=rerank_top_n)
        reranked = rerank(query_text, [c for c, _ in results[:rerank_top_n]], rerank_cfg)
        seen_ids: set[str] = {c["sha256"] for c in reranked[:top_k]}
        reranked_set = [(c, s) for (c, s) in results if c["sha256"] in seen_ids]
        if len(reranked_set) < top_k:
            for item in results:
                if item[0]["sha256"] in seen_ids:
                    continue
                seen_ids.add(item[0]["sha256"])
                reranked_set.append(item)
                if len(reranked_set) >= top_k:
                    break
        results = reranked_set

    if not dedupe_by_path:
        return results[:top_k]

    seen: set[str] = set()
    deduped = []
    for chunk, score in results:
        path = chunk.get("path")
        if path in seen:
            continue
        seen.add(path)
        deduped.append((chunk, score))
        if len(deduped) >= top_k:
            break
    return deduped


def update(
    root: Path,
    embed_config: EmbeddingConfig,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    weight_semantic: float | None = None,
    weight_lexical: float | None = None,
    remove_code: bool = False,
    file_types: set[str] | None = None,
    chunking_strategy: str = "token",
) -> None:
    _ensure_data_dir(root)

    if not _meta_path(root).exists():
        build(root, embed_config, chunk_size, chunk_overlap, weight_semantic, weight_lexical, remove_code, file_types, chunking_strategy=chunking_strategy)
        return

    meta = _load_json(_meta_path(root))
    validate_signature(meta)
    expected_rules = f"{CLEANING_RULES}|remove_code={remove_code}"
    if str(meta.get("cleaning_rules")) != expected_rules:
        raise SystemExit("Index config mismatch. Rebuild required.")

    current_paths = iter_files(root, file_types=file_types, exclude_dirs=DEFAULT_EXCLUDE_DIRS)
    full_manifest = _load_json(_manifest_path(root))
    entries = {e["path"]: e for e in full_manifest.get("entries", [])}

    current_set = {str(p) for p in current_paths}
    previous_set = set(entries.keys())
    removed_paths = previous_set - current_set

    changed_paths: list[Path] = []
    path_texts: dict[str, str] = {}
    for path in current_paths:
        text = read_text_strict(path)
        if text is None:
            continue
        path_texts[str(path)] = text
        sha = sha256_text(text)
        mtime = path.stat().st_mtime
        prev = entries.get(str(path))
        if not prev or prev["sha256"] != sha or prev["mtime"] != mtime:
            changed_paths.append(path)

    if not changed_paths and not removed_paths:
        return

    existing_chunks: list[dict] = full_manifest.get("chunks", [])
    existing_vectors = _load_vectors(root)
    existing_texts = _load_json(_lex_path(root)).get("texts", [])

    remove_set = {str(p) for p in changed_paths} | set(removed_paths)
    kept_chunks: list[dict] = []
    kept_vectors: list[list[float]] = []
    kept_texts: list[str] = []
    for idx, chunk in enumerate(existing_chunks):
        if chunk["path"] in remove_set:
            continue
        kept_chunks.append(chunk)
        kept_vectors.append(existing_vectors[idx])
        kept_texts.append(existing_texts[idx])

    new_chunks: list[Chunk] = []
    for path in changed_paths:
        text = path_texts.get(str(path))
        if text is None:
            continue
        cleaned = clean_text(text, remove_code=remove_code)
        if chunking_strategy == "semantic":
            embed_fn = lambda t: embed_texts(t, embed_config)
            new_chunks.extend(chunk_text_semantic(path, cleaned, embed_fn=embed_fn, max_tokens=chunk_size))
        else:
            new_chunks.extend(chunk_text(path, cleaned, max_tokens=chunk_size, overlap=chunk_overlap))

    if new_chunks:
        new_vectors = embed_texts([c.text for c in new_chunks], embed_config)
        for c, v in zip(new_chunks, new_vectors):
            kept_chunks.append({**asdict(c), "path": str(c.path)})
            kept_texts.append(c.text)
            kept_vectors.append(v)

    _save_vectors(kept_vectors, root)
    vindex = build_vector(kept_vectors)
    save_vector(vindex, _vec_path(root))
    _save_json(_lex_path(root), {"texts": kept_texts})

    chunk_map: dict[str, list[int]] = {}
    for idx, c in enumerate(kept_chunks):
        chunk_map.setdefault(c["path"], []).append(idx)

    manifest_entries = []
    for path in current_paths:
        text = path_texts.get(str(path))
        if text is None:
            continue
        manifest_entries.append(
            ManifestEntry(
                path=str(path),
                sha256=sha256_text(text),
                mtime=path.stat().st_mtime,
                chunk_indices=chunk_map.get(str(path), []),
            )
        )
    _save_json(_manifest_path(root), {"chunks": kept_chunks, **build_manifest(manifest_entries)})
