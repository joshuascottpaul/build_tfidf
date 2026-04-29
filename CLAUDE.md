# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A hybrid semantic + lexical search tool (`tfidf-search`) for local Markdown/text corpora. Despite the name, it uses dense vector embeddings (FAISS) + BM25 lexical search with score fusion -- not classical TF-IDF. CLI-first with an optional web UI and JSON API.

## Build and Run

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[web]"    # include web extra for serve command
```

The CLI entry point is `tfidf-search`, defined in `pyproject.toml` as `build_tfidf.cli:main`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q                    # run all tests
pytest tests/test_cli_smoke.py  # run a single test file
pytest -k test_name          # run a single test by name
```

Tests use pytest (pinned at 9.0.2). CI runs on Python 3.10 (`ubuntu-latest`) and also does a Homebrew smoke test on `macos-latest`.

## Architecture

The data flow is: **ingest -> clean -> chunk -> embed -> index -> query -> fuse -> (rerank)**

Key modules in `build_tfidf/`:

- `cli.py` -- Argument parsing, subcommand dispatch (`build`, `update`, `search`, `watch`, `inspect`, `serve`), shorthand injection (bare `"query"` args become `search "query"`)
- `index.py` -- Core orchestration: `build()`, `update()`, `query()`. This is where the pipeline is wired together.
- `embeddings.py` -- Provider abstraction for OpenAI, fastembed, and Ollama. `EmbeddingConfig` dataclass, `load_config_from_env()` for config from env vars.
- `ingest.py` -- File discovery (`iter_files`) and text reading across file types
- `cleaning.py` -- Strips YAML frontmatter, optional code fence removal, whitespace normalization
- `chunking.py` -- Token-based sliding window (tiktoken cl100k_base, 800 tokens default, 100 overlap) with Markdown heading context
- `vector_store.py` -- FAISS IndexFlatIP wrapper (L2-normalized vectors = cosine similarity)
- `lexical.py` -- BM25Okapi wrapper
- `scoring.py` -- Min-max normalization + weighted fusion (default 0.7 semantic / 0.3 lexical)
- `manifest.py` -- Tracks file metadata for incremental updates
- `metadata.py` -- Index signature computation and validation. `IndexMetadata` stores schema version, embedding model/dims, chunk size, weights, chunking strategy, and file_types.
- `web.py` -- Optional Flask web UI and JSON API (`serve` subcommand). Lazy-imports flask so the module is safe to skip when flask is not installed. All HTML/CSS/JS is inline (no external files or build step).

Index files are stored in `<corpus_root>/.tfidf-index/` (index.faiss, vectors.npy, metadata.json, manifest.json, lexical.json).

## Web UI and API

The `serve` subcommand starts a Flask server exposing:
- `GET /` -- single-page web UI with search, build controls, stats, and help modal
- `GET /api/status` -- index status and stats (reads metadata.json + manifest.json)
- `POST /api/build` -- incremental update (accepts file_types, chunking, remove_code)
- `POST /api/rebuild` -- delete index and build fresh
- `POST /api/search` -- search with configurable top_k, fusion, weights, hyde, all_chunks
- `POST /api/delete` -- remove a file from corpus and update index (path traversal protected)

Search results include `filename` (bare name) alongside `path` for mapping back to external record IDs (e.g. FileMaker integration).

See [API.md](API.md) for full endpoint reference. See [QUICKSTART.md](QUICKSTART.md) for FileMaker integration guide.

## Dependency Pinning

Versions are pinned in `requirements.txt` for Homebrew compatibility. Key constraints:
- `faiss-cpu==1.10.0` -- needs PyPI wheels; newer versions lack sdist
- `openai==1.61.0` -- 2.x pulls `jiter` which needs Rust to build from sdist
- Optional extras (`fastembed`, `flashrank`, `watchdog`, `unstructured`, `web`) are unpinned

Version is declared in both `pyproject.toml` and `build_tfidf/__init__.py` -- keep them in sync.

## Release Process

1. Bump version in `pyproject.toml` and `build_tfidf/__init__.py`
2. Update CHANGELOG.md
3. Commit, push, create GitHub release (`gh release create vX.Y.Z`)
4. Get tarball SHA: `curl -sL <tarball_url> | shasum -a 256`
5. Update `homebrew-build-tfidf` formula (URL + SHA) via GitHub API
6. Verify: `brew update && brew upgrade joshuascottpaul/build-tfidf/build-tfidf`

## Style and Communication

Per CONTRIBUTING.md: no em dash, no emoji, brief/factual/calm responses. CLI-first; optional web UI via `tfidf-search serve`.
