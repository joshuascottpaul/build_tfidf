# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A hybrid semantic + lexical search CLI (`tfidf-search`) for local Markdown/text corpora. Despite the name, it uses dense vector embeddings (FAISS) + BM25 lexical search with score fusion -- not classical TF-IDF.

## Build and Run

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
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

- `cli.py` -- Argument parsing, subcommand dispatch, shorthand injection (bare `"query"` args become `search "query"`)
- `index.py` -- Core orchestration: `build()`, `update()`, `query()`. This is where the pipeline is wired together.
- `embeddings.py` -- Provider abstraction for OpenAI, fastembed, and Ollama. Config loaded from env vars.
- `ingest.py` -- File discovery (`iter_files`) and text reading across file types
- `cleaning.py` -- Strips YAML frontmatter, optional code fence removal, whitespace normalization
- `chunking.py` -- Token-based sliding window (tiktoken cl100k_base, 800 tokens default, 100 overlap) with Markdown heading context
- `vector_store.py` -- FAISS IndexFlatIP wrapper (L2-normalized vectors = cosine similarity)
- `lexical.py` -- BM25Okapi wrapper
- `scoring.py` -- Min-max normalization + weighted fusion (default 0.7 semantic / 0.3 lexical)
- `manifest.py` -- Tracks file metadata for incremental updates
- `metadata.py` -- Index signature computation and validation

Index files are stored in `<corpus_root>/.tfidf-index/` (index.faiss, vectors.npy, metadata.json, manifest.json, lexical.json).

## Dependency Pinning

Versions are pinned in `requirements.txt` for Homebrew compatibility. Key constraints:
- `faiss-cpu==1.10.0` -- needs PyPI wheels; newer versions lack sdist
- `openai==1.61.0` -- 2.x pulls `jiter` which needs Rust to build from sdist
- Optional extras (`fastembed`, `flashrank`, `watchdog`, `unstructured`) are unpinned

Version is declared in both `pyproject.toml` and `build_tfidf/__init__.py` -- keep them in sync.

## Style and Communication

Per CONTRIBUTING.md: no em dash, no emoji, brief/factual/calm responses. CLI only -- no web UI or services.
