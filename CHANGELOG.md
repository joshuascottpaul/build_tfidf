# Changelog

## 0.1.12
- Add `serve` subcommand with a local web UI for building and searching indexes.
- Web UI displays index statistics (model, dimensions, chunk/file counts, weights, build date).
- Build panel with configurable file types, chunking strategy, and code fence stripping.
- Reindex button rebuilds using saved settings from the existing index metadata.
- Delete and rebuild option for changing settings from scratch.
- Search panel with adjustable top-k, fusion method, weight overrides, HyDE, and all-chunks toggle.
- Store `file_types` in index metadata for settings recall.
- New optional extra: `pip install "build-tfidf[web]"` (requires flask).

## 0.1.0
- Initial scaffolding.
## 0.1.5
- Pin openai to 1.61.0 to avoid jiter Rust build in Homebrew.
- Keep faiss-cpu pinned to 1.10.0 for PyPI availability.
- Add dependency pin rationale to README.

## 0.1.6
- Add explicit transitive runtime pins for Homebrew installs.
- Document relocation skip for tiktoken.

## 0.1.7
- Add query helpers for open, reveal, and clipboard.
- Add paths-only output and default file dedupe.

## 0.1.8
- Add exceptiongroup to runtime deps for Homebrew on Python 3.10.

## 0.1.9
- Show help on no args.
- Fix shorthand query parsing.

## 0.1.11
- Add Reciprocal Rank Fusion (`--fusion rrf`) as alternative to min-max score normalization.
- Batch Ollama embeddings via `/api/embed` endpoint with fallback to legacy API.
- Per-provider default fusion weights (openai 0.7/0.3, fastembed/ollama 0.6/0.4).
- Add `--weight-semantic` and `--weight-lexical` overrides for search.
- Add semantic chunking (`--chunking semantic`) for topic-boundary splitting.
- Add HyDE query expansion (`--hyde`) for better recall on short queries.
- Fix brew CI job to tolerate native extension relocation warnings.

## 0.1.10
- Preserve flags in shorthand queries like `--open` and `--pbcopy`.
- Expand CLI help examples and query options.
