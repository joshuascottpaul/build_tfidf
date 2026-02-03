# build_tfidf

High-quality semantic search for Markdown corpora.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build index
tfidf-search build

# Query
tfidf-search query "your query"
```

## CLI
- `tfidf-search build`
- `tfidf-search update`
- `tfidf-search query "..."`
- `tfidf-search inspect <chunk_id>`

## Dependency Pins and Rationale
We pin versions for reliability and Homebrew compatibility.

Current pins that matter most
- `faiss-cpu==1.10.0`
  - Reason: PyPI provides wheels but no source distribution for newer versions.
  - Impact: Homebrew resource vendoring requires sdists, so this keeps brew installs viable.

- `openai==1.61.0`
  - Reason: OpenAI 2.x depends on `jiter`, which requires Rust tooling to build from sdist.
  - Impact: Homebrew builds from sdists and fails without Rust, so pinning avoids that.

How we will address this in the future
1. If `faiss-cpu` publishes sdists again, we will raise the pin and update Homebrew resources.
2. If `openai` 2.x offers sdist that does not require Rust or ships wheels for all brew paths, we will upgrade.
3. We will validate upgrades by running `brew reinstall build-tfidf` and `tfidf-search --help`.

## Notes
- OpenAI embeddings require `OPENAI_API_KEY` in the environment.
- For offline mode, set `EMBEDDING_PROVIDER=ollama`.
