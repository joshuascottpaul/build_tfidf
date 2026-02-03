# build_tfidf

High‑quality semantic search for Markdown corpora.

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

## Notes
- OpenAI embeddings require `OPENAI_API_KEY` in the environment.
- For offline mode, set `EMBEDDING_PROVIDER=ollama`.
