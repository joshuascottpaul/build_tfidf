# build_tfidf Cheatsheet

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Build
```bash
tfidf-search build
tfidf-search build --root /path/to/corpus
tfidf-search build --remove-code
```

## Update
```bash
tfidf-search update
tfidf-search update --root /path/to/corpus
tfidf-search update --remove-code
```

## Query
```bash
tfidf-search query "your query"
tfidf-search query "your query" --top 10
tfidf-search query "your query" --rerank-model gpt-4o-mini --rerank-top 30
tfidf-search query "your query" --open 1
tfidf-search query "your query" --reveal 1
tfidf-search query "your query" --pbcopy 1
tfidf-search query "your query" --paths-only
```

## Inspect
```bash
tfidf-search inspect <chunk_id>
```

## Env
```bash
export OPENAI_API_KEY="..."
export EMBEDDING_PROVIDER=openai
export FALLBACK_TO_OLLAMA=true
export OLLAMA_MODEL=nomic-embed-text
```
