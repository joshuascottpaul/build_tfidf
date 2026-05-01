# tfidf-search Cheatsheet

## Setup
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .              # add .[web] to include the serve command
```

## Build
```bash
tfidf-search build
tfidf-search build --root /path/to/corpus
tfidf-search build --remove-code
tfidf-search build --chunking semantic
tfidf-search build --embedding-provider fastembed
tfidf-search build --embedding-provider fastembed --fastembed-threads 2
```

## Update
```bash
tfidf-search update
tfidf-search update --root /path/to/corpus
tfidf-search update --chunking semantic
tfidf-search update --embedding-provider fastembed --fastembed-threads 2
```

## Search
```bash
tfidf-search search "your query"
tfidf-search "your query"              # shorthand
tfidf-search --search "your query"     # shorthand
tfidf-search search "your query" --top 10
tfidf-search search "your query" --fusion rrf
tfidf-search search "your query" --hyde
tfidf-search search "your query" --weight-semantic 0.8 --weight-lexical 0.2
tfidf-search search "your query" --rerank-model ms-marco-MiniLM-L-12-v2 --rerank-top 30
tfidf-search search "your query" --open 1
tfidf-search search "your query" --reveal 1
tfidf-search search "your query" --pbcopy 1
tfidf-search search "your query" --paths-only
tfidf-search search "your query" --all-chunks
```

## Serve (web UI)
```bash
pip install "build-tfidf[web]"
tfidf-search serve
tfidf-search serve --root /path/to/corpus
tfidf-search serve --port 9090
tfidf-search serve --host 0.0.0.0 --port 8080
```

## Inspect
```bash
tfidf-search inspect <chunk_id>
```

## Env
```bash
export OPENAI_API_KEY="..."
export EMBEDDING_PROVIDER=openai          # openai, fastembed, ollama
export OPENAI_MODEL=text-embedding-3-large
export FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
export OLLAMA_MODEL=nomic-embed-text
export FALLBACK_TO_OLLAMA=true
export HYDE_MODEL=gpt-4o-mini             # or llama3.2 for Ollama
```
