# build_tfidf Index

This repo ships `build_tfidf`, a semantic search CLI for Markdown corpora. It indexes Markdown files, builds embeddings, and supports hybrid semantic plus lexical ranking. It is designed for high quality retrieval and repeatable results.

## Prerequisites
- Python 3.10
- OpenAI API key for best quality or local Ollama for offline mode
- Homebrew optional for install

## Install Options

### Option 1. Homebrew
```bash
brew install joshuascottpaul/build-tfidf/build-tfidf
tfidf-search --help
```

Note
- The formula skips relocation because native wheels like `tiktoken` and `jiter` are not relocatable.
- Risk. If you move the Cellar or use a nonstandard prefix, reinstall is required.

### Option 2. Python venv
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt
```

## Build the Index
```bash
tfidf-search build --root /path/to/corpus
```

Optional
```bash
tfidf-search build --remove-code
```

## Query the Index
```bash
tfidf-search query "your query"
tfidf-search query "your query" --top 10
```

## Update the Index
```bash
tfidf-search update --root /path/to/corpus
tfidf-search update --remove-code
```

## Inspect a Chunk
```bash
tfidf-search inspect <chunk_id>
```

## Configuration
```bash
export OPENAI_API_KEY="..."
export EMBEDDING_PROVIDER=openai
export OPENAI_MODEL=text-embedding-3-large
export FALLBACK_TO_OLLAMA=true
export OLLAMA_MODEL=nomic-embed-text
```

## Quality Notes
- Hybrid ranking uses semantic plus BM25 to preserve exact term recall.
- Optional rerank improves precision for ambiguous queries.

## Troubleshooting
- Missing deps. Activate venv and run `pip install -r requirements.txt && pip install -e .`.
- Missing index. Run `tfidf-search build` first.
- Homebrew warnings about relocation are expected. The install still succeeds.
