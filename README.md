# build_tfidf

High-quality hybrid semantic + lexical search for local Markdown corpora.


## Installation

### Quick Install with Package Managers

**Using [ubi](https://github.com/houseabsolute/ubi):**
```bash
ubi --project joshuascottpaul/build_tfidf --in ~/.local/bin
```

**Using [bin](https://github.com/marcosnils/bin):**
```bash
bin install github.com/joshuascottpaul/build_tfidf
```

### Manual Install

```bash
git clone https://github.com/joshuascottpaul/build_tfidf.git
cd build_tfidf
pip install -r requirements.txt  # if requirements.txt exists
```

### From Release

```bash
curl -L https://github.com/joshuascottpaul/build_tfidf/releases/latest/download/build_tfidf-v0.1.0-darwin-arm64.tar.gz | tar xz
cd build_tfidf-darwin-arm64
./install.sh
```

## Quickstart
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Build index
tfidf-search build

# Search
tfidf-search "your query"
tfidf-search search "your query"
tfidf-search --search "your query"
```

## CLI

### build
```
tfidf-search build [--root DIR] [--remove-code] [--file-types md,txt,html,docx]
                   [--embedding-provider {openai,fastembed,ollama}]
                   [--chunking {token,semantic}]
```
Builds the index from scratch. Defaults to `--root .` and `--file-types md`.

- `--chunking semantic` -- split text at topic boundaries using embedding similarity instead of fixed token windows

### update
```
tfidf-search update [--root DIR] [--remove-code] [--file-types md,html,docx]
                    [--chunking {token,semantic}]
```
Incrementally re-indexes only changed or new files.

### search
```
tfidf-search search TEXT [--top N] [--rerank-model MODEL] [--rerank-top N]
                         [--fusion {minmax,rrf}] [--hyde]
                         [--weight-semantic F] [--weight-lexical F]
                         [--all-chunks] [--open N] [--reveal N] [--pbcopy N]
                         [--paths-only]
tfidf-search TEXT        # shorthand
tfidf-search --search TEXT
```
- `--top N` -- number of results (default 10)
- `--fusion {minmax,rrf}` -- score fusion method (default: minmax). `rrf` uses Reciprocal Rank Fusion, which is rank-based and ignores raw score magnitudes.
- `--hyde` -- use HyDE (Hypothetical Document Embeddings) for query expansion. Generates a hypothetical answer via LLM and embeds that for semantic search. Requires OpenAI or Ollama.
- `--weight-semantic F` -- override semantic fusion weight (default: per-provider)
- `--weight-lexical F` -- override lexical fusion weight (default: per-provider)
- `--rerank-model MODEL` -- flashrank model name to re-rank results (e.g. `ms-marco-MiniLM-L-12-v2`)
- `--all-chunks` -- show multiple chunks per file instead of deduping by path
- `--open N` -- open result N in default app
- `--reveal N` -- reveal result N in Finder
- `--pbcopy N` -- copy result N path to clipboard
- `--paths-only` -- print file paths only (useful for scripts)

### watch
```
tfidf-search watch [--root DIR] [--file-types md,html,docx] [--debounce SECS]
```
Watches the corpus directory and automatically runs `update` when files change.
Rapid saves are debounced (default 1.5s) into a single update. Requires `watchdog`:
```bash
pip install "build-tfidf[watchdog]"
```

### inspect
```
tfidf-search inspect CHUNK_ID
```
Prints the stored chunk JSON for a given chunk sha256.

## Embedding Providers

### OpenAI (default)
```bash
export OPENAI_API_KEY=sk-...
export EMBEDDING_PROVIDER=openai        # default
export OPENAI_MODEL=text-embedding-3-large
export DIMENSIONS=                      # optional, reduces output dims
export BATCH_SIZE=32
export RPM_LIMIT=60
export FALLBACK_TO_OLLAMA=false
```

### fastembed — local, no API key
```bash
pip install "build-tfidf[fastembed]"
export EMBEDDING_PROVIDER=fastembed
export FASTEMBED_MODEL=BAAI/bge-small-en-v1.5   # default
```
Downloads model weights (~100MB) on first use. Fully offline after that.

### Ollama
```bash
export EMBEDDING_PROVIDER=ollama
export OLLAMA_MODEL=nomic-embed-text
```
Requires a running Ollama instance at `http://localhost:11434`.

## Re-ranking

Re-ranking uses [flashrank](https://github.com/PrithivirajDamodharan/FlashRank), a local cross-encoder. No API key required.

```bash
pip install "build-tfidf[flashrank]"
tfidf-search query "your query" --rerank-model ms-marco-MiniLM-L-12-v2 --rerank-top 30
```

Available models: `ms-marco-MiniLM-L-12-v2` (default), `ms-marco-MultiBERT-L-12`, `rank-T5-flan`.

## Score Fusion

Two fusion methods are available for combining semantic and lexical search results:

- **minmax** (default) -- min-max normalizes scores from each index, then combines with weighted sum. Default weights are per-provider: OpenAI 0.7/0.3, fastembed and Ollama 0.6/0.4.
- **rrf** -- Reciprocal Rank Fusion. Rank-based, ignores raw score magnitudes. More robust when score distributions differ between semantic and lexical results.

```bash
tfidf-search search "your query" --fusion rrf
tfidf-search search "your query" --weight-semantic 0.8 --weight-lexical 0.2
```

## HyDE (Hypothetical Document Embeddings)

HyDE improves recall for short or vague queries by generating a hypothetical answer via an LLM, then embedding that answer for semantic search. BM25 lexical search still uses the original query.

```bash
tfidf-search search "your query" --hyde
```

Requires OpenAI or Ollama (fastembed has no LLM, falls back to raw query). Configure the generation model with `HYDE_MODEL` env var (defaults to `gpt-4o-mini` for OpenAI, `llama3.2` for Ollama).

## Semantic Chunking

By default, text is split into fixed-size token windows. Semantic chunking detects topic boundaries by measuring embedding similarity between adjacent paragraphs and splits there instead.

```bash
tfidf-search build --chunking semantic
tfidf-search update --chunking semantic
```

This adds embedding calls at build time (one per paragraph). Falls back to token chunking for files with fewer than 3 paragraphs.

## Indexing non-Markdown files

Plain text (`.txt`) files are supported natively. HTML and DOCX files require [unstructured](https://github.com/Unstructured-IO/unstructured):

```bash
pip install "build-tfidf[unstructured]"
tfidf-search build --file-types md,html,docx
tfidf-search watch --file-types md,html,docx
```

## Optional extras summary

| Extra | Installs | Enables |
|---|---|---|
| `fastembed` | `fastembed` | Local embeddings, no API key |
| `flashrank` | `flashrank` | Local cross-encoder re-ranking |
| `watchdog` | `watchdog` | `tfidf-search watch` command |
| `unstructured` | `unstructured[docx,html]` | Index `.html` and `.docx` files |

Install multiple at once:
```bash
pip install "build-tfidf[fastembed,flashrank,watchdog]"
```

## Dependency Pins and Rationale
We pin versions for reliability and Homebrew compatibility.

Current pins that matter most:
- `faiss-cpu==1.10.0` — PyPI provides wheels but no sdist for newer versions; Homebrew requires sdists.
- `openai==1.61.0` — OpenAI 2.x pulls `jiter` which requires Rust to build from sdist; Homebrew fails without Rust.

Optional extras are unpinned — install them outside Homebrew.

## Notes
- If `tfidf-search` is not found, confirm your venv is active and run `pip install -e .`.
- For tests: `pip install -r requirements-dev.txt && python3.10 -m pytest tests/`

## Homebrew Install Strategy
- Homebrew installs binary wheels at install time using `pip --only-binary :all:`.
- `tiktoken` wheel relocation is skipped to avoid install errors.
- Optional extras (`fastembed`, `flashrank`, etc.) are not part of the Homebrew formula — install them manually after `brew install`.
