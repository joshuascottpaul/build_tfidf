# API Reference

The `tfidf-search serve` command starts a local HTTP server that exposes a JSON API for building, searching, and managing indexes.

```bash
pip install "build-tfidf[web]"
tfidf-search serve --root /path/to/corpus --port 8080
```

Base URL: `http://127.0.0.1:8080`

All responses are JSON. Errors return `{"error": "message"}` with an appropriate HTTP status code.

---

## GET /api/status

Check whether an index exists and get its configuration.

**Request:**
```
GET /api/status
```

**Response:**
```json
{
  "has_index": true,
  "root": "/absolute/path/to/corpus",
  "index_stats": {
    "schema_version": 1,
    "created_at": "2026-04-29T12:00:00+00:00",
    "embedding_model": "text-embedding-3-large",
    "embedding_dimensions": 3072,
    "chunk_size": 800,
    "chunk_overlap": 100,
    "chunking_strategy": "token",
    "file_types": "md,txt",
    "weight_semantic": 0.7,
    "weight_lexical": 0.3,
    "cleaning_rules": "front_matter,optional_code_fences,normalize_whitespace|remove_code=False",
    "num_chunks": 142,
    "num_files": 23
  }
}
```

When no index exists, `has_index` is `false` and `index_stats` is `null`.

---

## POST /api/build

Incrementally update the index. Creates a new index if none exists. Only re-embeds changed or new files.

**Request:**
```
POST /api/build
Content-Type: application/json

{
  "file_types": "md,txt",
  "chunking": "token",
  "remove_code": false
}
```

All fields are optional. Defaults: `file_types` = `"md"`, `chunking` = `"token"`, `remove_code` = `false`.

**Response:**
```json
{"ok": true}
```

---

## POST /api/rebuild

Delete the existing index and build from scratch. Use this when changing settings (embedding provider, chunking strategy, file types).

**Request:**
```
POST /api/rebuild
Content-Type: application/json

{
  "file_types": "md,txt",
  "chunking": "semantic",
  "remove_code": true
}
```

All fields are optional. Same defaults as `/api/build`.

**Response:**
```json
{"ok": true}
```

---

## GET /api/search

Search the index. Returns ranked results with scores.

**Request:**
```
GET /api/search?q=your+query&top_k=10&fusion=minmax
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | string | (required) | Search query |
| `top_k` | int | `10` | Number of results |
| `fusion` | string | `minmax` | Score fusion method: `minmax` or `rrf` |
| `weight_semantic` | float | auto | Semantic weight override (0-1) |
| `weight_lexical` | float | auto | Lexical weight override (0-1) |
| `hyde` | string | | Set to `1` to enable HyDE query expansion |
| `all_chunks` | string | | Set to `1` to show multiple chunks per file |

**Response:**
```json
{
  "results": [
    {
      "path": "notes/meeting-2026-04.txt",
      "filename": "meeting-2026-04.txt",
      "heading": "Action Items",
      "text": "First 300 characters of the matching chunk...",
      "score": 0.8421
    }
  ]
}
```

- `path` -- relative path from corpus root
- `filename` -- bare filename, useful for mapping back to external record IDs
- `heading` -- nearest Markdown heading above the chunk (empty string if none)
- `text` -- chunk text content
- `score` -- fused relevance score (higher is better, not normalized to any fixed range)

Returns `{"error": "No index found. Build the index first."}` with status 400 if no index exists.

---

## POST /api/delete

Delete a file from the corpus and update the index.

**Request:**
```
POST /api/delete
Content-Type: application/json

{"filename": "meeting-2026-04.txt"}
```

The `filename` is resolved relative to the corpus root. Path traversal (e.g. `../etc/passwd`) is blocked.

**Response:**
```json
{"ok": true}
```

| Status | Meaning |
|---|---|
| 200 | File deleted, index updated |
| 400 | Missing filename or path traversal attempt |
| 404 | File not found |
| 500 | Index update failed after deletion |

---

## Error format

All endpoints return errors in the same shape:

```json
{"error": "Description of what went wrong"}
```

HTTP status codes: 400 for bad requests, 404 for not found, 500 for server errors.

---

## curl examples

```bash
# Check status
curl http://localhost:8080/api/status

# Build index
curl -X POST http://localhost:8080/api/build \
  -H "Content-Type: application/json" \
  -d '{"file_types": "md,txt"}'

# Rebuild with new settings
curl -X POST http://localhost:8080/api/rebuild \
  -H "Content-Type: application/json" \
  -d '{"chunking": "semantic", "file_types": "md,txt"}'

# Search
curl "http://localhost:8080/api/search?q=embedding+models&top_k=5"

# Search with options
curl "http://localhost:8080/api/search?q=chunking&fusion=rrf&hyde=1&all_chunks=1"

# Delete a file
curl -X POST http://localhost:8080/api/delete \
  -H "Content-Type: application/json" \
  -d '{"filename": "old-notes.txt"}'
```
