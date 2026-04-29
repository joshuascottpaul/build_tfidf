# Quickstart

Get semantic search running on a local folder in under 5 minutes.

## 1. Install

**From source (recommended for web UI):**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[web]"
```

**From Homebrew:**
```bash
brew install joshuascottpaul/build-tfidf/build-tfidf
```
The Homebrew formula installs core dependencies only. To use the web UI (`serve` command), install flask into the brew venv:
```bash
/opt/homebrew/opt/build-tfidf/libexec/bin/pip install flask
```

## 2. Set up embeddings

Pick one provider:

**OpenAI (default):**
```bash
export OPENAI_API_KEY=sk-...
```

**Local with fastembed (no API key):**
```bash
pip install "build-tfidf[fastembed]"
export EMBEDDING_PROVIDER=fastembed
```

**Ollama:**
```bash
export EMBEDDING_PROVIDER=ollama
# Requires ollama running at localhost:11434
```

## 3. Build and search (CLI)

```bash
# Index a folder of Markdown/text files
tfidf-search build --root ~/notes

# Search
tfidf-search search "your query" --root ~/notes

# Or use the shorthand
tfidf-search "your query" --root ~/notes
```

## 4. Build and search (web UI)

```bash
tfidf-search serve --root ~/notes
# Open http://127.0.0.1:8080
```

The web UI lets you build/rebuild the index, adjust search options, and browse results.

## 5. Build and search (API)

```bash
# Start the server
tfidf-search serve --root ~/notes &

# Build the index
curl -X POST http://localhost:8080/api/build \
  -H "Content-Type: application/json" \
  -d '{"file_types": "md,txt"}'

# Search
curl "http://localhost:8080/api/search?q=meeting+notes&top_k=5"

# Delete a document
curl -X POST http://localhost:8080/api/delete \
  -H "Content-Type: application/json" \
  -d '{"filename": "old-doc.txt"}'
```

See [API.md](API.md) for the full API reference.

---

## FileMaker integration

Use tfidf-search as a search backend for FileMaker Pro (22+).

### Setup

1. Install tfidf-search on the FileMaker server machine
2. Create a corpus folder (e.g. `/opt/corpus/`)
3. Start the API server:
   ```bash
   tfidf-search serve --root /opt/corpus --host 0.0.0.0 --port 8080
   ```

### Indexing documents

When a user adds a PDF to a container field:

```
# FileMaker script
Set Variable [ $text ; GetTextFromPDF ( Documents::PDF_Container ) ]
If [ $text = "?" ]
  Show Custom Dialog [ "Could not extract text from this PDF." ]
  Exit Script
End If

# Save text to corpus folder, named by record ID
Set Variable [ $path ; "/opt/corpus/" & Documents::RecordID & ".txt" ]
Export Field Contents [ Documents::TextField ; $path ]

# Trigger index update
Insert from URL [ $$result ;
  "http://localhost:8080/api/build" ;
  cURL options: "-X POST -H \"Content-Type: application/json\" -d {}" ]
```

### Searching

```
# FileMaker script
Set Variable [ $query ; Documents::SearchField ]
Insert from URL [ $$result ;
  "http://localhost:8080/api/search?q=" & GetAsURLEncoded ( $query ) & "&top_k=10" ]

# Parse results
Set Variable [ $count ; JSONListCount ( JSONGetElement ( $$result ; "results" ) ; "" ) ]
Set Variable [ $i ; 0 ]
Loop
  Exit Loop If [ $i >= $count ]
  Set Variable [ $filename ; JSONGetElement ( $$result ; "results[" & $i & "].filename" ) ]
  Set Variable [ $score ; JSONGetElement ( $$result ; "results[" & $i & "].score" ) ]
  Set Variable [ $heading ; JSONGetElement ( $$result ; "results[" & $i & "].heading" ) ]
  Set Variable [ $text ; JSONGetElement ( $$result ; "results[" & $i & "].text" ) ]
  # $filename is "<RecordID>.txt" -- parse the ID to navigate to the source record
  Set Variable [ $recordID ; Substitute ( $filename ; ".txt" ; "" ) ]
  # ... add row to search results layout ...
  Set Variable [ $i ; $i + 1 ]
End Loop
```

### Deleting documents

```
# FileMaker script -- when a record is deleted
Insert from URL [ $$result ;
  "http://localhost:8080/api/delete" ;
  cURL options: "-X POST -H \"Content-Type: application/json\" -d {\"filename\": \"" & Documents::RecordID & ".txt\"}" ]
```

### Tips

- Name corpus files by record ID (e.g. `12345.txt`) so the `filename` field in search results maps directly back to FileMaker records.
- `GetTextFromPDF` returns `?` for scanned/image-only PDFs. Check for this and flag the record so the user knows OCR is needed.
- For large imports, batch the text file exports first, then call `/api/build` once at the end instead of after each file.
- Use `--host 0.0.0.0` if FileMaker Server and tfidf-search are on different machines. Use `127.0.0.1` (default) if they're on the same machine.
- The API is synchronous. A single-document update takes a few seconds. For bulk operations, consider running the build as a server-side scheduled script rather than blocking the FileMaker user.
