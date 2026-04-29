"""Minimal web UI for tfidf-search."""

from __future__ import annotations

from pathlib import Path

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tfidf-search</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; max-width: 740px; margin: 40px auto; padding: 0 16px; color: #222; }
h1 { font-size: 1.3rem; margin-bottom: 16px; }

/* toolbar */
#toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; font-size: 0.85rem; color: #6b7280; flex-wrap: wrap; }
#toolbar .root { font-family: monospace; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.8rem; }
.badge-ok { background: #dcfce7; color: #166534; }
.badge-none { background: #fef3c7; color: #92400e; }

/* stats */
#stats { font-size: 0.82rem; color: #6b7280; margin-bottom: 16px; line-height: 1.6; }
#stats span { margin-right: 16px; white-space: nowrap; }

/* collapsible panels */
details { margin-bottom: 16px; }
summary { cursor: pointer; font-size: 0.9rem; font-weight: 600; color: #374151; }
.panel { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; padding: 10px 0 0; font-size: 0.85rem; }
.panel label { display: flex; flex-direction: column; gap: 2px; }
.panel select, .panel input { padding: 4px 8px; border: 1px solid #d1d5db; border-radius: 4px; font-size: 0.85rem; }
.panel .full { grid-column: 1 / -1; }
.btn { padding: 6px 14px; font-size: 0.85rem; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; }
.btn:hover { background: #f3f4f6; }
.btn:disabled { opacity: 0.5; cursor: default; }
.btn-primary { background: #2563eb; color: #fff; border-color: #2563eb; }
.btn-primary:hover { background: #1d4ed8; }

/* search */
#searchRow { display: flex; gap: 8px; margin-bottom: 8px; }
#searchRow input[type=text] { flex: 1; padding: 8px 12px; font-size: 1rem; border: 1px solid #ccc; border-radius: 4px; }

/* results */
.result { margin-bottom: 14px; padding: 12px; border: 1px solid #e5e7eb; border-radius: 6px; }
.result-path { font-size: 0.85rem; color: #6b7280; margin-bottom: 4px; }
.result-heading { font-weight: 600; margin-bottom: 4px; }
.result-score { font-size: 0.8rem; color: #9ca3af; }
.result-text { font-size: 0.9rem; color: #444; margin-top: 6px; white-space: pre-wrap; }
#status { color: #6b7280; font-size: 0.9rem; margin-bottom: 12px; }

/* help */
#helpBtn { background: none; border: 1px solid #d1d5db; border-radius: 50%; width: 24px; height: 24px; font-size: 0.85rem; color: #6b7280; cursor: pointer; line-height: 22px; text-align: center; padding: 0; }
#helpBtn:hover { background: #f3f4f6; color: #374151; }
#helpOverlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.4); z-index: 100; }
#helpBox { position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%); background: #fff; border-radius: 8px; padding: 24px; max-width: 560px; width: 90%; max-height: 80vh; overflow-y: auto; z-index: 101; font-size: 0.85rem; line-height: 1.6; color: #374151; }
#helpBox h2 { font-size: 1rem; margin-bottom: 12px; }
#helpBox h3 { font-size: 0.9rem; margin: 12px 0 4px; color: #111; }
#helpBox p, #helpBox li { margin-bottom: 4px; }
#helpBox ul { padding-left: 18px; }
#helpBox code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 0.82rem; }
#helpBox .close { position: absolute; top: 12px; right: 16px; background: none; border: none; font-size: 1.2rem; cursor: pointer; color: #9ca3af; }
#helpBox .close:hover { color: #374151; }
</style>
</head>
<body>
<h1>tfidf-search</h1>

<!-- toolbar -->
<div id="toolbar">
  <span class="root" id="rootPath"></span>
  <span id="indexBadge" class="badge badge-none">no index</span>
  <button id="helpBtn" onclick="$('helpOverlay').style.display='block'" title="Help">?</button>
</div>
<div id="stats"></div>

<!-- help modal -->
<div id="helpOverlay" onclick="this.style.display='none'">
  <div id="helpBox" onclick="event.stopPropagation()">
    <button class="close" onclick="$('helpOverlay').style.display='none'">x</button>
    <h2>tfidf-search web UI</h2>
    <p>Hybrid semantic + lexical search for local text corpora.</p>

    <h3>Getting started</h3>
    <ul>
      <li>Place text files (.md, .txt) in the corpus folder shown above.</li>
      <li>Open <b>Build / Update Index</b> and click <b>Build / Update</b>.</li>
      <li>Type a query in the search box.</li>
    </ul>

    <h3>Build options</h3>
    <ul>
      <li><b>File types</b> -- comma-separated extensions to scan (e.g. <code>md,txt</code>).</li>
      <li><b>Chunking</b> -- <code>token</code> splits at fixed windows; <code>semantic</code> splits at topic boundaries.</li>
      <li><b>Strip code fences</b> -- remove fenced code blocks before indexing.</li>
      <li><b>Build / Update</b> -- incremental, only re-embeds changed files.</li>
      <li><b>Reindex</b> -- full rebuild using the settings saved in the existing index.</li>
      <li><b>Delete &amp; Rebuild</b> -- wipe the index and build fresh with the settings above.</li>
    </ul>

    <h3>Search options</h3>
    <ul>
      <li><b>Results</b> -- number of results to return.</li>
      <li><b>Fusion</b> -- <code>minmax</code> normalizes scores; <code>rrf</code> uses rank-based fusion.</li>
      <li><b>Semantic / Lexical weight</b> -- override the default fusion weights (leave blank for auto).</li>
      <li><b>HyDE</b> -- generates a hypothetical answer via LLM to improve recall on short queries.</li>
      <li><b>Show all chunks</b> -- display multiple matching chunks per file instead of one per file.</li>
    </ul>

    <h3>API</h3>
    <ul>
      <li><code>GET /api/status</code> -- index status and stats</li>
      <li><code>POST /api/build</code> -- incremental update</li>
      <li><code>POST /api/rebuild</code> -- delete and rebuild</li>
      <li><code>GET /api/search?q=...</code> -- search</li>
      <li><code>POST /api/delete</code> -- remove a file and update index</li>
    </ul>
    <p>See <a href="https://github.com/joshuascottpaul/build_tfidf/blob/main/API.md">API.md</a> for full details.</p>
  </div>
</div>

<!-- build options -->
<details id="buildPanel">
  <summary>Build / Update Index</summary>
  <div class="panel">
    <label>File types
      <input type="text" id="buildFileTypes" value="md" placeholder="md,txt,html,docx">
    </label>
    <label>Chunking
      <select id="buildChunking"><option value="token">token</option><option value="semantic">semantic</option></select>
    </label>
    <label class="full" style="flex-direction:row; align-items:center; gap:8px;">
      <input type="checkbox" id="buildRemoveCode"> Strip code fences
    </label>
    <div class="full" style="display:flex; gap:8px; flex-wrap:wrap;">
      <button class="btn" id="buildBtn" onclick="doBuild('build')">Build / Update</button>
      <button class="btn" id="reindexBtn" onclick="doBuild('reindex')" title="Re-read files, keep current settings">Reindex</button>
      <button class="btn" id="rebuildBtn" onclick="doRebuild()" style="color:#dc2626; border-color:#dc2626;" title="Delete index and build fresh with settings above">Delete &amp; Rebuild</button>
    </div>
  </div>
</details>

<!-- search -->
<form id="searchForm">
  <div id="searchRow">
    <input type="text" id="q" name="q" placeholder="Search..." autofocus>
    <button type="submit" class="btn btn-primary">Search</button>
  </div>
  <details id="searchOpts">
    <summary>Search options</summary>
    <div class="panel">
      <label>Results
        <input type="number" id="optTopK" value="10" min="1" max="100">
      </label>
      <label>Fusion
        <select id="optFusion"><option value="minmax">minmax</option><option value="rrf">rrf</option></select>
      </label>
      <label>Semantic weight
        <input type="number" id="optWeightSem" step="0.05" min="0" max="1" placeholder="auto">
      </label>
      <label>Lexical weight
        <input type="number" id="optWeightLex" step="0.05" min="0" max="1" placeholder="auto">
      </label>
      <label class="full" style="flex-direction:row; align-items:center; gap:8px;">
        <input type="checkbox" id="optHyde"> HyDE (hypothetical document embeddings)
      </label>
      <label class="full" style="flex-direction:row; align-items:center; gap:8px;">
        <input type="checkbox" id="optAllChunks"> Show all chunks (not just one per file)
      </label>
    </div>
  </details>
</form>

<div id="status"></div>
<div id="results"></div>

<script>
const $ = id => document.getElementById(id);
const statusDiv = $('status');
const resultsDiv = $('results');

/* --- status & stats --- */
let lastStats = null;
async function checkStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  $('rootPath').textContent = data.root;
  if (data.has_index) {
    $('indexBadge').textContent = 'index ready';
    $('indexBadge').className = 'badge badge-ok';
  } else {
    $('indexBadge').textContent = 'no index';
    $('indexBadge').className = 'badge badge-none';
  }
  lastStats = data.index_stats;
  const s = lastStats;
  if (s) {
    const parts = [
      s.embedding_model && '<span>model: ' + esc(s.embedding_model) + '</span>',
      s.embedding_dimensions != null && '<span>dims: ' + s.embedding_dimensions + '</span>',
      s.num_chunks != null && '<span>chunks: ' + s.num_chunks + '</span>',
      s.num_files != null && '<span>files: ' + s.num_files + '</span>',
      s.chunking_strategy && '<span>chunking: ' + esc(s.chunking_strategy) + '</span>',
      s.file_types && '<span>types: ' + esc(s.file_types) + '</span>',
      s.weight_semantic != null && '<span>weights: ' + s.weight_semantic + ' / ' + s.weight_lexical + '</span>',
      s.created_at && '<span>built: ' + esc(s.created_at.slice(0, 16)) + '</span>',
    ].filter(Boolean);
    $('stats').innerHTML = parts.join('');
    /* pre-fill build options from existing index */
    if (s.chunking_strategy) $('buildChunking').value = s.chunking_strategy;
    if (s.file_types) $('buildFileTypes').value = s.file_types;
    if (s.cleaning_rules && s.cleaning_rules.includes('remove_code=True'))
      $('buildRemoveCode').checked = true;
  } else {
    $('stats').innerHTML = '';
  }
}
checkStatus();

/* --- build helpers --- */
function getBuildOpts() {
  return {
    file_types: $('buildFileTypes').value.trim(),
    chunking: $('buildChunking').value,
    remove_code: $('buildRemoveCode').checked,
  };
}

function setBuildBusy(busy, label) {
  for (const id of ['buildBtn','reindexBtn','rebuildBtn']) {
    $(id).disabled = busy;
  }
  if (busy) statusDiv.textContent = label;
}

function savedOpts() {
  if (!lastStats) return null;
  const s = lastStats;
  const rc = s.cleaning_rules && s.cleaning_rules.includes('remove_code=True');
  return {
    chunking: s.chunking_strategy || 'token',
    remove_code: !!rc,
    file_types: s.file_types || 'md',
  };
}

async function postBuild(endpoint, body, successMsg) {
  setBuildBusy(true, 'Building index (this may take a while)...');
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) { statusDiv.textContent = 'Error: ' + data.error; return; }
    statusDiv.textContent = successMsg;
    checkStatus();
  } catch (err) {
    statusDiv.textContent = 'Failed: ' + err.message;
  } finally {
    setBuildBusy(false, '');
  }
}

async function doBuild(mode) {
  if (mode === 'reindex') {
    /* rebuild from scratch using saved index settings */
    const opts = savedOpts();
    if (!opts) { statusDiv.textContent = 'No existing index to reindex from.'; return; }
    postBuild('/api/rebuild', opts, 'Index rebuilt with saved settings.');
  } else {
    postBuild('/api/build', getBuildOpts(), 'Index updated.');
  }
}

async function doRebuild() {
  if (!confirm('Delete the existing index and rebuild from scratch with the settings above?')) return;
  postBuild('/api/rebuild', getBuildOpts(), 'Index rebuilt.');
}

/* --- search --- */
$('searchForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = $('q').value.trim();
  if (!q) return;
  statusDiv.textContent = 'Searching...';
  resultsDiv.innerHTML = '';
  const params = new URLSearchParams({ q });
  params.set('top_k', $('optTopK').value);
  params.set('fusion', $('optFusion').value);
  const ws = $('optWeightSem').value;
  const wl = $('optWeightLex').value;
  if (ws) params.set('weight_semantic', ws);
  if (wl) params.set('weight_lexical', wl);
  if ($('optHyde').checked) params.set('hyde', '1');
  if ($('optAllChunks').checked) params.set('all_chunks', '1');
  try {
    const res = await fetch('/api/search?' + params);
    const data = await res.json();
    if (data.error) { statusDiv.textContent = data.error; return; }
    statusDiv.textContent = data.results.length + ' result(s)';
    for (const r of data.results) {
      const div = document.createElement('div');
      div.className = 'result';
      let html = '<div class="result-path">' + esc(r.filename) + ' <span style="color:#9ca3af">' + esc(r.path) + '</span></div>';
      if (r.heading) html += '<div class="result-heading">' + esc(r.heading) + '</div>';
      html += '<div class="result-score">score: ' + r.score.toFixed(4) + '</div>';
      if (r.text) html += '<div class="result-text">' + esc(r.text.slice(0, 300)) + '</div>';
      div.innerHTML = html;
      resultsDiv.appendChild(div);
    }
  } catch (err) {
    statusDiv.textContent = 'Error: ' + err.message;
  }
});

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
</script>
</body>
</html>
"""


def create_app(root: Path, embed_config):
    import shutil

    from flask import Flask, jsonify, request
    from .index import (
        query as _query, build as _build, update as _update,
        _meta_path, _manifest_path, _data_dir, _load_json,
    )

    app = Flask(__name__)

    def _has_index():
        return _meta_path(root).exists()

    def _index_stats():
        if not _has_index():
            return None
        meta = _load_json(_meta_path(root))
        manifest = _load_json(_manifest_path(root))
        meta["num_chunks"] = len(manifest.get("chunks", []))
        meta["num_files"] = len(manifest.get("entries", []))
        return meta

    def _parse_build_opts(body):
        file_types_raw = body.get("file_types", "md")
        file_types = {ft.strip().lower().lstrip(".") for ft in file_types_raw.split(",") if ft.strip()}
        return {
            "remove_code": body.get("remove_code", False),
            "file_types": file_types or None,
            "chunking_strategy": body.get("chunking", "token"),
        }

    @app.route("/")
    def index():
        return _HTML

    @app.route("/api/status")
    def api_status():
        return jsonify(
            has_index=_has_index(),
            root=str(root.resolve()),
            index_stats=_index_stats(),
        )

    @app.route("/api/build", methods=["POST"])
    def api_build():
        body = request.get_json(silent=True) or {}
        opts = _parse_build_opts(body)
        try:
            _update(root, embed_config, **opts)
        except Exception as exc:
            return jsonify(error=str(exc)), 500
        return jsonify(ok=True)

    @app.route("/api/rebuild", methods=["POST"])
    def api_rebuild():
        body = request.get_json(silent=True) or {}
        opts = _parse_build_opts(body)
        idx_dir = _data_dir(root)
        if idx_dir.exists():
            shutil.rmtree(idx_dir)
        try:
            _build(root, embed_config, **opts)
        except Exception as exc:
            return jsonify(error=str(exc)), 500
        return jsonify(ok=True)

    @app.route("/api/delete", methods=["POST"])
    def api_delete():
        body = request.get_json(silent=True) or {}
        filename = body.get("filename", "").strip()
        if not filename:
            return jsonify(error="filename is required"), 400
        # Resolve against root, block path traversal
        target = (root / filename).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            return jsonify(error="Invalid filename"), 400
        if not target.exists():
            return jsonify(error="File not found"), 404
        target.unlink()
        # Rebuild index using settings from existing metadata
        if _has_index():
            meta = _load_json(_meta_path(root))
            ft_str = meta.get("file_types", "md")
            file_types = {ft.strip() for ft in ft_str.split(",") if ft.strip()}
            try:
                _update(root, embed_config, file_types=file_types or None)
            except Exception as exc:
                return jsonify(error=str(exc)), 500
        return jsonify(ok=True)

    @app.route("/api/search")
    def api_search():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify(results=[])
        if not _has_index():
            return jsonify(error="No index found. Build the index first."), 400
        top_k = request.args.get("top_k", 10, type=int)
        fusion = request.args.get("fusion", "minmax")
        ws = request.args.get("weight_semantic", None, type=float)
        wl = request.args.get("weight_lexical", None, type=float)
        hyde = request.args.get("hyde", "") == "1"
        all_chunks = request.args.get("all_chunks", "") == "1"
        try:
            results = _query(
                q, embed_config, root=root, top_k=top_k,
                fusion_method=fusion,
                weight_semantic=ws,
                weight_lexical=wl,
                hyde=hyde,
                dedupe_by_path=not all_chunks,
            )
        except Exception as exc:
            return jsonify(error=str(exc)), 500
        return jsonify(results=[
            {
                "path": chunk.get("path", ""),
                "filename": Path(chunk.get("path", "")).name,
                "heading": chunk.get("heading", ""),
                "text": chunk.get("text", ""),
                "score": round(score, 4),
            }
            for chunk, score in results
        ])

    return app


def start_server(
    root: Path,
    host: str,
    port: int,
    embed_config,
) -> None:
    app = create_app(root, embed_config)
    print(f"Serving on http://{host}:{port}  (root: {root})")
    app.run(host=host, port=port)
