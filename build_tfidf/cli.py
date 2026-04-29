"""CLI entry point for build_tfidf."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .embeddings import load_config_from_env
from .index import build as build_index
from .index import update as update_index
from .index import query as search_index


def _check_runtime() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required. Please upgrade your Python.")
    try:
        import numpy  # noqa: F401
        import faiss  # noqa: F401
        import tiktoken  # noqa: F401
        import pydantic  # noqa: F401
        import rank_bm25  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing required dependencies. Activate your venv and run: "
            "pip install -r requirements.txt && pip install -e ."
        ) from exc


def _parse_file_types(raw: str) -> set[str]:
    return {t.strip().lower().lstrip(".") for t in raw.split(",") if t.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic search for Markdown corpora.",
        epilog=(
            "Examples:\n"
            "\n"
            "  # Build index from current directory (Markdown only)\n"
            "  tfidf-search build\n"
            "\n"
            "  # Build from a specific corpus, strip code fences\n"
            "  tfidf-search build --root ~/notes --remove-code\n"
            "\n"
            "  # Build including HTML and DOCX files (requires unstructured)\n"
            "  tfidf-search build --root ~/notes --file-types md,txt,html,docx\n"
            "\n"
            "  # Build using local embeddings, no API key (requires fastembed)\n"
            "  tfidf-search build --embedding-provider fastembed\n"
            "\n"
            "  # Build using Ollama embeddings\n"
            "  tfidf-search build --embedding-provider ollama\n"
            "\n"
            "  # Build with semantic chunking (splits at topic boundaries)\n"
            "  tfidf-search build --chunking semantic\n"
            "\n"
            "  # Search — three equivalent shorthands\n"
            "  tfidf-search \"retrieval augmented generation\"\n"
            "  tfidf-search search \"retrieval augmented generation\"\n"
            "  tfidf-search --search \"retrieval augmented generation\"\n"
            "\n"
            "  # Search a specific corpus\n"
            "  tfidf-search search \"chunking strategies\" --root ~/notes\n"
            "\n"
            "  # Search with more results\n"
            "  tfidf-search search \"chunking strategies\" --top 20\n"
            "\n"
            "  # Search and re-rank with a local cross-encoder (requires flashrank)\n"
            "  tfidf-search search \"chunking strategies\" --rerank-model ms-marco-MiniLM-L-12-v2\n"
            "\n"
            "  # Search and re-rank with a larger candidate pool\n"
            "  tfidf-search search \"chunking strategies\" --rerank-model ms-marco-MiniLM-L-12-v2 --rerank-top 50\n"
            "\n"
            "  # Use Reciprocal Rank Fusion instead of min-max normalization\n"
            "  tfidf-search search \"chunking strategies\" --fusion rrf\n"
            "\n"
            "  # Override fusion weights (default: per-provider)\n"
            "  tfidf-search search \"chunking strategies\" --weight-semantic 0.8 --weight-lexical 0.2\n"
            "\n"
            "  # Use HyDE (hypothetical document embeddings) for better recall\n"
            "  tfidf-search search \"chunking strategies\" --hyde\n"
            "\n"
            "  # Show all matching chunks, not just one per file\n"
            "  tfidf-search search \"embedding models\" --all-chunks\n"
            "\n"
            "  # Open the top result in your default app\n"
            "  tfidf-search \"vector databases\" --open 1\n"
            "\n"
            "  # Reveal result 2 in Finder\n"
            "  tfidf-search \"vector databases\" --reveal 2\n"
            "\n"
            "  # Copy result path to clipboard\n"
            "  tfidf-search \"vector databases\" --pbcopy 1\n"
            "\n"
            "  # Print paths only (pipe-friendly)\n"
            "  tfidf-search search \"FAISS\" --paths-only | xargs grep -l 'IndexFlat'\n"
            "\n"
            "  # Incrementally update after editing files\n"
            "  tfidf-search update\n"
            "  tfidf-search update --root ~/notes --file-types md,txt,html,docx\n"
            "\n"
            "  # Watch corpus and auto-update on save (requires watchdog)\n"
            "  tfidf-search watch\n"
            "  tfidf-search watch --root ~/notes --debounce 2.0\n"
            "  tfidf-search watch --root ~/notes --file-types md,txt,html,docx\n"
            "\n"
            "  # Inspect a specific chunk by its sha256 id\n"
            "  tfidf-search inspect a3f1c2...\n"
            "\n"
            "Optional extras:\n"
            "  pip install 'build-tfidf[fastembed]'    local embeddings, no API key\n"
            "  pip install 'build-tfidf[flashrank]'    local cross-encoder re-ranking\n"
            "  pip install 'build-tfidf[watchdog]'     watch command\n"
            "  pip install 'build-tfidf[unstructured]' html/docx indexing\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the index")
    b.add_argument("--root", default=".", help="root directory to scan")
    b.add_argument("--remove-code", action="store_true", help="strip code fences")
    b.add_argument("--file-types", default="md", help="comma-separated file types (md,txt,html,docx)")
    b.add_argument("--embedding-provider", choices=["openai", "fastembed", "ollama"], default=None, help="embedding provider (overrides EMBEDDING_PROVIDER env var)")
    b.add_argument("--fastembed-threads", type=int, default=None, help="limit fastembed CPU threads (e.g. 2 for cron jobs)")
    b.add_argument("--chunking", choices=["token", "semantic"], default="token", help="chunking strategy (default: token)")

    u = sub.add_parser("update", help="incrementally update the index")
    u.add_argument("--root", default=".", help="root directory to scan")
    u.add_argument("--remove-code", action="store_true", help="strip code fences")
    u.add_argument("--file-types", default="md", help="comma-separated file types (md,txt,html,docx)")
    u.add_argument("--embedding-provider", choices=["openai", "fastembed", "ollama"], default=None, help="embedding provider (overrides EMBEDDING_PROVIDER env var)")
    u.add_argument("--fastembed-threads", type=int, default=None, help="limit fastembed CPU threads (e.g. 2 for cron jobs)")
    u.add_argument("--chunking", choices=["token", "semantic"], default="token", help="chunking strategy (default: token)")

    q = sub.add_parser("search", help="search the index")
    q.add_argument("text", help="query text")
    q.add_argument("--root", default=".", help="root directory where index lives")
    q.add_argument("--top", type=int, default=10, help="number of results")
    q.add_argument("--rerank-model", default="", help="flashrank model name for re-ranking")
    q.add_argument("--rerank-top", type=int, default=30, help="rerank candidate count")
    q.add_argument("--all-chunks", action="store_true", help="show multiple chunks per file")
    q.add_argument("--open", dest="open_index", type=int, help="open result number in default app")
    q.add_argument("--reveal", dest="reveal_index", type=int, help="reveal result in Finder")
    q.add_argument("--pbcopy", dest="pbcopy_index", type=int, help="copy result path to clipboard")
    q.add_argument("--paths-only", action="store_true", help="print only file paths")
    q.add_argument("--hyde", action="store_true", help="use HyDE (hypothetical document embeddings) for query expansion")
    q.add_argument("--fusion", choices=["minmax", "rrf"], default="minmax", help="score fusion method (default: minmax)")
    q.add_argument("--weight-semantic", type=float, default=None, help="semantic weight override (default: per-provider)")
    q.add_argument("--weight-lexical", type=float, default=None, help="lexical weight override (default: per-provider)")
    q.add_argument("--embedding-provider", choices=["openai", "fastembed", "ollama"], default=None, help="embedding provider (overrides EMBEDDING_PROVIDER env var)")

    w = sub.add_parser("watch", help="watch corpus and auto-update index on changes")
    w.add_argument("--root", default=".", help="root directory to watch")
    w.add_argument("--remove-code", action="store_true", help="strip code fences")
    w.add_argument("--file-types", default="md", help="comma-separated file types (md,txt,html,docx)")
    w.add_argument("--debounce", type=float, default=1.5, help="debounce window in seconds")
    w.add_argument("--embedding-provider", choices=["openai", "fastembed", "ollama"], default=None, help="embedding provider (overrides EMBEDDING_PROVIDER env var)")
    w.add_argument("--fastembed-threads", type=int, default=None, help="limit fastembed CPU threads (e.g. 2 for cron jobs)")

    insp = sub.add_parser("inspect", help="inspect a chunk by id")
    insp.add_argument("chunk_id", help="chunk id")
    insp.add_argument("--root", default=".", help="root directory where index lives")

    return parser


def _inject_shorthand_search(argv: list[str] | None) -> list[str]:
    if not argv:
        return []
    if "--search" in argv and "search" not in argv and "build" not in argv and "update" not in argv and "inspect" not in argv and "watch" not in argv:
        idx = argv.index("--search")
        if idx + 1 >= len(argv):
            return ["search"]
        value = argv[idx + 1]
        rest: list[str] = []
        skip = {idx, idx + 1}
        for i, token in enumerate(argv):
            if i in skip:
                continue
            rest.append(token)
        return ["search", value, *rest]
    if argv[0].startswith("-"):
        return argv
    if argv[0] in {"build", "update", "search", "inspect", "watch"}:
        return argv
    if any(token.startswith("-") for token in argv[1:]):
        return ["search", *argv]
    return ["search", " ".join(argv)]


def _run_watch(root: Path, cfg, remove_code: bool, file_types: set[str], debounce: float) -> None:
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError as exc:
        raise SystemExit("watchdog is not installed. Run: pip install watchdog") from exc

    import threading

    extensions = tuple(f".{ft}" for ft in file_types)
    timer: threading.Timer | None = None
    lock = threading.Lock()

    def _do_update() -> None:
        print(f"[watch] updating index...", flush=True)
        try:
            update_index(root, cfg, remove_code=remove_code, file_types=file_types)
            print("[watch] done.", flush=True)
        except Exception as e:
            print(f"[watch] update failed: {e}", flush=True)

    def _schedule() -> None:
        nonlocal timer
        with lock:
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(debounce, _do_update)
            timer.daemon = True
            timer.start()

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.is_directory:
                return
            if not str(event.src_path).endswith(extensions):
                return
            print(f"[watch] change: {event.src_path}", flush=True)
            _schedule()

    observer = Observer()
    observer.schedule(Handler(), str(root), recursive=True)
    observer.start()
    print(f"[watch] watching {root} for {extensions} — Ctrl-C to stop", flush=True)
    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def main(argv: list[str] | None = None) -> int:
    _check_runtime()
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    argv = _inject_shorthand_search(argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    provider = getattr(args, "embedding_provider", None)
    ft_threads = getattr(args, "fastembed_threads", None)
    cfg = load_config_from_env(provider_override=provider, fastembed_threads=ft_threads)

    if args.cmd == "build":
        build_index(
            Path(args.root), cfg,
            remove_code=args.remove_code,
            file_types=_parse_file_types(args.file_types),
            chunking_strategy=args.chunking,
        )
        return 0

    if args.cmd == "update":
        update_index(
            Path(args.root), cfg,
            remove_code=args.remove_code,
            file_types=_parse_file_types(args.file_types),
            chunking_strategy=args.chunking,
        )
        return 0

    if args.cmd == "watch":
        _run_watch(
            Path(args.root), cfg,
            remove_code=args.remove_code,
            file_types=_parse_file_types(args.file_types),
            debounce=args.debounce,
        )
        return 0

    if args.cmd == "search":
        query_text = args.text
        rerank_model = args.rerank_model.strip() or None
        results = search_index(
            query_text,
            cfg,
            root=Path(args.root),
            top_k=args.top,
            weight_semantic=args.weight_semantic,
            weight_lexical=args.weight_lexical,
            fusion_method=args.fusion,
            rerank_model=rerank_model,
            rerank_top_n=args.rerank_top,
            dedupe_by_path=not args.all_chunks,
            hyde=args.hyde,
        )
        for idx, (chunk, score) in enumerate(results, start=1):
            if args.paths_only:
                print(chunk["path"])
            else:
                print(f"{idx:02d}. {chunk['path']}  (score={score:.4f})")
        if args.open_index or args.reveal_index or args.pbcopy_index:
            import subprocess

            def _path_for(n: int) -> str | None:
                if n <= 0 or n > len(results):
                    return None
                return results[n - 1][0]["path"]

            if args.open_index:
                path = _path_for(args.open_index)
                if not path:
                    raise SystemExit("Invalid --open index.")
                subprocess.run(["open", path], check=False)
            if args.reveal_index:
                path = _path_for(args.reveal_index)
                if not path:
                    raise SystemExit("Invalid --reveal index.")
                subprocess.run(["open", "-R", path], check=False)
            if args.pbcopy_index:
                path = _path_for(args.pbcopy_index)
                if not path:
                    raise SystemExit("Invalid --pbcopy index.")
                subprocess.run(["pbcopy"], input=path, text=True, check=False)
        return 0

    if args.cmd == "inspect":
        from .index import _manifest_path, _load_json

        manifest = _load_json(_manifest_path(Path(args.root)))
        for chunk in manifest.get("chunks", []):
            if chunk["sha256"] == args.chunk_id:
                print(json.dumps(chunk, indent=2))
                return 0
        raise SystemExit("Chunk not found.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
