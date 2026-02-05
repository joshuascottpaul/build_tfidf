"""CLI entry point for build_tfidf."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .embeddings import load_config_from_env
from .index import build as build_index
from .index import update as update_index
from .index import query as query_index


def _check_runtime() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required. Please upgrade your Python.")
    try:
        import numpy  # noqa: F401
        import faiss  # noqa: F401
        import openai  # noqa: F401
        import tiktoken  # noqa: F401
        import pydantic  # noqa: F401
        import rank_bm25  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing required dependencies. Activate your venv and run: "
            "pip install -r requirements.txt && pip install -e ."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic search for Markdown corpora.",
        epilog=(
            "Examples:\n"
            "  tfidf-search build --root /path/to/corpus\n"
            "  tfidf-search query \"your query\"\n"
            "  tfidf-search \"your query\"  # shorthand\n"
            "  tfidf-search --query \"your query\"  # shorthand\n"
            "  tfidf-search query \"your query\" --open 1\n"
            "  tfidf-search query \"your query\" --pbcopy 1\n"
            "  tfidf-search query \"your query\" --paths-only\n"
            "  tfidf-search update --remove-code\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the index")
    b.add_argument("--root", default=".", help="root directory to scan")
    b.add_argument("--remove-code", action="store_true", help="strip code fences")

    u = sub.add_parser("update", help="incrementally update the index")
    u.add_argument("--root", default=".", help="root directory to scan")
    u.add_argument("--remove-code", action="store_true", help="strip code fences")

    q = sub.add_parser("query", help="query the index")
    q.add_argument("text", help="query text")
    q.add_argument("--top", type=int, default=10, help="number of results")
    q.add_argument("--rerank-model", default="", help="optional rerank model")
    q.add_argument("--rerank-top", type=int, default=30, help="rerank candidate count")
    q.add_argument("--all-chunks", action="store_true", help="show multiple chunks per file")
    q.add_argument("--open", dest="open_index", type=int, help="open result number in default app")
    q.add_argument("--reveal", dest="reveal_index", type=int, help="reveal result in Finder")
    q.add_argument("--pbcopy", dest="pbcopy_index", type=int, help="copy result path to clipboard")
    q.add_argument("--paths-only", action="store_true", help="print only file paths")

    insp = sub.add_parser("inspect", help="inspect a chunk by id")
    insp.add_argument("chunk_id", help="chunk id")

    return parser


def _inject_shorthand_query(argv: list[str] | None) -> list[str]:
    if not argv:
        return []
    if "--query" in argv and "query" not in argv and "build" not in argv and "update" not in argv and "inspect" not in argv:
        idx = argv.index("--query")
        if idx + 1 >= len(argv):
            return ["query"]
        return ["query", argv[idx + 1]]
    if argv[0].startswith("-"):
        return argv
    if argv[0] in {"build", "update", "query", "inspect"}:
        return argv
    return ["query", " ".join(argv)]


def main(argv: list[str] | None = None) -> int:
    _check_runtime()
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    argv = _inject_shorthand_query(argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    cfg = load_config_from_env()

    if args.cmd == "build":
        build_index(Path(args.root), cfg, remove_code=args.remove_code)
        return 0
    if args.cmd == "query":
        query_text = args.text
        rerank_model = args.rerank_model.strip() or None
        results = query_index(
            query_text,
            cfg,
            top_k=args.top,
            rerank_model=rerank_model,
            rerank_top_n=args.rerank_top,
            dedupe_by_path=not args.all_chunks,
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
    if args.cmd == "update":
        update_index(Path(args.root), cfg, remove_code=args.remove_code)
        return 0
    if args.cmd == "inspect":
        from .index import MANIFEST_PATH, _load_json

        manifest = _load_json(MANIFEST_PATH)
        for chunk in manifest.get("chunks", []):
            if chunk["sha256"] == args.chunk_id:
                print(json.dumps(chunk, indent=2))
                return 0
        raise SystemExit("Chunk not found.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
