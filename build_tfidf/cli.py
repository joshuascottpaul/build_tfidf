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
            "Missing required dependencies. Run: pip install -r requirements.txt"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic search for Markdown corpora.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the index")
    b.add_argument("--root", default=".", help="root directory to scan")
    b.add_argument("--remove-code", action="store_true", help="strip code fences")

    u = sub.add_parser("update", help="incrementally update the index")
    u.add_argument("--root", default=".", help="root directory to scan")

    q = sub.add_parser("query", help="query the index")
    q.add_argument("text", help="query text")
    q.add_argument("--top", type=int, default=10, help="number of results")
    q.add_argument("--rerank-model", default="", help="optional rerank model")
    q.add_argument("--rerank-top", type=int, default=30, help="rerank candidate count")

    insp = sub.add_parser("inspect", help="inspect a chunk by id")
    insp.add_argument("chunk_id", help="chunk id")

    return parser


def main(argv: list[str] | None = None) -> int:
    _check_runtime()
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config_from_env()

    if args.cmd == "build":
        build_index(Path(args.root), cfg, remove_code=args.remove_code)
        return 0
    if args.cmd == "query":
        rerank_model = args.rerank_model.strip() or None
        results = query_index(
            args.text,
            cfg,
            top_k=args.top,
            rerank_model=rerank_model,
            rerank_top_n=args.rerank_top,
        )
        for idx, (chunk, score) in enumerate(results, start=1):
            print(f"{idx:02d}. {chunk['path']}  (score={score:.4f})")
        return 0
    if args.cmd == "update":
        update_index(Path(args.root), cfg)
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
