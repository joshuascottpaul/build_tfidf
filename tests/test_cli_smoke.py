from __future__ import annotations

import build_tfidf.cli as cli


def test_cli_build_smoke(monkeypatch, tmp_path):
    def _noop_build(*args, **kwargs):
        return None

    monkeypatch.setattr(cli, "build_index", _noop_build)
    rc = cli.main(["build", "--root", str(tmp_path)])
    assert rc == 0


def test_cli_query_shorthand(monkeypatch):
    def _noop_query(*args, **kwargs):
        return []

    monkeypatch.setattr(cli, "query_index", _noop_query)
    rc = cli.main(["--query", "uncertainty"])
    assert rc == 0


def test_cli_query_all_chunks(monkeypatch):
    def _noop_query(*args, **kwargs):
        assert kwargs.get("dedupe_by_path") is False
        return []

    monkeypatch.setattr(cli, "query_index", _noop_query)
    rc = cli.main(["query", "uncertainty", "--all-chunks"])
    assert rc == 0


def test_cli_query_open_invalid(monkeypatch):
    def _noop_query(*args, **kwargs):
        return [({"path": "/tmp/a.md"}, 0.5)]

    monkeypatch.setattr(cli, "query_index", _noop_query)
    try:
        cli.main(["query", "uncertainty", "--open", "2"])
    except SystemExit:
        assert True
    else:
        assert False


def test_cli_query_pbcopy_invalid(monkeypatch):
    def _noop_query(*args, **kwargs):
        return [({"path": "/tmp/a.md"}, 0.5)]

    monkeypatch.setattr(cli, "query_index", _noop_query)
    try:
        cli.main(["query", "uncertainty", "--pbcopy", "2"])
    except SystemExit:
        assert True
    else:
        assert False
