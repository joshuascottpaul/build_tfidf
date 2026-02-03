from __future__ import annotations

import build_tfidf.cli as cli


def test_cli_build_smoke(monkeypatch, tmp_path):
    def _noop_build(*args, **kwargs):
        return None

    monkeypatch.setattr(cli, "build_index", _noop_build)
    rc = cli.main(["build", "--root", str(tmp_path)])
    assert rc == 0
