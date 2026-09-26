from __future__ import annotations

import subprocess
from pathlib import Path

from flightdeck.ledger_repos import (
    index_repos,
    inspect_repo,
    interesting_repos,
    normalize_github_remote,
)
from flightdeck.store import Store


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t.com",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(path),
        },
    )


def _init_repo(path: Path, *, commit: bool = True) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    if commit:
        (path / "README.md").write_text("hello\n")
        _git(path, "add", "README.md")
        _git(path, "commit", "-q", "-m", "init")
    return path


def test_normalize_github_remote_https_and_ssh():
    assert (
        normalize_github_remote("https://github.com/Cschaefer732/sparky.git")
        == "Cschaefer732/sparky"
    )
    assert (
        normalize_github_remote("git@github.com:Cschaefer732/sparky.git") == "Cschaefer732/sparky"
    )
    assert normalize_github_remote(None) is None
    assert normalize_github_remote("not-a-url") is None


def test_no_upstream_is_null_not_zero(tmp_path):
    repo = _init_repo(tmp_path / "solo")
    info = inspect_repo(repo)
    assert info["status"] == "ok"
    assert info["ahead"] is None
    assert info["behind"] is None


def test_untracked_only_counts_as_dirty(tmp_path):
    repo = _init_repo(tmp_path / "untracked")
    (repo / "new_file.txt").write_text("new\n")
    info = inspect_repo(repo)
    assert info["status"] == "ok"
    assert info["dirty"] is True


def test_no_commits_repo_reports_status(tmp_path):
    repo = _init_repo(tmp_path / "empty", commit=False)
    info = inspect_repo(repo)
    assert info["status"] == "no_commits"


def test_non_repo_dir_reports_status(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "file.txt").write_text("x\n")
    info = inspect_repo(plain)
    assert info["status"] == "not_a_repo"


def test_unreadable_path_reports_status(tmp_path):
    missing = tmp_path / "does_not_exist"
    info = inspect_repo(missing)
    assert info["status"] == "unreadable"


def test_index_repos_idempotent(tmp_path):
    root = tmp_path / "dev"
    root.mkdir()
    _init_repo(root / "repo_a")
    _init_repo(root / "repo_b", commit=False)

    with Store(tmp_path / "store") as store:
        summary1 = index_repos(store, roots=(str(root),))
        assert summary1["found"] == 2
        assert summary1["ok"] == 1
        assert summary1["no_commits"] == 1

        rows = store.conn.execute("SELECT COUNT(*) AS c FROM repo_index").fetchone()
        assert rows["c"] == 1

        summary2 = index_repos(store, roots=(str(root),))
        assert summary2["ok"] == 1
        rows_again = store.conn.execute("SELECT COUNT(*) AS c FROM repo_index").fetchone()
        assert rows_again["c"] == 1  # updated in place, not duplicated


def test_index_repos_dirty_and_no_remote_counted(tmp_path):
    root = tmp_path / "dev"
    root.mkdir()
    repo = _init_repo(root / "repo_c")
    (repo / "scratch.txt").write_text("x\n")

    with Store(tmp_path / "store") as store:
        summary = index_repos(store, roots=(str(root),))
        assert summary["dirty"] == 1
        assert summary["no_remote"] == 1

        interesting = interesting_repos(store)
        assert len(interesting) == 1
        assert interesting[0]["name"] == "repo_c"


def test_index_repos_skips_nested_skip_dirs(tmp_path):
    root = tmp_path / "dev"
    root.mkdir()
    (root / "node_modules").mkdir()
    (root / ".venv").mkdir()
    _init_repo(root / "repo_d")

    with Store(tmp_path / "store") as store:
        summary = index_repos(store, roots=(str(root),))
        assert summary["found"] == 1
