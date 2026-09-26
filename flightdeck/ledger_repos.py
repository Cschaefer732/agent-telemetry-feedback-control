"""Index local git repos into repo_index so an agent can see what exists at a glance.

`git diff` alone misses untracked files -- this codebase has a documented incident where a
patch silently omitted new files because of exactly that gap. `dirty` here is computed from
`git status --porcelain`, which reports untracked paths too.

`ahead`/`behind` are only meaningful relative to a tracking branch. A repo with no upstream
must report NULL for both, not 0 -- 0 means "in sync with a known upstream", NULL means "no
upstream to compare against". Conflating the two is the same class of bug this fleet keeps
finding (see the null-key-is-not-absent incident).

Every repo that cannot be fully inspected -- not-a-repo, no commits yet, detached HEAD, bare,
a submodule, unreadable, or a timed-out git call -- is COUNTED in the summary and never
raises. Silence is a finding.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

from flightdeck.store import Store, now_ms

SKIP_DIRS = {"node_modules", ".venv", "venv", "__pycache__", ".git"}

_SSH_RE = re.compile(r"^git@[^:]+:(?P<path>.+?)(?:\.git)?/?$")
_HTTPS_RE = re.compile(r"^https?://[^/]+/(?P<path>.+?)(?:\.git)?/?$")


def normalize_github_remote(url: str | None) -> str | None:
    """Return 'owner/name' for a github SSH or HTTPS remote, else None.

    Do not assume git@ shape: this machine's SSH key isn't registered on GitHub, so
    real-world remotes here are HTTPS. Both shapes are handled anyway since other
    machines/repos may still use SSH.
    """
    if not url:
        return None
    for pattern in (_SSH_RE, _HTTPS_RE):
        m = pattern.match(url.strip())
        if m:
            path = m.group("path").strip("/")
            parts = path.split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
    return None


def _run(path: Path, args: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None


def _is_bare(path: Path, timeout: float) -> bool:
    r = _run(path, ["rev-parse", "--is-bare-repository"], timeout)
    return bool(r and r.returncode == 0 and r.stdout.strip() == "true")


def inspect_repo(path: Path, *, timeout: float = 5.0) -> dict[str, Any]:
    """Inspect one git repo. Returns a dict with a 'status' key.

    status is one of: "ok", "not_a_repo", "no_commits", "bare", "submodule",
    "unreadable", "timeout".
    """
    path = Path(path)
    try:
        if not path.is_dir():
            return {"status": "unreadable"}
        list(path.iterdir())
    except OSError:
        return {"status": "unreadable"}

    top = _run(path, ["rev-parse", "--show-toplevel"], timeout)
    if top is None:
        return {"status": "timeout"}
    if top.returncode != 0:
        return {"status": "not_a_repo"}

    if _is_bare(path, timeout):
        return {"status": "bare"}

    git_common = _run(path, ["rev-parse", "--git-common-dir"], timeout)
    git_dir = _run(path, ["rev-parse", "--git-dir"], timeout)
    is_submodule = False
    if (
        git_common is not None
        and git_common.returncode == 0
        and git_dir is not None
        and git_dir.returncode == 0
    ):
        # a submodule's git-dir lives under the superproject's .git/modules, not <repo>/.git
        gd = git_dir.stdout.strip()
        if gd and ".git/modules" in gd.replace("\\", "/"):
            is_submodule = True

    head = _run(path, ["log", "-1", "--format=%H %ct %s"], timeout)
    if head is None:
        return {"status": "timeout"}
    if head.returncode != 0 or not head.stdout.strip():
        return {"status": "no_commits", "is_submodule": is_submodule}

    parts = head.stdout.strip().split(" ", 2)
    head_sha = parts[0]
    last_commit_at = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    last_commit_subject = parts[2] if len(parts) > 2 else ""

    branch_r = _run(path, ["symbolic-ref", "--short", "-q", "HEAD"], timeout)
    detached = branch_r is None or branch_r.returncode != 0
    branch = None if detached else branch_r.stdout.strip()

    status_r = _run(path, ["status", "--porcelain"], timeout)
    if status_r is None:
        return {"status": "timeout"}
    dirty = bool(status_r.stdout.strip())

    remote_r = _run(path, ["remote", "get-url", "origin"], timeout)
    remote_url = remote_r.stdout.strip() if remote_r and remote_r.returncode == 0 else None
    github_remote = normalize_github_remote(remote_url)

    ahead: int | None = None
    behind: int | None = None
    if branch:
        upstream_r = _run(
            path, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], timeout
        )
        if upstream_r is not None and upstream_r.returncode == 0 and upstream_r.stdout.strip():
            counts_r = _run(path, ["rev-list", "--left-right", "--count", "@{u}...HEAD"], timeout)
            if counts_r is not None and counts_r.returncode == 0 and counts_r.stdout.strip():
                nums = counts_r.stdout.split()
                if len(nums) == 2 and all(n.isdigit() for n in nums):
                    behind, ahead = int(nums[0]), int(nums[1])

    return {
        "status": "submodule" if is_submodule else "ok",
        "is_submodule": is_submodule,
        "detached": detached,
        "branch": branch,
        "head_sha": head_sha,
        "dirty": dirty,
        "ahead": ahead,
        "behind": behind,
        "last_commit_at": last_commit_at,
        "last_commit_subject": last_commit_subject,
        "github_remote": github_remote,
    }


def _candidate_dirs(root: Path, timeout: float) -> list[Path]:
    """Repo roots directly under `root`, plus root itself if it is a repo."""
    candidates: list[Path] = []
    if not root.is_dir():
        return candidates
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return candidates
    for entry in entries:
        if not entry.is_dir() or entry.name in SKIP_DIRS or entry.name.startswith("."):
            continue
        candidates.append(entry)
    return candidates


def index_repos(
    store: Store,
    *,
    roots: tuple[str, ...] = ("~/dev",),
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Scan `roots` for git repos (one level deep) and upsert repo_index.

    Idempotent: re-running UPSERTs by path, no duplicate rows.
    """
    started = time.monotonic()
    counts = {
        "found": 0,
        "ok": 0,
        "dirty": 0,
        "ahead": 0,
        "behind": 0,
        "no_remote": 0,
        "detached": 0,
        "submodules": 0,
        "not_a_repo": 0,
        "no_commits": 0,
        "bare": 0,
        "unreadable": 0,
        "timeout": 0,
    }

    dirs: list[Path] = []
    for raw in roots:
        expanded = Path(raw).expanduser()
        dirs.extend(_candidate_dirs(expanded, timeout))

    indexed_at = now_ms()
    for path in dirs:
        counts["found"] += 1
        info = inspect_repo(path, timeout=timeout)
        status = info["status"]
        if status in ("not_a_repo", "no_commits", "bare", "unreadable", "timeout"):
            counts[status] += 1
            if status == "no_commits" and info.get("is_submodule"):
                counts["submodules"] += 1
            continue

        if info.get("is_submodule"):
            counts["submodules"] += 1
        if info.get("detached"):
            counts["detached"] += 1
        if info["dirty"]:
            counts["dirty"] += 1
        if info["ahead"] not in (None, 0):
            counts["ahead"] += 1
        if info["behind"] not in (None, 0):
            counts["behind"] += 1
        if info["github_remote"] is None:
            counts["no_remote"] += 1
        counts["ok"] += 1

        store.conn.execute(
            """
            INSERT INTO repo_index
                (path, name, github_remote, branch, head_sha, dirty, ahead, behind,
                 last_commit_at, last_commit_subject, indexed_at)
            VALUES (:path, :name, :github_remote, :branch, :head_sha, :dirty, :ahead, :behind,
                    :last_commit_at, :last_commit_subject, :indexed_at)
            ON CONFLICT(path) DO UPDATE SET
                name=excluded.name,
                github_remote=excluded.github_remote,
                branch=excluded.branch,
                head_sha=excluded.head_sha,
                dirty=excluded.dirty,
                ahead=excluded.ahead,
                behind=excluded.behind,
                last_commit_at=excluded.last_commit_at,
                last_commit_subject=excluded.last_commit_subject,
                indexed_at=excluded.indexed_at
            """,
            {
                "path": str(path),
                "name": path.name,
                "github_remote": info["github_remote"],
                "branch": info["branch"],
                "head_sha": info["head_sha"],
                "dirty": int(info["dirty"]),
                "ahead": info["ahead"],
                "behind": info["behind"],
                "last_commit_at": info["last_commit_at"],
                "last_commit_subject": info["last_commit_subject"],
                "indexed_at": indexed_at,
            },
        )

    counts["duration_s"] = round(time.monotonic() - started, 3)
    return counts


def interesting_repos(store: Store) -> list[dict[str, Any]]:
    """Repos worth an agent's attention: dirty, ahead of remote, or with no remote at all."""
    rows = store.conn.execute(
        """
        SELECT path, name, github_remote, branch, head_sha, dirty, ahead, behind,
               last_commit_at, last_commit_subject, indexed_at
        FROM repo_index
        WHERE dirty = 1 OR (ahead IS NOT NULL AND ahead > 0) OR github_remote IS NULL
        ORDER BY name
        """
    ).fetchall()
    return [dict(r) for r in rows]
