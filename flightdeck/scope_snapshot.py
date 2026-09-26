"""Pin the transcript corpus a measurement was taken over, so the number can be re-derived.

The baseline reported rework 23.6% / correction 19.7% "measured on 90 real sessions". A
check today found 88 session files: transcripts age out on a rolling ~33-day retention, so
the corpus under the measurement changes every day and every rerun silently redefines its
own denominator. A rate with no addressable corpus behind it is a claim, not a result.

A snapshot is therefore content-addressed, never mtime-addressed. An earlier system here
signed generated artifacts with mtime and the signature went stale across checkouts while
the content was identical -- and, worse, matched while content differed. `corpus_sha` is a
hash over (relative path, content hash) pairs in sorted order: it changes when the corpus
changes and only then.

Snapshots record what was measured, not the measurement. Deleting one loses reproducibility,
not data.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("~/.claude/projects").expanduser()

#: Session transcripts sit one level down; subagent transcripts nest deeper and are
#: attributed to their parent session, so the snapshot pins both.
SESSION_GLOB = "*/*.jsonl"
SUBAGENT_GLOB = "*/**/*.jsonl"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class SnapshotEntry:
    path: str
    sha256: str
    bytes: int


@dataclass
class Snapshot:
    """What a measurement was taken over. `corpus_sha` is the identity of the corpus."""

    created_at: int
    root: str
    corpus_sha: str
    session_files: int
    total_files: int
    entries: list[SnapshotEntry] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Snapshot:
        raw = json.loads(text)
        raw["entries"] = [SnapshotEntry(**e) for e in raw.get("entries", [])]
        return cls(**raw)


def corpus_sha(entries: list[SnapshotEntry]) -> str:
    """Identity of a corpus: sorted (path, content-hash) pairs. Never size, never mtime --
    a size-tolerant check here would pass while the content underneath was swapped."""
    payload = "\n".join(f"{e.path}\t{e.sha256}" for e in sorted(entries, key=lambda e: e.path))
    return hashlib.sha256(payload.encode()).hexdigest()


def take(root: Path | str = DEFAULT_ROOT, *, note: str = "") -> Snapshot:
    base = Path(root).expanduser()
    seen: dict[str, SnapshotEntry] = {}
    sessions = 0
    for path in sorted(base.glob(SUBAGENT_GLOB)):
        if not path.is_file():
            continue
        rel = str(path.relative_to(base))
        seen[rel] = SnapshotEntry(path=rel, sha256=_sha256_file(path), bytes=path.stat().st_size)
    for path in base.glob(SESSION_GLOB):
        if path.is_file():
            sessions += 1
    entries = list(seen.values())
    return Snapshot(
        created_at=int(time.time()),
        root=str(base),
        corpus_sha=corpus_sha(entries),
        session_files=sessions,
        total_files=len(entries),
        entries=entries,
        note=note,
    )


def verify(snapshot: Snapshot, root: Path | str | None = None) -> dict[str, Any]:
    """Compare a snapshot against the corpus on disk now.

    Returns the three ways a corpus moves, separately -- aged-out sessions are the expected
    drift and must not be conflated with a file whose content changed underneath a
    measurement, which would mean a transcript was rewritten.
    """
    base = Path(root or snapshot.root).expanduser()
    now = {e.path: e for e in take(base).entries}
    was = {e.path: e for e in snapshot.entries}
    missing = sorted(set(was) - set(now))
    added = sorted(set(now) - set(was))
    changed = sorted(p for p in set(was) & set(now) if was[p].sha256 != now[p].sha256)
    return {
        "corpus_sha_then": snapshot.corpus_sha,
        "corpus_sha_now": corpus_sha(list(now.values())),
        "identical": not (missing or added or changed),
        "missing": missing,
        "added": added,
        "changed": changed,
        "counts": {"missing": len(missing), "added": len(added), "changed": len(changed)},
    }


def save(snapshot: Snapshot, directory: Path | str) -> Path:
    target = Path(directory).expanduser() / "snapshots"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"corpus-{snapshot.corpus_sha[:12]}.json"
    path.write_text(snapshot.to_json())
    return path


def load(path: Path | str) -> Snapshot:
    return Snapshot.from_json(Path(path).expanduser().read_text())
