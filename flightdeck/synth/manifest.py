"""The reproducibility contract: seed, generator version, params, and a content hash
written next to every generated corpus.

Content, not mtime, not size -- this repo has been burned once by an mtime-keyed
signature (mtime-signature-stale-across-checkouts) and once by a length check that
missed a content swap (length-checks-hide-content-swaps). `content_hash` hashes the
serialized row VALUES; `verify_manifest` recomputes and compares, never warn-and-continue.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flightdeck.synth.provenance import GENERATOR_VERSION


@dataclass(frozen=True)
class Manifest:
    kind: str  # "scope_rows" | "prompt_labels"
    seed: int
    params: dict[str, Any]
    count: int
    content_hash: str
    generator_version: str = GENERATOR_VERSION
    created_at: int = field(default_factory=lambda: int(time.time()))

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _canonicalize(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(rows: list[dict[str, Any]]) -> str:
    """sha256 over each row canonicalized and joined with '\\n', in the given order.
    Order matters -- callers must pass rows in the deterministic generation order, not a
    resorted one, or two identical generations could hash differently for no reason."""
    body = "\n".join(_canonicalize(row) for row in rows)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def write_manifest(path: Path, manifest: Manifest) -> None:
    path = Path(path)
    path.write_text(json.dumps(manifest.to_row(), sort_keys=True, indent=2) + "\n")


def read_manifest(path: Path) -> Manifest:
    data = json.loads(Path(path).read_text())
    return Manifest(**data)


def verify_manifest(path: Path, rows: list[dict[str, Any]]) -> bool:
    """Recompute content_hash from rows and compare to the manifest on disk. False on any
    mismatch -- including a manifest that fails to parse."""
    try:
        manifest = read_manifest(path)
    except (OSError, ValueError, TypeError):
        return False
    return content_hash(rows) == manifest.content_hash
