#!/usr/bin/env python3
"""Single entrypoint wired into every Claude Code hook event turnlog cares about.

Must never fail a Claude Code turn: always exits 0, regardless of stdin content, store state, or
bugs in the collector. Resolves the repo root from __file__ so it runs correctly from any cwd
with no PYTHONPATH set — Claude Code invokes hook commands with the user's shell cwd, not this
repo's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flightdeck.collect_claude import handle  # noqa: E402


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return handle(payload)


if __name__ == "__main__":
    sys.exit(main())
