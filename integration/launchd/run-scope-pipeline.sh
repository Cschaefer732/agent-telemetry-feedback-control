#!/bin/bash
# Runs `flightdeck scope ingest` + `flightdeck kpi` on a schedule (see
# com.sparky.flightdeck.scope-pipeline.plist in this directory), because before this nothing
# scheduled either command at all -- both required a human to remember to type them.
#
# Two things this script exists specifically to prevent, both drawn from CLAUDE.md's own
# failure log:
#   1. A silently-dead job. launchd running the unit proves nothing about the pipeline
#      inside it; every run writes a status file with its exit code and timestamp, and a
#      STALE status file (not updated in > 2x the interval) is itself a finding, whether the
#      cause is launchd not firing, flock never releasing, or the python process wedged.
#   2. An overlapping run corrupting the append-only store. flock on a lockfile makes two
#      concurrent invocations (a slow run bumping into the next scheduled tick) a no-op
#      instead of two writers touching sqlite at once.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="${SPARKY_TURNLOG_DIR:-$HOME/.local/state/sparky/turnlog}"
LOCK_DIR="$STATE_DIR/scope-pipeline.lock.d"
STATUS_FILE="$STATE_DIR/scope-pipeline-status.json"
PYTHON_BIN="${FLIGHTDECK_PYTHON:-python3}"
# macOS ships no `flock(1)` (that's util-linux), so overlap protection is a `mkdir` lock --
# mkdir is atomic on every filesystem this runs on. A lock older than 2 intervals is treated
# as abandoned (a killed/crashed prior run) rather than left to wedge the job forever.
STALE_LOCK_SECONDS=3600

mkdir -p "$STATE_DIR"

if [ -d "$LOCK_DIR" ]; then
    lock_age=$(( $(date +%s) - $(stat -f %m "$LOCK_DIR" 2>/dev/null || echo 0) ))
    if [ "$lock_age" -lt "$STALE_LOCK_SECONDS" ]; then
        echo "$(date -u +%FT%TZ) scope-pipeline: previous run still holds the lock (${lock_age}s old), skipping" >&2
        exit 0
    fi
    echo "$(date -u +%FT%TZ) scope-pipeline: removing stale lock (${lock_age}s old)" >&2
    rmdir "$LOCK_DIR" 2>/dev/null
fi
mkdir "$LOCK_DIR" || exit 0   # someone else won the race just now -- fine, they run instead
trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

cd "$REPO_DIR" || exit 1

overall_rc=0
run_step() {
    local name="$1"
    shift
    echo "=== $(date -u +%FT%TZ) scope-pipeline: $name ==="
    "$PYTHON_BIN" -m flightdeck --json "$@"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "$(date -u +%FT%TZ) scope-pipeline: $name FAILED rc=$rc" >&2
        overall_rc=1
    fi
    return 0
}

run_step "scope ingest" scope ingest
# `scope kpi` (not the top-level `kpi`): this is the one that runs aggregate() through
# health_verdict/health_exit_code, i.e. the only call that can turn a long silence into a
# nonzero exit. The top-level `kpi` reports turn metrics and would never notice.
run_step "scope kpi" scope kpi

python3 - "$STATUS_FILE" "$overall_rc" <<'PY'
import json
import sys
import time

status_path, rc = sys.argv[1], int(sys.argv[2])
with open(status_path, "w") as fh:
    json.dump({"last_run_utc": time.time(), "exit_code": rc}, fh)
PY

echo "$(date -u +%FT%TZ) scope-pipeline: done, overall_rc=$overall_rc"
exit "$overall_rc"
