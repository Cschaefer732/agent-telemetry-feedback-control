#!/usr/bin/env bash
# Pull every box's turnlog into the canonical store on the review host, then merge.
#
# Runs ON the review host, pulling. Pushing from each box would mean a sleeping laptop
# silently skips its own sync and nobody notices; pulling makes an unreachable box a
# recorded, visible gap.
#
# An unreachable host is NOT fatal. A laptop sleeps by design, and a review that aborts
# because a laptop was closed is a review that never runs.

set -uo pipefail

FLIGHTDECK_DIR="${FLIGHTDECK_DIR:-$HOME/dev/agent-telemetry-feedback-control}"
STATE_DIR="${SPARKY_TURNLOG_DIR:-$HOME/.local/state/sparky/turnlog}"
INBOX="$STATE_DIR/inbox"
LOG="$STATE_DIR/sync.log"

# host:label pairs, "user@host:label" separated by spaces. Prefer stable (tailnet/VPN) names
# over LAN IPs -- LAN addresses differ per network segment and a laptop is not always on the
# same one. The review host itself needs no entry here.
#
# EXAMPLE below: replace with your own fleet. Only list boxes that actually run agent
# sessions worth syncing -- a box that never produced turnlog data doesn't need an entry.
HOSTS="${SPARKY_SYNC_HOSTS:-user@203.0.113.10:laptop user@203.0.113.11:linux-host}"

mkdir -p "$INBOX"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

reached=0
missed=0

for entry in $HOSTS; do
    target="${entry%:*}"
    label="${entry##*:}"
    dest="$INBOX/$label"
    mkdir -p "$dest"

    # BatchMode so a host that wants a password fails fast instead of hanging the timer.
    # Deliberately plain rsync flags: macOS ships an old rsync that exits usage-error on
    # --info and friends while still returning a success-looking status in some pipelines.
    if rsync -az --timeout=60 \
        -e "ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new" \
        --include='turnlog.db' --include='events-*.jsonl' --exclude='*' \
        "$target:.local/state/sparky/turnlog/" "$dest/" >>"$LOG" 2>&1; then
        log "synced $label"
        reached=$((reached + 1))

        # repo-index.md and registry.json are an OPTIONAL per-host project registry produced
        # by a separate project-snapshot script, if you run one -- a different tree from the
        # turnlog dir above. Pull both explicitly by path so they land flat in $dest/, where
        # fleet_status.py's reader expects them (host_dir / "registry.json"). Best-effort: a
        # host that has never run such a script is not a sync failure.
        rsync -az --timeout=60 \
            -e "ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new" \
            "$target:.personal-agent/state/repo-index.md" \
            "$target:.personal-agent/state/projects/registry.json" \
            "$dest/" >>"$LOG" 2>&1 \
            || log "note: repo-index/registry not yet available on $label"
    else
        log "UNREACHABLE $label (rsync exit $?) — its turns are missing from tonight's review"
        missed=$((missed + 1))
    fi
done

log "sync complete: $reached reached, $missed unreachable"

# Merge into the canonical store. Done in Python so the merge uses the same dedupe rules as every
# other write path rather than a second, divergent implementation in SQL here.
cd "$FLIGHTDECK_DIR" || exit 1
python3 -m flightdeck sync --inbox "$INBOX" --hosts-reached "$reached" --hosts-missed "$missed" \
    >>"$LOG" 2>&1

# Always succeed. A partial sync is a finding for the reviewer to report, not a reason to abort
# the chain and produce no review at all.
exit 0
