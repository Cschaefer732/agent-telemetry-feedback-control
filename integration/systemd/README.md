# Nightly review units (review host) + per-host metrics collection (every box)

Four `systemd --user` units rather than one. A single unit doing sync → rollup → review → verify
would report one exit code for four different failures, and the one that matters most (the review
produced nothing) would be indistinguishable from a laptop being asleep.

| Unit | Time | Does |
|---|---|---|
| `sparky-review-sync` | 01:40 | Pull turnlogs from every box, merge into the canonical store |
| `sparky-review-rollup` | 01:50 | Replay JSONL, score unscored turns, expire text past TTL, run probes, build samples, render `flightdeck/state/fleet-status.md` |
| `sparky-review` | 02:00 | Real Claude Code headless, reads the samples, applies + commits + pushes |
| `sparky-review-verify` | 03:10 | Regression check, auto-revert, heartbeat |

All four set `Persistent=true`, so a missed window (box off, suspended) backfills on next boot
rather than skipping a night silently.

`sparky-collect-host-metrics` is a fifth unit, but not part of that chain and not review-host-only:
it runs on **every** box in your fleet every 5 minutes, snapshotting that host's own CPU/RAM/GPU/disk
into its local `host_metrics` table. `sync-turnlogs.sh` (01:40, review host only) then pulls each
host's rows into the canonical store, and `sparky-review-rollup` renders them into
`fleet-status.md`. Without this unit running on a box, that box's row in `fleet-status.md` stays
empty forever — install it on every box, not just the review host.

| Unit | Where | Time | Does |
|---|---|---|---|
| `sparky-review-sync` | review host only | 01:40 | Pull turnlogs from every box, merge into the canonical store |
| `sparky-review-rollup` | review host only | 01:50 | Replay JSONL, score unscored turns, expire text past TTL, run probes, build samples, render `flightdeck/state/fleet-status.md` |
| `sparky-review` | review host only | 02:00 | Real Claude Code headless, reads the samples, applies + commits + pushes |
| `sparky-review-verify` | review host only | 03:10 | Regression check, auto-revert, heartbeat |
| `sparky-collect-host-metrics` | every box | every 5min | Snapshot this host's CPU/RAM/GPU/disk into `host_metrics` |

## Install (review host: the nightly review chain)

```sh
mkdir -p ~/.config/systemd/user
cp ~/dev/agent-telemetry-feedback-control/integration/systemd/*.service ~/.config/systemd/user/
cp ~/dev/agent-telemetry-feedback-control/integration/systemd/*.timer   ~/.config/systemd/user/
install -m 0755 ~/dev/agent-telemetry-feedback-control/integration/systemd/sync-turnlogs.sh ~/.local/bin/

systemctl --user daemon-reload
systemctl --user enable --now sparky-review-sync.timer sparky-review-rollup.timer \
                              sparky-review.timer sparky-review-verify.timer

# Timers only fire while a user session exists. Without lingering, closing the SSH session stops
# the whole chain — and it stops silently, which is the worst failure mode available here.
loginctl enable-linger "$USER"
```

## Install (every box: host metrics)

Run this on every box where `~/dev/agent-telemetry-feedback-control` is checked out and you want that box's
resource usage showing up in `fleet-status.md` — the review host and every other agent-running
box in your fleet.

```sh
mkdir -p ~/.config/systemd/user
cp ~/dev/agent-telemetry-feedback-control/integration/systemd/sparky-collect-host-metrics.service ~/.config/systemd/user/
cp ~/dev/agent-telemetry-feedback-control/integration/systemd/sparky-collect-host-metrics.timer   ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now sparky-collect-host-metrics.timer
loginctl enable-linger "$USER"
```

## Prerequisite: real Claude Code on the review host (milestone M0)

`sparky-review.service` runs `claude -p` against a real, authenticated Claude Code install, not a
local model wrapper. A local model cannot reason over a night of traces and rewrite skills. This
is the one prerequisite that can fail for reasons outside this repo's control, so verify it before
enabling the review timer:

```sh
systemd-run --user --wait --pipe claude -p 'reply with exactly: ok'
```

It must succeed non-interactively, inside a systemd user session, with no TTY. If it prompts for
auth, the review stage will hang every night and the timer will look healthy while producing
nothing.

## Verify the chain actually ran

```sh
systemctl --user list-timers 'sparky-review*'
journalctl --user -u sparky-review.service -n 50 --no-pager
python3 -m flightdeck doctor
```

`doctor` is the real check: it reports whether last night produced turns, probes, and a digest.
A green timer with an empty store is the exact failure this project was built to surface.
