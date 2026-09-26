# Deploying flightdeck

Order matters. Each step is verifiable on its own, and a step that cannot be verified is not done.

| Step | Where | Verify with |
|---|---|---|
| 0. Real Claude Code on the review host | review host | `systemd-run --user --wait --pipe claude -p 'reply: ok'` |
| 1. Install the package | every box | `python3 -m flightdeck init` |
| 2. Claude Code collector | every dev box | `integration/claude-code/README.md` |
| 3. crush emitter | every dev box + review host | `integration/crush/INTEGRATION.md` |
| 4. Probes + doctor baseline | every box | `python3 -m flightdeck doctor` |
| 5. Nightly units | review host | `integration/systemd/README.md` |
| 6. Governor stays in shadow | review host | `python3 -m flightdeck governor status` |

## Step 0 is first for a reason

`sparky-review.service` runs real Claude Code, not a local-model wrapper. A local model cannot
reason over a night of traces and rewrite skills. Auth on the review host is the one prerequisite
that can fail for reasons outside this repo, and if it fails the entire nightly review is dead
while every timer still reports green. Prove it works before building on it.

## Step 1 — install

```sh
cd ~/dev/closed-loop-agent-tuning
pip install -e '.[dev]'
python3 -m flightdeck init
```

The store lives at `~/.local/state/sparky/turnlog/` (override with `SPARKY_TURNLOG_DIR`). It is
deliberately not inside any repo: telemetry contains redacted-but-still-sensitive text and must
never be committable by accident. The repo's `.gitignore` blocks `*.db` and `*.jsonl` as a second
line of defence.

## Step 4 — establish a doctor baseline

Run `doctor` **before** you believe any number this system produces:

```sh
python3 -m flightdeck doctor
```

Exit code 1 means something is unhealthy. On a fresh install expect findings — no turns yet, the
collectors silent. What matters is that each finding disappears as you complete the step that
should fix it. A finding that survives its own fix is the real signal.

Findings worth expecting on a first run, none of them flightdeck bugs:

- A symlink `doctor` expects (config, skills, a shared vault) is missing until the surrounding
  install script that's supposed to create it has actually been run.
- The upstream crush fork ships telemetry to its own vendor by default (`disable_metrics` unset in
  `crush.json`) until you turn it off as part of the crush integration step.
- A helper hook that degrades gracefully when its dependency is absent (e.g. a local memory/vault
  tool not installed on every box) will report that absence as a finding rather than fail silently.

## Step 6 — the governor stays in shadow

Every domain ships shadowed. It records what it would have chosen and changes nothing. Do not flip
a domain live by hand — let the nightly reviewer do it once `graduation_report` shows every arm
past `min_samples` with a positive KPI delta over seven days. Flipping early is how you end up
tuning on a table that only ever saw one arm.

## Turning it off

```sh
export SPARKY_TURNLOG=0     # emitter off for one invocation
export SPARKY_JUDGE=0       # no judge calls
systemctl --user disable --now 'sparky-review*.timer'
```

Nothing in the system requires telemetry to be on. Every wire point is nil-safe and every hook
exits zero.
