---
name: nightly-review
description: Use at 02:00 on the review host to review the day's agent turns and tune the system. Reads the flightdeck brief, samples complex and low-KPI turns, sweeps every session for problems, then edits skills, prompts, and governor weights within the guardrails and commits each change to main.
---

# Nightly review

You are tuning a system that will run unattended tomorrow on whatever you leave behind. Every
change you make is read by tomorrow night's reviewer as its baseline, so a wrong change does not
just cost one day — it compounds until a regression check catches it.

Work from evidence in the store. Do not tune from intuition about what "should" help.

## 1. Read the brief

`python3 -m flightdeck sample --write-brief` has already produced the brief. It contains:

- **Aggregates**: KPI by tier, source and host; tool failure leaderboard; compaction frequency;
  token spend; flagged-vs-judged counts.
- **Sample A (complexity)**: the most complicated turns of the day.
- **Sample B (low KPI)**: the worst-scoring turns not already reviewed.
- **Sample C (sweep)**: problem signals across *every* session.
- **Probe results**: symlink integrity, migration state, hook liveness, collector heartbeat.
- **Governor shadow report**: what the governor would have chosen versus what ran.

For any turn you want to look at properly:

```sh
python3 -m flightdeck show <turn_id>          # full record, events timeline, texts, judgment
```

## 2. Check for silence before you check for problems

Do this first, every night, without exception.

A collector that produced nothing, a hook that never fired, a probe that never ran, a judge queue
that dropped more than it processed — each of these means every other number in the brief is
understated. This system exists because a `Stop` hook once sat dead for weeks and nothing said so.

If `aggregates.empty_window` is true, or a box that was awake reported zero turns, or
`probe_never_ran` appears in the sweep: **that is tonight's finding**. Report it at the top of the
digest and do not tune weights from data you now know is partial.

## 3. Diagnose

For each sample-A, sample-B, and escalated sample-C turn, read the record and answer:

1. What was the turn asked to do, and did it do it?
2. Where did the time and the tokens actually go?
3. Was the model tier right — overkill, right, or underpowered?
4. Were the loaded skills used? Did a missing skill cause the failure?
5. Was the failure the model's, or the harness's (tool error, permission denial, missing context,
   compaction at the wrong moment)?

The judge verdict in `judgments` is **advisory evidence, not a conclusion**. It comes from a 30B
and is frequently confidently wrong. Read the trace yourself before you agree with it.

A single bad turn is an anecdote. Look for the pattern across the day before changing anything —
if you cannot point at three turns or a clear aggregate, you are guessing.

## 4. Change things

You may write **only** these paths -- see `flightdeck.review.WRITABLE_PATTERNS` (adjust both
to your own repo's layout):

- `crush-fork/skills/**/*.md`, `crush-fork/agents/*.md`, `crush-fork/modes/*.md`
- `crush-fork/CRUSH.md`, `crush-fork/MEMORY.md`
- `governor.toml` (numbers only, each clamped to the min/max declared in its own comment)
- `claude/skills/**`, `claude/rules/**`
- `docs/reviews/YYYY-MM-DD.md`

You may **never** write `*.go`, `*.patch`, `build.sh`, `install.sh`, systemd units, `crush.json`,
`settings.json`, or anything under `.git/`. You cannot rebuild and cross-compile the crush fork at
3am, so you must not be able to break it. `flightdeck/review.py` enforces this; do not attempt to
work around it. If a fix genuinely requires a denied path, write the diagnosis and the proposed
patch into the digest and leave it for a human.

Before committing, verify your own paths:

```sh
python3 -m flightdeck guard --check <path> [<path>...]
```

Kinds of change, in rough order of how often they are the right answer:

- **Skill text**: a skill that loads often but is never used is costing tokens for nothing —
  tighten its description so it stops matching, or fix it so it earns its place.
- **Prompt/mode text**: repeated harness failures usually trace to an instruction the model keeps
  misreading, not to the model.
- **Governor weights**: only with sample counts behind them. Small steps.
- **Domain graduation**: flip a governor domain out of shadow only when `graduation_report` shows
  every arm past `min_samples` AND a positive mean KPI delta over 7 days. This is the single
  highest-risk change available to you — one domain per night, never two.
- **Symlink repair**: re-run `install.sh`'s link step for a probe-failed path. Report it; a broken
  symlink usually means something moved and other things point at it too.

## 5. Commit

One change, one commit. Never batch unrelated tuning into a single commit — the regression check
reverts by commit, and a batched commit forces it to throw away good changes with the bad.

```
tune(<domain>): <what changed>

Evidence: <n> turns, KPI <before> → <expected>, sample ids <...>
Reverts-if: KPI(7d) drops >5% with n>=30
```

Record each one so the regression check can find it:

```sh
python3 -m flightdeck record-change --domain <domain> --path <path> \
    --summary "<one line>" --commit "$(git rev-parse HEAD)" --evidence '<json>'
```

Then push to `main`.

## 6. Check the do-not-retry list

```sh
python3 -m flightdeck reverted
```

A change that was reverted for regressing KPI must not be re-derived from the same evidence. If
tonight's analysis points at a previously reverted change, say so in the digest and explain what is
different now — or drop it.

## 7. Write the digest

`docs/reviews/YYYY-MM-DD.md`, containing:

- Silence findings first (§2), or an explicit statement that all collectors reported.
- What changed, with evidence and commit shas.
- **What you considered and rejected, and why.** Next month's reviewer needs to know an idea was
  already tried.
- **Every cap that dropped data** — sample truncations, judge-queue skips, unreachable hosts,
  emitter drops. A silent truncation reads as full coverage, which is a lie the next reviewer will
  believe.
- Open questions for a human.

## Rules

- Evidence before changes. If you cannot cite turns, do not change it.
- Small steps. You get another run tomorrow night; the system does not get another chance to
  un-break itself if you overshoot.
- Never claim a change worked. You applied it; the regression check decides whether it worked.
- If the store looks wrong — impossible values, missing days, duplicated turns — stop tuning and
  report it. Tuning on corrupt telemetry is worse than not tuning.
