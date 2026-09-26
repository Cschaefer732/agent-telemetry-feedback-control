---
type: topic
updated: 2026-08-14
sources: [flightdeck/review.py, integration/review-skill/SKILL.md]
---

# Guardrails on the nightly reviewer

The reviewer writes to `main` unattended at 03:00. The boundary on its authority is enforced by
tested code in `flightdeck/review.py`, **not** by an instruction in a prompt that a model may
reinterpret at 3am.

## Deny beats allow, always

`Guardrails.is_writable` checks `DENIED_PATTERNS` before `WRITABLE_PATTERNS`. If the order were
reversed, the reviewer could widen its own authority by adding an allow entry — which is exactly
the failure mode that makes unattended writes dangerous.

Resolution rules: paths are resolved without requiring existence (the reviewer creates new files,
and a not-yet-existing path must still be checked), and anything that resolves outside the repo —
via `..`, an absolute path, or a symlink that escapes the tree — is rejected as `outside_repo`.

## What it may write

- `spark/crush/skills/**/*.md`, `agents/*.md`, `modes/*.md`
- `spark/crush/CRUSH.md`, `spark/crush/MEMORY.md`
- `governor.toml` — numbers only, each clamped to the min/max declared in its own comment
- `claude/skills/**`, `claude/rules/**`
- `docs/reviews/*.md`

## What it may never write

`*.go`, `*.patch`, `build.sh`, `install.sh`, `*.service`, `*.timer`, `crush.json`,
`settings.json`, anything under `.git/`, and `flightdeck/*.py`.

The reason is narrow and practical: **it cannot rebuild and cross-compile the crush fork at 3am,
so it must not be able to break it.** It also may not edit the module that defines its own limits.

If a fix genuinely requires a denied path, the reviewer writes the diagnosis and the proposed patch
into the digest and leaves it for a human.

## Regression auto-revert

Each change is one commit, recorded in `tuning_changes`. `regression_check` compares the trailing
7-day mean `kpi_score` before and after each change that is 2–14 days old. A drop greater than 5%
with at least 30 samples on each side marks it reverted.

Younger than 2 days there is not enough post-change evidence; older than 14 days, attributing a
drift to one commit is guesswork. Anything reverted joins a do-not-retry list so the next night
cannot re-derive the same bad change from the same evidence.

One change per commit is not a style preference — the revert works by commit, so a batched commit
forces it to throw away good changes along with the bad.

Related: [[kpi-definitions]], [[governor]], [[schema-contract]].
