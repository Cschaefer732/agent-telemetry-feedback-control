---
type: topic
updated: 2026-08-14
sources: [flightdeck/kpi.py, flightdeck/models.py]
---

# KPI definitions

Every turn is scored at close, deterministically, with no model calls. "Low performance" has to be
a number before the nightly review can sample on it.

`kpi_score = Σ wᵢ · componentᵢ`, weights in `[kpi.weights]` of `governor.toml`, default equal
(1/6 each). All six components are clamped to `[0, 1]` by a single `_clamp` helper, and each is
defined so that an *unmeasurable* component scores neutral rather than punishing the turn.

## The six components

| Component | Computed from | Unmeasurable when |
|---|---|---|
| `completion` | `todo` events: closed ÷ opened. No todo events → 1.0 if `outcome == "ok"`, else 0.0. A `critic` event with `verdict: fail` caps it at 0.5; `outcome` of error/cancelled forces 0.0 | never — falls back to outcome |
| `tool_reliability` | 1 − failed ÷ total over `tool_call` events (`ok == 0` is failed) | no tool calls → 1.0 |
| `efficiency` | baseline ÷ this turn's `total_tokens`, clamped. At or below baseline scores 1.0 | no trailing baseline, or zero tokens → 1.0 |
| `focus` | 1 − churn, churn = (repeat_edits + 2×reverts) ÷ edits. Reverts weigh double — an edit-then-revert inside one turn is the clearest flailing signal there is | no edits → 1.0 |
| `context_health` | 1 − occupancy, minus 0.2 per real compaction | occupancy is NULL (see below) |
| `autonomy` | 1 − min(1, (interrupts + permission denials) ÷ 3) | never |

## The context_snapshot trap

`context_health` counts `compaction` events only. Snapshots are a **separate event kind**
(`context_snapshot`). They shared a kind briefly during construction, which meant every periodic
window reading counted as a compaction — driving this component to zero on precisely the
best-instrumented turns and inverting what the score rewarded. See [[schema-contract]].

When occupancy is NULL — which is always true for the `claude-code` source, since hooks cannot see
the window — the component starts from 1.0 and *still* applies the compaction penalty. Skipping the
penalty entirely would score Claude Code turns artificially high against crush turns, which defeats
the reason both sources share one table.

## Prefill signals (not yet scored)

`ttft_ms` and `tools_hash_changes` are recorded per turn as of schema v4, merged in from the
fork's own timing work (a small patch to the fork adding per-request timing). They are **not** part of the
composite score yet, deliberately — they are strong candidates for [[governor]] features, but
adding a feature before there is evidence for its weight is how a selector overfits.

- `ttft_ms` — lowest time-to-first-token in the turn. Separates prefill from generation; on a
  local model a warm KV prefix returns a small TTFT even when total latency is long.
- `tools_hash_changes` — how many times the active tool set changed mid-turn. Each change
  invalidates the prompt prefix, so this measures what tool-search churn actually costs.

Both are NULL when unobserved, and `tools_hash_changes` is emitted as 0 only when the tool set
genuinely never changed — "never changed" and "never observed" must not both read as zero.

## Cross-source comparison

Compare **component-wise**, not on the composite, when comparing `crush` against `claude-code`.
The composite silently rewards a source for what it cannot measure. `kpi_components` is stored as
JSON on every turn precisely so this comparison is possible.

## Flag rule

A turn is flagged, and so queued for the [[governor]]-adjacent judge, if **any** of:

- `outcome != "ok"`
- critic verdict `fail`
- tool failure rate > 0.25 (`thresholds.tool_error_rate`)
- `wall_ms` above the trailing-7-day p95 for that (host, tier) — skipped when no p95 exists yet
- a compaction fired mid-turn
- an edit-then-revert was detected
- the user interrupted
- `kpi_score < 0.5` (`thresholds.low_kpi`)

Flag reasons are stored as slugs, not just a boolean, so the reviewer knows *why* a turn surfaced.

Related: [[turnlog-emitter]], [[guardrails]].
