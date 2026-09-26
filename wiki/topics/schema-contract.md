---
type: topic
updated: 2026-08-14
sources: [flightdeck/schema.py, flightdeck/models.py, flightdeck/store.py, go/turnlog/record.go]
---

# The schema contract

The collector, the scorer, and the reviewer are separated by a **schema**, not by function calls.
Anything that can write the schema is a valid source; anything that can read it is a valid
consumer. The Go emitter and the Python tooling share **no code at all** — only a JSONL format.

That is deliberate: `go/turnlog` must vendor into the crush fork with zero dependencies, and the
Python side must be readable by Claude Code, which has no crush and no Go.

## Tables

| Table | Holds |
|---|---|
| `turns` | one row per turn: identity, model/provider/tier, mode, usage, context peak, outcome, `kpi_score`, `kpi_components`, `flagged`, `judged` |
| `events` | the timeline: 20 kinds, each with optional name, duration, ok flag, JSON payload |
| `texts` | captured bodies, redacted at write, with `expires_at` (14 days) |
| `judgments` | the detached judge's advisory verdict per flagged turn |
| `governor_decisions` | what the [[governor]] chose or would have chosen, with its feature vector |
| `probes` | symlink integrity, migration state, hook liveness, collector heartbeat |
| `tuning_changes` | what the nightly reviewer changed, so a regression can revert it |

Schema version 3. Migrations are forward-only in `MIGRATIONS`; v2 adds the `events` dedupe index
that makes the cross-box merge idempotent, v3 adds `tuning_changes`.

## Two writes per record

sqlite for queries, JSONL for durability. sqlite can be locked by the nightly rollup, corrupted, or
mid-migration; the JSONL append is a single `write()` to an `O_APPEND` handle. **sqlite is a
derived index that can always be rebuilt by replaying JSONL** — `Store.ingest_jsonl` is idempotent,
so replay never double-counts.

The store is never `crush.db`. A crush migration, a session delete, or a db reset must not be able
to destroy telemetry history.

## The payload-is-a-JSON-string trap

`models.Event.to_row()` serializes `payload` to a **JSON string**, and `from_row()` `json.loads()`
it back. The Go emitter must therefore emit `payload` as a string containing JSON, not as a bare
object. Emitting an object silently breaks replay. `TestEventPayloadIsJSONString` and
`test_event_payloads_decode` guard both ends.

## Optional means absent, not zero

Go optional fields are pointers with `omitempty`, so an omitted key falls back to the Python
dataclass default. "The provider reported 0 tokens" and "no provider call happened" are different
facts, and [[kpi-definitions]] treats them differently. The `claude-code` collector leans on this:
every field a hook cannot observe stays NULL.

## Keeping the two sides honest

`tests/test_cross_language.py` runs the **real** Go emitter via `go/cmd/emit-fixture` and replays
its output through the **real** Python ingest path. A hand-written fixture would pass forever while
the two implementations drifted; this fails the moment either side renames a field or loses an
event kind.

Related: [[turnlog-emitter]], [[guardrails]].
