---
type: entity
updated: 2026-08-14
sources: [go/turnlog/turnlog.go, go/turnlog/turn.go, go/turnlog/redact.go, integration/crush/INTEGRATION.md]
---

# turnlog emitter

The Go package vendored into the crush fork as `internal/turnlog`. Zero external dependencies,
specifically so it can be copied in without touching `go.mod`.

It is built around one rule: **it may lose data, but it may never slow down or fail a turn.**

## Drop, never block

Every `enqueue` is a non-blocking channel send with a `default` branch that increments a drop
counter. A telemetry backpressure stall inside the agent loop would be a user-visible regression; a
counted drop is not.

Turn-close records are the exception — they bypass the queue and write synchronously with an
`fsync`. Losing an event costs detail; losing a turn record costs the whole row and every KPI
derived from it.

The emitter writes a `collector_heartbeat` probe on shutdown carrying its own written/dropped/error
counts. Without it a queue that overflowed under load would be invisible, and the KPIs from that
period would read as complete when they were not. See [[silence-is-a-finding]] in the fleet notes.

## Nil-safe by design

A `nil *Emitter` is a valid receiver for every method, and `nil *TurnRecorder` likewise. `New`
returns nil when telemetry is disabled or the directory is unusable — never an error. This means
the crush wire points need no nil checks, no build tags, and no decision about what to do when
telemetry cannot start. None of them should abort a turn over it.

## Redaction at write time

Scrubbing later — in the rollup, or in the nightly job — would mean the raw secret existed on disk
in between, on a box whose logs get rsynced to another machine and read by an agent. So it happens
in-process, before anything touches disk, and it **fails closed**: if redaction panics, the text is
discarded rather than written raw.

The Go and Python scrubbers are deliberately duplicated rather than shared, because the Go side
must stay dependency-free. Where they disagree, the Python side is authoritative — it also scrubs
on ingest.

## What it captures that hooks cannot

Per-request latency, true token accounting including the `estimated` flag, the remaining-token math
that decides compaction, tool-search activations, and window occupancy at each step boundary. crush
already computes occupancy inside its auto-summarize check and discards it; keeping it is the
single highest-value line in the integration.

Config via `crush.json`; kill switch via `SPARKY_TURNLOG=0` in the environment, so telemetry can be
disabled for one invocation without editing config other boxes have symlinked.

Related: [[schema-contract]], [[kpi-definitions]], [[governor]].
