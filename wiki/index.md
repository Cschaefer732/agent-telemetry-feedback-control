# Index

<!-- One line per wiki page: [Title](path) — one-sentence summary -->

## Topics

- [KPI definitions](topics/kpi-definitions.md) — the six scored components, what makes each unmeasurable per source, and the flag rule thresholds
- [The schema contract](topics/schema-contract.md) — the tables, the two-writes durability model, and the payload-is-a-JSON-string trap between Go and Python
- [Guardrails on the nightly reviewer](topics/guardrails.md) — what the 3am reviewer may and may not write, why deny beats allow, and how regression auto-revert works

## Entities

- [Governor](entities/governor.md) — the selector: arms, features, shadow mode, and the graduation criteria for going live
- [turnlog emitter](entities/turnlog-emitter.md) — the Go collector: drop-never-block, nil-safe receivers, redaction at write time
