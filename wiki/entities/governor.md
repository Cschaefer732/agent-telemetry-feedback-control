---
type: entity
updated: 2026-08-14
sources: [flightdeck/governor.py, flightdeck/governor.toml]
---

# Governor

The selector. Decides what the next turn should use, from weights the nightly reviewer tunes.

```
score(a) = Σ wᵢ,ₐ · fᵢ  +  β · success(bucket, a)  −  γ · cost(a)
```

Chosen over real ML (logistic regression, LinUCB) because it works from roughly 20 samples per arm
and because the reviewer can read and hand-edit the weights file. A trained model needs hundreds of
samples per arm before it beats the defaults, and is opaque to the thing doing the tuning. The KPI
table is shaped so a real model can be swapped in later without re-collecting anything.

## Domains and arms

| Domain | Arms |
|---|---|
| `model_tier` | fast / balanced / deep / frontier |
| `skills` | per-skill include or suppress |
| `compaction` | offset ×3 crossed with recall on/off → six named arms |
| `mode_delegation` | plan_first, critic, delegate |

## Features

All cheap and computed pre-turn: prompt token length, repo primary language, tool verbs in the
prompt, file references, whether the previous turn failed, `is_subagent`, context occupancy, and a
six-hour time-of-day bucket. `Features.bucket()` collapses these into a coarse, stable key for the
success table.

## Shadow mode

**Every domain ships shadowed.** The governor computes its choice, records it to
`governor_decisions`, and the system behaves exactly as it did before. That produces a
counterfactual record — "when it would have picked `deep` and we ran `fast`, what happened?" —
at zero behavioural risk.

`graduation_report` is the evidence for going live: every arm past `min_samples`, and a positive
mean KPI delta over the trailing 7 days. One domain per night, never two.

## Guards

- `min_samples` floor with an `optimistic_prior` so unexplored arms get tried
- ε-greedy exploration at 0.10 keeps the table from collapsing onto one arm
- recency decay by half-life, so a tier that was good three months ago does not outrank current
  evidence
- `clamp_config` enforces every declared min/max and *reports* violations rather than silently
  correcting them
- `choose` never raises — a broken config falls back to the first arm with `reason="config_error"`,
  because a governor that throws would break every turn

Kill switch: `SPARKY_GOVERNOR=0`, or `[governor] enabled = false`.

Related: [[kpi-definitions]], [[guardrails]], [[turnlog-emitter]].
