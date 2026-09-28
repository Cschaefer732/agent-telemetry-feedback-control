# Claude Code collector

Captures Claude Code turns into the same schema as the crush emitter, so the nightly review can
compare a local 30B and a frontier model on equivalent task shapes rather than on vibes.

This collector is deliberately lossy. Hooks cannot see per-request latency, retry counts, cache
hits, or window occupancy, so those columns stay **NULL** — never zero. A zero would read as a
real measurement and quietly corrupt the cross-source comparison this collector exists for.

## Install

Add to the `hooks` block of `~/.claude/settings.json`, alongside the entries already there. Every
existing hook keeps working; these are additional entries in the same arrays.

```json
{
  "UserPromptSubmit": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python3 \"$HOME/dev/agent-telemetry-feedback-control/integration/claude-code/turnlog-hook.py\"",
          "timeout": 5
        }
      ]
    }
  ],
  "PreToolUse": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python3 \"$HOME/dev/agent-telemetry-feedback-control/integration/claude-code/turnlog-hook.py\"",
          "timeout": 5
        }
      ]
    }
  ],
  "PostToolUse": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python3 \"$HOME/dev/agent-telemetry-feedback-control/integration/claude-code/turnlog-hook.py\"",
          "timeout": 5
        }
      ]
    }
  ],
  "PostToolUseFailure": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python3 \"$HOME/dev/agent-telemetry-feedback-control/integration/claude-code/turnlog-hook.py\"",
          "timeout": 5
        }
      ]
    }
  ],
  "Stop": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python3 \"$HOME/dev/agent-telemetry-feedback-control/integration/claude-code/turnlog-hook.py\"",
          "timeout": 10
        }
      ]
    }
  ]
}
```

One entrypoint for all five events: the hook dispatches on `hook_event_name` from the payload.
A single script means one thing to verify and one thing that can be silently wrong, instead of five.

`PostToolUseFailure` matters as much as `PostToolUse` — without it, `tool_reliability` scores 1.0
forever because only successes are ever recorded.

## Verify it actually fired

Configuration is not evidence. Prove the hook ran:

```sh
# 1. Note the current count
python3 -m flightdeck kpi --since 1h | grep -A3 kpi_by_source

# 2. Run any Claude Code turn in another terminal, then:
python3 -m flightdeck doctor --since 1h
```

`kpi_by_source` must now contain a `claude-code` entry. If `doctor` reports
`collector_heartbeat` with `claude-code: silent`, the hook is configured but not running — check
`~/.local/state/sparky/turnlog/hook-errors.log`, which is where the hook writes tracebacks
instead of failing your turn.

## The contract this hook keeps

- **It never fails a turn.** Every exception is caught, logged to `hook-errors.log`, and the
  process exits 0 — including on malformed or empty stdin. Verified by test and by
  `echo 'not json' | python3 turnlog-hook.py; echo $?`.
- **Unobservable fields stay NULL.** `model_ms`, `retries`, `cached_tokens`, `context_peak`,
  `context_window`.
- **Text is redacted at write time** through the same scrubber as the Go emitter, with the same
  14-day TTL.
- **Turn correlation survives across processes.** Each hook invocation is its own process, so the
  open turn id lives in a per-session state file under the turnlog directory. Concurrent sessions
  do not collide, and a `Stop` with no matching open turn still writes a minimal row — a turn we
  half-saw is still evidence.
