# Wiring turnlog into the crush fork

The fork is not a checked-out tree. Your own config repo's build script (e.g.
`<your-config-repo>/crush-fork/build.sh`) clones upstream crush at a pinned tag, applies your
fork's patch, and builds. So integrating turnlog means: apply the patch to a scratch clone, add
the package and the wire points, rebuild the patch from the diff, and cross-compile for your
inference host.

## 0. Prerequisites

```sh
cd <your-config-repo>/crush-fork
# read build.sh for the pinned TAG before starting; the wire points below are line-anchored to it
```

## 1. Vendor the package

`go/turnlog/` in this repo has **zero external dependencies** precisely so it can be copied in
without touching `go.mod`:

```sh
cp -R ~/dev/sparky-flightdeck/go/turnlog <clone>/internal/turnlog
```

Then delete `internal/turnlog/turnlog_test.go`'s module path assumptions if any appear — the
package itself imports nothing from this repo.

Keep the copy one-directional. `go/turnlog` here is the source of truth; edits made inside a crush
clone are lost the next time `build.sh` regenerates from the patch.

## 2. Wire points

Every call is nil-safe, so none of these need a guard. The emitter is created once and the
recorder is carried on the existing per-turn context.

### 2.1 Emitter lifecycle — `internal/cmd/root.go`

Create alongside the existing `event.Init()` / `shouldEnableMetrics(cfg)` call, and **disable
PostHog at the same time** (see §4):

```go
tl := turnlog.NewFromEnv()
defer tl.Close()
```

Hang `tl` off the same struct that already carries the coordinator's dependencies.

### 2.2 Turn open/close — `internal/agent/coordinator.go`, `run()`

Open the recorder where the turn begins; finish it at the same site as the `Stop` hook, which
already runs on `context.WithoutCancel` with a 60s cap. Use that same detached context — **a
cancelled turn is exactly the kind this system exists to explain**, so it must still record:

```go
rec := tl.StartTurn(turnlog.TurnMeta{
    SessionID:       sessionID,
    ParentSessionID: parentSessionID,
    CWD:             cwd,
    GitSHA:          gitSHA,
    TurnIdx:         turnIdx,
    AgentName:       agentName,
    IsSubagent:      isSubAgent,
    Mode:            string(currentMode),
    Provider:        providerID,
    Model:           modelID,
    Tier:            tierName,
    ContextWindow:   int64(contextWindow),
})
// ... turn runs ...
rec.Finish(outcome, finishReason, errorClass)
```

`outcome` maps from the same values the `Stop` hook already computes for `CRUSH_OUTCOME`.

### 2.3 Model requests — `internal/agent/agent.go`, step-finish callback

**This wire point already exists, on one particular fork.** A prior patch to that fork added an `llm.step` slog
line at exactly this site, plus the plumbing that makes it possible: `stepStart`/`stepFirstToken`,
a `markFirstToken()` closure called from `OnReasoningStart`, `OnReasoningDelta`, `OnTextDelta` and
`OnToolInputStart`, and an fnv hash of the active tool names. Keep all of that — it is the hard
part — and add the structured call beside it:

```go
rec.ModelRequest(turnlog.StepUsage{
    LatencyMS:        time.Since(stepStart).Milliseconds(),
    TTFTMS:           ttftMs,          // already computed for the llm.step line; -1 if no token
    PromptTokens:     int64(usage.InputTokens),
    CompletionTokens: int64(usage.OutputTokens),
    CachedTokens:     int64(usage.CacheReadTokens),
    FinishReason:     string(finishReason),
    Estimated:        estimated,
    Retries:          retries,
    ActiveTools:      stepToolCount,
    ToolsHash:        fmt.Sprintf("%016x", stepToolsHash),
})
```

The `slog.Info("llm.step", …)` line can stay — it is useful when tailing logs by eye — but it is
now redundant with the structured path, and only the structured path is queryable, scoreable, or
visible to the nightly review. Do not build a second analysis on top of the log line.

Why these two extra fields earn their place:

- **`TTFTMS`** separates prefill from generation. On a local model a warm KV prefix returns a small
  TTFT even when total latency is long, so this is the prompt-cache hit/miss signal — and it is a
  far better basis for tier selection than raw latency, which conflates the two.
- **`ToolsHash`** makes cache behaviour explainable. When it changes between steps the prompt
  prefix changed and the provider's cache was invalidated, so tool-search churn becomes a measured
  cost rather than a suspicion. turnlog counts the changes per turn into `tools_hash_changes`.

`estimated` is the existing flag set by `internal/agent/usage_fallback.go` when a provider reports
zero usage. Propagating it matters: a KPI computed from chars/4 approximations is not the same
measurement as one from real provider counts, and the scorer needs to know which it has.

### 2.3b Tool latency — same commit, `OnToolResult`

`6056f58` also added a `tool.latency` slog line, with a `toolStarts` map keyed by `ToolCallID`
under its own mutex. Reuse that timing rather than measuring again, and pass the id through so a
tool call can be correlated across events:

```go
rec.ToolCall(result.ToolName, time.Since(tcStart).Milliseconds(), ok, map[string]any{
    "tool_call_id": result.ToolCallID,
})
```

### 2.4 Context pressure — `internal/agent/agent.go`, `StopWhen`

This is the highest-value single line in the integration. `StopWhen` already computes

```go
remaining := contextWindow - (session.CompletionTokens + session.PromptTokens)
```

to decide whether to auto-summarize, and then throws it away. Keep it:

```go
rec.ContextSnapshot(int64(session.CompletionTokens+session.PromptTokens), int64(contextWindow))
```

Without this, context pressure can only be inferred after the fact from token totals; with it,
the peak occupancy per turn is a measured number.

### 2.5 Compaction — `internal/agent/agent.go` `Summarize()` and the `CompactHooks` bracket

```go
rec.Compaction(source, tokensBefore, tokensAfter, vetoed, len(summary))
```

`vetoed` is true when `PreCompact` returned deny — the fork already supports a hook halting
compaction, and a vetoed compaction that then blows the window is a distinct failure worth seeing.

### 2.6 Tools — `internal/agent/hooked_tool.go`

The existing wrapper already brackets every tool call and knows the permission and hook decisions:

```go
start := time.Now()
resp, err := inner.Run(ctx, params)
rec.ToolCall(toolName, time.Since(start).Milliseconds(), err == nil, map[string]any{
    "path": pathFromParams(params),   // only for edit/write/read tools
    "arg_bytes": len(rawParams),
})
```

Also emit from the same wrapper:
- `rec.Permission(toolName, decision)` on the permission-service result
- `rec.Hook(hookEvent, hookName, durationMS, ok, decision)` from `internal/hooks/runner.go`

Do **not** put whole tool arguments in the payload. `arg_bytes` plus the path is enough for churn
and failure analysis; full bodies go through `rec.Text(TextToolArg, …)` only when
`TextCapture=full`, and are redacted and TTL'd like everything else.

### 2.7 Skills — `internal/skills/tracker.go` + `coordinator.logTurnSkillUsage`

The fork already logs a per-turn skill summary. Emit it structurally at the same site:

```go
rec.SkillsLoaded(tracker.LoadedNames(), activeTotal)
```

`SkillUsed` is the one that needs new plumbing and is the reason skill precision is measurable at
all: emit it when a tool call or a search term matches a skill's declared tools/procedures. An
approximation is fine here — "loaded and something from it was invoked" beats no signal — but note
in the patch that it is an approximation so the governor's skill domain is not over-trusted.

### 2.8 Everything else

| Call | Site |
|---|---|
| `rec.ModeChange(from, to)` | `internal/agent/mode.go`, the `modeStore` set path |
| `rec.Critic(verdict, revisions)` | `coordinator.reviseWithCritic` |
| `rec.Todos(opened, closed)` | diff `session.Todos` between turn open and close |
| `rec.Queue(depth)` | `coordinator.QueuedPrompts(sessionID)` at turn open |
| `rec.Edit(path, added, removed)` / `rec.Revert(path, reason)` | edit tools; reverts from `crush.db` `files` version deltas within one turn |
| `rec.MCPCall(server, tool, ms, ok)` | `internal/agent/tools/mcp-tools.go` wrapper |
| `rec.LSPEvent(server, event, ok)` | `internal/lsp/manager.go` start/diagnostic paths |
| `rec.Recall(kind, count, budget)` | `coordinator.RecallDigests` / fact recall |
| `rec.Delegate(agent, task)` | `delegate_task` tool |
| `rec.Worktree(action, path)` | `internal/agent/worktree.go` |
| `rec.Interrupt(source)` | cancellation path |
| `rec.Prompt(text)` / `rec.Response(text)` | `UserPromptSubmit` equivalent and turn close |

## 3. Config

Add a `turnlog` block to `crush.json` for the persistent settings, but keep the kill switch in the
environment (`SPARKY_TURNLOG=0`) so telemetry can be disabled for one invocation without editing a
config file that other boxes have symlinked:

```json
"turnlog": {
  "enabled": true,
  "dir": "~/.local/state/sparky/turnlog",
  "retention_days": 14,
  "text_capture": "full",
  "queue_size": 4096
}
```

## 4. Turn off PostHog in the same change

`internal/cmd/root.go`'s `shouldEnableMetrics(cfg)` currently returns true on this install:
`disable_metrics` is unset, so usage events ship to `data.charm.land`. Set it:

```json
"options": { "disable_metrics": true }
```

Doing it in this change rather than separately is the point — flightdeck is the local replacement,
and running both means the fleet's usage data leaves the fleet for no benefit.

## 5. Rebuild the patch and cross-compile

```sh
cd <clone>
git add -A
git diff --cached > <your-config-repo>/crush-fork/crush-fork.patch

cd <your-config-repo>/crush-fork
./build.sh                                          # macOS host build
GOOS=linux GOARCH=arm64 ./build.sh /tmp/crush-arm64   # for your Linux inference host
```

`GOOS=linux` is required, not just `GOARCH=arm64` — omitting it produces a Mach-O binary that
fails on a Linux target with `Exec format error`.

## 6. Verify the wiring is live

Verification is deliberately behavioural, not textual. A symbol present in the binary only proves
it linked; it does not prove the wire point fires.

```sh
export SPARKY_TURNLOG_DIR=/tmp/turnlog-verify
sparky run "list the files in this directory"
python -m flightdeck doctor --dir /tmp/turnlog-verify
```

Expect: exactly one `turn` record, a nonzero `model_req` count, at least one `tool_call`, a
`context_peak` that is not null, and `dropped == 0`. Any of those being empty means the
corresponding wire point did not fire — which is the failure this project exists to make loud.

## 7. Latency acceptance check

Before and after the patch, run the same prompt ten times and compare median wall time. The
emitter is non-blocking by construction, but "by construction" is not a measurement. If the
delta exceeds a few milliseconds, the likely cause is `SyncEvery` set too low or text capture
running on bodies larger than `MaxTextLen`.
