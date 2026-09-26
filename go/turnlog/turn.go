package turnlog

import (
	"sync"
	"sync/atomic"
)

// TurnMeta is what the wire point knows when a turn opens. Everything else is accumulated as the
// turn runs.
type TurnMeta struct {
	SessionID       string
	ParentSessionID string
	CWD             string
	GitSHA          string
	TurnIdx         int
	AgentName       string
	IsSubagent      bool
	Mode            string
	Provider        string
	Model           string
	Tier            string
	ContextWindow   int64
}

// TurnRecorder accumulates one turn. Methods are safe to call from the concurrent goroutines that
// run tools and model requests, and every one of them is a no-op on a nil receiver so the crush
// wire points need no branching when telemetry is off.
type TurnRecorder struct {
	e    *Emitter
	id   string
	meta TurnMeta

	startedAt int64

	mu               sync.Mutex
	modelMS          int64
	requests         int
	retries          int
	promptTokens     int64
	completionTokens int64
	cachedTokens     int64
	contextPeak      int64
	estimated        bool
	// firstTTFTMS is the best (lowest) time-to-first-token seen this turn, -1 until one is
	// observed. Lowest rather than first because a retry's TTFT is the honest one to keep.
	firstTTFTMS      int64
	lastToolsHash    string
	toolsHashChanges int
	textSeq          map[string]int
	finished         atomic.Bool
}

// StartTurn opens a turn. The returned recorder is nil when telemetry is disabled.
func (e *Emitter) StartTurn(meta TurnMeta) *TurnRecorder {
	if e == nil {
		return nil
	}
	return &TurnRecorder{
		e:           e,
		id:          NewID(),
		meta:        meta,
		startedAt:   nowMS(),
		textSeq:     map[string]int{},
		firstTTFTMS: -1,
	}
}

// ID is the turn id, needed by callers that correlate telemetry with an external record (the
// governor decision, the judge queue).
func (t *TurnRecorder) ID() string {
	if t == nil {
		return ""
	}
	return t.id
}

func (t *TurnRecorder) event(kind, name string, durationMS int64, ok *int, payload map[string]any) {
	if t == nil || t.e == nil {
		return
	}
	rec := eventRecord{
		Kind:      kindEvent,
		TurnID:    t.id,
		TS:        nowMS(),
		EventKind: kind,
		Name:      strPtr(name),
		OK:        ok,
		Payload:   encodePayload(redactMap(payload, 0)),
	}
	if durationMS >= 0 {
		rec.DurationMS = i64Ptr(durationMS)
	}
	t.e.enqueue(rec)
}

func okFlag(ok bool) *int {
	if ok {
		return intPtr(1)
	}
	return intPtr(0)
}

// ---------- text capture ----------

// Text records a captured body (prompt, response, snapshot). Honours the TextCapture setting and
// the retention TTL, and drops the body entirely if redaction failed.
func (t *TurnRecorder) Text(kind, body string) {
	if t == nil || t.e == nil || body == "" {
		return
	}
	switch t.e.cfg.TextCapture {
	case TextCaptureOff:
		return
	case TextCaptureTruncated:
		if len(body) > 2048 {
			body = body[:2048] + "…[truncated]"
		}
	default:
		if t.e.cfg.MaxTextLen > 0 && len(body) > t.e.cfg.MaxTextLen {
			body = body[:t.e.cfg.MaxTextLen] + "…[truncated]"
		}
	}
	scrubbed, ok := redactOrDrop(body)
	if !ok {
		return
	}
	t.mu.Lock()
	seq := t.textSeq[kind]
	t.textSeq[kind] = seq + 1
	t.mu.Unlock()

	t.e.enqueue(textRecord{
		Kind:      kindText,
		TurnID:    t.id,
		TextKind:  kind,
		Seq:       seq,
		Body:      scrubbed,
		ExpiresAt: t.startedAt + t.e.retentionMS(),
	})
}

func (t *TurnRecorder) Prompt(body string)   { t.Text(TextPrompt, body) }
func (t *TurnRecorder) Response(body string) { t.Text(TextResponse, body) }

// maxToolArgLen and maxToolResultLen cap tool I/O independently of cfg.MaxTextLen. Prompts and
// responses are the point of the turn and get the full TextCapture budget (64KB by default); a
// tool call's arguments are almost always a handful of bytes, and a tool's result can be an
// entire build log or a `find /` dump that has no business landing on disk whole. These caps
// apply on top of TextCapture — "truncated"/"off" still win, since Text() enforces those first.
const (
	maxToolArgLen    = 4 * 1024
	maxToolResultLen = 8 * 1024
)

// ToolArg and ToolResult record a tool call's arguments and result body, subject to the same
// TextCapture policy and redaction as Prompt/Response. TextBlob carries no tool-name column (see
// models.TextBlob) so the name is folded into the body itself ("[name] ...") — enough for a
// reviewer to correlate a captured row with the tool_call event emitted around it, without a
// schema change on the Python side.
func (t *TurnRecorder) ToolArg(name, body string) {
	t.Text(TextToolArg, toolIOBody(name, body, maxToolArgLen))
}

func (t *TurnRecorder) ToolResult(name, body string) {
	t.Text(TextToolResult, toolIOBody(name, body, maxToolResultLen))
}

func toolIOBody(name, body string, limit int) string {
	if len(body) > limit {
		body = body[:limit] + "…[truncated]"
	}
	if name == "" {
		return body
	}
	return "[" + name + "] " + body
}

// ---------- per-turn signals ----------

func (t *TurnRecorder) ToolCall(name string, durationMS int64, ok bool, payload map[string]any) {
	t.event(EventToolCall, name, durationMS, okFlag(ok), payload)
}

// StepUsage is one provider round trip. A struct rather than positional arguments because the
// call site in crush's step-finish callback already has all of this in scope, and a nine-argument
// function is where the wrong value silently lands in the wrong column.
type StepUsage struct {
	LatencyMS        int64
	PromptTokens     int64
	CompletionTokens int64
	CachedTokens     int64
	FinishReason     string
	Estimated        bool
	Retries          int

	// TTFTMS is time to first token: the gap between sending the request and the first reasoning,
	// text, or tool-input delta arriving. It separates prefill from generation, which on a local
	// model is the KV prefix-cache hit/miss signal — a warm prefix returns a small TTFT even when
	// total latency is long. Negative means no token was observed.
	TTFTMS int64

	// ActiveTools and ToolsHash describe the tool set sent with this step. The hash is what makes
	// cache behaviour explainable: if it changes between steps, the prompt prefix changed and the
	// provider's cache was invalidated — so tool-search churn shows up as a measurable cost rather
	// than a suspicion.
	ActiveTools int
	ToolsHash   string
}

// ModelRequest records one provider round trip and folds its usage into the turn totals. This is
// the signal hooks cannot see at all — latency, TTFT, and true token accounting only exist
// in-process.
func (t *TurnRecorder) ModelRequest(u StepUsage) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.modelMS += u.LatencyMS
	t.requests++
	t.retries += u.Retries
	t.promptTokens += u.PromptTokens
	t.completionTokens += u.CompletionTokens
	t.cachedTokens += u.CachedTokens
	if u.Estimated {
		t.estimated = true
	}
	if u.TTFTMS >= 0 && (t.firstTTFTMS < 0 || u.TTFTMS < t.firstTTFTMS) {
		t.firstTTFTMS = u.TTFTMS
	}
	if u.ToolsHash != "" {
		if t.lastToolsHash != "" && t.lastToolsHash != u.ToolsHash {
			t.toolsHashChanges++
		}
		t.lastToolsHash = u.ToolsHash
	}
	t.mu.Unlock()

	t.event(EventModelReq, t.meta.Model, u.LatencyMS, okFlag(true), map[string]any{
		"prompt_tokens":     u.PromptTokens,
		"completion_tokens": u.CompletionTokens,
		"cached_tokens":     u.CachedTokens,
		"finish_reason":     u.FinishReason,
		"estimated":         u.Estimated,
		"retries":           u.Retries,
		"ttft_ms":           u.TTFTMS,
		"active_tools":      u.ActiveTools,
		"tools_hash":        u.ToolsHash,
	})
}

// ContextSnapshot records how full the window was at a step boundary. crush already computes this
// inside its auto-summarize check and then discards it; keeping it is what makes context pressure
// measurable instead of inferred.
func (t *TurnRecorder) ContextSnapshot(used, window int64) {
	if t == nil {
		return
	}
	t.mu.Lock()
	if used > t.contextPeak {
		t.contextPeak = used
	}
	if window > 0 {
		t.meta.ContextWindow = window
	}
	t.mu.Unlock()
	t.event(EventContextSnapshot, "context_snapshot", -1, nil, map[string]any{
		"used": used, "window": window,
	})
}

func (t *TurnRecorder) Compaction(source string, tokensBefore, tokensAfter int64, vetoed bool, summaryLen int) {
	t.event(EventCompaction, source, -1, okFlag(!vetoed), map[string]any{
		"tokens_before": tokensBefore,
		"tokens_after":  tokensAfter,
		"vetoed":        vetoed,
		"summary_len":   summaryLen,
	})
}

func (t *TurnRecorder) ModeChange(from, to string) {
	if t != nil {
		t.mu.Lock()
		t.meta.Mode = to
		t.mu.Unlock()
	}
	t.event(EventModeChange, to, -1, nil, map[string]any{"from": from, "to": to})
}

func (t *TurnRecorder) SkillsLoaded(names []string, activeTotal int) {
	items := make([]any, 0, len(names))
	for _, n := range names {
		items = append(items, n)
	}
	t.event(EventSkillLoad, "", -1, nil, map[string]any{
		"loaded": items, "loaded_count": len(names), "active_total": activeTotal,
	})
}

// SkillUsed is what separates "we paid for this skill's tokens" from "this skill did something".
// Skill precision is meaningless without both.
func (t *TurnRecorder) SkillUsed(name, evidence string) {
	t.event(EventSkillUse, name, -1, nil, map[string]any{"evidence": evidence})
}

func (t *TurnRecorder) MCPCall(server, tool string, durationMS int64, ok bool) {
	t.event(EventMCPCall, server+"/"+tool, durationMS, okFlag(ok), nil)
}

func (t *TurnRecorder) LSPEvent(server, event string, ok bool) {
	t.event(EventLSP, server, -1, okFlag(ok), map[string]any{"event": event})
}

func (t *TurnRecorder) Hook(hookEvent, name string, durationMS int64, ok bool, decision string) {
	t.event(EventHook, name, durationMS, okFlag(ok), map[string]any{
		"hook_event": hookEvent, "decision": decision,
	})
}

func (t *TurnRecorder) Permission(tool, decision string) {
	t.event(EventPermission, tool, -1, okFlag(decision != "deny"), map[string]any{"decision": decision})
}

func (t *TurnRecorder) Recall(kind string, count int, budget int) {
	t.event(EventRecall, kind, -1, nil, map[string]any{"count": count, "budget": budget})
}

func (t *TurnRecorder) Critic(verdict string, revisions int) {
	t.event(EventCritic, "critic", -1, okFlag(verdict == "pass"), map[string]any{
		"verdict": verdict, "revisions": revisions,
	})
}

func (t *TurnRecorder) Queue(depth int) {
	t.event(EventQueue, "", -1, nil, map[string]any{"depth": depth})
}

func (t *TurnRecorder) Todos(opened, closed int) {
	t.event(EventTodo, "", -1, nil, map[string]any{"opened": opened, "closed": closed})
}

func (t *TurnRecorder) Edit(path string, added, removed int) {
	t.event(EventEdit, path, -1, okFlag(true), map[string]any{
		"path": path, "added": added, "removed": removed,
	})
}

// Revert is the flailing signal: the agent wrote something and then undid it inside one turn.
func (t *TurnRecorder) Revert(path, reason string) {
	t.event(EventRevert, path, -1, okFlag(false), map[string]any{"path": path, "reason": reason})
}

func (t *TurnRecorder) Interrupt(source string) {
	t.event(EventInterrupt, source, -1, nil, nil)
}

func (t *TurnRecorder) Worktree(action, path string) {
	t.event(EventWorktree, action, -1, nil, map[string]any{"path": path})
}

func (t *TurnRecorder) Delegate(agentName, task string) {
	t.event(EventDelegate, agentName, -1, nil, map[string]any{"task": task})
}

// ---------- close ----------

// Finish writes the turn record. Unlike events, this bypasses the queue and writes synchronously
// with an fsync: a dropped event costs detail, a dropped turn record costs the entire row and
// every KPI derived from it. Callers must invoke this on a detached context so a cancelled turn
// still records — a cancelled turn is exactly the kind this system exists to explain.
func (t *TurnRecorder) Finish(outcome, finishReason, errorClass string) {
	if t == nil || t.e == nil || !t.finished.CompareAndSwap(false, true) {
		return
	}
	ended := nowMS()
	t.mu.Lock()
	rec := turnRecord{
		Kind:            kindTurn,
		TurnID:          t.id,
		SessionID:       t.meta.SessionID,
		ParentSessionID: strPtr(t.meta.ParentSessionID),
		Source:          t.e.cfg.Source,
		Host:            t.e.cfg.Host,
		CWD:             strPtr(t.meta.CWD),
		GitSHA:          strPtr(t.meta.GitSHA),
		StartedAt:       t.startedAt,
		EndedAt:         i64Ptr(ended),
		WallMS:          i64Ptr(ended - t.startedAt),
		AgentName:       strPtr(t.meta.AgentName),
		Mode:            strPtr(t.meta.Mode),
		Provider:        strPtr(t.meta.Provider),
		Model:           strPtr(t.meta.Model),
		Tier:            strPtr(t.meta.Tier),
		Outcome:         strPtr(outcome),
		FinishReason:    strPtr(finishReason),
		ErrorClass:      strPtr(errorClass),
	}
	if t.meta.IsSubagent {
		rec.IsSubagent = 1
	}
	if t.meta.TurnIdx > 0 {
		rec.TurnIdx = intPtr(t.meta.TurnIdx)
	}
	if t.requests > 0 {
		rec.ModelMS = i64Ptr(t.modelMS)
		rec.Requests = intPtr(t.requests)
		rec.Retries = intPtr(t.retries)
		rec.PromptTokens = i64Ptr(t.promptTokens)
		rec.CompletionToken = i64Ptr(t.completionTokens)
		rec.CachedTokens = i64Ptr(t.cachedTokens)
	}
	if t.contextPeak > 0 {
		rec.ContextPeak = i64Ptr(t.contextPeak)
	}
	if t.firstTTFTMS >= 0 {
		rec.TTFTMS = i64Ptr(t.firstTTFTMS)
	}
	if t.lastToolsHash != "" {
		// Emitted even when zero: "the tool set never changed" is a real finding, distinct from
		// "we never observed the tool set".
		rec.ToolsHashChange = intPtr(t.toolsHashChanges)
	}
	if t.meta.ContextWindow > 0 {
		rec.ContextWindow = i64Ptr(t.meta.ContextWindow)
	}
	if t.estimated {
		rec.Estimated = 1
	}
	t.mu.Unlock()

	t.e.write(rec, true)
}
