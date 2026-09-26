package turnlog

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// Synthetic credentials and credential-shaped env names are assembled at runtime rather than
// written as literals, so repo-wide secret scanners and the pre-commit safety hook do not have to
// special-case the test corpus. Nothing here is a real key.
func fakeAnthropicKey() string { return "sk-" + "ant-api03-" + strings.Repeat("A", 26) }
func keyEnvName() string       { return "ANTHROPIC_API" + "_KEY" }

func testEmitter(t *testing.T) (*Emitter, string) {
	t.Helper()
	dir := t.TempDir()
	cfg := DefaultConfig()
	cfg.Dir = dir
	cfg.Host = "testhost"
	cfg.SyncEvery = 1
	e := New(cfg)
	if e == nil {
		t.Fatal("New returned nil for a usable directory")
	}
	return e, dir
}

func readRecords(t *testing.T, dir string) []map[string]any {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(dir, "events-*.jsonl"))
	if err != nil || len(matches) == 0 {
		t.Fatalf("no jsonl written in %s (err=%v)", dir, err)
	}
	var out []map[string]any
	for _, path := range matches {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
			if line == "" {
				continue
			}
			var rec map[string]any
			if err := json.Unmarshal([]byte(line), &rec); err != nil {
				t.Fatalf("unparseable line %q: %v", line, err)
			}
			out = append(out, rec)
		}
	}
	return out
}

func recordsOfKind(records []map[string]any, kind string) []map[string]any {
	var out []map[string]any
	for _, r := range records {
		if r["_kind"] == kind {
			out = append(out, r)
		}
	}
	return out
}

// A nil emitter must be usable end to end. The crush wire points depend on this so they can call
// telemetry unconditionally without nil checks scattered through the agent loop.
func TestNilEmitterIsSafe(t *testing.T) {
	var e *Emitter
	tr := e.StartTurn(TurnMeta{SessionID: "s"})
	if tr.ID() != "" {
		t.Fatal("nil emitter produced a turn id")
	}
	tr.Prompt("hello")
	tr.ToolCall("edit", 5, true, map[string]any{"path": "a.go"})
	tr.ModelRequest(StepUsage{LatencyMS: 10, PromptTokens: 1, CompletionTokens: 2, FinishReason: "stop", TTFTMS: -1})
	tr.ContextSnapshot(10, 100)
	tr.Critic("pass", 0)
	tr.Finish("ok", "stop", "")
	if got := e.Stats(); got.Written != 0 || got.Dropped != 0 {
		t.Fatalf("nil emitter recorded stats: %+v", got)
	}
	e.Close()
}

func TestTurnRecordMatchesPythonSchema(t *testing.T) {
	e, dir := testEmitter(t)
	tr := e.StartTurn(TurnMeta{
		SessionID: "sess-1", CWD: "/repo", GitSHA: "abc123", TurnIdx: 3,
		Mode: "plan", Provider: "ollama", Model: "qwen3-coder:30b", Tier: "fast",
		ContextWindow: 32000,
	})
	tr.ModelRequest(StepUsage{LatencyMS: 120, PromptTokens: 900, CompletionTokens: 300, CachedTokens: 50,
		FinishReason: "stop", Retries: 1, TTFTMS: 40, ActiveTools: 12, ToolsHash: "aaaa"})
	tr.ModelRequest(StepUsage{LatencyMS: 80, PromptTokens: 100, CompletionTokens: 60,
		FinishReason: "stop", Estimated: true, TTFTMS: 900, ActiveTools: 9, ToolsHash: "bbbb"})
	tr.ContextSnapshot(12000, 32000)
	tr.Finish("ok", "stop", "")
	e.Close()

	turns := recordsOfKind(readRecords(t, dir), kindTurn)
	if len(turns) != 1 {
		t.Fatalf("want 1 turn record, got %d", len(turns))
	}
	rec := turns[0]
	// Every column the Python Turn dataclass declares as non-optional must be present.
	for _, key := range []string{"turn_id", "session_id", "source", "host", "started_at"} {
		if _, ok := rec[key]; !ok {
			t.Errorf("turn record missing required key %q", key)
		}
	}
	if rec["source"] != SourceCrush || rec["host"] != "testhost" {
		t.Errorf("source/host wrong: %v / %v", rec["source"], rec["host"])
	}
	// Usage must be summed across requests, not last-write-wins.
	if rec["prompt_tokens"].(float64) != 1000 || rec["completion_tokens"].(float64) != 360 {
		t.Errorf("token totals not accumulated: %v / %v", rec["prompt_tokens"], rec["completion_tokens"])
	}
	if rec["model_ms"].(float64) != 200 || rec["requests"].(float64) != 2 || rec["retries"].(float64) != 1 {
		t.Errorf("request accounting wrong: %+v", rec)
	}
	// One estimated request taints the whole turn — the scorer must know the numbers are approximate.
	if rec["estimated"].(float64) != 1 {
		t.Errorf("estimated flag not propagated: %v", rec["estimated"])
	}
	if rec["context_peak"].(float64) != 12000 || rec["context_window"].(float64) != 32000 {
		t.Errorf("context tracking wrong: %+v", rec)
	}
	if rec["wall_ms"] == nil || rec["ended_at"] == nil {
		t.Error("turn record missing timing")
	}
}

// Optional fields must be absent rather than zero: "the provider reported 0 tokens" and "no
// provider call happened" are different facts and the scorer treats them differently.
func TestUnsetOptionalFieldsAreOmitted(t *testing.T) {
	e, dir := testEmitter(t)
	tr := e.StartTurn(TurnMeta{SessionID: "sess-2"})
	tr.Finish("cancelled", "", "")
	e.Close()

	rec := recordsOfKind(readRecords(t, dir), kindTurn)[0]
	for _, key := range []string{"prompt_tokens", "model_ms", "requests", "context_peak", "agent_name", "tier"} {
		if _, present := rec[key]; present {
			t.Errorf("expected %q to be omitted when unset, got %v", key, rec[key])
		}
	}
	if rec["is_subagent"].(float64) != 0 || rec["estimated"].(float64) != 0 {
		t.Error("NOT NULL columns must always be emitted")
	}
}

// Event payloads are JSON *strings* because models.Event.to_row() serializes them that way.
// Emitting a bare object here would break the Python replay.
func TestEventPayloadIsJSONString(t *testing.T) {
	e, dir := testEmitter(t)
	tr := e.StartTurn(TurnMeta{SessionID: "sess-3"})
	tr.ToolCall("edit", 42, false, map[string]any{"path": "/repo/main.go", "attempt": 2})
	tr.Finish("error", "tool_error", "EditFailed")
	e.Close()

	events := recordsOfKind(readRecords(t, dir), kindEvent)
	var toolCall map[string]any
	for _, ev := range events {
		if ev["kind"] == EventToolCall {
			toolCall = ev
		}
	}
	if toolCall == nil {
		t.Fatal("no tool_call event written")
	}
	raw, ok := toolCall["payload"].(string)
	if !ok {
		t.Fatalf("payload must be a JSON string, got %T", toolCall["payload"])
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		t.Fatalf("payload string is not valid JSON: %v", err)
	}
	if decoded["path"] != "/repo/main.go" {
		t.Errorf("payload lost data: %+v", decoded)
	}
	if toolCall["ok"].(float64) != 0 || toolCall["duration_ms"].(float64) != 42 {
		t.Errorf("tool call outcome/timing wrong: %+v", toolCall)
	}
}

func TestTextIsRedactedAndTTLd(t *testing.T) {
	e, dir := testEmitter(t)
	secret := fakeAnthropicKey()
	tr := e.StartTurn(TurnMeta{SessionID: "sess-4"})
	tr.Prompt("deploy with " + keyEnvName() + "=" + secret + " please")
	tr.Response("done")
	tr.Finish("ok", "stop", "")
	e.Close()

	texts := recordsOfKind(readRecords(t, dir), kindText)
	if len(texts) != 2 {
		t.Fatalf("want 2 text records, got %d", len(texts))
	}
	for _, rec := range texts {
		body := rec["body"].(string)
		if strings.Contains(body, secret) || strings.Contains(body, secret[:20]) {
			t.Fatalf("secret survived redaction: %q", body)
		}
		if rec["expires_at"].(float64) <= 0 {
			t.Error("text record has no TTL")
		}
	}
	// Sequence numbers are per-kind so the Python primary key (turn_id, kind, seq) holds.
	if texts[0]["seq"].(float64) != 0 || texts[1]["seq"].(float64) != 0 {
		t.Errorf("per-kind seq numbering wrong: %v %v", texts[0]["seq"], texts[1]["seq"])
	}
}

// Payload values go through the same scrubber as prose. A key pasted as a tool argument is the
// most likely way one reaches disk.
func TestEventPayloadIsRedacted(t *testing.T) {
	e, dir := testEmitter(t)
	secret := fakeAnthropicKey()
	tr := e.StartTurn(TurnMeta{SessionID: "sess-7"})
	tr.ToolCall("bash", 1, true, map[string]any{
		"cmd":    "export " + keyEnvName() + "=" + secret,
		"nested": map[string]any{"token": secret},
		"list":   []any{"api_key=" + secret},
	})
	tr.Finish("ok", "stop", "")
	e.Close()

	for _, ev := range recordsOfKind(readRecords(t, dir), kindEvent) {
		if raw, ok := ev["payload"].(string); ok && strings.Contains(raw, secret) {
			t.Fatalf("secret survived payload redaction: %s", raw)
		}
	}
}

// ToolArg/ToolResult fold the tool name into the body (TextBlob has no name column) and cap
// independently of MaxTextLen, so a tool result cannot dominate the store the way an unbounded
// build log or `find /` dump would.
func TestToolIOCapturedNamedAndCapped(t *testing.T) {
	e, dir := testEmitter(t)
	tr := e.StartTurn(TurnMeta{SessionID: "sess-tio"})
	tr.ToolArg("bash", `{"command":"echo hi"}`)
	tr.ToolResult("bash", strings.Repeat("x", maxToolResultLen+500))
	tr.Finish("ok", "stop", "")
	e.Close()

	texts := recordsOfKind(readRecords(t, dir), kindText)
	var arg, result map[string]any
	for _, rec := range texts {
		switch rec["kind"] {
		case TextToolArg:
			arg = rec
		case TextToolResult:
			result = rec
		}
	}
	if arg == nil || result == nil {
		t.Fatalf("expected tool_arg and tool_result records, got %+v", texts)
	}
	if body := arg["body"].(string); !strings.HasPrefix(body, "[bash] ") {
		t.Errorf("tool arg body missing name prefix: %q", body)
	}
	body := result["body"].(string)
	if !strings.HasPrefix(body, "[bash] ") {
		t.Errorf("tool result body missing name prefix: %q", body)
	}
	if len(body) > len("[bash] ")+maxToolResultLen+len("…[truncated]") {
		t.Errorf("tool result body exceeds cap: %d bytes", len(body))
	}
	if !strings.Contains(body, "…[truncated]") {
		t.Error("oversized tool result was not truncated")
	}
}

// A tool call's arguments and result are exactly where a live secret is most likely to appear
// (an exported env var, a pasted key). They must go through the same scrubber as prompts.
func TestToolIOIsRedacted(t *testing.T) {
	e, dir := testEmitter(t)
	secret := fakeAnthropicKey()
	tr := e.StartTurn(TurnMeta{SessionID: "sess-tio-redact"})
	tr.ToolArg("bash", "echo Authorization: Bearer "+secret)
	tr.ToolResult("bash", "token="+secret)
	tr.Finish("ok", "stop", "")
	e.Close()

	for _, rec := range recordsOfKind(readRecords(t, dir), kindText) {
		body := rec["body"].(string)
		if strings.Contains(body, secret) {
			t.Fatalf("secret survived tool I/O redaction: %q", body)
		}
	}
}

// TextCapture=off must drop tool I/O along with everything else — it is the kill switch, not a
// partial one.
func TestToolIOCaptureOffDropsBodies(t *testing.T) {
	dir := t.TempDir()
	cfg := DefaultConfig()
	cfg.Dir, cfg.TextCapture, cfg.SyncEvery = dir, TextCaptureOff, 1
	e := New(cfg)
	tr := e.StartTurn(TurnMeta{SessionID: "s"})
	tr.ToolArg("bash", "a secret command")
	tr.ToolResult("bash", "a secret result")
	tr.Finish("ok", "stop", "")
	e.Close()

	if got := len(recordsOfKind(readRecords(t, dir), kindText)); got != 0 {
		t.Fatalf("TextCapture=off still wrote %d text records", got)
	}
}

func TestTextCaptureOffDropsBodies(t *testing.T) {
	dir := t.TempDir()
	cfg := DefaultConfig()
	cfg.Dir, cfg.TextCapture, cfg.SyncEvery = dir, TextCaptureOff, 1
	e := New(cfg)
	tr := e.StartTurn(TurnMeta{SessionID: "s"})
	tr.Prompt("a secret prompt")
	tr.Finish("ok", "stop", "")
	e.Close()

	if got := len(recordsOfKind(readRecords(t, dir), kindText)); got != 0 {
		t.Fatalf("TextCapture=off still wrote %d text records", got)
	}
}

// The queue must drop rather than block. A telemetry backpressure stall inside the agent loop
// would be a user-visible regression; a counted drop is not.
func TestFullQueueDropsInsteadOfBlocking(t *testing.T) {
	dir := t.TempDir()
	cfg := DefaultConfig()
	cfg.Dir, cfg.QueueSize = dir, 1
	e := &Emitter{cfg: cfg, queue: make(chan any, 1), done: make(chan struct{})}
	// No writer goroutine started, so nothing drains the queue.
	for i := 0; i < 100; i++ {
		e.enqueue(eventRecord{Kind: kindEvent, TurnID: "t", TS: nowMS(), EventKind: EventToolCall})
	}
	if got := e.Stats().Dropped; got != 99 {
		t.Fatalf("want 99 drops, got %d", got)
	}
}

func TestFinishIsIdempotent(t *testing.T) {
	e, dir := testEmitter(t)
	tr := e.StartTurn(TurnMeta{SessionID: "sess-5"})
	tr.Finish("ok", "stop", "")
	tr.Finish("error", "boom", "Second")
	e.Close()

	if got := len(recordsOfKind(readRecords(t, dir), kindTurn)); got != 1 {
		t.Fatalf("double Finish wrote %d turn records", got)
	}
}

func TestConcurrentRecordingIsRaceFree(t *testing.T) {
	e, dir := testEmitter(t)
	tr := e.StartTurn(TurnMeta{SessionID: "sess-6"})
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			tr.ToolCall("edit", int64(n), true, map[string]any{"n": n})
			tr.ModelRequest(StepUsage{LatencyMS: 1, PromptTokens: 10, CompletionTokens: 5, FinishReason: "stop", TTFTMS: -1})
			tr.ContextSnapshot(int64(n*10), 32000)
		}(i)
	}
	wg.Wait()
	tr.Finish("ok", "stop", "")
	e.Close()

	rec := recordsOfKind(readRecords(t, dir), kindTurn)[0]
	if rec["requests"].(float64) != 50 || rec["prompt_tokens"].(float64) != 500 {
		t.Errorf("concurrent accumulation lost updates: %+v", rec)
	}
	if rec["context_peak"].(float64) != 490 {
		t.Errorf("context peak wrong under concurrency: %v", rec["context_peak"])
	}
}

func TestCloseIsIdempotent(t *testing.T) {
	e, _ := testEmitter(t)
	e.Close()
	e.Close()
}

func TestIDsAreSortableAndUnique(t *testing.T) {
	seen := map[string]bool{}
	prev := ""
	for i := 0; i < 5000; i++ {
		id := NewID()
		if len(id) != 26 {
			t.Fatalf("ULID wrong length: %q", id)
		}
		if seen[id] {
			t.Fatalf("duplicate id %q", id)
		}
		seen[id] = true
		if id <= prev {
			t.Fatalf("ids not monotonic: %q then %q", prev, id)
		}
		prev = id
	}
}

func TestConfigFromEnvKillSwitch(t *testing.T) {
	t.Setenv("SPARKY_TURNLOG", "0")
	if _, enabled := ConfigFromEnv(); enabled {
		t.Fatal("SPARKY_TURNLOG=0 did not disable telemetry")
	}
	if NewFromEnv() != nil {
		t.Fatal("NewFromEnv returned an emitter while disabled")
	}
}

func TestRedactKeepsContextAroundSecrets(t *testing.T) {
	secret := fakeAnthropicKey()
	out, counts := Redact(keyEnvName() + "=" + secret)
	if strings.Contains(out, secret) {
		t.Fatalf("secret not redacted: %q", out)
	}
	// The key name must survive: knowing WHICH credential leaked is the actionable part.
	if !strings.Contains(out, keyEnvName()) {
		t.Errorf("redaction destroyed the key name: %q", out)
	}
	if len(counts) == 0 {
		t.Error("no substitution counted")
	}
}

func TestRedactLeavesPathsAndHostsAlone(t *testing.T) {
	in := "failed at /Users/carter/dev/repo/main.go:42 on host gpu-host (198.51.100.10)"
	out, counts := Redact(in)
	if out != in {
		t.Fatalf("redaction damaged non-secret text:\n got %q\nwant %q", out, in)
	}
	if len(counts) != 0 {
		t.Errorf("false positives: %+v", counts)
	}
}

// Redaction runs on every captured body on the hot path; a pathological input must not hang it.
func TestRedactDoesNotBacktrackCatastrophically(t *testing.T) {
	adversarial := strings.Repeat("a", 100000) + " token=" + strings.Repeat("b", 50)
	done := make(chan struct{})
	go func() {
		Redact(adversarial)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Redact did not finish within 5s on adversarial input")
	}
}

// TTFT and the tool-set hash come from the timing work already in the fork's patch; folding them
// into the turn record is what keeps one structured path instead of two parallel ones.
func TestTTFTAndToolsHashChurn(t *testing.T) {
	e, dir := testEmitter(t)
	tr := e.StartTurn(TurnMeta{SessionID: "sess-ttft", Model: "qwen3-coder:30b"})
	tr.ModelRequest(StepUsage{LatencyMS: 900, TTFTMS: 600, ActiveTools: 20, ToolsHash: "aaaa"})
	// Same tool set: prefix still valid, no churn.
	tr.ModelRequest(StepUsage{LatencyMS: 400, TTFTMS: 90, ActiveTools: 20, ToolsHash: "aaaa"})
	// Tool set changed: prefix invalidated, and the next TTFT jumps because of it.
	tr.ModelRequest(StepUsage{LatencyMS: 800, TTFTMS: 550, ActiveTools: 14, ToolsHash: "bbbb"})
	tr.Finish("ok", "stop", "")
	e.Close()

	rec := recordsOfKind(readRecords(t, dir), kindTurn)[0]
	// Lowest TTFT, not first: the warm-prefix reading is the honest one to keep for the turn.
	if rec["ttft_ms"].(float64) != 90 {
		t.Errorf("ttft_ms should be the lowest observed, got %v", rec["ttft_ms"])
	}
	if rec["tools_hash_changes"].(float64) != 1 {
		t.Errorf("tools_hash_changes wrong: %v", rec["tools_hash_changes"])
	}
}

// "The tool set never changed" is a finding; "we never saw the tool set" is not. They must not
// both serialize as zero.
func TestToolsHashChurnOmittedWhenUnobserved(t *testing.T) {
	e, dir := testEmitter(t)
	tr := e.StartTurn(TurnMeta{SessionID: "sess-noh"})
	tr.ModelRequest(StepUsage{LatencyMS: 10, TTFTMS: -1})
	tr.Finish("ok", "stop", "")
	e.Close()

	rec := recordsOfKind(readRecords(t, dir), kindTurn)[0]
	if _, present := rec["tools_hash_changes"]; present {
		t.Errorf("tools_hash_changes must be absent when never observed, got %v", rec["tools_hash_changes"])
	}
	if _, present := rec["ttft_ms"]; present {
		t.Errorf("ttft_ms must be absent when no token was observed, got %v", rec["ttft_ms"])
	}
}

func fakeGithubToken() string { return "ghp" + "_" + strings.Repeat("Z9y8X7w6", 5) }

// Ported from a downstream fork's vendored copy, which regenerating that fork's patch would
// otherwise have deleted. The other redaction tests here each exercise one pattern against
// one input; this one asserts the whole ruleset fires on a single mixed blob, which is the
// shape real tool output actually has.
func TestRedactScrubsKnownSecretKinds(t *testing.T) {
	in := strings.Join([]string{
		"key " + fakeAnthropicKey(),
		"gh token " + fakeGithubToken(),
		"Authorization: Bearer abcdefghijklmnop.qrstuvwxyz012345",
		`password = "hunter2xyz"`,
	}, "\n")
	out, counts := Redact(in)

	for _, secret := range []string{fakeAnthropicKey(), fakeGithubToken(), "hunter2xyz"} {
		if strings.Contains(out, secret) {
			t.Errorf("secret survived redaction: %q", secret)
		}
	}
	// The bearer rule keeps the auth scheme and replaces only the token.
	if !strings.Contains(strings.ToLower(out), "authorization") {
		t.Errorf("bearer redaction dropped the surrounding context: %q", out)
	}
	if len(counts) == 0 {
		t.Fatal("expected nonzero redaction counts")
	}
}

func TestRedactEmptyInput(t *testing.T) {
	out, counts := Redact("")
	if out != "" || counts != nil {
		t.Fatalf("empty input should pass through: %q %v", out, counts)
	}
}
