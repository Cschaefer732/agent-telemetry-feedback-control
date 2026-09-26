package turnlog

// Record shapes. These mirror flightdeck/models.py exactly: the Python rollup replays this JSONL
// straight into sqlite, so a renamed field here silently drops a column there. The `_kind`
// discriminator and the field names are the contract — change them in both places or not at all.
//
// Optional fields are pointers with omitempty so an omitted key falls back to the Python
// dataclass default rather than writing a zero that reads as real data. A turn that reports
// prompt_tokens=0 because the provider was silent is a different fact from one that never
// reported at all, and the scorer treats them differently.

import "encoding/json"

const (
	kindTurn  = "turn"
	kindEvent = "event"
	kindText  = "text"
	kindProbe = "probe"
)

// probeRecord mirrors models.Probe. The emitter writes one on shutdown so its own drop count
// reaches the store. Without it, a queue that overflowed under load would be invisible and the
// KPIs computed from that period would read as complete when they were not.
type probeRecord struct {
	Kind string `json:"_kind"`

	TS     int64   `json:"ts"`
	Host   string  `json:"host"`
	Probe  string  `json:"kind"`
	OK     int     `json:"ok"`
	Total  int     `json:"total"`
	Detail *string `json:"detail,omitempty"`
}

// SourceCrush is the only source this emitter produces. The arch loop runs through crush and is
// distinguished by the SPARKY_TURNLOG_SOURCE override rather than by a separate code path.
const SourceCrush = "crush"

type turnRecord struct {
	Kind string `json:"_kind"`

	TurnID          string  `json:"turn_id"`
	SessionID       string  `json:"session_id"`
	ParentSessionID *string `json:"parent_session_id,omitempty"`
	Source          string  `json:"source"`
	Host            string  `json:"host"`
	CWD             *string `json:"cwd,omitempty"`
	GitSHA          *string `json:"git_sha,omitempty"`
	TurnIdx         *int    `json:"turn_idx,omitempty"`
	StartedAt       int64   `json:"started_at"`
	EndedAt         *int64  `json:"ended_at,omitempty"`
	WallMS          *int64  `json:"wall_ms,omitempty"`
	AgentName       *string `json:"agent_name,omitempty"`
	IsSubagent      int     `json:"is_subagent"`
	Mode            *string `json:"mode,omitempty"`
	Provider        *string `json:"provider,omitempty"`
	Model           *string `json:"model,omitempty"`
	Tier            *string `json:"tier,omitempty"`
	ModelMS         *int64  `json:"model_ms,omitempty"`
	Requests        *int    `json:"requests,omitempty"`
	Retries         *int    `json:"retries,omitempty"`
	PromptTokens    *int64  `json:"prompt_tokens,omitempty"`
	CompletionToken *int64  `json:"completion_tokens,omitempty"`
	CachedTokens    *int64  `json:"cached_tokens,omitempty"`
	Estimated       int     `json:"estimated"`
	ContextPeak     *int64  `json:"context_peak,omitempty"`
	ContextWindow   *int64  `json:"context_window,omitempty"`
	TTFTMS          *int64  `json:"ttft_ms,omitempty"`
	ToolsHashChange *int    `json:"tools_hash_changes,omitempty"`
	Outcome         *string `json:"outcome,omitempty"`
	FinishReason    *string `json:"finish_reason,omitempty"`
	ErrorClass      *string `json:"error_class,omitempty"`
}

type eventRecord struct {
	Kind string `json:"_kind"`

	TurnID     string  `json:"turn_id"`
	TS         int64   `json:"ts"`
	EventKind  string  `json:"kind"`
	Name       *string `json:"name,omitempty"`
	DurationMS *int64  `json:"duration_ms,omitempty"`
	OK         *int    `json:"ok,omitempty"`
	// Payload is a JSON *string*, not an object. models.Event.to_row() serializes it that way and
	// from_row json.loads() it back; emitting a bare object here would break the replay.
	Payload *string `json:"payload,omitempty"`
}

type textRecord struct {
	Kind string `json:"_kind"`

	TurnID    string `json:"turn_id"`
	TextKind  string `json:"kind"`
	Seq       int    `json:"seq"`
	Body      string `json:"body"`
	ExpiresAt int64  `json:"expires_at"`
}

// Event kinds, mirroring EVENT_KINDS in models.py.
const (
	EventToolCall = "tool_call"
	EventModelReq = "model_req"
	// EventCompaction is an actual compaction. EventContextSnapshot is a periodic reading of
	// window occupancy — reusing one kind for both made every observation look like a compaction
	// to the scorer.
	EventCompaction      = "compaction"
	EventContextSnapshot = "context_snapshot"
	EventModeChange      = "mode_change"
	EventSkillLoad       = "skill_load"
	EventSkillUse        = "skill_use"
	EventMCPCall         = "mcp_call"
	EventLSP             = "lsp_event"
	EventHook            = "hook"
	EventPermission      = "permission"
	EventRecall          = "recall"
	EventCritic          = "critic"
	EventQueue           = "queue"
	EventTodo            = "todo"
	EventEdit            = "edit"
	EventRevert          = "revert"
	EventInterrupt       = "interrupt"
	EventWorktree        = "worktree"
	EventDelegate        = "delegate"
)

// Text kinds, mirroring TEXT_KINDS in models.py.
const (
	TextPrompt          = "prompt"
	TextResponse        = "response"
	TextContextSnapshot = "context_snapshot"
	TextSummary         = "summary"
	TextToolArg         = "tool_arg"
	TextToolResult      = "tool_result"
)

func encodePayload(payload map[string]any) *string {
	if len(payload) == 0 {
		return nil
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		// A payload that will not marshal is dropped rather than failing the event. The event's
		// existence, timing and outcome are the load-bearing parts; the payload is detail.
		return nil
	}
	s := string(raw)
	return &s
}

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func intPtr(v int) *int     { return &v }
func i64Ptr(v int64) *int64 { return &v }
