// Command emit-fixture writes one fully-populated turn to a turnlog directory.
//
// It exists so the Python test suite can prove the cross-language contract on real output rather
// than on a hand-written fixture that would drift from the emitter the moment either side changed.
package main

import (
	"fmt"
	"os"

	"github.com/Cschaefer732/closed-loop-agent-tuning/go/turnlog"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: emit-fixture <dir>")
		os.Exit(2)
	}
	cfg := turnlog.DefaultConfig()
	cfg.Dir = os.Args[1]
	cfg.Host = "fixture-host"
	cfg.SyncEvery = 1
	e := turnlog.New(cfg)
	if e == nil {
		fmt.Fprintln(os.Stderr, "emitter did not start")
		os.Exit(1)
	}

	tr := e.StartTurn(turnlog.TurnMeta{
		SessionID:       "fixture-session",
		ParentSessionID: "fixture-parent",
		CWD:             "/repo",
		GitSHA:          "deadbeef",
		TurnIdx:         2,
		AgentName:       "code-reviewer",
		IsSubagent:      true,
		Mode:            "plan",
		Provider:        "ollama",
		Model:           "qwen3-coder:30b",
		Tier:            "fast",
		ContextWindow:   32000,
	})
	tr.Prompt("refactor the parser")
	tr.SkillsLoaded([]string{"systematic-debugging", "find-docs"}, 12)
	tr.SkillUsed("systematic-debugging", "tool:grep")
	tr.ToolCall("edit", 12, true, map[string]any{"path": "/repo/parser.go"})
	tr.ToolCall("bash", 900, false, map[string]any{"cmd": "go test ./..."})
	tr.ModelRequest(turnlog.StepUsage{
		LatencyMS: 1200, PromptTokens: 8000, CompletionTokens: 700, CachedTokens: 200,
		FinishReason: "stop", Retries: 1, TTFTMS: 85, ActiveTools: 14, ToolsHash: "0f1e2d3c4b5a6978",
	})
	tr.ContextSnapshot(21000, 32000)
	tr.Compaction("auto", 21000, 6000, false, 1400)
	tr.ModeChange("plan", "default")
	tr.Todos(4, 3)
	tr.Edit("/repo/parser.go", 20, 4)
	tr.Revert("/repo/parser.go", "tests failed")
	tr.Permission("bash", "deny")
	tr.Critic("fail", 1)
	tr.MCPCall("context7", "docs", 300, true)
	tr.LSPEvent("gopls", "diagnostics", true)
	tr.Hook("PreToolUse", "block-secrets", 4, true, "allow")
	tr.Recall("digest", 3, 2000)
	tr.Delegate("code-reviewer", "review the diff")
	tr.Worktree("created", "/tmp/wt")
	tr.Queue(2)
	tr.Interrupt("user")
	tr.Response("done")
	tr.Finish("error", "tool_error", "TestFailure")

	e.Close()

	stats := e.Stats()
	if stats.Dropped != 0 || stats.WriteErrs != 0 {
		fmt.Fprintf(os.Stderr, "emitter lost records: %+v\n", stats)
		os.Exit(1)
	}
	fmt.Printf("%s\n", tr.ID())
}
