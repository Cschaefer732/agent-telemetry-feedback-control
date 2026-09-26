package turnlog

import (
	"regexp"
	"strings"
)

// Redaction runs at write time, in-process, before anything touches disk. Scrubbing later — in the
// rollup, or in the nightly job — would mean the raw secret existed on disk in between, on a box
// whose logs get rsynced to another machine and read by an agent.
//
// This mirrors flightdeck/redact.py. The two implementations are deliberately duplicated rather
// than shared: the Go side must have zero dependencies to vendor cleanly into the crush fork.
// Where they disagree the Python side is authoritative, because it also scrubs on ingest.

type redactPattern struct {
	kind string
	re   *regexp.Regexp
	// group is the submatch index to replace. 0 replaces the whole match; a positive index keeps
	// the surrounding context (the key name, the auth scheme) and replaces only the secret.
	group int
}

var redactPatterns = []redactPattern{
	// Order is specific-to-general. A JWT inside an Authorization header must be labeled once, by
	// the bearer rule, not substituted twice.
	{kind: "pem", re: regexp.MustCompile(`(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----`), group: 0},
	{kind: "bearer", re: regexp.MustCompile(`(?i)\b(authorization\s*:\s*|bearer\s+)([A-Za-z0-9\-._~+/]{12,}={0,2})`), group: 2},
	{kind: "anthropic_key", re: regexp.MustCompile(`sk-ant-[A-Za-z0-9\-_]{20,}`), group: 0},
	{kind: "openai_key", re: regexp.MustCompile(`sk-(?:proj-)?[A-Za-z0-9\-_]{20,}`), group: 0},
	{kind: "github_token", re: regexp.MustCompile(`gh[pousr]_[A-Za-z0-9]{36,}`), group: 0},
	{kind: "aws_key", re: regexp.MustCompile(`\b(?:AKIA|ASIA)[0-9A-Z]{16}\b`), group: 0},
	{kind: "slack_token", re: regexp.MustCompile(`xox[baprs]-[A-Za-z0-9\-]{10,}`), group: 0},
	{kind: "jwt", re: regexp.MustCompile(`\beyJ[A-Za-z0-9\-_]{6,}\.[A-Za-z0-9\-_]{6,}\.[A-Za-z0-9\-_]{6,}`), group: 0},
	{kind: "url_userinfo", re: regexp.MustCompile(`([a-zA-Z][a-zA-Z0-9+.\-]*://)([^/\s:@]+:[^/\s@]+)@`), group: 2},
	{kind: "kv_secret", re: regexp.MustCompile(`(?i)\b(pass(?:word|wd)?|secret|token|api[_-]?key|auth|credential)\b(\s*[:=]\s*"?)([^\s"',;)]{4,})`), group: 3},
}

// Redact returns the scrubbed text and a count of substitutions by pattern kind.
func Redact(text string) (string, map[string]int) {
	if text == "" {
		return text, nil
	}
	counts := map[string]int{}
	out := text
	for _, p := range redactPatterns {
		pattern := p
		out = pattern.re.ReplaceAllStringFunc(out, func(match string) string {
			groups := pattern.re.FindStringSubmatch(match)
			if groups == nil {
				return match
			}
			counts[pattern.kind]++
			placeholder := "[REDACTED:" + pattern.kind + "]"
			if pattern.group == 0 || pattern.group >= len(groups) {
				return placeholder
			}
			// Keep everything except the secret submatch, so the reviewer still sees which key
			// leaked without seeing its value.
			secret := groups[pattern.group]
			idx := strings.LastIndex(match, secret)
			if idx < 0 {
				return placeholder
			}
			return match[:idx] + placeholder + match[idx+len(secret):]
		})
	}
	if len(counts) == 0 {
		return out, nil
	}
	return out, counts
}

// redactOrDrop fails closed: if redaction panics for any reason, the text is discarded rather
// than written raw. There is no third option here worth having.
func redactOrDrop(text string) (result string, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			result, ok = "", false
		}
	}()
	scrubbed, _ := Redact(text)
	return scrubbed, true
}

// redactMap scrubs string values in an event payload. Depth-limited so a pathological structure
// cannot recurse without bound on the hot path.
func redactMap(payload map[string]any, depth int) map[string]any {
	if payload == nil || depth > 8 {
		return payload
	}
	out := make(map[string]any, len(payload))
	for k, v := range payload {
		switch typed := v.(type) {
		case string:
			if scrubbed, ok := redactOrDrop(typed); ok {
				out[k] = scrubbed
			}
		case map[string]any:
			out[k] = redactMap(typed, depth+1)
		case []any:
			items := make([]any, 0, len(typed))
			for _, item := range typed {
				switch inner := item.(type) {
				case string:
					if scrubbed, ok := redactOrDrop(inner); ok {
						items = append(items, scrubbed)
					}
				case map[string]any:
					items = append(items, redactMap(inner, depth+1))
				default:
					items = append(items, item)
				}
			}
			out[k] = items
		default:
			out[k] = v
		}
	}
	return out
}
