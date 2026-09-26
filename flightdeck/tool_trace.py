"""Measure how Claude Code actually uses tools, straight from the transcript JSONL.

WHY THIS EXISTS: the user wants "batched tools" and "preemptive tool-calling". A
speculative-execution engine already exists for the opencode/ollama wire, but it cannot apply to
Claude Code -- Claude Code talks to the Anthropic API directly, there is no interceptable wire to
sit in front of. The only honest contribution on this side is MEASUREMENT: quantify what batching
or prefetching would actually have saved, from real transcripts, so any future harness change is
justified by data instead of assumption.

Reuses `ingest_transcript`'s hard-won parsing facts rather than re-deriving them:
  - The transcript FILENAME (`<session_id>.jsonl`) is the join key, not any inner `session_id`
    field (a resumed session can carry a stale one). Irrelevant to per-file analysis here, but
    the same file discovery (`ingest_transcript.find_transcript` / `DEFAULT_ROOT`) is reused so
    this module looks in the same place.
  - One assistant API response is split into one JSONL row PER CONTENT BLOCK (thinking / tool_use
    / text), all sharing the same `message.id`. Counting rows instead of grouping by `message.id`
    overcounts "tool calls" and "assistant turns" by the block-count factor. Every count here is
    grouped by `message.id`.
  - A tool_use block's result is a `tool_result` content block in a later `user`-role message,
    matched by `tool_use_id`. A tool_use with no matching tool_result (interrupted mid-call,
    truncated transcript) is counted, not dropped -- see `unmatched_tool_use`.

READ-ONLY: this module never writes to the store or to any transcript. It returns plain dicts;
persistence, if wanted, is a decision for the caller.

SIDECHAIN (subagent) TRAFFIC: Claude Code marks subagent-turn rows `isSidechain: true`. Whether
those are counted is an explicit parameter (`include_sidechains`), never a silent default that
could mix two populations with different tool-usage shapes (a subagent's tool trace answers a
narrower question than the top-level agent's) into one rate.

DEPENDENCY HEURISTIC -- READ THIS BEFORE TRUSTING THE "MISSED BATCH" NUMBER:
`_looks_dependent` is a STRING-CONTAINMENT check: a later single tool call is judged "dependent"
on an earlier one if some token extracted from the later call's input (a path-like or
quoted-string-like substring) appears verbatim in the earlier call's result text. This is
deliberately conservative in one direction and deliberately loose in another, and both biases
matter:
  - Conservative (undercounts dependence, so OVERSTATES the opportunity): a call can depend on
    an earlier one's result semantically (e.g. "now edit the file whose content I just read")
    without any literal substring match (e.g. it reasons over line numbers, or decides *whether*
    to act based on the result's meaning, not its text). This heuristic cannot see that.
  - Loose (falsely flags dependence, so UNDERSTATES the opportunity): a short common substring
    (e.g. "test", "src", a version number) coincidentally appearing in both is treated as a
    dependency even when there is none.
Because both errors run in opposite directions and neither is measured here, `missed_batches`
reports TWO numbers, not one:
  - `strict`: any token overlap at all counts as "dependent" (excluded from the opportunity) --
    this UNDERCOUNTS the opportunity (biased toward token overlap = false dependency, but errs
    toward NOT claiming an opportunity when unsure by using a low bar for "dependent").
  - `loose`: only a longer, more specific token overlap (>= _LOOSE_MIN_TOKEN_LEN chars, filtering
    out short common words) counts as "dependent" -- this OVERCOUNTS the opportunity, since some
    of what it calls independent may in fact depend on the earlier result's meaning rather than
    its literal text.
Treat `strict` as a floor and `loose` as a ceiling on the true missed-batch opportunity. Neither
is the true number. Do not report a single point estimate without both bounds and this caveat.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flightdeck.ingest_transcript import DEFAULT_ROOT, _iso_to_ms, find_transcript

# Read-only, repeatable tools: calling them again with the same input cannot change anything
# about the world, only what the agent knows. These are the only ones that could ever be
# speculatively pre-executed without risk. Matched case-sensitively against the transcript's
# tool_use `name` field (Claude Code's own tool names).
READ_ONLY_TOOLS = frozenset(
    {
        "Read",
        "Grep",
        "Glob",
        "NotebookRead",
        "WebFetch",
        "WebSearch",
        "TodoRead",
    }
)

# Bash is NOT in READ_ONLY_TOOLS: its command can be anything from `git status` to `rm -rf`.
# Effectful-by-default; a curated read-only-looking prefix allowlist below is the only carve-out,
# and it is deliberately narrow -- a false "this is read-only" verdict is the dangerous direction.
_READ_ONLY_BASH_PREFIXES = (
    "git status",
    "git diff",
    "git log",
    "git show",
    "git branch",
    "ls",
    "cat ",
    "head ",
    "tail ",
    "wc ",
    "find ",
    "grep ",
    "rg ",
    "pwd",
    "echo ",
    "which ",
    "file ",
)


def _bash_is_read_only(command: str) -> bool:
    """Conservative: only a command whose FIRST pipeline stage matches a known read-only prefix
    is treated as prefetchable. `cmd1 && cmd2` or `cmd1 | cmd2` is judged by `cmd1` only, which
    means a read-only-looking prefix piped into something effectful (rare, but possible) can be
    misjudged read-only -- that risk is accepted here only for measurement purposes; this
    function must never be reused to actually gate real speculative execution."""
    command = command.strip()
    return any(command.startswith(p) for p in _READ_ONLY_BASH_PREFIXES)


def is_prefetchable(tool_name: str, tool_input: dict[str, Any]) -> bool:
    """Whether a tool call is read-only and repeatable -- the only category that could ever be
    speculatively pre-executed. See module docstring: Bash gets a narrow, conservative allowlist
    rather than a blanket verdict either way."""
    if tool_name in READ_ONLY_TOOLS:
        return True
    if tool_name == "Bash":
        command = tool_input.get("command")
        return isinstance(command, str) and _bash_is_read_only(command)
    return False


def tool_target(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """The thing a tool call operates on, for repeat-call / caching analysis (e.g. "Read the same
    file 5 times"). Best-effort: falls back to None (not "unknown") rather than a fabricated
    value when a tool's input shape isn't one of the ones this function recognizes -- an
    unrecognized shape must not silently collapse into a fake shared target."""
    for key in ("file_path", "path", "notebook_path"):
        val = tool_input.get(key)
        if isinstance(val, str):
            return val
    if tool_name == "Bash":
        command = tool_input.get("command")
        if isinstance(command, str):
            return command.strip()
    if tool_name in ("Grep", "WebSearch"):
        val = tool_input.get("pattern") or tool_input.get("query")
        if isinstance(val, str):
            return val
    if tool_name == "WebFetch":
        val = tool_input.get("url")
        if isinstance(val, str):
            return val
    return None


@dataclass
class ToolCall:
    """One tool_use block, matched with its result if one was found."""

    tool_use_id: str
    name: str
    input: dict[str, Any]
    message_id: str
    ts_ms: int
    batch_size: int  # number of tool_use blocks in the SAME assistant message
    result_text: str | None = None
    is_error: bool = False
    has_result: bool = False


@dataclass
class ParseIssue:
    path: str
    line_no: int
    reason: str


@dataclass
class SessionTrace:
    session_id: str
    path: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    parse_issues: list[ParseIssue] = field(default_factory=list)
    sidechain_calls_excluded: int = 0


# A path-or-string-literal-shaped token, extracted from a tool call's JSON input, used by the
# dependency heuristic. Deliberately simple: substrings of at least 4 characters containing a
# '/', '.', or looking like an identifier -- see module docstring for why this is conservative
# in one direction and loose in the other.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]{4,}")
_LOOSE_MIN_TOKEN_LEN = 8


def _tokens(text: str, *, min_len: int) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text) if len(t) >= min_len}


def _looks_dependent(earlier_result: str, later_input: dict[str, Any], *, min_len: int) -> bool:
    """See module docstring's DEPENDENCY HEURISTIC section for the full caveat. `min_len`
    controls strict (4) vs loose (8) mode -- a lower bound flags more overlaps as "dependent",
    which shrinks the reported opportunity (strict mode); a higher bound requires a longer, more
    specific shared token (loose mode), which grows it."""
    later_text = json.dumps(later_input, sort_keys=True)
    later_tokens = _tokens(later_text, min_len=min_len)
    if not later_tokens:
        return False
    earlier_tokens = _tokens(earlier_result, min_len=min_len)
    return not earlier_tokens.isdisjoint(later_tokens)


def _extract_tool_use_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def _extract_tool_result_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


def _result_text(block: dict[str, Any]) -> str:
    """`tool_result.content` is either a plain string or a list of content blocks (text/image).
    Non-text blocks (images) contribute nothing to the text used for the dependency heuristic --
    that is a real information loss (an image result could still make a later call "dependent"),
    accepted because there is no text to token-match against."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
        return "\n".join(parts)
    return ""


def parse_transcript(
    path: Path, *, session_id: str | None = None, include_sidechains: bool = False
) -> SessionTrace:
    """Single streaming pass extracting every tool_use block (grouped into per-assistant-message
    batches by `message.id`) and matching each to its tool_result by `tool_use_id`, in file order.

    Every unparseable line and every structurally odd record (missing message, tool_use with no
    later matching tool_result, etc.) is COUNTED via `parse_issues` / `sidechain_calls_excluded`,
    never silently skipped -- a caller that ignores those fields is choosing to, not being denied
    the information.
    """
    trace = SessionTrace(session_id=session_id or path.stem, path=str(path))
    pending_results: dict[str, dict[str, Any]] = {}
    calls_by_id: dict[str, ToolCall] = {}

    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError as exc:
        trace.parse_issues.append(ParseIssue(str(path), 0, f"open failed: {exc}"))
        return trace

    with handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except ValueError as exc:
                trace.parse_issues.append(ParseIssue(str(path), line_no, f"json error: {exc}"))
                continue
            if not isinstance(record, dict):
                trace.parse_issues.append(ParseIssue(str(path), line_no, "record is not an object"))
                continue

            is_sidechain = bool(record.get("isSidechain"))

            tool_use_blocks = _extract_tool_use_blocks(record)
            if tool_use_blocks:
                if is_sidechain and not include_sidechains:
                    trace.sidechain_calls_excluded += len(tool_use_blocks)
                else:
                    message = record.get("message") or {}
                    mid = message.get("id")
                    ts_ms = record.get("timestamp")
                    ts_val = _iso_to_ms(ts_ms or "") if isinstance(ts_ms, str) else None
                    if ts_val is None:
                        trace.parse_issues.append(
                            ParseIssue(str(path), line_no, "tool_use row missing/bad timestamp")
                        )
                        ts_val = 0
                    if not mid:
                        trace.parse_issues.append(
                            ParseIssue(str(path), line_no, "tool_use row missing message.id")
                        )
                        mid = f"__unknown_{line_no}"
                    batch_size = len(tool_use_blocks)
                    for block in tool_use_blocks:
                        tuid = block.get("id")
                        if not tuid:
                            trace.parse_issues.append(
                                ParseIssue(str(path), line_no, "tool_use block missing id")
                            )
                            continue
                        call = ToolCall(
                            tool_use_id=tuid,
                            name=block.get("name") or "<unknown>",
                            input=block.get("input") or {},
                            message_id=mid,
                            ts_ms=ts_val,
                            batch_size=batch_size,
                        )
                        calls_by_id[tuid] = call
                        trace.tool_calls.append(call)
                        if tuid in pending_results:
                            result = pending_results.pop(tuid)
                            call.has_result = True
                            call.result_text = _result_text(result)
                            call.is_error = bool(result.get("is_error"))

            result_blocks = _extract_tool_result_blocks(record)
            for rblock in result_blocks:
                tuid = rblock.get("tool_use_id")
                if not tuid:
                    trace.parse_issues.append(
                        ParseIssue(str(path), line_no, "tool_result block missing tool_use_id")
                    )
                    continue
                call = calls_by_id.get(tuid)
                if call is not None:
                    call.has_result = True
                    call.result_text = _result_text(rblock)
                    call.is_error = bool(rblock.get("is_error"))
                else:
                    # Result arrived before (or without) its tool_use -- hold it in case the
                    # tool_use shows up later in the file; if it never does, it is simply never
                    # matched (fine: `unmatched_tool_use`/orphan-result counts are for tool_uses,
                    # not for stray results).
                    pending_results[tuid] = rblock

    return trace


def batch_stats(trace: SessionTrace) -> dict[str, Any]:
    """Requirement 1: batching ACTUALLY ACHIEVED. Groups by `message_id` (one assistant
    response), not by row -- a message with N tool_use blocks is one batch of size N."""
    by_message: dict[str, int] = {}
    for call in trace.tool_calls:
        by_message[call.message_id] = by_message.get(call.message_id, 0) + 1

    sizes = list(by_message.values())
    dist = Counter(sizes)
    total_calls = sum(sizes)
    batched_calls = sum(s for s in sizes if s >= 2)
    return {
        "assistant_messages_with_tools": len(sizes),
        "messages_single_tool": dist.get(1, 0),
        "messages_multi_tool": sum(c for size, c in dist.items() if size >= 2),
        "batch_size_distribution": dict(sorted(dist.items())),
        "total_tool_calls": total_calls,
        "calls_traveling_in_a_batch": batched_calls,
        "achieved_batch_rate": (batched_calls / total_calls) if total_calls else None,
    }


def missed_batch_opportunities(trace: SessionTrace) -> dict[str, Any]:
    """Requirement 2: the real prize. Finds consecutive pairs of SINGLE-tool assistant messages
    (batch_size == 1 on both sides) and classifies the later call as dependent or independent of
    the earlier one using `_looks_dependent` at two thresholds. See the module docstring's
    DEPENDENCY HEURISTIC section before trusting either number -- `strict` is a floor, `loose` is
    a ceiling, and the true count is unmeasured and lies somewhere in between (possibly outside,
    for reasons the heuristic cannot see: e.g. this pairs only ADJACENT singles, so an
    independent call three messages later is never counted as an opportunity at all, which
    UNDERcounts regardless of threshold).
    """
    singles = [c for c in trace.tool_calls if c.batch_size == 1]
    # Keep only one call per message (batch_size==1 messages have exactly one call by
    # definition, but guard anyway in case of a parse anomaly).
    seen_messages: set[str] = set()
    ordered_singles: list[ToolCall] = []
    for c in singles:
        if c.message_id not in seen_messages:
            seen_messages.add(c.message_id)
            ordered_singles.append(c)

    strict_independent = 0
    loose_independent = 0
    pairs_considered = 0
    for earlier, later in zip(ordered_singles, ordered_singles[1:], strict=False):
        pairs_considered += 1
        earlier_result = earlier.result_text or ""
        if not _looks_dependent(earlier_result, later.input, min_len=4):
            strict_independent += 1
        if not _looks_dependent(earlier_result, later.input, min_len=_LOOSE_MIN_TOKEN_LEN):
            loose_independent += 1

    return {
        "consecutive_single_pairs_considered": pairs_considered,
        "missed_batches_strict": strict_independent,
        "missed_batches_loose": loose_independent,
        "note": (
            "strict/loose bound the true opportunity; both undercount cases where an "
            "independent call is not ADJACENT to the one before it. See module docstring."
        ),
    }


def prefetchability_stats(trace: SessionTrace) -> dict[str, Any]:
    """Requirement 3: share of calls that are read-only/repeatable, and the most frequent
    repeated (tool, target) pairs -- a concrete, measurable caching win independent of batching."""
    total = len(trace.tool_calls)
    prefetchable = 0
    target_counts: Counter[tuple[str, str]] = Counter()
    for call in trace.tool_calls:
        ro = is_prefetchable(call.name, call.input)
        if ro:
            prefetchable += 1
        target = tool_target(call.name, call.input)
        if target is not None:
            target_counts[(call.name, target)] += 1

    repeated = [(k, v) for k, v in target_counts.items() if v >= 2]
    repeated.sort(key=lambda kv: kv[1], reverse=True)
    return {
        "total_calls": total,
        "prefetchable_calls": prefetchable,
        "prefetchable_share": (prefetchable / total) if total else None,
        "top_repeated_targets": [
            {"tool": tool, "target": target, "count": count}
            for (tool, target), count in repeated[:20]
        ],
    }


def failure_stats(trace: SessionTrace) -> dict[str, Any]:
    """Requirement 4: failure/retry cascades. `errored` = `is_error` explicitly true on the
    tool_result. `unmatched` = a tool_use with NO tool_result ever seen in this file (turn
    interrupted, transcript truncated, or a genuine harness bug losing a result) -- reported
    separately from `errored` because it is a different failure mode (never ran to completion at
    all, vs ran and failed) and conflating them would hide which one dominates."""
    by_tool: dict[str, dict[str, int]] = {}
    for call in trace.tool_calls:
        stats = by_tool.setdefault(call.name, {"total": 0, "errored": 0, "unmatched": 0})
        stats["total"] += 1
        if not call.has_result:
            stats["unmatched"] += 1
        elif call.is_error:
            stats["errored"] += 1

    by_tool_rates = {
        name: {
            **s,
            "error_rate": (s["errored"] / s["total"]) if s["total"] else None,
            "unmatched_rate": (s["unmatched"] / s["total"]) if s["total"] else None,
        }
        for name, s in by_tool.items()
    }
    total = sum(s["total"] for s in by_tool.values())
    total_errored = sum(s["errored"] for s in by_tool.values())
    total_unmatched = sum(s["unmatched"] for s in by_tool.values())
    return {
        "by_tool": by_tool_rates,
        "total_calls": total,
        "total_errored": total_errored,
        "total_unmatched": total_unmatched,
        "overall_error_rate": (total_errored / total) if total else None,
    }


def analyze_session(
    path: Path, *, session_id: str | None = None, include_sidechains: bool = False
) -> dict[str, Any]:
    """One session's full report: parse it, then run every measurement over the same trace."""
    trace = parse_transcript(path, session_id=session_id, include_sidechains=include_sidechains)
    return {
        "session_id": trace.session_id,
        "path": trace.path,
        "parse_issues": [asdict(i) for i in trace.parse_issues],
        "sidechain_calls_excluded": trace.sidechain_calls_excluded,
        "include_sidechains": include_sidechains,
        "batching": batch_stats(trace),
        "missed_batches": missed_batch_opportunities(trace),
        "prefetchability": prefetchability_stats(trace),
        "failures": failure_stats(trace),
    }


def analyze_corpus(
    root: Path | str = DEFAULT_ROOT, *, include_sidechains: bool = False
) -> dict[str, Any]:
    """Every `*.jsonl` transcript under `root`, aggregated. `root` and `find_transcript`'s
    `DEFAULT_ROOT` are the same constant reused from `ingest_transcript` -- one place names
    where Claude Code writes transcripts.
    """
    base = Path(root).expanduser()
    paths = sorted(base.glob("*/*.jsonl"))

    per_session: list[dict[str, Any]] = []
    total_parse_issues = 0
    total_sidechain_excluded = 0
    agg_batch_sizes: Counter[int] = Counter()
    agg_total_calls = 0
    agg_batched_calls = 0
    agg_strict_missed = 0
    agg_loose_missed = 0
    agg_pairs = 0
    agg_prefetchable = 0
    agg_target_counts: Counter[tuple[str, str]] = Counter()
    agg_by_tool: dict[str, dict[str, int]] = {}

    for path in paths:
        report = analyze_session(path, include_sidechains=include_sidechains)
        per_session.append(report)
        total_parse_issues += len(report["parse_issues"])
        total_sidechain_excluded += report["sidechain_calls_excluded"]

        for size_str, count in report["batching"]["batch_size_distribution"].items():
            agg_batch_sizes[int(size_str)] += count
        agg_total_calls += report["batching"]["total_tool_calls"]
        agg_batched_calls += report["batching"]["calls_traveling_in_a_batch"]

        agg_strict_missed += report["missed_batches"]["missed_batches_strict"]
        agg_loose_missed += report["missed_batches"]["missed_batches_loose"]
        agg_pairs += report["missed_batches"]["consecutive_single_pairs_considered"]

        agg_prefetchable += report["prefetchability"]["prefetchable_calls"]
        for item in report["prefetchability"]["top_repeated_targets"]:
            agg_target_counts[(item["tool"], item["target"])] += item["count"]

        for tool, stats in report["failures"]["by_tool"].items():
            acc = agg_by_tool.setdefault(tool, {"total": 0, "errored": 0, "unmatched": 0})
            acc["total"] += stats["total"]
            acc["errored"] += stats["errored"]
            acc["unmatched"] += stats["unmatched"]

    top_targets = sorted(agg_target_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
    by_tool_rates = {
        tool: {
            **s,
            "error_rate": (s["errored"] / s["total"]) if s["total"] else None,
            "unmatched_rate": (s["unmatched"] / s["total"]) if s["total"] else None,
        }
        for tool, s in agg_by_tool.items()
    }

    return {
        "sessions_scanned": len(paths),
        "include_sidechains": include_sidechains,
        "total_parse_issues": total_parse_issues,
        "total_sidechain_calls_excluded": total_sidechain_excluded,
        "batching": {
            "total_tool_calls": agg_total_calls,
            "calls_traveling_in_a_batch": agg_batched_calls,
            "achieved_batch_rate": (agg_batched_calls / agg_total_calls)
            if agg_total_calls
            else None,
            "batch_size_distribution": dict(sorted(agg_batch_sizes.items())),
        },
        "missed_batches": {
            "consecutive_single_pairs_considered": agg_pairs,
            "missed_batches_strict": agg_strict_missed,
            "missed_batches_loose": agg_loose_missed,
        },
        "prefetchability": {
            "prefetchable_calls": agg_prefetchable,
            "prefetchable_share": (agg_prefetchable / agg_total_calls) if agg_total_calls else None,
            "top_repeated_targets": [
                {"tool": tool, "target": target, "count": count}
                for (tool, target), count in top_targets
            ],
        },
        "failures": {
            "by_tool": by_tool_rates,
            "total_errored": sum(s["errored"] for s in agg_by_tool.values()),
            "total_unmatched": sum(s["unmatched"] for s in agg_by_tool.values()),
        },
        "per_session": per_session,
    }


__all__ = [
    "ToolCall",
    "SessionTrace",
    "ParseIssue",
    "parse_transcript",
    "analyze_session",
    "analyze_corpus",
    "batch_stats",
    "missed_batch_opportunities",
    "prefetchability_stats",
    "failure_stats",
    "is_prefetchable",
    "tool_target",
    "find_transcript",
]
