"""Self-checks for the failure mode this system keeps hitting: a component that is configured but
silently doing nothing — a symlink pointing at a moved file, a migration never applied, a hook
registered but never firing, a collector producing zero rows on a box that was awake.

Every probe here must turn silence into a finding instead of an absence. That means two hard
rules: no probe may raise (a crashing probe defeats the nightly job that runs it), and every
comparison that touches a filesystem path resolves both sides first, because macOS routes /tmp
through /private/tmp and an unresolved comparison invents a false "wrong_target".
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from flightdeck.models import SOURCES, Probe
from flightdeck.schema import SCHEMA_VERSION
from flightdeck.scope_ingest import gate_log_path, read_gate_log
from flightdeck.store import DEFAULT_DIR, Store, hostname, now_ms

# Hook event names collect_claude.handle() actually dispatches on (see collect_claude.py). A name
# absent here for a full window means the hook is registered but not running. SessionStart and
# SubagentStart are deliberately excluded: handle() receives them but no-ops, so they can never
# show "ok" and would make this probe permanently red.
_DEFAULT_HOOKS = [
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "Stop",
]


def _run(
    kind: str, host: str | None, compute: Callable[[], tuple[int, int, dict[str, Any]]]
) -> Probe:
    """Every probe funnels through here: fixes ts/host, and converts any exception into a failed
    probe instead of letting it kill the nightly job."""
    ts = now_ms()
    resolved_host = "unknown"
    try:
        resolved_host = host if host is not None else hostname()
        ok, total, detail = compute()
        return Probe(ts=ts, host=resolved_host, kind=kind, ok=ok, total=total, detail=detail)
    except Exception as exc:  # noqa: BLE001 - deliberate: a probe must never propagate
        return Probe(
            ts=ts,
            host=resolved_host,
            kind=kind,
            ok=0,
            total=0,
            detail={"error": type(exc).__name__, "message": str(exc)},
        )


def _classify_symlink(link: Path, target: Path) -> tuple[str, Path | None]:
    if link.is_symlink():
        # resolve(strict=False) walks the whole chain, including a dangling final hop, without
        # raising. Resolving `target` too is what neutralizes /tmp vs /private/tmp style
        # symlinked-parent noise on macOS.
        resolved = link.resolve()
        if not resolved.exists():
            return "broken", resolved
        if resolved == target.resolve():
            return "ok", resolved
        return "wrong_target", resolved
    if link.exists():
        return "not_symlink", None
    return "missing", None


def probe_symlinks(expected: dict[Path, Path], *, host: str | None = None) -> Probe:
    """expected maps link path -> the path it must resolve to."""

    def compute() -> tuple[int, int, dict[str, Any]]:
        detail: dict[str, Any] = {}
        ok = 0
        for link, target in expected.items():
            status, resolved = _classify_symlink(link, target)
            detail[str(link)] = {
                "status": status,
                "resolved": str(resolved) if resolved is not None else None,
                "expected": str(target),
            }
            if status == "ok":
                ok += 1
        return ok, len(expected), detail

    return _run("symlink", host, compute)


def _read_schema_version(db_path: Path) -> int | None:
    """MAX(version) from schema_version if present, else MAX(version_id) from
    goose_db_version. None means neither table exists."""
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "schema_version" in tables:
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        if "goose_db_version" in tables:
            row = conn.execute("SELECT MAX(version_id) FROM goose_db_version").fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        return None
    finally:
        conn.close()


def probe_migrations(
    db_paths: dict[str, Path], expected: dict[str, int], *, host: str | None = None
) -> Probe:
    """db label -> sqlite file; expected label -> minimum schema/goose version."""

    def compute() -> tuple[int, int, dict[str, Any]]:
        detail: dict[str, Any] = {}
        ok = 0
        for label, min_version in expected.items():
            path = db_paths.get(label)
            if path is None:
                detail[label] = {"status": "unconfigured"}
                continue
            db_path = Path(path)
            if not db_path.exists():
                detail[label] = {"status": "absent"}
                continue
            try:
                version = _read_schema_version(db_path)
            except Exception as exc:  # noqa: BLE001 - one bad db must not sink the others
                detail[label] = {"status": "error", "error": type(exc).__name__}
                continue
            if version is None:
                detail[label] = {"status": "no_version_table"}
                continue
            passed = version >= min_version
            detail[label] = {
                "status": "ok" if passed else "stale",
                "version": version,
                "expected_min": min_version,
            }
            if passed:
                ok += 1
        return ok, len(expected), detail

    return _run("migration", host, compute)


def probe_hook_liveness(
    store: Store, *, window_hours: int = 24, expected_hooks: list[str], host: str | None = None
) -> Probe:
    """A hook in expected_hooks with zero liveness evidence in the window is a failure.

    Liveness evidence is source-specific. The Go emitter records kind="hook" events named after
    the hook itself, so a matching event name is direct evidence. collect_claude.py has no
    hook-shaped analog — it never emits kind="hook" (see its handle() dispatch) — so a Claude Code
    box proves liveness the way that collector actually writes data instead: at least one
    claude-code turn in the window with a tool_call event and captured text. Either signal marks
    every expected hook 'ok' for that probe run; a box with neither is genuinely silent.
    """

    def compute() -> tuple[int, int, dict[str, Any]]:
        since = now_ms() - window_hours * 3600_000
        rows = store.conn.execute(
            "SELECT DISTINCT name FROM events WHERE kind='hook' AND ts >= ?", (since,)
        )
        hook_names = {name for (name,) in rows if name}

        claude_code_active = bool(
            store.conn.execute(
                "SELECT 1 FROM turns t WHERE t.source='claude-code' AND t.started_at >= ? "
                "AND EXISTS (SELECT 1 FROM events e WHERE e.turn_id=t.turn_id "
                "AND e.kind='tool_call' AND e.ts >= ?) "
                "AND EXISTS (SELECT 1 FROM texts x WHERE x.turn_id=t.turn_id) LIMIT 1",
                (since, since),
            ).fetchone()
        )

        detail: dict[str, Any] = {}
        ok = 0
        for hook in expected_hooks:
            if hook in hook_names:
                evidence = "hook_event"
            elif claude_code_active:
                evidence = "claude_code_collector"
            else:
                evidence = None
            detail[hook] = {"status": "ok" if evidence else "silent", "evidence": evidence}
            if evidence:
                ok += 1
        return ok, len(expected_hooks), detail

    return _run("hook_liveness", host, compute)


def probe_collector_heartbeat(
    store: Store, *, window_hours: int = 24, expected_sources: list[str], host: str | None = None
) -> Probe:
    def compute() -> tuple[int, int, dict[str, Any]]:
        since = now_ms() - window_hours * 3600_000
        rows = store.conn.execute(
            "SELECT DISTINCT source FROM turns WHERE started_at >= ?", (since,)
        )
        seen = {source for (source,) in rows if source}
        detail: dict[str, Any] = {}
        ok = 0
        for source in expected_sources:
            active = source in seen
            detail[source] = {"status": "ok" if active else "silent"}
            if active:
                ok += 1
        return ok, len(expected_sources), detail

    return _run("collector_heartbeat", host, compute)


def _last_gate_entry_epoch(path: Path) -> int | None:
    """Epoch seconds of the newest well-formed row in the gate log, or None if the file is
    absent/empty/entirely malformed. Reuses `read_gate_log` so a truncated tail behaves the
    same way here as it does for real ingestion."""
    import datetime as dt

    last: int | None = None
    for row in read_gate_log(path):
        stamp = row.get("ts")
        if not stamp:
            continue
        try:
            epoch = int(dt.datetime.fromisoformat(stamp).timestamp())
        except (TypeError, ValueError):
            continue
        if last is None or epoch > last:
            last = epoch
    return last


def probe_scope_liveness(
    store: Store,
    *,
    directory: Path | str | None = None,
    window_hours: int = 24,
    host: str | None = None,
) -> Probe:
    """Is the scope gate hook still firing, and is ingestion still turning its decisions into
    scope_records? `doctor` had no scope check at all before this -- a broken hook read exactly
    like a quiet week in every existing probe.

    Two checks, run independently because they fail independently:

    - `gate_log`: has the gate log advanced in the window? If not, was the machine doing
      anything else (any turn recorded in the same window, any source)? Turns-without-decisions
      is real evidence the hook stopped firing. Neither-happened is NOT evidence of anything --
      it is indistinguishable from a legitimately idle machine with the data this store has, and
      this probe says so in `detail` rather than quietly reporting either "ok" or "fail" for it.
      Distinguishing "asleep" from "broken" would need OS-level activity data (last input event,
      screen wake) that nothing here collects; if that ever gets wired up, this is the place to
      consume it.
    - `ingestion`: given the gate log DID advance, did `scope_records` grow to match? A "no"
      here is the ingest step being broken (not run, erroring, or pointed at the wrong
      directory) while the hook itself is fine -- a different failure than the one above, and
      conflating them would send someone to fix the wrong component.

    Neither check can ever return 'ok' AND leave a genuine break unreported: 'inconclusive' and
    'not_applicable' count toward `ok` (so an idle machine, or a machine with nothing yet to
    ingest, is not a nightly false alarm) but are never returned when there's decisive evidence
    of a break -- `total` stays fixed at 2 so the two failure modes above still both show up in
    `Probe.healthy`.
    """

    def compute() -> tuple[int, int, dict[str, Any]]:
        target_dir = directory if directory is not None else store.directory
        path = gate_log_path(target_dir)
        since_ms = now_ms() - window_hours * 3600_000
        since_s = since_ms // 1000

        last_entry = _last_gate_entry_epoch(path)
        gate_advancing = last_entry is not None and last_entry >= since_s

        machine_active = bool(
            store.conn.execute(
                "SELECT 1 FROM turns WHERE started_at >= ? LIMIT 1", (since_ms,)
            ).fetchone()
        )
        records_in_window = store.conn.execute(
            "SELECT COUNT(*) FROM scope_records WHERE created_at >= ?", (since_s,)
        ).fetchone()[0]

        checks: dict[str, dict[str, Any]] = {}
        if gate_advancing:
            checks["gate_log"] = {"status": "ok", "reason": "gate-log.jsonl advanced in window"}
        elif machine_active:
            checks["gate_log"] = {
                "status": "fail",
                "reason": "turns recorded but the gate made no decisions -- hook not firing",
            }
        else:
            checks["gate_log"] = {
                "status": "inconclusive",
                "reason": "no gate decisions and no turn activity -- cannot tell a broken hook "
                "from an idle machine with the data this store has",
            }

        if not gate_advancing:
            checks["ingestion"] = {
                "status": "not_applicable",
                "reason": "no fresh gate decisions to ingest",
            }
        elif records_in_window > 0:
            checks["ingestion"] = {
                "status": "ok",
                "reason": f"{records_in_window} scope_records written in window",
            }
        else:
            checks["ingestion"] = {
                "status": "fail",
                "reason": "gate log advanced but scope_records did not grow -- ingestion is broken",
            }

        ok = sum(1 for c in checks.values() if c["status"] != "fail")
        detail = {
            "path": str(path),
            "window_hours": window_hours,
            "gate_last_entry_epoch": last_entry,
            "machine_active": machine_active,
            "records_in_window": records_in_window,
            "checks": checks,
            "limitation": (
                "This probe cannot distinguish 'hook broken' from 'nobody was working' when "
                "both the gate log and turns are quiet -- there is no activity signal available "
                "here besides turns. It only calls a hook broken when turns prove the machine "
                "was in use while the gate stayed silent."
            ),
        }
        return ok, len(checks), detail

    return _run("scope_liveness", host, compute)


def probe_judge_queue(
    store: Store,
    *,
    window_hours: int = 24,
    host: str | None = None,
    cap: int = 20,
) -> Probe:
    """flagged-but-unjudged backlog and drop count; ok when backlog is within cap.

    `cap` isn't in the caller-facing signature the spec lists but has to come from somewhere —
    ProbeConfig.judge_backlog_cap flows in through run_all. Kept as a defaulted kwarg so a plain
    `probe_judge_queue(store)` call still works.
    """

    def compute() -> tuple[int, int, dict[str, Any]]:
        since = now_ms() - window_hours * 3600_000
        backlog = store.conn.execute(
            "SELECT COUNT(*) FROM turns WHERE flagged=1 AND judged=0 AND started_at >= ?", (since,)
        ).fetchone()[0]
        # A full judge queue drops events rather than blocking (see CLAUDE.md); those show up as
        # ok=0 `queue` kind events.
        dropped = store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind='queue' AND ok=0 AND ts >= ?", (since,)
        ).fetchone()[0]
        within_cap = backlog <= cap
        detail = {
            "backlog": backlog,
            "cap": cap,
            "dropped": dropped,
            "window_hours": window_hours,
        }
        return (1 if within_cap else 0), 1, detail

    return _run("judge_queue", host, compute)


@dataclass
class ProbeConfig:
    symlinks: dict[Path, Path]
    databases: dict[str, Path]
    expected_versions: dict[str, int]
    expected_hooks: list[str]
    expected_sources: list[str]
    judge_backlog_cap: int = 20
    scope_window_hours: int = 24


def default_config() -> ProbeConfig:
    """AN EXAMPLE fleet expectation, shaped after one particular agent-config repo's
    install.sh -- edit `symlinks` below for your own dotfiles/config repo layout.

    Describes what *should* be true of a fully installed box; does not check any of it, so this
    must never fail just because the paths don't exist on the box running it (e.g. CI).
    """
    home = Path.home()
    repo = Path(os.environ.get("FLIGHTDECK_CONFIG_REPO", str(home / "dev" / "my-agent-config")))
    claude_dir = home / ".claude"
    crush_dir = home / ".config" / "crush"
    local_bin = home / ".local" / "bin"
    pa_dir = home / ".personal-agent"

    symlinks = {
        claude_dir / "CLAUDE.md": repo / "claude" / "CLAUDE.md",
        claude_dir / "settings.json": repo / "claude" / "settings.json",
        local_bin / "sparky": repo / "spark" / "cli" / "spark",
        local_bin / "spark": repo / "spark" / "cli" / "spark",
        crush_dir / "crush.json": repo / "spark" / "crush" / "crush.json",
        crush_dir / "CRUSH.md": repo / "spark" / "crush" / "CRUSH.md",
        crush_dir / "MEMORY.md": repo / "spark" / "crush" / "MEMORY.md",
        crush_dir / "MODELS.md": repo / "spark" / "crush" / "MODELS.md",
        crush_dir / "brain-vault": repo / "spark" / "brain-vault",
        crush_dir / "skills": repo / "spark" / "crush" / "skills",
        crush_dir / "hooks": repo / "spark" / "crush" / "hooks",
        crush_dir / "agents": repo / "spark" / "crush" / "agents",
        crush_dir / "modes": repo / "spark" / "crush" / "modes",
        pa_dir / "scripts": repo / "personal-agent" / "scripts",
        pa_dir / "DESIGN.md": repo / "personal-agent" / "DESIGN.md",
    }

    databases = {"turnlog": DEFAULT_DIR / "turnlog.db"}
    expected_versions = {"turnlog": SCHEMA_VERSION}

    return ProbeConfig(
        symlinks=symlinks,
        databases=databases,
        expected_versions=expected_versions,
        expected_hooks=list(_DEFAULT_HOOKS),
        expected_sources=list(SOURCES),
        judge_backlog_cap=20,
        scope_window_hours=24,
    )


def run_all(store: Store, config: ProbeConfig) -> list[Probe]:
    host = hostname()
    probes = [
        probe_symlinks(config.symlinks, host=host),
        probe_migrations(config.databases, config.expected_versions, host=host),
        probe_hook_liveness(store, expected_hooks=config.expected_hooks, host=host),
        probe_collector_heartbeat(store, expected_sources=config.expected_sources, host=host),
        probe_judge_queue(store, cap=config.judge_backlog_cap, host=host),
        probe_scope_liveness(store, window_hours=config.scope_window_hours, host=host),
    ]
    for probe in probes:
        store.add_probe(probe)
    return probes
