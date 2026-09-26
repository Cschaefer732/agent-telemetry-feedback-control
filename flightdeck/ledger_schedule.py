"""Index every scheduler on the fleet into one view: launchd, cron, and systemd user timers.

Three sources previously answered "what is scheduled" independently, or not at all. The
motivating find: this laptop's crontab holds five comments describing five jobs and ZERO
job lines -- a scheduler that fires nothing reads exactly like a scheduler with nothing to
do, unless the comments are surfaced as `orphaned_comments`.

"Zero jobs" and "unreachable" are different facts. schedule_sources.jobs_found is NULL
(never 0) when reachable=0 -- see the null-key-is-not-absent incident this fleet already
hit once. Every source failure is counted and reported; nothing here raises.
"""

from __future__ import annotations

import os
import plistlib
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from flightdeck.store import Store, now_ms

LAUNCH_AGENTS_DIR = Path("~/Library/LaunchAgents").expanduser()
#: Label prefixes this project's own jobs use -- edit for your own reverse-DNS namespace.
OWN_PREFIXES = ("com.sparky.", "dev.example.", "com.example.")

_CRON_COMMENT_RE = re.compile(r"^\s*#\s*(.+)$")
_CRON_JOB_RE = re.compile(r"^\s*[^#\s]")


def _local_host() -> str:
    return socket.gethostname().split(".")[0]


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None


def _is_own(name: str) -> bool:
    return name.startswith(OWN_PREFIXES)


# --------------------------------------------------------------------------
# launchd
# --------------------------------------------------------------------------


def _parse_plist_schedule(plist: dict[str, Any]) -> str | None:
    if "StartInterval" in plist:
        return f"StartInterval={plist['StartInterval']}"
    if "StartCalendarInterval" in plist:
        return f"StartCalendarInterval={plist['StartCalendarInterval']!r}"
    if plist.get("RunAtLoad"):
        return "RunAtLoad"
    if "WatchPaths" in plist:
        return f"WatchPaths={plist['WatchPaths']!r}"
    return None


def _read_plist(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as fh:
            return plistlib.load(fh)
    except (OSError, plistlib.InvalidFileException):
        return None


def collect_launchd(*, launch_agents_dir: Path = LAUNCH_AGENTS_DIR) -> dict[str, Any]:
    """Merge `launchctl list` (loaded) with on-disk plists (installed).

    Returns {"reachable": bool, "error": str|None, "jobs": [...]}. A job present in only
    one of the two is reported with a note, never silently dropped.
    """
    jobs: dict[str, dict[str, Any]] = {}

    listed = _run(["launchctl", "list"], timeout=5.0)
    if listed is None:
        return {"reachable": False, "error": "launchctl list failed or timed out", "jobs": []}
    if listed.returncode != 0:
        return {
            "reachable": False,
            "error": f"launchctl list exit {listed.returncode}: {listed.stderr.strip()}",
            "jobs": [],
        }

    for line in listed.stdout.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        pid_s, last_exit_s, label = parts
        if not label:
            continue
        pid = None if pid_s == "-" else _to_int(pid_s)
        last_exit = _to_int(last_exit_s)
        jobs[label] = {
            "name": label,
            "loaded": True,
            "pid": pid,
            "last_exit": last_exit,
            "status": "running" if pid is not None else ("failed" if last_exit else "waiting"),
            "schedule": None,
            "command": None,
            "note": None,
        }

    plist_dir_ok = launch_agents_dir.is_dir()
    if plist_dir_ok:
        for plist_path in sorted(launch_agents_dir.glob("*.plist")):
            data = _read_plist(plist_path)
            label = data.get("Label") or plist_path.stem if data is not None else plist_path.stem

            schedule = _parse_plist_schedule(data) if data else None
            command = None
            if data and data.get("ProgramArguments"):
                command = " ".join(str(a) for a in data["ProgramArguments"])

            if label in jobs:
                jobs[label]["schedule"] = schedule
                jobs[label]["command"] = command
                jobs[label]["on_disk"] = True
            else:
                jobs[label] = {
                    "name": label,
                    "loaded": False,
                    "pid": None,
                    "last_exit": None,
                    "status": "unknown",
                    "schedule": schedule,
                    "command": command,
                    "note": "installed_not_loaded",
                    "on_disk": True,
                }

    for label, job in jobs.items():
        job.setdefault("on_disk", False)
        job["own"] = _is_own(label)
        # We only scan ~/Library/LaunchAgents. `launchctl list` also surfaces agents/
        # daemons loaded from /System/Library and /Library (Apple + vendor installers) and
        # ephemeral per-window `application.*` entries -- none of those are covered by our
        # scan, so "no plist on disk" is only a meaningful finding for the user's own jobs,
        # the ones this scan CAN see the full picture for.
        if job["loaded"] and not job["on_disk"] and job["own"]:
            job["note"] = "loaded_no_plist_on_disk"
        job["enabled"] = 1 if job["loaded"] else 0

    return {"reachable": True, "error": None, "jobs": list(jobs.values())}


def _to_int(s: str) -> int | None:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# cron
# --------------------------------------------------------------------------


def collect_cron() -> dict[str, Any]:
    """`crontab -l` only. Comment-only lines describing jobs with no job line are surfaced
    as orphaned_comments -- see module docstring."""
    r = _run(["crontab", "-l"], timeout=5.0)
    if r is None:
        return {
            "reachable": False,
            "error": "crontab -l failed or timed out",
            "jobs": [],
            "orphaned_comments": [],
        }
    if r.returncode != 0:
        # "no crontab for user" is a normal empty state, not an error.
        stderr = r.stderr.strip()
        if "no crontab" in stderr.lower():
            return {"reachable": True, "error": None, "jobs": [], "orphaned_comments": []}
        return {
            "reachable": False,
            "error": f"crontab -l exit {r.returncode}: {stderr}",
            "jobs": [],
            "orphaned_comments": [],
        }

    return _parse_crontab(r.stdout)


def _parse_crontab(text: str, *, host: str | None = None) -> dict[str, Any]:
    """Split a crontab into job lines and orphaned description comments.

    Shared by the local and remote collectors so a comment-only crontab is reported the
    same way whichever machine it is on -- that is the finding this module was built for.
    """
    jobs: list[dict[str, Any]] = []
    comments: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _CRON_COMMENT_RE.match(line)
        if m:
            comment = m.group(1).strip()
            if comment:
                comments.append(comment)
            continue
        # job line: "min hour dom mon dow command..."
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        schedule = " ".join(fields[:5])
        command = fields[5]
        jobs.append(
            {
                "name": command.split()[0] if command.split() else command,
                "schedule": schedule,
                "command": command,
                "enabled": 1,
                "status": "unknown",
            }
        )

    # Comments only count as ORPHANED when the crontab describes work and schedules none.
    # A comment above a live job line is documentation, not a finding.
    orphaned = comments if not jobs else []
    out: dict[str, Any] = {
        "reachable": True,
        "error": None,
        "jobs": jobs,
        "orphaned_comments": orphaned,
    }
    if host:
        out["host"] = host
    return out


# --------------------------------------------------------------------------
# systemd user timers (remote, over ssh)
# --------------------------------------------------------------------------

_TIMER_LINE_RE = re.compile(
    r"^(?P<next>\S.*?\S|\S)\s{2,}(?P<left>\S+(?: \S+)*?)\s{2,}"
    r"(?P<last>\S.*?\S|\S|n/a)\s{2,}(?P<passed>\S+(?: \S+)*?)\s{2,}"
    r"(?P<unit>\S+)\s{2,}(?P<activates>\S+)\s*$"
)


def _ssh(host: str, remote_cmd: str, timeout: float) -> subprocess.CompletedProcess[str] | None:
    return _run(
        ["ssh", "-o", f"ConnectTimeout={int(timeout)}", "-o", "BatchMode=yes", host, remote_cmd],
        timeout=timeout + 3,
    )


def collect_systemd_timers(
    host: str = "review-host", *, ssh_host: str = "user@review-host", timeout: float = 6.0
) -> dict[str, Any]:
    """systemd --user timers on `host` (default: the box running the nightly review chain),
    fetched read-only over ssh. Override `host`/`ssh_host` for your own fleet."""
    r = _ssh(ssh_host, "systemctl --user list-timers --all --no-legend --no-pager", timeout)
    if r is None:
        return {"reachable": False, "error": f"ssh to {ssh_host} timed out", "jobs": []}
    if r.returncode != 0:
        return {
            "reachable": False,
            "error": f"list-timers exit {r.returncode}: {r.stderr.strip()}",
            "jobs": [],
        }

    jobs: list[dict[str, Any]] = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        # Column boundaries are whitespace-run-delimited (NEXT/LAST can be multi-word
        # dates); regex-match the fixed trailing columns and fall back to a token scan
        # for UNIT/ACTIVATES if the line shape is unexpected.
        m = _TIMER_LINE_RE.match(line)
        unit = m.group("unit") if m else None
        activates = m.group("activates") if m else None
        left_text = m.group("left") if m else None
        passed_text = m.group("passed") if m else None
        if unit is None:
            for tok in reversed(fields):
                if tok.endswith(".service"):
                    activates = tok
                elif tok.endswith(".timer"):
                    unit = tok
                    break
        if unit is None:
            continue
        jobs.append(
            {
                "name": unit,
                "activates": activates,
                "schedule": None,
                "command": activates,
                "enabled": 1,
                "status": "unknown",
                "left_text": left_text,
                "passed_text": passed_text,
                "raw_line": line,
            }
        )

    # per-unit exit status, best-effort, bounded total time.
    deadline = time.monotonic() + timeout
    for job in jobs:
        if time.monotonic() > deadline:
            job["note"] = "status_lookup_skipped_budget"
            continue
        unit = job["activates"] or job["name"]
        sr = _ssh(
            ssh_host,
            f"systemctl --user show {unit} -p ExecMainStatus,Result,ActiveState",
            min(2.0, timeout),
        )
        if sr is None or sr.returncode != 0:
            job["note"] = "status_lookup_failed"
            continue
        props = dict(line.split("=", 1) for line in sr.stdout.splitlines() if "=" in line)
        exit_code = _to_int(props.get("ExecMainStatus"))
        result = props.get("Result")
        active = props.get("ActiveState")
        job["last_exit"] = exit_code
        if result == "failed" or (exit_code not in (None, 0)):
            job["status"] = "failed"
        elif active == "active":
            job["status"] = "running"
        elif active:
            job["status"] = "waiting"

    _annotate_stale_systemd(jobs)
    return {"reachable": True, "error": None, "jobs": jobs}


def _annotate_stale_systemd(jobs: list[dict[str, Any]]) -> None:
    """Flag a timer whose last run (PASSED) is much older than its next-fire countdown
    (LEFT) -- it should have fired again in between and didn't. Runs in `index_schedules`
    too, not just the live collector, so fixture-fed jobs get the same check."""
    for job in jobs:
        left_s = _parse_left_or_passed(job.get("left_text") or "")
        passed_s = _parse_left_or_passed(job.get("passed_text") or "")
        # 3x is a deliberately loose margin: "clearly overdue", not "slightly late".
        if left_s is not None and passed_s is not None and left_s > 0 and passed_s > 3 * left_s:
            existing_note = job.get("note")
            job["note"] = "stale" if not existing_note else f"{existing_note};stale"


# --------------------------------------------------------------------------
# indexing
# --------------------------------------------------------------------------


def _upsert_source(
    store: Store,
    *,
    source: str,
    host: str,
    reachable: bool,
    jobs_found: int | None,
    error: str | None,
    checked_at: int,
) -> None:
    store.conn.execute(
        """
        INSERT INTO schedule_sources (source, host, reachable, jobs_found, error, checked_at)
        VALUES (:source, :host, :reachable, :jobs_found, :error, :checked_at)
        ON CONFLICT(source, host) DO UPDATE SET
            reachable=excluded.reachable,
            jobs_found=excluded.jobs_found,
            error=excluded.error,
            checked_at=excluded.checked_at
        """,
        {
            "source": source,
            "host": host,
            "reachable": int(reachable),
            "jobs_found": jobs_found,
            "error": error,
            "checked_at": checked_at,
        },
    )


def _upsert_job(
    store: Store, *, source: str, host: str, job: dict[str, Any], indexed_at: int
) -> None:
    name = str(job.get("name") or "unknown")
    job_id = f"{source}|{host}|{name}"
    store.conn.execute(
        """
        INSERT INTO scheduled_jobs
            (job_id, source, host, name, schedule, command, enabled,
             last_run_at, next_run_at, last_exit, status, note, indexed_at)
        VALUES (:job_id, :source, :host, :name, :schedule, :command, :enabled,
                :last_run_at, :next_run_at, :last_exit, :status, :note, :indexed_at)
        ON CONFLICT(job_id) DO UPDATE SET
            schedule=excluded.schedule,
            command=excluded.command,
            enabled=excluded.enabled,
            last_run_at=excluded.last_run_at,
            next_run_at=excluded.next_run_at,
            last_exit=excluded.last_exit,
            status=excluded.status,
            note=excluded.note,
            indexed_at=excluded.indexed_at
        """,
        {
            "job_id": job_id,
            "source": source,
            "host": host,
            "name": name,
            "schedule": job.get("schedule"),
            "command": job.get("command"),
            "enabled": job.get("enabled"),
            "last_run_at": job.get("last_run_at"),
            "next_run_at": job.get("next_run_at"),
            "last_exit": job.get("last_exit"),
            "status": job.get("status"),
            "note": job.get("note"),
            "indexed_at": indexed_at,
        },
    )


def collect_remote_cron(host: str, *, ssh_host: str, timeout: float = 6.0) -> dict[str, Any]:
    """`crontab -l` on another box, read-only over ssh.

    Exists so a fleet member that is DOWN is recorded as unreachable rather than omitted.
    A host that is never collected has no row at all, and an absent row reads to every
    downstream consumer exactly like a host with nothing scheduled -- which is the same
    collapse of "unknown" into "zero" that `schedule_sources.reachable` exists to prevent.
    A box that was unreachable for an entire session and never got a row is exactly the
    failure this function exists to make visible instead.
    """
    r = _ssh(ssh_host, "crontab -l", timeout)
    if r is None:
        return {"reachable": False, "error": f"ssh to {ssh_host} timed out", "jobs": []}
    # `crontab -l` exits 1 with "no crontab for <user>" -- an EMPTY crontab, which is a
    # real answer (zero jobs), not an unreachable host. Only a transport failure is
    # unreachable.
    if r.returncode != 0 and "no crontab" not in (r.stderr or "").lower():
        return {
            "reachable": False,
            "error": f"crontab -l exit {r.returncode}: {(r.stderr or '').strip()}",
            "jobs": [],
        }
    return _parse_crontab(r.stdout or "", host=host)


def index_schedules(
    store: Store,
    *,
    include_remote: bool = True,
    hosts: dict[str, Any] | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Collect launchd + cron (local) and systemd timers (the review host, if include_remote)
    into scheduled_jobs / schedule_sources. Idempotent: upserts by job_id / (source, host).

    `hosts` overrides collector callables for testing:
        {"launchd": callable() -> dict, "cron": callable() -> dict,
         "systemd": callable() -> dict}

    The remote hosts checked are configurable, not hardcoded to any particular fleet:
    `FLIGHTDECK_REVIEW_HOST` / `FLIGHTDECK_REVIEW_SSH_HOST` name the box that runs the
    nightly review chain's systemd timers (defaults: "review-host" / "user@review-host").
    An optional second remote host's crontab can also be indexed -- set
    `FLIGHTDECK_SECONDARY_HOST` (and optionally `FLIGHTDECK_SECONDARY_SSH_HOST`) if you have
    one; it is skipped entirely when unset.
    """
    started = time.monotonic()
    indexed_at = now if now is not None else now_ms()
    local_host = _local_host()

    collectors: dict[str, Any] = {
        "launchd": (collect_launchd, local_host),
        "cron": (collect_cron, local_host),
    }
    if include_remote:
        review_host = os.environ.get("FLIGHTDECK_REVIEW_HOST", "review-host")
        review_ssh = os.environ.get("FLIGHTDECK_REVIEW_SSH_HOST", f"user@{review_host}")
        collectors["systemd"] = (
            lambda: collect_systemd_timers(host=review_host, ssh_host=review_ssh),
            review_host,
        )
        # An optional second remote host with its own schedule -- reachable only sometimes;
        # collect it anyway so an unreachable run leaves a row saying so, rather than a
        # silent gap that reads exactly like "nothing scheduled".
        secondary_host = os.environ.get("FLIGHTDECK_SECONDARY_HOST")
        if secondary_host:
            secondary_ssh = os.environ.get(
                "FLIGHTDECK_SECONDARY_SSH_HOST", f"user@{secondary_host}"
            )
            collectors[f"cron@{secondary_host}"] = (
                lambda: collect_remote_cron(secondary_host, ssh_host=secondary_ssh),
                secondary_host,
            )

    if hosts:
        for source, fn in hosts.items():
            if source in collectors:
                collectors[source] = (fn, collectors[source][1])

    counts: dict[str, Any] = {
        "sources": {},
        "jobs_by_source": {},
        "reachable": {},
        "unreachable": {},
        "orphaned_comments_by_source": {},
        "own_jobs": 0,
        "third_party_jobs": 0,
        "duration_s": 0.0,
    }

    for source, (collector, host) in collectors.items():
        try:
            result = collector()
        except Exception as exc:  # collector must never take down the indexer
            result = {"reachable": False, "error": f"collector raised: {exc}", "jobs": []}

        reachable = bool(result.get("reachable"))
        error = result.get("error")
        jobs = result.get("jobs") or []
        if source == "systemd":
            _annotate_stale_systemd(jobs)
        orphaned = result.get("orphaned_comments") or []

        # Orphaned cron comments describe jobs that don't exist as job lines. There is no
        # job row to hang this off, so it rides on schedule_sources.error even though the
        # source is reachable -- stale_or_failing() treats this case distinctly from an
        # actual reachability error.
        if orphaned and error is None:
            error = "orphaned_comments: " + " | ".join(orphaned)

        jobs_found = len(jobs) if reachable else None
        _upsert_source(
            store,
            source=source,
            host=host,
            reachable=reachable,
            jobs_found=jobs_found,
            error=error,
            checked_at=indexed_at,
        )

        counts["sources"][f"{source}|{host}"] = "reachable" if reachable else "unreachable"
        if reachable:
            counts["reachable"][source] = counts["reachable"].get(source, 0) + 1
        else:
            counts["unreachable"][source] = counts["unreachable"].get(source, 0) + 1

        if reachable:
            counts["jobs_by_source"][source] = counts["jobs_by_source"].get(source, 0) + len(jobs)
        else:
            # None, not 0. An unreachable source reported nothing; it did not report that
            # nothing is scheduled. schedule_sources.jobs_found keeps this distinction and
            # the summary must not quietly undo it one layer up.
            counts["jobs_by_source"].setdefault(source, None)
        if orphaned:
            counts["orphaned_comments_by_source"][source] = orphaned

        for job in jobs:
            _upsert_job(store, source=source, host=host, job=job, indexed_at=indexed_at)
            if job.get("own"):
                counts["own_jobs"] += 1
            elif source == "launchd":
                counts["third_party_jobs"] += 1

    counts["duration_s"] = round(time.monotonic() - started, 3)
    return counts


# --------------------------------------------------------------------------
# stale / failing view
# --------------------------------------------------------------------------


def _parse_left_or_passed(text: str) -> float | None:
    """Best-effort parse of systemd's human duration ('3h 12min left') into seconds."""
    if not text or text.lower() in ("n/a", "-"):
        return None
    units = {"d": 86400, "h": 3600, "min": 60, "s": 1, "ms": 0.001, "us": 1e-6, "y": 31557600}
    total = 0.0
    matched = False
    for value, unit in re.findall(r"([\d.]+)\s*([a-zµ]+)", text):
        mult = units.get(unit)
        if mult is None:
            continue
        total += float(value) * mult
        matched = True
    return total if matched else None


def stale_or_failing(store: Store) -> list[dict[str, Any]]:
    """The subset a human should look at: failed/nonzero-exit jobs, jobs whose last run
    looks far older than their period, plist-vs-loaded mismatches, and orphaned cron
    comments (surfaced via schedule_sources.error, since they have no job row)."""
    findings: list[dict[str, Any]] = []

    rows = store.conn.execute(
        """
        SELECT job_id, source, host, name, schedule, command, enabled,
               last_run_at, next_run_at, last_exit, status, note, indexed_at
        FROM scheduled_jobs
        ORDER BY source, host, name
        """
    ).fetchall()

    for r in rows:
        row = dict(r)
        reasons: list[str] = []
        if row["status"] == "failed":
            reasons.append("status=failed")
        if row["last_exit"] not in (None, 0):
            reasons.append(f"last_exit={row['last_exit']}")
        note_parts = (row["note"] or "").split(";")
        if "installed_not_loaded" in note_parts:
            reasons.append("installed on disk but not loaded")
        if "loaded_no_plist_on_disk" in note_parts:
            reasons.append("loaded but no plist on disk")
        if "stale" in note_parts:
            reasons.append("last run far older than its stated interval")

        if reasons:
            findings.append({**row, "reasons": reasons})

    src_rows = store.conn.execute(
        "SELECT source, host, reachable, jobs_found, error, checked_at FROM schedule_sources"
    ).fetchall()
    for r in src_rows:
        row = dict(r)
        if not row["reachable"]:
            findings.append(
                {
                    "job_id": None,
                    "source": row["source"],
                    "host": row["host"],
                    "name": None,
                    "reasons": [f"source unreachable: {row['error']}"],
                }
            )
        elif row["error"] and row["error"].startswith("orphaned_comments:"):
            findings.append(
                {
                    "job_id": None,
                    "source": row["source"],
                    "host": row["host"],
                    "name": None,
                    "reasons": [
                        f"crontab describes work it does not run: {row['error']}",
                    ],
                }
            )

    return findings
