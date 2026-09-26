"""flightdeck CLI — the surface the systemd units and the nightly reviewer actually call.

Commands are thin. Every one of them delegates to a module that is tested on its own; this file
exists to parse arguments and print, not to hold logic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from flightdeck import probes as probes_mod
from flightdeck import review as review_mod
from flightdeck.governor import Governor, load_kpi_tuning
from flightdeck.ids import ulid
from flightdeck.kpi import score_and_persist, trailing_baseline
from flightdeck.models import TuningChange, Turn
from flightdeck.store import DEFAULT_DIR, Store, hostname, now_ms

DURATION = re.compile(r"^(\d+)([smhd])$")
_UNITS = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}


def parse_duration(value: str) -> int:
    match = DURATION.match(value.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"expected a duration like 24h or 7d, got {value!r}")
    return int(match.group(1)) * _UNITS[match.group(2)]


def open_store(args: argparse.Namespace) -> Store:
    return Store(args.dir or DEFAULT_DIR)


def emit(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


# ---------- commands ----------


def cmd_init(args: argparse.Namespace) -> int:
    store = open_store(args)
    emit(
        {"dir": str(store.directory), "db": str(store.db_path), "schema": store.version},
        as_json=args.json,
    )
    return 0


def cmd_rollup(args: argparse.Namespace) -> int:
    store = open_store(args)
    result: dict[str, Any] = {
        "ingested": {},
        "scored": 0,
        "governor_decisions": 0,
        "expired": 0,
        "probes": [],
    }

    for path in store.jsonl.files():
        counts = store.ingest_jsonl(path)
        for kind, count in counts.items():
            result["ingested"][kind] = result["ingested"].get(kind, 0) + count

    # Score anything the emitter wrote but nothing has scored yet. The emitter deliberately does
    # not compute KPIs itself — scoring needs trailing-window context it has no business loading
    # inside a turn. Turns are grouped by (source, tier): the trailing baseline barely moves
    # turn-to-turn, so one baseline lookup per group replaces one per turn.
    since = now_ms() - args.since
    unscored = [t for t in store.iter_turns(since_ms=since) if t.kpi_score is None]
    groups: dict[tuple[str | None, str | None], list[Turn]] = {}
    for turn in unscored:
        groups.setdefault((turn.source, turn.tier), []).append(turn)

    # governor.toml's [kpi.weights]/[thresholds] are the nightly reviewer's self-tuning knobs;
    # load once per rollup rather than per turn — the file doesn't change mid-run.
    weights, thresholds = load_kpi_tuning()

    for (source, tier), turns in groups.items():
        baseline = trailing_baseline(
            store, source=source, tier=tier, before_ms=max(t.started_at for t in turns)
        )
        for turn in turns:
            score_and_persist(
                store, turn, baseline=baseline, weights=weights, thresholds=thresholds
            )
            result["scored"] += 1

    # Scoring only fills in kpi_score; nothing above ever consults the governor. Backfill its
    # shadow decisions over whatever is now scored in this window — see
    # Governor.record_shadow_decisions for why this has to be a replay rather than a live call.
    gov = Governor(store=store)
    result["governor_decisions"] = gov.record_shadow_decisions(since_ms=since)

    if args.expire:
        result["expired"] = store.expire_texts()

    if args.probe:
        for probe in probes_mod.run_all(store, probes_mod.default_config()):
            result["probes"].append(
                {"kind": probe.kind, "ok": probe.ok, "total": probe.total, "healthy": probe.healthy}
            )

    emit(result, as_json=args.json)
    return 0


def cmd_backfill_tier(args: argparse.Namespace) -> int:
    """Fill `tier` on existing rows where it's derivable from `source`/`model` but was never
    written — the collector fix in cmd_rollup's ingest path and collect_claude.py only covers
    turns recorded from here on; this is the one-time catch-up for everything recorded before it.
    Mirrors the correction to JSONL (mirror=True, the upsert_turn default) so a store rebuilt from
    JSONL from scratch keeps the backfilled tier instead of losing it."""
    from flightdeck.tiers import derive_tier

    store = open_store(args)
    since = now_ms() - args.since if args.since else None
    scanned = updated = unmapped = 0
    for turn in store.iter_turns(since_ms=since):
        if turn.tier is not None:
            continue
        scanned += 1
        new_tier = derive_tier(turn.source, turn.model)
        if new_tier is None:
            unmapped += 1
            continue
        updated += 1
        if not args.dry_run:
            turn.tier = new_tier
            store.upsert_turn(turn)
    emit(
        {"scanned": scanned, "updated": updated, "unmapped": unmapped, "dry_run": args.dry_run},
        as_json=args.json,
    )
    return 0


def cmd_backfill_transcript(args: argparse.Namespace) -> int:
    """Fill token counts, `model`, a derived `model_ms`, and (only where unambiguous) `outcome`
    on claude-code turns from the Claude Code transcript JSONL — the data collect_claude.py's
    hook payloads structurally cannot carry. See flightdeck/ingest_transcript.py's module
    docstring for the join-key and dedupe pitfalls this had to get right on real data."""
    from flightdeck.ingest_transcript import DEFAULT_ROOT, backfill

    store = open_store(args)
    since = now_ms() - args.since if args.since else None
    report = backfill(store, root=args.root or DEFAULT_ROOT, since_ms=since, dry_run=args.dry_run)
    emit(report, as_json=args.json)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Is anything silently not collecting?

    Exit code is meaningful: nonzero when something is unhealthy, so a systemd unit or a shell
    check can act on it without parsing output.
    """
    store = open_store(args)
    since = now_ms() - args.since
    agg = review_mod.aggregates(store, since_ms=since)
    probe_results = probes_mod.run_all(store, probes_mod.default_config())

    findings: list[str] = []
    if agg.get("empty_window"):
        findings.append(f"no turns recorded in the last {args.since // 3_600_000}h")
    for probe in probe_results:
        if not probe.healthy:
            findings.append(f"probe {probe.kind}: {probe.ok}/{probe.total} ok — {probe.detail}")
    if agg.get("unjudged_flagged", 0) > 20:
        findings.append(f"judge backlog {agg['unjudged_flagged']} exceeds cap")

    report = {
        "host": hostname(),
        "dir": str(store.directory),
        "schema": store.version,
        "window_hours": args.since // 3_600_000,
        "turns": agg.get("turns", 0),
        "kpi_by_source": agg.get("kpi_by_source", {}),
        "probes": [{"kind": p.kind, "ok": p.ok, "total": p.total} for p in probe_results],
        "findings": findings,
        "healthy": not findings,
    }
    emit(report, as_json=args.json)
    return 0 if not findings else 1


def cmd_sample(args: argparse.Namespace) -> int:
    store = open_store(args)
    sampler = review_mod.Sampler(store, window_ms=args.since)
    complex_sample = sampler.complex_turns(args.complex)
    low_sample = sampler.low_kpi_turns(args.low)
    signals = sampler.sweep()
    agg = review_mod.aggregates(store, since_ms=now_ms() - args.since)

    payload = {
        "aggregates": agg,
        "complex": {
            "note": complex_sample.note,
            "dropped": complex_sample.dropped,
            "turns": [t.turn_id for t in complex_sample.turns],
        },
        "low_kpi": {
            "note": low_sample.note,
            "dropped": low_sample.dropped,
            "turns": [t.turn_id for t in low_sample.turns],
        },
        "sweep": [{"signal": s.signal, "session": s.session_id, **s.detail} for s in signals],
    }

    if args.write_brief:
        brief = render_brief(payload)
        target = Path(args.brief_path or (store.directory / "review-brief.md"))
        target.write_text(brief, encoding="utf-8")
        payload["brief"] = str(target)

    emit(payload, as_json=args.json)
    return 0


def render_brief(payload: dict[str, Any]) -> str:
    agg = payload["aggregates"]
    lines = [
        "# Nightly review brief",
        "",
        "Read `integration/review-skill/SKILL.md` before acting on any of this.",
        "",
        "## Aggregates",
        "",
        "```json",
        json.dumps(agg, indent=2, default=str),
        "```",
        "",
    ]
    if agg.get("empty_window"):
        lines[3:3] = [
            "> **The window is empty.** No turns were recorded. Treat this as tonight's finding",
            "> and do not tune from the numbers below.",
            "",
        ]
    for key, title in (
        ("complex", "Sample A — most complex turns"),
        ("low_kpi", "Sample B — lowest KPI turns"),
    ):
        block = payload[key]
        lines += [f"## {title}", "", block["note"], ""]
        if block["dropped"]:
            lines += [
                f"**{block['dropped']} turns were dropped by the cap** — coverage is partial.",
                "",
            ]
        lines += [f"- `{turn_id}`" for turn_id in block["turns"]] or ["- (none)"]
        lines += [""]
    lines += ["## Sample C — sweep signals", ""]
    if payload["sweep"]:
        lines += [
            f"- **{s['signal']}** in `{s['session']}` — "
            f"{json.dumps({k: v for k, v in s.items() if k not in ('signal', 'session')})}"
            for s in payload["sweep"]
        ]
    else:
        lines += ["- none"]
    lines += ["", "Inspect any turn with `python3 -m flightdeck show <turn_id>`.", ""]
    lines += _pending_longrun_lines()
    return "\n".join(lines)


def _pending_longrun_lines() -> list[str]:
    """Long training runs finish unattended and wait for exactly this review's verdict —
    without a brief section they sit invisible until doctor's 72h warning."""
    runs_dir = (
        Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        / "sparky"
        / "training"
        / "runs"
    )
    pending = []
    for mf in sorted(runs_dir.glob("*/manifest.json")):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if (
            m.get("state") == "done"
            and (mf.parent / "REVIEW.md").exists()
            and not (mf.parent / "verdict.json").exists()
        ):
            pending.append((m.get("run_id", mf.parent.name), mf.parent / "REVIEW.md"))
    if not pending:
        return []
    lines = ["## Pending long-run reviews", ""]
    lines += [f"- `{run_id}` — read `{review}`" for run_id, review in pending]
    lines += [
        "",
        "Read each run's REVIEW.md, then record the verdict with your own training-run "
        "tooling (this gates artifact promotion) -- flightdeck itself does not ship one.",
        "",
    ]
    return lines


def cmd_show(args: argparse.Namespace) -> int:
    store = open_store(args)
    turn = store.get_turn(args.turn_id)
    if turn is None:
        print(f"no such turn: {args.turn_id}", file=sys.stderr)
        return 1
    payload = {
        "turn": turn.to_row(),
        "events": [
            {
                "ts": e.ts,
                "kind": e.kind,
                "name": e.name,
                "ok": e.ok,
                "duration_ms": e.duration_ms,
                "payload": e.payload,
            }
            for e in store.events_for(args.turn_id)
        ],
        "texts": [
            {"kind": t.kind, "seq": t.seq, "body": t.body} for t in store.texts_for(args.turn_id)
        ],
        "judgment": store.judgment_for(args.turn_id),
        "governor_decisions": store.decisions_for(args.turn_id),
    }
    emit(payload, as_json=True)
    return 0


def cmd_kpi(args: argparse.Namespace) -> int:
    store = open_store(args)
    emit(review_mod.aggregates(store, since_ms=now_ms() - args.since), as_json=True)
    return 0


def cmd_governor(args: argparse.Namespace) -> int:
    from flightdeck.governor import Governor

    store = open_store(args)
    gov = Governor(store=store)
    if args.action == "status":
        emit(
            {
                "weights_version": gov.weights_version,
                "domains": {
                    domain: {
                        "live": gov.enabled(domain),
                        "graduation": gov.graduation_report(domain),
                    }
                    for domain in ("model_tier", "skills", "compaction", "mode_delegation")
                },
            },
            as_json=True,
        )
        return 0
    emit(gov.graduation_report(args.domain), as_json=True)
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    """Path authority check the reviewer runs before committing.

    Exit code is the contract: 0 means every path may be written, 1 means at least one may not.
    """
    guard = review_mod.Guardrails(Path(args.repo).resolve())
    rejected = guard.check(args.check)
    emit({"checked": args.check, "rejected": rejected, "allowed": not rejected}, as_json=True)
    return 0 if not rejected else 1


def cmd_sync(args: argparse.Namespace) -> int:
    store = open_store(args)
    merged: dict[str, dict[str, int]] = {}
    inbox = Path(args.inbox).expanduser()
    for host_dir in sorted(p for p in inbox.iterdir() if p.is_dir()) if inbox.exists() else []:
        db = host_dir / "turnlog.db"
        if db.exists():
            merged[host_dir.name] = store.merge_from(db)
        for log in sorted(host_dir.glob("events-*.jsonl")):
            store.ingest_jsonl(log)
    payload = {
        "merged": merged,
        "hosts_reached": args.hosts_reached,
        "hosts_missed": args.hosts_missed,
    }
    if args.hosts_missed:
        # Surfaced rather than swallowed: a review built on a partial fleet must say so.
        payload["warning"] = f"{args.hosts_missed} host(s) unreachable; tonight's review is partial"
    emit(payload, as_json=True)
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    """Snapshot this host's CPU/RAM/GPU/disk into host_metrics. Meant to run on every box, on a
    cadence, via its own systemd timer (integration/systemd/sparky-collect-host-metrics.*) — not
    part of the sync/rollup/review chain, which only runs on spark."""
    from flightdeck.collectors.host_metrics import collect_host_metrics

    store = open_store(args)
    record = collect_host_metrics(hostname())
    store.add_host_metrics(record)
    emit(record, as_json=args.json)
    return 0


def cmd_fleet_status(args: argparse.Namespace) -> int:
    from flightdeck.fleet_status import FLEET_STATUS_PATH, render_fleet_status

    store = open_store(args)
    inbox = Path(args.inbox).expanduser()
    md = render_fleet_status(store, inbox_dir=inbox)
    target = Path(args.out).expanduser() if args.out else FLEET_STATUS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(md, encoding="utf-8")
    emit({"path": str(target)}, as_json=args.json)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    store = open_store(args)
    verdicts = review_mod.regression_check(store)
    reverted: list[str] = []
    for verdict in verdicts:
        if verdict.should_revert and args.auto_revert:
            store.conn.execute(
                "UPDATE tuning_changes SET reverted_at=?, revert_reason=? WHERE change_id=?",
                (now_ms(), verdict.reason, verdict.change_id),
            )
            reverted.append(verdict.change_id)
    emit(
        {
            "verdicts": [v.__dict__ for v in verdicts],
            "marked_reverted": reverted,
            # The git revert itself is left to the caller: this process must not rewrite history
            # it cannot then verify.
            "next": [
                f"git revert --no-edit {v.commit_sha}"
                for v in verdicts
                if v.should_revert and v.commit_sha
            ],
        },
        as_json=True,
    )
    return 0


def cmd_record_change(args: argparse.Namespace) -> int:
    store = open_store(args)
    change = TuningChange(
        change_id=ulid(),
        applied_at=now_ms(),
        domain=args.domain,
        path=args.path,
        summary=args.summary,
        commit_sha=args.commit,
        evidence=json.loads(args.evidence) if args.evidence else {},
    )
    store.add_tuning_change(change)
    emit({"change_id": change.change_id}, as_json=True)
    return 0


def cmd_reverted(args: argparse.Namespace) -> int:
    store = open_store(args)
    rows = store.conn.execute(
        "SELECT change_id, domain, path, summary, commit_sha, revert_reason, reverted_at "
        "FROM tuning_changes WHERE reverted_at IS NOT NULL ORDER BY reverted_at DESC"
    ).fetchall()
    emit([dict(row) for row in rows], as_json=True)
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    from flightdeck.judge import JudgeConfig, run_queue

    store = open_store(args)
    emit(run_queue(store, JudgeConfig.from_env(), limit=args.limit), as_json=True)
    return 0


# ---------- parser ----------


def cmd_scope(args: argparse.Namespace) -> int:
    from flightdeck.scope_gate import classify
    from flightdeck.scope_kpi import zero_streak_verdict

    if args.action == "classify":
        decision = classify(args.prompt, has_active_scope=args.active)
        emit(
            {
                "tier": decision.tier,
                "reasons": decision.reasons,
                "ceremonial": decision.ceremonial,
                "signals": decision.signals,
            },
            as_json=args.json,
        )
        return 0

    if args.action == "registry":
        from flightdeck.scope_registry import summary

        result = summary()
        emit(result, as_json=args.json)
        return 0

    if args.action == "learn":
        # Mines next-turn corrections after un-scoped task starts, replays the gate
        # counterfactually, and promotes only what clears the bar. Shipping nothing is the
        # expected outcome most weeks: the clean-signal slice is ~1 turn per session.
        from flightdeck.scope_learn import apply as learn_apply

        result = learn_apply(
            args.root,
            registry=args.registry,
            store=open_store(args) if args.write else None,
            dry_run=not args.write,
        )
        emit(result, as_json=args.json)
        return 0 if result["verdict"] != "fail" else 1

    if args.action == "eval":
        # Reproduces the classifier numbers instead of quoting them. Held-out spends are
        # logged: a held-out set consulted repeatedly while a prompt is tuned is a training
        # set with extra steps, and the log is what makes that visible afterwards.
        from flightdeck import scope_eval

        rows = scope_eval.load_gold()
        predict = (
            scope_eval.regex_predict if args.classifier == "regex" else scope_eval.judge_predict
        )
        # The spend is recorded inside evaluate() now, not here: a held-out look taken from
        # a REPL, a test, or a future script bypassed this call site entirely, which made
        # the log a convention rather than the peek-prevention mechanism it claims to be.
        report = scope_eval.evaluate(
            rows,
            predict,
            name=args.classifier,
            split=args.split,
            spend_dir=args.dir or DEFAULT_DIR,
        )
        emit(report, as_json=args.json)
        return 0 if report["verdict"] == "pass" else 1

    if args.action == "ingest":
        # Gate decisions become thin scope_records so the KPI layer reads something real.
        # Ingested rows carry verdict "silent": the gate fired, and nothing observed whether
        # the scoping that followed was any good. Calling that "pass" would manufacture the
        # measurement this subsystem exists to earn.
        from flightdeck.scope_ingest import ingest

        store = open_store(args)
        summary = ingest(store, directory=args.dir or None)
        emit(summary, as_json=args.json)
        return 0 if summary["rows_now"] else 1

    if args.action == "kpi":
        # aggregate() had no caller anywhere in the CLI: the module that decides whether the
        # scope stream is healthy was unreachable from the command line, so its three-valued
        # verdict could never reach a human or an exit code. health_verdict() is what turns a
        # long silence into a nonzero exit -- 'silent' alone stays 0, because a quiet week is
        # not a failure and false alarms are how a check gets ignored.
        from flightdeck.scope_kpi import aggregate, health_exit_code, health_verdict

        store = open_store(args)
        summary = aggregate(store.scope_records())
        history = [int(n) for n in (args.history or "").split(",") if n.strip()]
        verdict = health_verdict(summary, rows_history=history or None)
        summary["health_verdict"] = verdict
        emit(summary, as_json=args.json)
        return health_exit_code(verdict)

    if args.action == "chain":
        # The row hashes were written from the first commit and verified nowhere, which
        # makes them decoration: an edited or deleted row was undetectable.
        from flightdeck.scope_ingest import verify_chain

        result = verify_chain(open_store(args))
        emit(result, as_json=args.json)
        return 0 if result["verdict"] == "pass" else 1

    if args.action == "snapshot":
        # Pin the corpus a measurement was taken over. Retention is rolling, so a rate with
        # no addressable corpus behind it cannot be re-derived and is a claim, not a result.
        from flightdeck import scope_snapshot

        snapshot = scope_snapshot.take(note=args.prompt or "")
        path = scope_snapshot.save(snapshot, args.dir or DEFAULT_DIR)
        emit(
            {
                "corpus_sha": snapshot.corpus_sha,
                "session_files": snapshot.session_files,
                "total_files": snapshot.total_files,
                "path": str(path),
            },
            as_json=args.json,
        )
        return 0

    # `log` -- what the gate has actually been deciding. Zero rows is the interesting
    # answer, not an empty one: it means the hook is not firing, which is how a capture
    # layer sat wired to nothing while every other check stayed green.
    path = Path(args.dir or DEFAULT_DIR).expanduser() / "scope" / "gate-log.jsonl"
    cutoff = time.time() - args.since
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            stamp = row.get("ts", "")
            try:
                seen = dt.datetime.fromisoformat(stamp).timestamp()
            except ValueError:
                continue
            if seen >= cutoff:
                rows.append(row)

    if not rows:
        emit(
            {
                "verdict": "silent",
                "rows": 0,
                "path": str(path),
                "reason": "no gate decisions recorded in this window; is the hook wired?",
            },
            as_json=args.json,
        )
        return 1

    tiers: dict[str, int] = {}
    for row in rows:
        tiers[row.get("tier", "?")] = tiers.get(row.get("tier", "?"), 0) + 1
    ceremonial = sum(count for tier, count in tiers.items() if tier != "none")

    # zero_streak_verdict compares a TRAILING RUN of zeroes against ZERO_STREAK_LIMIT, so a
    # one-element history can never reach a limit of 3 -- the old `[ceremonial]` call was
    # mathematically incapable of returning 'fail'. It is the check-that-cannot-fail shape
    # this function exists to detect, wired to detect it in itself. Bucket by UTC day so the
    # streak means "three consecutive days the gate saw nothing ceremonial", not "three rows".
    per_day: dict[str, int] = {}
    for row in rows:
        stamp = str(row.get("ts", ""))[:10]
        tier = row.get("tier", "?")
        per_day.setdefault(stamp, 0)
        if tier != "none":
            per_day[stamp] += 1
    history = [per_day[day] for day in sorted(per_day)]

    emit(
        {
            "verdict": zero_streak_verdict(history),
            "rows": len(rows),
            "tiers": tiers,
            "ceremonial": ceremonial,
            "days": len(history),
            "history": history,
            "ceremonial_rate": round(ceremonial / len(rows), 4),
        },
        as_json=args.json,
    )
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    """The active ledger: who is working on what, where, right now.

    Liveness is heartbeat recency, never the status column -- a killed session, a crashed
    process, or a sleeping laptop never gets to deregister itself, so a registry that trusts
    a self-reported status fills up with ghosts that all claim to be running.
    """
    from flightdeck import ledger

    store = open_store(args)
    if args.action == "show":
        view = ledger.render(store, within_seconds=args.within)
        emit(view, as_json=args.json)
        # An empty ledger is not a failure: nothing running is a legitimate state. What must
        # not read as healthy is an empty ledger with no data behind it at all.
        return 0

    if args.action == "index":
        from flightdeck.ledger_content import reindex_all
        from flightdeck.ledger_repos import index_repos

        summary = {
            "content": reindex_all(store),
            "repos": index_repos(store, roots=tuple(args.roots or ("~/dev",))),
        }
        emit(summary, as_json=args.json)
        return 0

    if args.action == "search":
        from flightdeck.ledger_content import search

        emit({"query": args.query, "rows": search(store, args.query or "")}, as_json=args.json)
        return 0

    if args.action == "repos":
        from flightdeck.ledger_repos import interesting_repos

        emit({"repos": interesting_repos(store)}, as_json=args.json)
        return 0

    if args.action == "schedule":
        # Three schedulers, no previous surface that answered "what is scheduled". The
        # find that motivated it: a crontab holding five comments and zero job lines.
        from flightdeck.ledger_schedule import index_schedules, stale_or_failing

        summary = index_schedules(store, include_remote=not args.no_remote)
        summary["stale_or_failing"] = stale_or_failing(store)
        emit(summary, as_json=args.json)
        # Findings are not a crash: exit 0 so this can run unattended without a scheduler
        # treating "there is something to look at" as "the indexer broke".
        return 0

    return 1


def cmd_todo(args: argparse.Namespace) -> int:
    """The durable, cross-session todo queue -- not a mirror of the in-flight TodoWrite list."""
    from flightdeck import ledger

    store = open_store(args)
    if args.action == "add":
        scope = "global" if args.use_global else (args.scope or "local")
        todo_id = ledger.add_todo(
            store,
            text=args.text,
            scope=scope,
            repo=args.repo,
            source=args.source,
            key=args.key,
        )
        emit(todo_id, as_json=args.json)
        return 0

    if args.action == "list":
        # Unlike `add`, list has no reason to default to 'local' -- an unfiltered `todo list`
        # should show everything, same as ledger.list_todos' own scope=None default.
        rows = ledger.list_todos(store, scope=args.scope, repo=args.repo, status=args.status)
        if args.json:
            emit(rows, as_json=True)
            return 0
        for row in rows:
            age_s = (now_ms() - row["created_at"]) / 1000
            print(f"{row['todo_id']} · {row['scope']} · {age_s:.0f}s · {row['text']}")
        return 0

    if args.action == "done":
        if not args.todo_id:
            print("todo done requires a todo_id", file=sys.stderr)
            return 2
        ok = ledger.done_todo(store, args.todo_id)
        emit({"todo_id": args.todo_id, "done": ok}, as_json=args.json)
        return 0 if ok else 1

    return 1


def _read_plan_payload(args: argparse.Namespace) -> dict[str, Any]:
    """--file, or stdin when --file is omitted. A plain JSON object whose keys are add_plan's
    keyword arguments -- the CLI does no field-by-field parsing of its own."""
    text = Path(args.file).expanduser().read_text() if args.file else sys.stdin.read()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("plan payload must be a JSON object")
    return data


def _parse_plan_set_fields(pairs: list[str]) -> dict[str, Any]:
    """--set field=json, repeatable. Each value is JSON so `--set stages=[...]` and
    `--set goal=\"text\"` both round-trip the same way add_plan's fields already do."""
    fields: dict[str, Any] = {}
    for pair in pairs:
        name, sep, raw = pair.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--set expects field=json, got {pair!r}")
        fields[name] = json.loads(raw)
    return fields


def cmd_plan(args: argparse.Namespace) -> int:
    """A plan IS a todo -- stages are fields on it, never separate todo rows."""
    from flightdeck import ledger

    store = open_store(args)

    if args.action == "add":
        payload = _read_plan_payload(args)
        todo_id = ledger.add_plan(store, **payload)
        emit(todo_id, as_json=args.json)
        return 0

    if args.action == "show":
        plan = ledger.get_plan(store, args.todo_id)
        if plan is None:
            print(f"no plan for todo_id {args.todo_id}", file=sys.stderr)
            return 1
        emit(plan, as_json=True)
        return 0

    if args.action == "list":
        rows = ledger.list_plans(store, status=args.status)
        emit(rows, as_json=True)
        return 0

    if args.action == "amend":
        fields = _parse_plan_set_fields(args.set or [])
        try:
            new_revision = ledger.amend_plan(
                store,
                args.todo_id,
                expected_revision=args.revision,
                changed_by=args.by,
                note=args.note,
                **fields,
            )
        except (ledger.PlanConflict, ValueError) as exc:
            payload: dict[str, Any] = {"error": str(exc)}
            if isinstance(exc, ledger.PlanConflict):
                payload["current_revision"] = exc.actual
            emit(payload, as_json=args.json)
            return 1
        emit({"todo_id": args.todo_id, "revision": new_revision}, as_json=args.json)
        return 0

    if args.action == "status":
        try:
            new_revision = ledger.set_plan_status(
                store,
                args.todo_id,
                args.status_value,
                expected_revision=args.revision,
                changed_by=args.by,
                superseded_by=args.superseded_by,
                note=args.note,
            )
        except (ledger.PlanConflict, ValueError) as exc:
            payload = {"error": str(exc)}
            if isinstance(exc, ledger.PlanConflict):
                payload["current_revision"] = exc.actual
            emit(payload, as_json=args.json)
            return 1
        emit({"todo_id": args.todo_id, "revision": new_revision}, as_json=args.json)
        return 0

    if args.action == "check":
        plan = ledger.get_plan(store, args.todo_id)
        if plan is None:
            print(f"no plan for todo_id {args.todo_id}", file=sys.stderr)
            return 1
        problems = ledger.plan_problems(plan)
        emit(problems, as_json=args.json)
        return 1 if problems else 0

    if args.action == "header":
        print(ledger.plan_header(store, session_id=args.session, repo=args.repo))
        return 0

    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flightdeck", description=__doc__)
    parser.add_argument("--dir", type=Path, default=None, help="turnlog directory")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create the local store")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("rollup", help="replay logs, score turns, expire text, run probes")
    p.add_argument("--since", type=parse_duration, default=parse_duration("48h"))
    p.add_argument("--expire", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.set_defaults(func=cmd_rollup)

    p = sub.add_parser(
        "backfill-tier", help="fill turns.tier on existing rows where source/model derives it"
    )
    p.add_argument(
        "--since", type=parse_duration, default=None, help="only rows started within this window"
    )
    p.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    p.set_defaults(func=cmd_backfill_tier)

    p = sub.add_parser(
        "backfill-transcript",
        help="fill claude-code tokens/model/model_ms/outcome from Claude Code transcripts",
    )
    p.add_argument(
        "--since", type=parse_duration, default=None, help="only rows started within this window"
    )
    p.add_argument("--root", default=None, help="transcript corpus (default ~/.claude/projects)")
    p.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    p.set_defaults(func=cmd_backfill_transcript)

    p = sub.add_parser("doctor", help="is anything silently not collecting?")
    p.add_argument("--since", type=parse_duration, default=parse_duration("24h"))
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("sample", help="build the nightly review samples")
    p.add_argument("--since", type=parse_duration, default=parse_duration("24h"))
    p.add_argument("--complex", type=int, default=10)
    p.add_argument("--low", type=int, default=15)
    p.add_argument("--write-brief", action="store_true")
    p.add_argument("--brief-path", type=Path, default=None)
    p.set_defaults(func=cmd_sample)

    from flightdeck.synth import cli as synth_cli

    synth_cli.build_subparser(sub)

    p = sub.add_parser("ledger", help="the active ledger: sessions, memories, skills, repos")
    p.add_argument(
        "action",
        choices=["show", "index", "search", "repos", "schedule"],
        nargs="?",
        default="show",
    )
    p.add_argument("--within", type=int, default=180, help="heartbeat staleness, seconds")
    p.add_argument("--roots", nargs="*", default=None, help="repo roots to index")
    p.add_argument("--query", default="", help="substring to search memories and skills")
    p.add_argument("--no-remote", action="store_true", help="skip spark systemd timers")
    p.set_defaults(func=cmd_ledger)

    p = sub.add_parser("todo", help="the durable cross-session todo queue")
    p.add_argument("action", choices=["add", "list", "done"])
    p.add_argument("todo_id", nargs="?", default=None, help="for `done`")
    p.add_argument("--text", default=None, help="for `add`")
    p.add_argument("--scope", choices=["local", "global"], default=None)
    p.add_argument(
        "--global", dest="use_global", action="store_true", help="shorthand for --scope global"
    )
    p.add_argument("--repo", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--key", default=None, help="dedupe key: re-filing (source, key) is a no-op")
    p.add_argument("--status", default="open", help="for `list`; pass '' for every status")
    p.set_defaults(func=cmd_todo)

    p = sub.add_parser("plan", help="one plan = one todo; stages are fields on it")
    p.add_argument("action", choices=["add", "show", "list", "amend", "status", "check", "header"])
    p.add_argument("todo_id", nargs="?", default=None)
    p.add_argument("status_value", nargs="?", default=None, help="new status, for `plan status`")
    p.add_argument("--file", default=None, help="JSON plan payload; stdin if omitted (`add`)")
    p.add_argument("--revision", type=int, default=None, help="expected_revision for CAS writes")
    p.add_argument("--by", default=None, help="changed_by, for `amend`/`status`")
    p.add_argument("--note", default=None)
    p.add_argument("--set", action="append", default=None, help="field=json, repeatable (`amend`)")
    p.add_argument("--status", default=None, help="filter, for `plan list`")
    p.add_argument("--superseded-by", dest="superseded_by", default=None)
    p.add_argument("--session", default=None, help="for `plan header`")
    p.add_argument("--repo", default=None, help="for `plan header`")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("scope", help="the scoping gate: classify a prompt, or audit decisions")
    p.add_argument(
        "action",
        choices=[
            "classify",
            "log",
            "ingest",
            "chain",
            "snapshot",
            "eval",
            "learn",
            "registry",
            "kpi",
        ],
    )
    p.add_argument("--classifier", choices=["regex", "judge"], default="regex")
    p.add_argument("--registry", default="VAGUE_MARKERS", help="which registry to learn")
    p.add_argument("--write", action="store_true", help="apply the update; default is a dry run")
    p.add_argument("--root", default="~/.claude/projects", help="transcript corpus to mine")
    p.add_argument("--split", choices=["train", "holdout"], default="train")
    p.add_argument("--history", default="", help="comma-separated prior window row counts")
    p.add_argument("prompt", nargs="?", default="", help="prompt text, for `classify`")
    p.add_argument("--active", action="store_true", help="a scope already exists for the session")
    p.add_argument("--since", type=parse_duration, default=parse_duration("24h"))
    p.set_defaults(func=cmd_scope)

    p = sub.add_parser("show", help="everything recorded about one turn")
    p.add_argument("turn_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("kpi", help="KPI rollup")
    p.add_argument("--since", type=parse_duration, default=parse_duration("24h"))
    p.set_defaults(func=cmd_kpi)

    p = sub.add_parser("governor", help="shadow/live state and graduation evidence")
    p.add_argument("action", choices=["status", "report"], nargs="?", default="status")
    p.add_argument("--domain", default="model_tier")
    p.set_defaults(func=cmd_governor)

    p = sub.add_parser("guard", help="may the reviewer write these paths?")
    p.add_argument("--repo", default=".")
    p.add_argument("--check", nargs="+", required=True)
    p.set_defaults(func=cmd_guard)

    p = sub.add_parser("sync", help="merge other boxes' stores from the sync inbox")
    p.add_argument("--inbox", required=True)
    p.add_argument("--hosts-reached", type=int, default=0)
    p.add_argument("--hosts-missed", type=int, default=0)
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("collect", help="snapshot this host's CPU/RAM/GPU/disk into host_metrics")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser(
        "fleet-status", help="render fleet-status.md from host metrics + registry (M6 minimal)"
    )
    p.add_argument("--inbox", default=str(DEFAULT_DIR / "inbox"))
    p.add_argument("--out", default=None, help="defaults to flightdeck/state/fleet-status.md")
    p.set_defaults(func=cmd_fleet_status)

    p = sub.add_parser("verify", help="regression check on recent tuning changes")
    p.add_argument("--auto-revert", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("record-change", help="record a tuning change for later regression checking")
    p.add_argument("--domain", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--commit", default=None)
    p.add_argument("--evidence", default=None)
    p.set_defaults(func=cmd_record_change)

    p = sub.add_parser(
        "reverted", help="changes reverted for regressing KPI — the do-not-retry list"
    )
    p.set_defaults(func=cmd_reverted)

    p = sub.add_parser("judge", help="run the detached judge over flagged turns")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_judge)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
