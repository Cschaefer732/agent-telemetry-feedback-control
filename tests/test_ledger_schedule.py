from __future__ import annotations

from flightdeck.ledger_schedule import index_schedules, stale_or_failing
from flightdeck.store import Store


def _rows(store, table):
    return [dict(r) for r in store.conn.execute(f"SELECT * FROM {table}").fetchall()]


def test_unreachable_host_records_reachable_zero_and_null_jobs(tmp_path):
    store = Store(tmp_path)

    def dead_systemd():
        return {"reachable": False, "error": "ssh to user@review-host timed out", "jobs": []}

    counts = index_schedules(
        store,
        hosts={
            "launchd": lambda: {"reachable": True, "error": None, "jobs": []},
            "cron": lambda: {"reachable": True, "error": None, "jobs": [], "orphaned_comments": []},
            "systemd": dead_systemd,
        },
    )
    assert counts["unreachable"]["systemd"] == 1

    sources = {(r["source"], r["host"]): r for r in _rows(store, "schedule_sources")}
    row = sources[("systemd", "review-host")]
    assert row["reachable"] == 0
    assert row["jobs_found"] is None  # never 0 -- unreachable is not "zero jobs"
    assert "timed out" in row["error"]


def test_orphaned_cron_comments_reported_not_swallowed(tmp_path):
    store = Store(tmp_path)

    def cron_only_comments():
        return {
            "reachable": True,
            "error": None,
            "jobs": [],
            "orphaned_comments": [
                "nightly backup",
                "weekly cleanup",
                "hourly sync",
                "daily report",
                "log rotate",
            ],
        }

    index_schedules(
        store,
        hosts={
            "launchd": lambda: {"reachable": True, "error": None, "jobs": []},
            "cron": cron_only_comments,
            "systemd": lambda: {"reachable": True, "error": None, "jobs": []},
        },
    )

    sources = {(r["source"], r["host"]): r for r in _rows(store, "schedule_sources")}
    cron_row = next(r for (s, _h), r in sources.items() if s == "cron")
    assert cron_row["reachable"] == 1
    assert cron_row["jobs_found"] == 0
    assert cron_row["error"].startswith("orphaned_comments:")
    for desc in ["nightly backup", "weekly cleanup", "hourly sync", "daily report", "log rotate"]:
        assert desc in cron_row["error"]

    findings = stale_or_failing(store)
    assert any("crontab describes work it does not run" in r["reasons"][0] for r in findings)


def test_stale_systemd_timer_detected(tmp_path):
    store = Store(tmp_path)

    def one_stale_timer():
        return {
            "reachable": True,
            "error": None,
            "jobs": [
                {
                    "name": "sparky-nightly.timer",
                    "activates": "sparky-nightly.service",
                    "schedule": None,
                    "command": "sparky-nightly.service",
                    "enabled": 1,
                    "status": "waiting",
                    "left_text": "12h left",
                    "passed_text": "9d ago",
                }
            ],
        }

    index_schedules(
        store,
        hosts={
            "launchd": lambda: {"reachable": True, "error": None, "jobs": []},
            "cron": lambda: {"reachable": True, "error": None, "jobs": [], "orphaned_comments": []},
            "systemd": one_stale_timer,
        },
    )

    findings = stale_or_failing(store)
    stale = [f for f in findings if f.get("name") == "sparky-nightly.timer"]
    assert len(stale) == 1
    assert any("far older than its stated interval" in reason for reason in stale[0]["reasons"])


def test_failed_unit_detected(tmp_path):
    store = Store(tmp_path)

    def one_failed_timer():
        return {
            "reachable": True,
            "error": None,
            "jobs": [
                {
                    "name": "broken.timer",
                    "activates": "broken.service",
                    "schedule": None,
                    "command": "broken.service",
                    "enabled": 1,
                    "status": "failed",
                    "last_exit": 1,
                }
            ],
        }

    index_schedules(
        store,
        hosts={
            "launchd": lambda: {"reachable": True, "error": None, "jobs": []},
            "cron": lambda: {"reachable": True, "error": None, "jobs": [], "orphaned_comments": []},
            "systemd": one_failed_timer,
        },
    )

    findings = stale_or_failing(store)
    failed = [f for f in findings if f.get("name") == "broken.timer"]
    assert len(failed) == 1
    assert "status=failed" in failed[0]["reasons"]
    assert "last_exit=1" in failed[0]["reasons"]


def test_plist_on_disk_not_loaded_detected(tmp_path):
    store = Store(tmp_path)

    def launchd_with_installed_not_loaded():
        return {
            "reachable": True,
            "error": None,
            "jobs": [
                {
                    "name": "com.sparky.dreamer",
                    "loaded": False,
                    "pid": None,
                    "last_exit": None,
                    "status": "unknown",
                    "schedule": "RunAtLoad",
                    "command": "/usr/local/bin/dreamer",
                    "note": "installed_not_loaded",
                    "enabled": 0,
                    "own": True,
                }
            ],
        }

    index_schedules(
        store,
        hosts={
            "launchd": launchd_with_installed_not_loaded,
            "cron": lambda: {"reachable": True, "error": None, "jobs": [], "orphaned_comments": []},
            "systemd": lambda: {"reachable": True, "error": None, "jobs": []},
        },
    )

    findings = stale_or_failing(store)
    matches = [f for f in findings if f.get("name") == "com.sparky.dreamer"]
    assert len(matches) == 1
    assert "installed on disk but not loaded" in matches[0]["reasons"]


def test_loaded_no_plist_on_disk_detected(tmp_path):
    store = Store(tmp_path)

    def launchd_orphan_loaded():
        return {
            "reachable": True,
            "error": None,
            "jobs": [
                {
                    "name": "com.google.keystone.agent",
                    "loaded": True,
                    "pid": 123,
                    "last_exit": 0,
                    "status": "running",
                    "schedule": None,
                    "command": None,
                    "note": "loaded_no_plist_on_disk",
                    "enabled": 1,
                    "own": False,
                }
            ],
        }

    index_schedules(
        store,
        hosts={
            "launchd": launchd_orphan_loaded,
            "cron": lambda: {"reachable": True, "error": None, "jobs": [], "orphaned_comments": []},
            "systemd": lambda: {"reachable": True, "error": None, "jobs": []},
        },
    )

    findings = stale_or_failing(store)
    matches = [f for f in findings if f.get("name") == "com.google.keystone.agent"]
    assert len(matches) == 1
    assert "loaded but no plist on disk" in matches[0]["reasons"]


def test_idempotent_no_duplicate_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("FLIGHTDECK_SECONDARY_HOST", "secondary-host")
    store = Store(tmp_path)

    def launchd():
        return {
            "reachable": True,
            "error": None,
            "jobs": [
                {
                    "name": "com.sparky.dreamer",
                    "loaded": True,
                    "pid": 1,
                    "last_exit": 0,
                    "status": "running",
                    "schedule": "RunAtLoad",
                    "command": "/usr/local/bin/dreamer",
                    "note": None,
                    "enabled": 1,
                    "own": True,
                }
            ],
        }

    hosts = {
        "launchd": launchd,
        "cron": lambda: {"reachable": True, "error": None, "jobs": [], "orphaned_comments": []},
        "systemd": lambda: {"reachable": True, "error": None, "jobs": []},
        # Stubbed too: without it this test ssh'd to a real, currently-unreachable host and
        # spent its ConnectTimeout doing it. A unit test must not depend on the network.
        "cron@secondary-host": lambda: {
            "reachable": False,
            "error": "stubbed unreachable",
            "jobs": [],
        },
    }

    index_schedules(store, hosts=hosts)
    index_schedules(store, hosts=hosts)

    jobs = _rows(store, "scheduled_jobs")
    sources = _rows(store, "schedule_sources")
    assert len(jobs) == 1
    assert len(sources) == 4

    # The unreachable host must record NULL jobs, never 0 -- that distinction is the
    # entire reason schedule_sources is a separate table.
    secondary = [r for r in sources if r["host"] == "secondary-host"][0]
    assert secondary["reachable"] == 0
    assert secondary["jobs_found"] is None


def test_include_remote_false_skips_systemd(tmp_path):
    store = Store(tmp_path)
    calls = []

    def cron():
        calls.append("cron")
        return {"reachable": True, "error": None, "jobs": [], "orphaned_comments": []}

    index_schedules(
        store,
        include_remote=False,
        hosts={
            "launchd": lambda: {"reachable": True, "error": None, "jobs": []},
            "cron": cron,
        },
    )

    sources = _rows(store, "schedule_sources")
    assert not any(r["source"] == "systemd" for r in sources)
