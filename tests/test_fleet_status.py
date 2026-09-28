from __future__ import annotations

import json
from pathlib import Path

from flightdeck.fleet_status import render_fleet_status
from flightdeck.store import Store


def _insert_host_metrics(store: Store, host: str, ts: str = "2026-08-15T11:55:00Z") -> None:
    store.conn.execute(
        "INSERT INTO host_metrics (host, ts, cpu_pct, mem_used_mb, mem_total_mb, "
        "gpu_pct, gpu_mem_used_mb, disk_used_gb, disk_total_gb) VALUES (?,?,?,?,?,?,?,?,?)",
        (host, ts, 42.0, 8000, 16000, None, None, 100.0, 400.0),
    )
    store.conn.commit()


def test_render_includes_each_host_with_metrics(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.db")
    _insert_host_metrics(store, "spark")
    _insert_host_metrics(store, "laptop")
    md = render_fleet_status(store, inbox_dir=tmp_path)
    assert "spark" in md
    assert "laptop" in md
    assert "42.0" in md  # cpu_pct surfaced


def test_render_handles_host_with_no_metrics_gracefully(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.db")
    md = render_fleet_status(store, inbox_dir=tmp_path)
    assert "No host metrics" in md or md.strip() != ""


def test_render_only_surfaces_latest_row_per_host(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.db")
    _insert_host_metrics(store, "spark", ts="2026-08-15T10:00:00Z")
    store.conn.execute(
        "INSERT INTO host_metrics (host, ts, cpu_pct, mem_used_mb, mem_total_mb, "
        "gpu_pct, gpu_mem_used_mb, disk_used_gb, disk_total_gb) VALUES (?,?,?,?,?,?,?,?,?)",
        ("spark", "2026-08-15T11:55:00Z", 77.0, 9000, 16000, None, None, 100.0, 400.0),
    )
    store.conn.commit()
    md = render_fleet_status(store, inbox_dir=tmp_path)
    assert "77.0" in md
    assert "42.0" not in md


def test_render_includes_registry_when_present(tmp_path: Path) -> None:
    """Real producer key is "repos", not "projects" — see an external
    project-snapshot script that writes this file."""
    store = Store(tmp_path / "test.db")
    _insert_host_metrics(store, "spark")
    host_dir = tmp_path / "spark"
    host_dir.mkdir()
    (host_dir / "registry.json").write_text(
        '{"repos": [{"name": "agent-telemetry-feedback-control", "path": "~/dev/agent-telemetry-feedback-control"}]}',
        encoding="utf-8",
    )
    md = render_fleet_status(store, inbox_dir=tmp_path)
    assert "agent-telemetry-feedback-control" in md


def test_render_includes_registry_real_producer_shape(tmp_path: Path) -> None:
    """The exact record shape snapshot-projects.sh's repo_json() emits (name/path plus git
    status fields) must render without error and surface the repo name."""
    store = Store(tmp_path / "test.db")
    _insert_host_metrics(store, "spark")
    host_dir = tmp_path / "spark"
    host_dir.mkdir()
    (host_dir / "registry.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-08-15T12:00:00Z",
                "root": "/home/user/dev",
                "repos": [
                    {
                        "name": "agent-telemetry-feedback-control",
                        "path": "/home/user/dev/agent-telemetry-feedback-control",
                        "branch": "main",
                        "dirty": True,
                        "uncommittedChanges": 3,
                        "ahead": 1,
                        "behind": 0,
                        "lastCommit": "abc1234",
                        "lastCommitSubject": "fix: registry key",
                        "lastCommitDate": "2026-08-15T11:00:00Z",
                        "todoFiles": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    md = render_fleet_status(store, inbox_dir=tmp_path)
    assert "agent-telemetry-feedback-control" in md
    assert "/home/user/dev/agent-telemetry-feedback-control" in md
