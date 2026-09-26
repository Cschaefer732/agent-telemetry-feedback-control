"""host_metrics is the new table backing fleet-status: one row per host per collection tick.
UNIQUE(host, ts) is load-bearing — it's what lets merge_from's INSERT OR IGNORE dedup this table
the same way it dedups every other table, without a special case.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from flightdeck.schema import SCHEMA_VERSION, migrate


def test_host_metrics_table_created(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    migrate(conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(host_metrics)")}
    assert cols == {
        "host",
        "ts",
        "cpu_pct",
        "mem_used_mb",
        "mem_total_mb",
        "gpu_pct",
        "gpu_mem_used_mb",
        "disk_used_gb",
        "disk_total_gb",
    }


def test_host_metrics_unique_host_ts(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO host_metrics (host, ts, cpu_pct, mem_used_mb, mem_total_mb, "
        "gpu_pct, gpu_mem_used_mb, disk_used_gb, disk_total_gb) VALUES "
        "('spark', '2026-08-15T00:00:00Z', 10.0, 1000, 2000, 0.0, 0, 100.0, 200.0)"
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO host_metrics (host, ts, cpu_pct, mem_used_mb, mem_total_mb, "
            "gpu_pct, gpu_mem_used_mb, disk_used_gb, disk_total_gb) VALUES "
            "('spark', '2026-08-15T00:00:00Z', 99.0, 1000, 2000, 0.0, 0, 100.0, 200.0)"
        )
        conn.commit()


def test_schema_version_includes_host_metrics_migration() -> None:
    assert SCHEMA_VERSION >= 5


def test_governor_decisions_domain_index_created(tmp_path: Path) -> None:
    """Index-only migration (SCHEMA_VERSION 5->6): governor_decisions gets a (domain, turn_id)
    index so Governor.success_table's join can filter on domain without a full table scan — its
    primary key is (turn_id, domain), domain-last, so it couldn't serve that filter on its own."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    migrate(conn)
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(governor_decisions)")}
    assert "idx_governor_decisions_domain" in indexes
    cols = [row[2] for row in conn.execute("PRAGMA index_info(idx_governor_decisions_domain)")]
    assert cols == ["domain", "turn_id"]


def test_schema_version_includes_governor_decisions_index_migration() -> None:
    assert SCHEMA_VERSION >= 6
