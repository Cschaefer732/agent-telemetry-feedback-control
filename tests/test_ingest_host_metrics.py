"""ingest_jsonl must dispatch `host_metrics` records into the host_metrics table the same way it
already dispatches turn/event/text/judgment/decision/probe/tuning_change — so the collector
(flightdeck/collectors/host_metrics.py) flows through the existing JSONL->sqlite path rather than
needing a second ingest mechanism.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flightdeck.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path)


def _write_log(tmp_path: Path, *lines: str) -> Path:
    log = tmp_path / "events-2026-08-15.jsonl"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


def test_ingest_jsonl_writes_host_metrics_row(store: Store, tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        '{"_kind":"host_metrics","host":"spark","ts":"2026-08-15T00:00:00+00:00",'
        '"cpu_pct":12.5,"mem_used_mb":8000,"mem_total_mb":16000,"gpu_pct":5.0,'
        '"gpu_mem_used_mb":1000,"disk_used_gb":100.0,"disk_total_gb":500.0}',
    )
    counts = store.ingest_jsonl(log)
    assert counts.get("host_metrics") == 1
    row = store.conn.execute("SELECT * FROM host_metrics WHERE host='spark'").fetchone()
    assert row is not None
    assert row["cpu_pct"] == 12.5
    assert row["mem_used_mb"] == 8000
    assert row["gpu_pct"] == 5.0


def test_ingest_jsonl_host_metrics_handles_no_gpu(store: Store, tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        '{"_kind":"host_metrics","host":"laptop","ts":"2026-08-15T00:00:00+00:00",'
        '"cpu_pct":3.0,"mem_used_mb":4000,"mem_total_mb":16000,"gpu_pct":null,'
        '"gpu_mem_used_mb":null,"disk_used_gb":50.0,"disk_total_gb":250.0}',
    )
    store.ingest_jsonl(log)
    row = store.conn.execute("SELECT * FROM host_metrics WHERE host='laptop'").fetchone()
    assert row["gpu_pct"] is None
    assert row["gpu_mem_used_mb"] is None


def test_ingest_jsonl_host_metrics_is_idempotent(store: Store, tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        '{"_kind":"host_metrics","host":"spark","ts":"2026-08-15T00:00:00+00:00",'
        '"cpu_pct":12.5,"mem_used_mb":8000,"mem_total_mb":16000,"gpu_pct":null,'
        '"gpu_mem_used_mb":null,"disk_used_gb":100.0,"disk_total_gb":500.0}',
    )
    store.ingest_jsonl(log)
    store.ingest_jsonl(log)
    count = store.conn.execute("SELECT COUNT(*) FROM host_metrics WHERE host='spark'").fetchone()[0]
    assert count == 1
