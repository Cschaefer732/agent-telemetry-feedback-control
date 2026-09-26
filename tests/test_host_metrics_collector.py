from __future__ import annotations

import subprocess

import pytest

from flightdeck.collectors import host_metrics
from flightdeck.collectors.host_metrics import collect_host_metrics


def test_collect_host_metrics_returns_expected_shape() -> None:
    m = collect_host_metrics("test-host")
    assert m["host"] == "test-host"
    assert isinstance(m["ts"], str) and "T" in m["ts"]
    assert isinstance(m["cpu_pct"], float)
    assert 0.0 <= m["cpu_pct"] <= 100.0
    assert isinstance(m["mem_used_mb"], int)
    assert isinstance(m["mem_total_mb"], int)
    assert m["mem_used_mb"] <= m["mem_total_mb"]
    # GPU fields present but may be None on non-GPU hosts
    assert "gpu_pct" in m
    assert "gpu_mem_used_mb" in m
    assert isinstance(m["disk_used_gb"], float)
    assert isinstance(m["disk_total_gb"], float)


# ---------- _gpu_metrics robustness ----------


def test_gpu_metrics_none_when_nvidia_smi_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_metrics.shutil, "which", lambda name: None)

    util, mem = host_metrics._gpu_metrics()

    assert util is None
    assert mem is None


def test_gpu_metrics_none_when_nvidia_smi_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present-but-erroring nvidia-smi (driver hiccup, no GPU visible in this cgroup) must not
    raise out of the collector — it degrades to a NULL GPU reading."""
    monkeypatch.setattr(host_metrics.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def boom(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(host_metrics.subprocess, "run", boom)

    util, mem = host_metrics._gpu_metrics()

    assert util is None
    assert mem is None


def test_gpu_metrics_aggregates_multiple_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-GPU hosts (e.g. spark's dual GB10) must not silently drop every GPU past the
    first CSV line."""
    monkeypatch.setattr(host_metrics.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    class _Result:
        stdout = "10, 1000\n30, 2000\n"

    monkeypatch.setattr(host_metrics.subprocess, "run", lambda *a, **k: _Result())

    util, mem = host_metrics._gpu_metrics()

    assert util == 20.0  # average of 10 and 30
    assert mem == 3000  # sum of 1000 and 2000


def test_gpu_metrics_tolerates_na_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """GB10 unified-memory GPUs report memory.used as "[N/A]" — that is a NULL memory
    reading, not a collector crash, and must not take utilization down with it."""
    monkeypatch.setattr(host_metrics.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    class _Result:
        stdout = "45, [N/A]\n"

    monkeypatch.setattr(host_metrics.subprocess, "run", lambda *a, **k: _Result())

    util, mem = host_metrics._gpu_metrics()

    assert util == 45.0
    assert mem is None


def test_gpu_metrics_mixed_na_and_numeric_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_metrics.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    class _Result:
        stdout = "10, 1000\n30, [N/A]\n"

    monkeypatch.setattr(host_metrics.subprocess, "run", lambda *a, **k: _Result())

    util, mem = host_metrics._gpu_metrics()

    assert util == 20.0
    assert mem == 1000


def test_gpu_metrics_all_unparsable_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_metrics.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    class _Result:
        stdout = "[N/A], [N/A]\nnot,a,number\n"

    monkeypatch.setattr(host_metrics.subprocess, "run", lambda *a, **k: _Result())

    util, mem = host_metrics._gpu_metrics()

    assert util is None
    assert mem is None


# ---------- _disk_gb robustness ----------


def test_disk_gb_uses_disk_usage_used_not_total_minus_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """total-free double-counts filesystem-reserved blocks as used; shutil.disk_usage().used is
    the OS's own figure and must be what's reported."""

    class _Usage:
        total = 1_000_000_000
        used = 400_000_000
        free = 550_000_000  # total - free (600MB) != used (400MB): a reserved-blocks gap

    monkeypatch.setattr(host_metrics.shutil, "disk_usage", lambda path: _Usage())

    used_gb, total_gb = host_metrics._disk_gb()

    assert used_gb == round(400_000_000 / (1024**3), 1)
    assert total_gb == round(1_000_000_000 / (1024**3), 1)
