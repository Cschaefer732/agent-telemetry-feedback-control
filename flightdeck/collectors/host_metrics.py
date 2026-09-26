"""Point-in-time host resource metrics — CPU/RAM/GPU/disk — for fleet-status.

Reads live system state at call time; consumes nothing external at build time. Feeds the
`host_metrics` table (schema v5) through the same JSONL→ingest path as every other record kind.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

MEMINFO_PATH = Path("/proc/meminfo")


def _read_meminfo_linux() -> tuple[int, int]:
    total_kb = avail_kb = 0
    for line in MEMINFO_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            avail_kb = int(line.split()[1])
    used_mb = (total_kb - avail_kb) // 1024
    total_mb = total_kb // 1024
    return used_mb, total_mb


def _read_mem_macos() -> tuple[int, int]:
    total_bytes = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    vm_stat = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page_size = 4096
    pages_active = pages_wired = 0
    for line in vm_stat.splitlines():
        if "page size of" in line:
            page_size = int(line.split()[-2])
        elif line.startswith("Pages active:"):
            pages_active = int(line.split()[-1].rstrip("."))
        elif line.startswith("Pages wired down:"):
            pages_wired = int(line.split()[-1].rstrip("."))
    used_mb = (pages_active + pages_wired) * page_size // (1024 * 1024)
    total_mb = total_bytes // (1024 * 1024)
    return used_mb, total_mb


def _cpu_pct() -> float:
    load1 = os.getloadavg()[0]
    ncpu = os.cpu_count() or 1
    return round(min(load1 / ncpu * 100.0, 100.0), 1)


def _disk_gb(path: str = "/") -> tuple[float, float]:
    usage = shutil.disk_usage(path)
    # total-free double-counts filesystem-reserved blocks (e.g. ext4's 5% root reserve) as
    # "used" by us; .used is what the OS itself considers occupied.
    used_gb = round(usage.used / (1024**3), 1)
    total_gb = round(usage.total / (1024**3), 1)
    return used_gb, total_gb


def _gpu_metrics() -> tuple[float | None, int | None]:
    """Averages utilization and sums memory across all GPUs nvidia-smi reports (multi-GPU hosts,
    e.g. spark's GB10 pair) rather than reading only the first CSV line. A present-but-erroring
    nvidia-smi (driver hiccup, no GPU visible in this cgroup) is a NULL reading, not a collector
    crash — GPU is one signal among several this collector reports.
    """
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return None, None

    lines = [line for line in out.strip().splitlines() if line.strip()]
    if not lines:
        return None, None

    # GB10/unified-memory GPUs report "[N/A]" for memory.used — a NULL field, not a crash.
    utils: list[float] = []
    mems: list[int] = []
    for line in lines:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        with contextlib.suppress(ValueError):
            utils.append(float(parts[0].strip()))
        with contextlib.suppress(ValueError):
            mems.append(int(parts[1].strip()))

    util_avg = round(sum(utils) / len(utils), 1) if utils else None
    mem_total = sum(mems) if mems else None
    return util_avg, mem_total


def collect_host_metrics(host: str) -> dict:
    """Returns exactly the host_metrics schema columns (see flightdeck/schema.py migration 5)."""
    if platform.system() == "Darwin":
        mem_used_mb, mem_total_mb = _read_mem_macos()
    else:
        mem_used_mb, mem_total_mb = _read_meminfo_linux()
    gpu_pct, gpu_mem_used_mb = _gpu_metrics()
    disk_used_gb, disk_total_gb = _disk_gb()
    return {
        "host": host,
        "ts": datetime.now(UTC).isoformat(),
        "cpu_pct": _cpu_pct(),
        "mem_used_mb": mem_used_mb,
        "mem_total_mb": mem_total_mb,
        "gpu_pct": gpu_pct,
        "gpu_mem_used_mb": gpu_mem_used_mb,
        "disk_used_gb": disk_used_gb,
        "disk_total_gb": disk_total_gb,
    }
