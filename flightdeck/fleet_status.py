"""Renders fleet-status.md — host metrics + project registry, no turn-activity narrative.

This is a separate, simpler artifact from the nightly review digest
(docs/reviews/YYYY-MM-DD.md). It gets its own path so the two never collide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightdeck.store import Store

STATE_DIR = Path(__file__).resolve().parent / "state"
FLEET_STATUS_PATH = STATE_DIR / "fleet-status.md"


def _latest_host_metrics(store: Store) -> list[dict[str, Any]]:
    rows = store.conn.execute(
        "SELECT h.* FROM host_metrics h "
        "JOIN (SELECT host, MAX(ts) AS ts FROM host_metrics GROUP BY host) latest "
        "ON h.host = latest.host AND h.ts = latest.ts "
        "ORDER BY h.host"
    ).fetchall()
    return [dict(row) for row in rows]


def _fmt(value: Any) -> str:
    return "—" if value is None else str(value)


def _render_metrics_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No host metrics recorded yet.", ""]
    lines = [
        "| Host | CPU % | Mem (MB) | GPU % | GPU Mem (MB) | Disk (GB) | Last seen |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        mem = f"{_fmt(row['mem_used_mb'])}/{_fmt(row['mem_total_mb'])}"
        disk = f"{_fmt(row['disk_used_gb'])}/{_fmt(row['disk_total_gb'])}"
        lines.append(
            f"| {row['host']} | {_fmt(row['cpu_pct'])} | {mem} | {_fmt(row['gpu_pct'])} | "
            f"{_fmt(row['gpu_mem_used_mb'])} | {disk} | {row['ts']} |"
        )
    lines.append("")
    return lines


def _render_registry(inbox_dir: Path) -> list[str]:
    if not inbox_dir.exists():
        return ["No registry data synced yet.", ""]
    lines: list[str] = []
    for host_dir in sorted(p for p in inbox_dir.iterdir() if p.is_dir()):
        registry_path = host_dir / "registry.json"
        if not registry_path.exists():
            continue
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A torn write from a killed rsync mid-transfer — skip it, don't fail the digest.
            continue
        # Producer is an external project-snapshot script that emits {"repos": [...]}
        # (its own OUT=$STATE/projects/registry.json) — not "projects".
        projects = data.get("repos", []) if isinstance(data, dict) else data
        if not projects:
            continue
        lines.append(f"### {host_dir.name}")
        lines.append("")
        for project in projects:
            name = project.get("name", "?") if isinstance(project, dict) else str(project)
            path = project.get("path") if isinstance(project, dict) else None
            lines.append(f"- **{name}**" + (f" — `{path}`" if path else ""))
        lines.append("")
    return lines or ["No registry data synced yet.", ""]


def render_fleet_status(store: Store, inbox_dir: Path) -> str:
    lines = [
        "# Fleet Status",
        "",
        "## Host Metrics",
        "",
        *_render_metrics_table(_latest_host_metrics(store)),
        "## Registry",
        "",
        *_render_registry(Path(inbox_dir)),
    ]
    return "\n".join(lines).rstrip() + "\n"
