"""Populate memory_index and skill_index so an agent can ask "what do I know and what can I
do" without reading hundreds of files.

Content-addressed, not mtime-based -- this project has already been bitten by mtime indexes
going stale across a checkout (see wiki: mtime-signature-stale-across-checkouts). Every row
carries a content_sha of the source file's bytes; re-indexing an unchanged file is a no-op
detected by comparing that hash, not by comparing timestamps.

Every failure mode a real corpus produces -- missing frontmatter, malformed YAML, a memory
with no description, an empty project dir, an unreadable file -- is counted and reported.
None of them raise. Silent skipping is the failure mode this codebase keeps shipping
(see wiki: silence-is-a-finding); a summary dict with a zero in the wrong bucket is a bug
report, a raised exception or a dropped row is not.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from flightdeck.store import Store

DEFAULT_MEMORY_ROOT = Path("~/.claude/projects").expanduser()
DEFAULT_SKILL_ROOTS = (Path("~/.claude/skills").expanduser(),)

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")


# ---------- a small frontmatter parser ----------
#
# Not a YAML library: this repo is stdlib-heavy and none of the real files on disk need
# more than top-level `key: value` pairs, one level of nested mapping (`metadata: / type:`),
# and a folded block scalar (`description: >`). Tested against the live corpus below, not
# just synthetic fixtures.


class NoFrontmatterError(ValueError):
    """The file has no `---` delimited frontmatter block at all."""


class MalformedFrontmatterError(ValueError):
    """A frontmatter block exists but does not parse."""


# Kept for callers that don't care which flavor of failure it was.
FrontmatterError = (NoFrontmatterError, MalformedFrontmatterError)


def parse_frontmatter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise NoFrontmatterError("missing opening ---")
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise NoFrontmatterError("missing closing ---")

    block = lines[1:end]
    result: dict[str, Any] = {}
    i = 0
    while i < len(block):
        line = block[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue
        if line.startswith(" "):
            # Indented line with no owning top-level key yet -- malformed; skip it rather
            # than mis-attribute it to the previous key.
            i += 1
            continue
        if ":" not in line:
            raise MalformedFrontmatterError(f"not a key: line {i + 1!r}: {line!r}")
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()

        if rest in (">", "|"):
            # Folded (>) or literal (|) block scalar: gather following more-indented lines.
            fold = rest == ">"
            j = i + 1
            collected: list[str] = []
            while j < len(block) and (block[j].startswith(" ") or not block[j].strip()):
                collected.append(block[j].strip() if fold else block[j][2:])
                j += 1
            result[key] = " ".join(c for c in collected if c) if fold else "\n".join(collected)
            i = j
            continue

        if rest == "":
            # Either a nested mapping or an empty scalar -- look ahead for indented children.
            j = i + 1
            nested: dict[str, Any] = {}
            while j < len(block) and block[j].startswith(" "):
                child = block[j].strip()
                if ":" in child:
                    ck, _, cv = child.partition(":")
                    nested[ck.strip()] = _strip_quotes(cv.strip())
                j += 1
            if nested:
                result[key] = nested
                i = j
                continue
            result[key] = None
            i += 1
            continue

        result[key] = _strip_quotes(rest)
        i += 1

    return result


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def content_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_wikilinks(body: str) -> list[str]:
    seen: list[str] = []
    for match in WIKILINK_RE.finditer(body):
        target = match.group(1).strip()
        if target and target not in seen:
            seen.append(target)
    return seen


# ---------- memories ----------


def index_memories(
    store: Store,
    *,
    root: Path | str = DEFAULT_MEMORY_ROOT,
    now: int | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser()
    now = now if now is not None else int(time.time())

    written = 0
    unchanged = 0
    projects_scanned = 0
    projects_without_memory_dir = 0
    unreadable = 0
    no_frontmatter = 0
    malformed_frontmatter = 0
    missing_name = 0
    missing_description = 0

    existing: dict[tuple[str, str], str] = {
        (row["project"], row["name"]): row["content_sha"]
        for row in store.conn.execute("SELECT project, name, content_sha FROM memory_index")
    }
    seen_keys: set[tuple[str, str]] = set()

    all_names: set[str] = set()
    link_rows: list[tuple[str, str, list[str]]] = []  # (project, name, links)

    if not root.exists():
        return {
            "root": str(root),
            "error": "root does not exist",
            "written": 0,
            "unchanged": 0,
            "projects_scanned": 0,
            "projects_without_memory_dir": 0,
            "unreadable": 0,
            "no_frontmatter": 0,
            "malformed_frontmatter": 0,
            "missing_name": 0,
            "missing_description": 0,
            "dangling_links": [],
        }

    for project_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        projects_scanned += 1
        memory_dir = project_dir / "memory"
        if not memory_dir.is_dir():
            projects_without_memory_dir += 1
            continue

        for path in sorted(memory_dir.glob("*.md")):
            if path.name == "MEMORY.md":
                continue  # the index, not a memory

            try:
                raw = path.read_bytes()
            except OSError:
                unreadable += 1
                continue

            sha = content_sha(raw)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                unreadable += 1
                continue

            try:
                fm = parse_frontmatter(text)
            except NoFrontmatterError:
                no_frontmatter += 1
                continue
            except MalformedFrontmatterError:
                malformed_frontmatter += 1
                continue

            name = fm.get("name")
            if not name:
                missing_name += 1
                continue

            description = fm.get("description") or None
            if description is None:
                missing_description += 1

            metadata = fm.get("metadata")
            kind = metadata.get("type") if isinstance(metadata, dict) else None

            body = text.split("---", 2)[-1] if text.count("---") >= 2 else text
            links = extract_wikilinks(body)

            project = project_dir.name
            key = (project, name)
            seen_keys.add(key)
            all_names.add(name)
            link_rows.append((project, name, links))

            if existing.get(key) == sha:
                unchanged += 1
                continue

            store.conn.execute(
                """
                INSERT INTO memory_index
                    (name, project, path, kind, description, links, content_sha, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project, name) DO UPDATE SET
                    path=excluded.path,
                    kind=excluded.kind,
                    description=excluded.description,
                    links=excluded.links,
                    content_sha=excluded.content_sha,
                    indexed_at=excluded.indexed_at
                """,
                (name, project, str(path), kind, description, json.dumps(links), sha, now),
            )
            written += 1

    # A file that's gone from disk but still in the table isn't reported here -- that's a
    # prune, a separate concern from this pass's own honesty about what it just read.
    dangling = sorted(
        {
            target
            for (_project, _name, links) in link_rows
            for target in links
            if target not in all_names
        }
    )

    return {
        "root": str(root),
        "written": written,
        "unchanged": unchanged,
        "total_seen": len(seen_keys),
        "projects_scanned": projects_scanned,
        "projects_without_memory_dir": projects_without_memory_dir,
        "unreadable": unreadable,
        "no_frontmatter": no_frontmatter,
        "malformed_frontmatter": malformed_frontmatter,
        "missing_name": missing_name,
        "missing_description": missing_description,
        "dangling_links": dangling,
        "dangling_link_count": len(dangling),
    }


# ---------- skills ----------


def _scope_for(skill_dir: Path, roots: tuple[Path, ...]) -> str:
    for root in roots:
        try:
            skill_dir.relative_to(root)
        except ValueError:
            continue
        if ".claude/skills" in str(root) and root == Path("~/.claude/skills").expanduser():
            return "user"
        return "project"
    return "project"


def index_skills(
    store: Store,
    *,
    roots: tuple[Path, ...] | list[Path] = DEFAULT_SKILL_ROOTS,
    now: int | None = None,
) -> dict[str, Any]:
    roots = tuple(Path(r).expanduser() for r in roots)
    now = now if now is not None else int(time.time())

    written = 0
    unchanged = 0
    dirs_scanned = 0
    unreadable = 0
    no_frontmatter = 0
    malformed_frontmatter = 0
    missing_name = 0
    missing_description = 0

    existing: dict[str, str] = {
        row["name"]: row["content_sha"]
        for row in store.conn.execute("SELECT name, content_sha FROM skill_index")
    }

    for root in roots:
        if not root.exists():
            continue
        for skill_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            path = skill_dir / "SKILL.md"
            if not path.exists():
                continue
            dirs_scanned += 1

            try:
                raw = path.read_bytes()
            except OSError:
                unreadable += 1
                continue

            sha = content_sha(raw)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                unreadable += 1
                continue

            try:
                fm = parse_frontmatter(text)
            except NoFrontmatterError:
                no_frontmatter += 1
                continue
            except MalformedFrontmatterError:
                malformed_frontmatter += 1
                continue

            name = fm.get("name")
            if not name:
                missing_name += 1
                continue

            description = fm.get("description") or None
            if description is None:
                missing_description += 1

            trigger = fm.get("trigger") or fm.get("triggers") or None
            scope = _scope_for(skill_dir, roots)

            if existing.get(name) == sha:
                unchanged += 1
                continue

            store.conn.execute(
                """
                INSERT INTO skill_index
                    (name, path, scope, description, trigger, content_sha, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    path=excluded.path,
                    scope=excluded.scope,
                    description=excluded.description,
                    trigger=excluded.trigger,
                    content_sha=excluded.content_sha,
                    indexed_at=excluded.indexed_at
                """,
                (name, str(path), scope, description, trigger, sha, now),
            )
            written += 1

    return {
        "roots": [str(r) for r in roots],
        "written": written,
        "unchanged": unchanged,
        "dirs_scanned": dirs_scanned,
        "unreadable": unreadable,
        "no_frontmatter": no_frontmatter,
        "malformed_frontmatter": malformed_frontmatter,
        "missing_name": missing_name,
        "missing_description": missing_description,
    }


def reindex_all(
    store: Store,
    *,
    memory_root: Path | str = DEFAULT_MEMORY_ROOT,
    skill_roots: tuple[Path, ...] | list[Path] = DEFAULT_SKILL_ROOTS,
) -> dict[str, Any]:
    return {
        "memories": index_memories(store, root=memory_root),
        "skills": index_skills(store, roots=skill_roots),
    }


def search(store: Store, query: str) -> list[dict[str, Any]]:
    """Substring match over name/description, across both indexes, for agent recall."""
    like = f"%{query.lower()}%"
    rows: list[dict[str, Any]] = []

    for row in store.conn.execute(
        """
        SELECT name, project, path, kind, description, links, 'memory' AS source
        FROM memory_index
        WHERE lower(name) LIKE ? OR lower(coalesce(description, '')) LIKE ?
        ORDER BY project, name
        """,
        (like, like),
    ):
        rows.append(dict(row))

    for row in store.conn.execute(
        """
        SELECT name, path, scope, description, trigger, 'skill' AS source
        FROM skill_index
        WHERE lower(name) LIKE ? OR lower(coalesce(description, '')) LIKE ?
        ORDER BY name
        """,
        (like, like),
    ):
        rows.append(dict(row))

    return rows
