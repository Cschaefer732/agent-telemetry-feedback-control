from __future__ import annotations

from pathlib import Path

from flightdeck.ledger_content import (
    index_memories,
    index_skills,
    reindex_all,
    search,
)
from flightdeck.store import Store


def _write_memory(root: Path, project: str, filename: str, body: str) -> Path:
    d = root / project / "memory"
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    path.write_text(body)
    return path


GOOD_MEMORY = """---
name: {name}
description: {description}
metadata:
  type: project
---

Body text{links}
"""


def _memory(name: str, description: str = "a memory", links: str = "") -> str:
    return GOOD_MEMORY.format(name=name, description=description, links=links)


def test_index_memories_basic(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    _write_memory(root, "proj-a", "one.md", _memory("one", "first memory"))
    _write_memory(root, "proj-a", "two.md", _memory("two", "second memory"))

    with Store(directory=tmp_path / "store") as store:
        summary = index_memories(store, root=root)
        assert summary["written"] == 2
        assert summary["unchanged"] == 0
        rows = store.conn.execute("SELECT * FROM memory_index ORDER BY name").fetchall()
        assert [r["name"] for r in rows] == ["one", "two"]
        assert rows[0]["project"] == "proj-a"
        assert rows[0]["description"] == "first memory"
        assert rows[0]["kind"] == "project"


def test_memory_index_file_is_not_ingested(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    _write_memory(root, "proj-a", "one.md", _memory("one"))
    (root / "proj-a" / "memory" / "MEMORY.md").write_text("# index\n- [one](one.md)\n")

    with Store(directory=tmp_path / "store") as store:
        summary = index_memories(store, root=root)
        assert summary["written"] == 1
        names = {r["name"] for r in store.conn.execute("SELECT name FROM memory_index")}
        assert "MEMORY" not in names


def test_idempotent_reindex_is_noop(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    _write_memory(root, "proj-a", "one.md", _memory("one"))

    with Store(directory=tmp_path / "store") as store:
        first = index_memories(store, root=root)
        assert first["written"] == 1

        before = dict(
            store.conn.execute(
                "SELECT content_sha, indexed_at FROM memory_index WHERE name='one'"
            ).fetchone()
        )

        second = index_memories(store, root=root)
        assert second["written"] == 0
        assert second["unchanged"] == 1

        after = dict(
            store.conn.execute(
                "SELECT content_sha, indexed_at FROM memory_index WHERE name='one'"
            ).fetchone()
        )
        assert before == after  # no churn: same hash, same indexed_at

        rows = store.conn.execute("SELECT COUNT(*) AS n FROM memory_index").fetchone()
        assert rows["n"] == 1  # no duplicate row


def test_changed_content_reindexes(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    path = _write_memory(root, "proj-a", "one.md", _memory("one", "v1"))

    with Store(directory=tmp_path / "store") as store:
        index_memories(store, root=root)
        path.write_text(_memory("one", "v2"))
        summary = index_memories(store, root=root)
        assert summary["written"] == 1
        row = store.conn.execute("SELECT description FROM memory_index WHERE name='one'").fetchone()
        assert row["description"] == "v2"


def test_dangling_wikilink_reported_not_raised(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    _write_memory(
        root,
        "proj-a",
        "one.md",
        _memory("one", "refs another", links="\n\nSee [[does-not-exist]] for more."),
    )

    with Store(directory=tmp_path / "store") as store:
        summary = index_memories(store, root=root)
        assert summary["written"] == 1
        assert "does-not-exist" in summary["dangling_links"]

        row = store.conn.execute("SELECT links FROM memory_index WHERE name='one'").fetchone()
        assert "does-not-exist" in row["links"]


def test_valid_link_is_not_dangling(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    _write_memory(root, "proj-a", "one.md", _memory("one", links="\n\n[[two]]"))
    _write_memory(root, "proj-a", "two.md", _memory("two"))

    with Store(directory=tmp_path / "store") as store:
        summary = index_memories(store, root=root)
        assert summary["dangling_links"] == []


def test_malformed_frontmatter_counted_not_raised(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    d = root / "proj-a" / "memory"
    d.mkdir(parents=True)
    (d / "broken.md").write_text("---\nnot a key value pair at all\n---\nbody\n")
    _write_memory(root, "proj-a", "good.md", _memory("good"))

    with Store(directory=tmp_path / "store") as store:
        summary = index_memories(store, root=root)
        assert summary["malformed_frontmatter"] == 1
        assert summary["written"] == 1  # good.md still indexed


def test_no_frontmatter_counted_not_raised(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    d = root / "proj-a" / "memory"
    d.mkdir(parents=True)
    (d / "plain.md").write_text("just a markdown file, no frontmatter at all\n")

    with Store(directory=tmp_path / "store") as store:
        summary = index_memories(store, root=root)
        assert summary["no_frontmatter"] == 1
        assert summary["written"] == 0


def test_missing_description_counted_and_left_null(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    d = root / "proj-a" / "memory"
    d.mkdir(parents=True)
    (d / "nodesc.md").write_text("---\nname: nodesc\n---\nbody\n")

    with Store(directory=tmp_path / "store") as store:
        summary = index_memories(store, root=root)
        assert summary["missing_description"] == 1
        row = store.conn.execute(
            "SELECT description FROM memory_index WHERE name='nodesc'"
        ).fetchone()
        assert row["description"] is None


def test_project_dir_with_no_memory_subdir(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    (root / "empty-proj").mkdir(parents=True)
    _write_memory(root, "proj-a", "one.md", _memory("one"))

    with Store(directory=tmp_path / "store") as store:
        summary = index_memories(store, root=root)
        assert summary["projects_without_memory_dir"] == 1
        assert summary["written"] == 1


def test_missing_root_reported_not_raised(tmp_path: Path) -> None:
    with Store(directory=tmp_path / "store") as store:
        summary = index_memories(store, root=tmp_path / "does-not-exist")
        assert summary["written"] == 0
        assert "error" in summary


def test_unreadable_file_counted(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    d = root / "proj-a" / "memory"
    d.mkdir(parents=True)
    path = d / "binary.md"
    path.write_bytes(b"---\nname: x\n---\n\xff\xfe\x00\x80not utf8")

    with Store(directory=tmp_path / "store") as store:
        summary = index_memories(store, root=root)
        assert summary["unreadable"] == 1


# ---------- skills ----------


SKILL_MD = """---
name: {name}
description: {description}
triggers: {triggers}
---

# {name}
"""


def _skill(name: str, description: str = "does a thing", triggers: str = "do the thing") -> str:
    return SKILL_MD.format(name=name, description=description, triggers=triggers)


def test_index_skills_basic(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "SKILL.md").write_text(_skill("alpha"))

    with Store(directory=tmp_path / "store") as store:
        summary = index_skills(store, roots=[root])
        assert summary["written"] == 1
        row = store.conn.execute("SELECT * FROM skill_index WHERE name='alpha'").fetchone()
        assert row["description"] == "does a thing"
        assert row["trigger"] == "do the thing"
        assert row["scope"] == "project"


def test_index_skills_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "SKILL.md").write_text(_skill("alpha"))

    with Store(directory=tmp_path / "store") as store:
        index_skills(store, roots=[root])
        second = index_skills(store, roots=[root])
        assert second["written"] == 0
        assert second["unchanged"] == 1
        rows = store.conn.execute("SELECT COUNT(*) AS n FROM skill_index").fetchone()
        assert rows["n"] == 1


def test_index_skills_dir_without_skill_md_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    (root / "empty").mkdir(parents=True)
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "SKILL.md").write_text(_skill("alpha"))

    with Store(directory=tmp_path / "store") as store:
        summary = index_skills(store, roots=[root])
        assert summary["written"] == 1
        assert summary["dirs_scanned"] == 1


def test_reindex_all_covers_both(tmp_path: Path) -> None:
    proj_root = tmp_path / "projects"
    _write_memory(proj_root, "proj-a", "one.md", _memory("one"))
    skill_root = tmp_path / "skills"
    (skill_root / "alpha").mkdir(parents=True)
    (skill_root / "alpha" / "SKILL.md").write_text(_skill("alpha"))

    with Store(directory=tmp_path / "store") as store:
        summary = reindex_all(store, memory_root=proj_root, skill_roots=[skill_root])
        assert summary["memories"]["written"] == 1
        assert summary["skills"]["written"] == 1


def test_search_matches_name_and_description(tmp_path: Path) -> None:
    proj_root = tmp_path / "projects"
    _write_memory(proj_root, "proj-a", "one.md", _memory("one", "about widgets"))
    skill_root = tmp_path / "skills"
    (skill_root / "widget-skill").mkdir(parents=True)
    (skill_root / "widget-skill" / "SKILL.md").write_text(_skill("widget-skill", "handles widgets"))

    with Store(directory=tmp_path / "store") as store:
        reindex_all(store, memory_root=proj_root, skill_roots=[skill_root])
        results = search(store, "widget")
        sources = {r["source"] for r in results}
        assert sources == {"memory", "skill"}
        assert len(results) == 2
