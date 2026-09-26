"""Single source of truth for what "synthetic" means at the row/file/store/KPI layers.

# generator

No import of `flightdeck.scope_gate` or `flightdeck.scope_judge` is permitted anywhere in
`flightdeck/synth/` -- see `tests/test_synth.py::test_no_classifier_import` for the
mechanical enforcement (an ast-walk over this package's own source). A generator that
imports the classifier it stands in for can silently launder a circular label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "1"  # bump on any change to generation logic; part of the hash inputs

#: Every label must name where it came from. "classifier_self" is deliberately absent --
#: a row cannot even be represented as self-labeled by the thing it evaluates.
LABEL_SOURCES = ("human_confirmed", "llm_crosschecked", "generator_asserted", "template_intent")


@dataclass(frozen=True)
class SyntheticProvenance:
    synthetic: bool = True
    generator: str = "flightdeck.synth"
    generator_version: str = GENERATOR_VERSION
    manifest_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def stamp_row(
    provenance_dict: dict[str, Any] | None = None, *, manifest_hash: str = ""
) -> dict[str, Any]:
    """Merge synthetic provenance into an existing provenance dict. Refuses to override a
    caller-supplied `synthetic: False` -- that would silently launder a real row as fake or
    vice versa, so fail loud instead."""
    provenance_dict = dict(provenance_dict or {})
    if provenance_dict.get("synthetic") is False:
        raise ValueError("refusing to stamp synthetic=True over an explicit synthetic=False")
    merged = SyntheticProvenance(manifest_hash=manifest_hash).as_dict()
    merged.update(provenance_dict)
    merged["synthetic"] = True
    return merged


def is_default_store_dir(directory: Path | str, default_dir: Path) -> bool:
    """True if `directory` resolves to the live store. Compared by realpath, not string
    equality, so a symlink or relative path can't sneak past the guard."""
    return Path(directory).expanduser().resolve() == Path(default_dir).expanduser().resolve()


def refuse_if_live_store(
    directory: Path | str, default_dir: Path, *, what: str = "synthetic data"
) -> None:
    """The generalized guard sample_data.py used to apply only inside `generate()`. Every
    write path in this package calls this before touching a Store.

    NOTE for a future reader of flightdeck/store.py: this same check belongs INSIDE
    `Store.add_scope_record` (and any bulk loader) so it applies no matter which caller
    forgets to check first -- see report 06's containment layer 3. That edit is out of
    scope here (store.py is fenced off to another concurrent task); this function is the
    synth-side half of that defense until store.py grows its own.
    """
    if is_default_store_dir(directory, default_dir):
        raise ValueError(
            f"refusing to write {what} into the live store ({default_dir}); "
            "pass a Store pointed at a scratch directory"
        )


def row_is_synthetic(row: dict[str, Any]) -> bool:
    provenance = row.get("provenance")
    if isinstance(provenance, str):
        import json

        try:
            provenance = json.loads(provenance)
        except (ValueError, TypeError):
            return False
    return bool(isinstance(provenance, dict) and provenance.get("synthetic") is True)
