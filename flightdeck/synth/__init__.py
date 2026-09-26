"""Synthetic-data generation for scope_records and gate/judge prompt labels.

Replaces `flightdeck/sample_data.py`. See `flightdeck/synth/manifest.py` for the
reproducibility contract and `flightdeck/synth/provenance.py` for containment.

# generator (package-wide: no module under flightdeck/synth/ may import
# flightdeck.scope_gate or flightdeck.scope_judge -- see provenance.py docstring and
# tests/test_synth.py::test_no_classifier_import)
"""

from __future__ import annotations

from flightdeck.synth.manifest import Manifest, content_hash, verify_manifest, write_manifest
from flightdeck.synth.prompt_labels import (
    PromptLabel,
    PromptLabelParams,
    generate_labels,
    seed_cases,
)
from flightdeck.synth.scope_rows import ScopeRowParams, generate_rows, invariants_hold

__all__ = [
    "Manifest",
    "content_hash",
    "verify_manifest",
    "write_manifest",
    "PromptLabel",
    "PromptLabelParams",
    "generate_labels",
    "seed_cases",
    "ScopeRowParams",
    "generate_rows",
    "invariants_hold",
]
