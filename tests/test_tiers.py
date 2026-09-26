from __future__ import annotations

from flightdeck.models import TIERS
from flightdeck.tiers import MODEL_TIER, derive_tier


def test_claude_code_is_always_frontier_regardless_of_model() -> None:
    """claude-code always runs Anthropic Claude; hook payloads never carry a model id (see
    collect_claude.py), so source alone must be enough."""
    assert derive_tier("claude-code", None) == "frontier"
    assert derive_tier("claude-code", "anything") == "frontier"


def test_known_local_models_map_to_a_tier() -> None:
    assert derive_tier("crush", "qwen3.8:27b") == "fast"
    assert derive_tier("crush", "qwen3-coder-next:latest") == "deep"
    assert derive_tier("crush", "qwen3-coder:30b") == "balanced"


def test_ollama_prefix_is_stripped() -> None:
    """crush turns can carry the model both bare ("qwen3.8:27b") and provider-prefixed
    ("ollama/qwen3.8:27b") -- both must resolve to the same tier."""
    assert derive_tier("crush", "ollama/qwen3.8:27b") == derive_tier("crush", "qwen3.8:27b")


def test_case_insensitive() -> None:
    assert derive_tier("crush", "QWEN3.8:27B") == "fast"


def test_unmapped_model_returns_none_not_a_guess() -> None:
    assert derive_tier("crush", "some-new-model-nobody-has-mapped-yet") is None


def test_no_model_no_source_rule_returns_none() -> None:
    assert derive_tier("crush", None) is None
    assert derive_tier(None, None) is None


def test_claude_passthrough_model_id_maps_to_frontier() -> None:
    """A crush/opencode turn that explicitly routed to a Claude passthrough model (the
    "frontier/<model>" naming) is a real frontier turn even though its source isn't
    literally "claude-code"."""
    assert derive_tier("crush", "frontier/claude-opus-5") == "frontier"
    assert derive_tier("crush", "anthropic/claude-3-5-sonnet") == "frontier"


def test_every_mapped_tier_is_a_real_tier_name() -> None:
    """MODEL_TIER must only ever produce values the rest of the system (models.TIERS,
    governor.toml's [arms.model_tier]) actually recognizes."""
    assert set(MODEL_TIER.values()) <= set(TIERS)
