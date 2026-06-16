"""Unit tests for event-anchored cumulative CVD divergence detection."""
from __future__ import annotations

import pytest
from src.strategies.reclaim_anchored_divergence import (
    AnchoredDivergenceDecision,
    evaluate_anchored_divergence,
)


# ── LONG confirmed ───────────────────────────────────────────────────

def test_long_divergence_confirmed() -> None:
    """Price new low + CVD recovered = confirmed (bullish divergence)."""
    decision = evaluate_anchored_divergence(
        side="LONG",
        previous_extreme_price=100.0,
        previous_anchored_cvd=-1_000_000.0,
        current_extreme_price=99.0,
        current_anchored_cvd=-800_000.0,
    )
    assert decision.confirmed is True
    assert decision.reason == "ok"
    assert decision.price_extension_pct > 0
    assert decision.cvd_recovery > 0


# ── LONG no divergence (CVD follows price lower) ─────────────────────

def test_long_no_divergence_cvd_follows() -> None:
    """Price new low but CVD makes new low too — no divergence."""
    decision = evaluate_anchored_divergence(
        side="LONG",
        previous_extreme_price=100.0,
        previous_anchored_cvd=-1_000_000.0,
        current_extreme_price=99.0,
        current_anchored_cvd=-1_200_000.0,
    )
    assert decision.confirmed is False
    assert decision.reason == "cvd_not_recovered"


# ── LONG no new low ──────────────────────────────────────────────────

def test_long_no_new_low() -> None:
    """Price not making a new low — no divergence evaluation."""
    decision = evaluate_anchored_divergence(
        side="LONG",
        previous_extreme_price=100.0,
        previous_anchored_cvd=-1_000_000.0,
        current_extreme_price=101.0,
        current_anchored_cvd=-500_000.0,
    )
    assert decision.confirmed is False
    assert decision.reason == "no_new_low"


# ── SHORT confirmed ──────────────────────────────────────────────────

def test_short_divergence_confirmed() -> None:
    """Price new high + CVD reversed down = confirmed (bearish divergence)."""
    decision = evaluate_anchored_divergence(
        side="SHORT",
        previous_extreme_price=100.0,
        previous_anchored_cvd=1_000_000.0,
        current_extreme_price=101.0,
        current_anchored_cvd=800_000.0,
    )
    assert decision.confirmed is True
    assert decision.reason == "ok"
    assert decision.price_extension_pct > 0
    assert decision.cvd_recovery > 0


# ── SHORT no divergence (CVD follows price higher) ───────────────────

def test_short_no_divergence_cvd_follows() -> None:
    """Price new high but CVD makes new high too — no divergence."""
    decision = evaluate_anchored_divergence(
        side="SHORT",
        previous_extreme_price=100.0,
        previous_anchored_cvd=1_000_000.0,
        current_extreme_price=101.0,
        current_anchored_cvd=1_200_000.0,
    )
    assert decision.confirmed is False
    assert decision.reason == "cvd_not_reversed"


# ── SHORT no new high ────────────────────────────────────────────────

def test_short_no_new_high() -> None:
    """Price not making a new high — no divergence evaluation."""
    decision = evaluate_anchored_divergence(
        side="SHORT",
        previous_extreme_price=100.0,
        previous_anchored_cvd=1_000_000.0,
        current_extreme_price=99.0,
        current_anchored_cvd=500_000.0,
    )
    assert decision.confirmed is False
    assert decision.reason == "no_new_high"


# ── Missing data ─────────────────────────────────────────────────────

def test_missing_price_data() -> None:
    decision = evaluate_anchored_divergence(
        side="LONG",
        previous_extreme_price=None,
        previous_anchored_cvd=-1_000_000.0,
        current_extreme_price=99.0,
        current_anchored_cvd=-800_000.0,
    )
    assert decision.confirmed is False
    assert decision.reason == "price_data_missing"


def test_missing_cvd_data() -> None:
    decision = evaluate_anchored_divergence(
        side="LONG",
        previous_extreme_price=100.0,
        previous_anchored_cvd=None,
        current_extreme_price=99.0,
        current_anchored_cvd=-800_000.0,
    )
    assert decision.confirmed is False
    assert decision.reason == "cvd_data_missing"


# ── Price extension field ────────────────────────────────────────────

def test_price_extension_pct_long() -> None:
    """price_extension_pct should reflect how much lower the current extreme is."""
    decision = evaluate_anchored_divergence(
        side="LONG",
        previous_extreme_price=100.0,
        previous_anchored_cvd=-1_000_000.0,
        current_extreme_price=98.0,
        current_anchored_cvd=-800_000.0,
    )
    assert decision.confirmed is True
    assert abs(decision.price_extension_pct - 0.02) < 0.0001


def test_price_extension_pct_short() -> None:
    """price_extension_pct should reflect how much higher the current extreme is."""
    decision = evaluate_anchored_divergence(
        side="SHORT",
        previous_extreme_price=100.0,
        previous_anchored_cvd=1_000_000.0,
        current_extreme_price=102.0,
        current_anchored_cvd=800_000.0,
    )
    assert decision.confirmed is True
    assert abs(decision.price_extension_pct - 0.02) < 0.0001


# ── CVD recovery field ───────────────────────────────────────────────

def test_cvd_recovery_long() -> None:
    """cvd_recovery = current_cvd - previous_cvd (positive = improvement)."""
    decision = evaluate_anchored_divergence(
        side="LONG",
        previous_extreme_price=100.0,
        previous_anchored_cvd=-1_000_000.0,
        current_extreme_price=99.0,
        current_anchored_cvd=-700_000.0,
    )
    assert decision.confirmed is True
    assert decision.cvd_recovery == 300_000.0


def test_cvd_recovery_short() -> None:
    """cvd_recovery = previous_cvd - current_cvd (positive = improvement)."""
    decision = evaluate_anchored_divergence(
        side="SHORT",
        previous_extreme_price=100.0,
        previous_anchored_cvd=1_000_000.0,
        current_extreme_price=101.0,
        current_anchored_cvd=700_000.0,
    )
    assert decision.confirmed is True
    assert decision.cvd_recovery == 300_000.0


# ── Config: min_price_extension_pct ─────────────────────────────────────

def test_divergence_respects_min_price_extension_pct() -> None:
    """Price extension below min_price_extension_pct must NOT confirm."""
    from src.strategies.reclaim_anchored_divergence import AnchoredDivergenceConfig

    config = AnchoredDivergenceConfig(
        min_price_extension_pct=0.001,  # 0.1% required
        min_cvd_recovery=0,
    )
    # Price only extends 0.05% below previous (below 0.1% threshold)
    decision = evaluate_anchored_divergence(
        side="LONG",
        previous_extreme_price=100.0,
        previous_anchored_cvd=-1_000_000.0,
        current_extreme_price=99.95,  # only 0.05% lower
        current_anchored_cvd=-800_000.0,  # CVD recovery is good
        config=config,
    )
    assert decision.confirmed is False
    assert decision.reason == "no_new_low"


def test_divergence_respects_min_cvd_recovery() -> None:
    """CVD recovery below min_cvd_recovery must NOT confirm."""
    from src.strategies.reclaim_anchored_divergence import AnchoredDivergenceConfig

    config = AnchoredDivergenceConfig(
        min_price_extension_pct=0.0,
        min_cvd_recovery=500_000,  # need at least 500k CVD recovery
    )
    # CVD recovers only 200k (from -1M to -800k), below 500k threshold
    decision = evaluate_anchored_divergence(
        side="LONG",
        previous_extreme_price=100.0,
        previous_anchored_cvd=-1_000_000.0,
        current_extreme_price=99.0,  # good price extension
        current_anchored_cvd=-800_000.0,  # but only 200k recovery
        config=config,
    )
    assert decision.confirmed is False
    assert decision.reason == "cvd_not_recovered"


def test_divergence_short_respects_min_cvd_recovery() -> None:
    """SHORT: CVD reversal below min_cvd_recovery must NOT confirm."""
    from src.strategies.reclaim_anchored_divergence import AnchoredDivergenceConfig

    config = AnchoredDivergenceConfig(
        min_price_extension_pct=0.0,
        min_cvd_recovery=500_000,
    )
    decision = evaluate_anchored_divergence(
        side="SHORT",
        previous_extreme_price=100.0,
        previous_anchored_cvd=1_000_000.0,
        current_extreme_price=101.0,
        current_anchored_cvd=800_000.0,  # only 200k reversal
        config=config,
    )
    assert decision.confirmed is False
    assert decision.reason == "cvd_not_reversed"


def test_divergence_default_config_no_minimums() -> None:
    """Without config, defaults (0 min_price_extension, 0 min_cvd_recovery) apply."""
    decision = evaluate_anchored_divergence(
        side="LONG",
        previous_extreme_price=100.0,
        previous_anchored_cvd=-1_000_000.0,
        current_extreme_price=99.999,  # tiny extension
        current_anchored_cvd=-999_999.0,  # tiny recovery
    )
    # Default config has both at 0 → any extension + any recovery = confirmed
    assert decision.confirmed is True
