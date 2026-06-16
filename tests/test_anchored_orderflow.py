"""Unit tests for the shared AnchoredOrderflowTracker."""
from __future__ import annotations

import pytest
from src.strategies.anchored_orderflow import (
    AnchoredOrderflowSnapshot,
    AnchoredOrderflowTracker,
)


# ── UP anchor / update ───────────────────────────────────────────────

def test_up_anchor_and_update() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="UP", ts_ms=1000, price=100.0,
        cumulative_cvd=500_000.0,
        cumulative_buy_volume=1_000_000.0,
        cumulative_sell_volume=500_000.0,
    )
    assert t.initialised is True
    assert t.direction == "UP"
    assert t.anchor_price == 100.0

    snap = t.update(
        ts_ms=2000, price=102.0,
        cumulative_cvd=600_000.0,
        cumulative_buy_volume=1_100_000.0,
        cumulative_sell_volume=500_000.0,
    )
    assert snap.anchored_cvd == 100_000.0
    assert snap.buy_volume == 100_000.0
    assert snap.sell_volume == 0.0
    assert snap.total_volume == 100_000.0
    assert snap.new_extreme_count == 1
    assert snap.last_extreme_price == 102.0


# ── DOWN anchor / update ─────────────────────────────────────────────

def test_down_anchor_and_update() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="DOWN", ts_ms=1000, price=100.0,
        cumulative_cvd=-500_000.0,
        cumulative_buy_volume=500_000.0,
        cumulative_sell_volume=1_000_000.0,
    )
    assert t.initialised is True
    assert t.direction == "DOWN"

    snap = t.update(
        ts_ms=2000, price=98.0,
        cumulative_cvd=-400_000.0,
        cumulative_buy_volume=600_000.0,
        cumulative_sell_volume=1_000_000.0,
    )
    assert snap.anchored_cvd == 100_000.0  # CVD improved (less bearish)
    assert snap.buy_volume == 100_000.0
    assert snap.sell_volume == 0.0
    assert snap.new_extreme_count == 1
    assert snap.last_extreme_price == 98.0


# ── anchored_cvd calculation ─────────────────────────────────────────

def test_anchored_cvd_calculation() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="UP", ts_ms=1000, price=100.0,
        cumulative_cvd=1_000_000.0,
        cumulative_buy_volume=2_000_000.0,
        cumulative_sell_volume=1_000_000.0,
    )
    # CVD moves higher
    snap = t.update(
        ts_ms=2000, price=101.0,
        cumulative_cvd=1_200_000.0,
        cumulative_buy_volume=2_200_000.0,
        cumulative_sell_volume=1_000_000.0,
    )
    assert snap.anchored_cvd == 200_000.0

    # CVD moves lower
    snap2 = t.update(
        ts_ms=3000, price=100.5,
        cumulative_cvd=900_000.0,
        cumulative_buy_volume=2_100_000.0,
        cumulative_sell_volume=1_200_000.0,
    )
    assert snap2.anchored_cvd == -100_000.0


# ── Buy/sell volume deltas ───────────────────────────────────────────

def test_buy_sell_volume_deltas() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="UP", ts_ms=1000, price=100.0,
        cumulative_cvd=0.0,
        cumulative_buy_volume=500.0,
        cumulative_sell_volume=300.0,
    )
    snap = t.update(
        ts_ms=2000, price=101.0,
        cumulative_cvd=0.0,
        cumulative_buy_volume=700.0,
        cumulative_sell_volume=350.0,
    )
    assert snap.buy_volume == 200.0
    assert snap.sell_volume == 50.0
    assert snap.total_volume == 250.0


# ── new_extreme_count ────────────────────────────────────────────────

def test_new_extreme_count_up() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="UP", ts_ms=1000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    # No new extreme (same price)
    snap = t.update(
        ts_ms=2000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert snap.new_extreme_count == 0

    # New high
    snap = t.update(
        ts_ms=3000, price=101.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert snap.new_extreme_count == 1

    # Not a new high
    snap = t.update(
        ts_ms=4000, price=100.5,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert snap.new_extreme_count == 1

    # Another new high
    snap = t.update(
        ts_ms=5000, price=102.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert snap.new_extreme_count == 2


def test_new_extreme_count_down() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="DOWN", ts_ms=1000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    # New low
    snap = t.update(
        ts_ms=2000, price=99.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert snap.new_extreme_count == 1

    # Not a new low
    snap = t.update(
        ts_ms=3000, price=99.5,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert snap.new_extreme_count == 1

    # Another new low
    snap = t.update(
        ts_ms=4000, price=98.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert snap.new_extreme_count == 2


# ── price_move_pct ───────────────────────────────────────────────────

def test_price_move_pct_up() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="UP", ts_ms=1000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    snap = t.update(
        ts_ms=2000, price=102.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert abs(snap.price_move_pct - 0.02) < 0.0001


def test_price_move_pct_down() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="DOWN", ts_ms=1000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    snap = t.update(
        ts_ms=2000, price=97.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert abs(snap.price_move_pct - 0.03) < 0.0001


def test_price_move_pct_no_move() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="UP", ts_ms=1000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    snap = t.update(
        ts_ms=2000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert snap.price_move_pct == 0.0


# ── cvd_efficiency ───────────────────────────────────────────────────

def test_cvd_efficiency() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="UP", ts_ms=1000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    snap = t.update(
        ts_ms=2000, price=102.0,
        cumulative_cvd=50_000.0,
        cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    # price_move_pct = 0.02, anchored_cvd = 50_000
    # efficiency = 0.02 / 50_000 = 4e-7
    expected = 0.02 / 50_000.0
    assert abs(snap.cvd_efficiency - expected) < 0.0001


# ── volume_efficiency ────────────────────────────────────────────────

def test_volume_efficiency() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="UP", ts_ms=1000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=500.0, cumulative_sell_volume=300.0,
    )
    snap = t.update(
        ts_ms=2000, price=102.0,
        cumulative_cvd=0.0,
        cumulative_buy_volume=1500.0, cumulative_sell_volume=300.0,
    )
    # price_move_pct = 0.02, total_volume = 1000.0
    # efficiency = 0.02 / 1000 = 2e-5
    assert snap.total_volume == 1000.0
    expected = 0.02 / 1000.0
    assert abs(snap.volume_efficiency - expected) < 0.0001


# ── Zero division protection ─────────────────────────────────────────

def test_cvd_efficiency_zero_cvd() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="UP", ts_ms=1000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    snap = t.update(
        ts_ms=2000, price=102.0,
        cumulative_cvd=0.0,  # anchored_cvd = 0
        cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    # price_move_pct = 0.02 / eps (very small) = large but finite
    assert snap.cvd_efficiency > 0
    assert snap.cvd_efficiency != float("inf")


def test_volume_efficiency_zero_volume() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="UP", ts_ms=1000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    snap = t.update(
        ts_ms=2000, price=102.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert snap.volume_efficiency > 0
    assert snap.volume_efficiency != float("inf")


# ── reset ────────────────────────────────────────────────────────────

def test_reset() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="DOWN", ts_ms=1000, price=100.0,
        cumulative_cvd=-500_000.0,
        cumulative_buy_volume=500_000.0, cumulative_sell_volume=1_000_000.0,
    )
    t.update(
        ts_ms=2000, price=98.0,
        cumulative_cvd=-400_000.0,
        cumulative_buy_volume=600_000.0, cumulative_sell_volume=1_000_000.0,
    )
    assert t.initialised is True
    assert t.new_extreme_count == 1

    t.reset()
    assert t.initialised is False
    assert t.direction is None
    assert t.anchor_ts_ms == 0
    assert t.new_extreme_count == 0
    assert t.last_extreme_price == 0.0


# ── Uninitialised update ─────────────────────────────────────────────

def test_uninitialised_update_returns_empty_snapshot() -> None:
    t = AnchoredOrderflowTracker()
    snap = t.update(
        ts_ms=1000, price=100.0,
        cumulative_cvd=0.0, cumulative_buy_volume=0.0, cumulative_sell_volume=0.0,
    )
    assert snap.anchored_cvd == 0.0
    assert snap.new_extreme_count == 0
    assert snap.last_extreme_price == 0.0


# ── last_extreme_anchored_cvd tracking ───────────────────────────────

def test_last_extreme_anchored_cvd() -> None:
    t = AnchoredOrderflowTracker()
    t.anchor(
        direction="DOWN", ts_ms=1000, price=100.0,
        cumulative_cvd=-1_000_000.0,
        cumulative_buy_volume=500_000.0, cumulative_sell_volume=1_500_000.0,
    )
    # First update is extreme (new low)
    snap = t.update(
        ts_ms=2000, price=99.0,
        cumulative_cvd=-950_000.0,
        cumulative_buy_volume=550_000.0, cumulative_sell_volume=1_500_000.0,
    )
    assert snap.new_extreme_count == 1
    assert snap.last_extreme_price == 99.0
    # anchored_cvd at extreme = -950_000 - (-1_000_000) = 50_000
    assert snap.last_extreme_anchored_cvd == 50_000.0
