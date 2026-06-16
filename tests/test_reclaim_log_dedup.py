"""Tests for reclaim confirmed log deduplication and TREND_METRICS_MISSING throttling.

Covers:
1. LOWER_RECLAIM_CONFIRMED only printed once per reclaim cycle
2. UPPER_RECLAIM_CONFIRMED only printed once per reclaim cycle
3. Reset allows re-printing in next cycle
4. TREND_METRICS_MISSING throttled to 30s per direction+reason key
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig


# ======================================================================
# Helpers
# ======================================================================


@dataclass
class FakeBollSnapshot:
    upper: float = 3200.0
    middle: float = 3000.0
    lower: float = 2800.0
    alert_switch_on: bool = True
    band_std: float = 2.0
    sma: float = 3000.0
    ts_ms: int = 0


@dataclass
class FakeCvdSnapshot:
    ts_ms: int = 0
    fast_cvd: float = 0.0
    buy_ratio: float = 0.5
    sell_ratio: float = 0.5
    cvd: float = 0.0
    cvd_ma: float = 0.0
    net_taker_volume: float = 0.0
    # ── CVD direction check attributes ───────────────────────────────
    cross_negative: bool = False
    cvd_decreasing: bool = True
    no_new_high: bool = True
    no_new_low: bool = True
    buy_spike: bool = False
    sell_spike: bool = True
    cross_positive: bool = False
    cvd_increasing: bool = True
    min_buy_ratio: float = 0.58
    min_sell_ratio: float = 0.58


def _make_strategy(**config_overrides):
    """Build a minimal BollCvdReclaimStrategy for log testing."""
    cfg = BollCvdReclaimStrategyConfig(entry_reclaim_v2_enabled=False)
    for k, v in config_overrides.items():
        setattr(cfg, k, v)
    sizer_cfg = SimplePositionSizerConfig()
    sizer = SimplePositionSizer(sizer_cfg)
    strategy = BollCvdReclaimStrategy(cfg, sizer)
    return strategy


def _arm_lower_reclaim(strategy, price=2700.0, ts_ms=1000000):
    """Force lower_armed + reclaim_seen + reclaim_pending state."""
    state = strategy.state
    state.lower_armed = True
    state.lower_extreme_price = 2650.0
    state.lower_deep_enough = True
    state.lower_extreme_ts_ms = ts_ms - 5000

    # Fake CVD structure OK
    state.lower_cvd_divergence_confirmed = True

    # Set reclaim seen with timestamp in the past (just before confirm window)
    state.lower_reclaim_seen = True
    state.lower_reclaim_ts_ms = ts_ms - int(strategy.config.entry_reclaim_confirm_seconds * 1000) - 1


def _arm_upper_reclaim(strategy, price=3300.0, ts_ms=1000000):
    """Force upper_armed + reclaim_seen + reclaim_pending state."""
    state = strategy.state
    state.upper_armed = True
    state.upper_extreme_price = 3350.0
    state.upper_deep_enough = True
    state.upper_extreme_ts_ms = ts_ms - 5000

    # Fake CVD structure OK
    state.upper_cvd_divergence_confirmed = True

    # Set reclaim seen with timestamp in the past (just before confirm window)
    state.upper_reclaim_seen = True
    state.upper_reclaim_ts_ms = ts_ms - int(strategy.config.entry_reclaim_confirm_seconds * 1000) - 1


# ======================================================================
# 1. LOWER_RECLAIM_CONFIRMED dedup
# ======================================================================


def test_lower_reclaim_confirmed_logged_once_per_cycle(caplog):
    """LOWER_RECLAIM_CONFIRMED must only log once per reclaim cycle."""
    strategy = _make_strategy()
    cfg = strategy.config
    ts_base = 1000000
    confirm_ms = int(strategy.config.entry_reclaim_confirm_seconds * 1000)

    _arm_lower_reclaim(strategy, price=2700.0, ts_ms=ts_base)

    # Advance time past confirm window and feed 20 ticks
    caplog.set_level(logging.INFO)
    for i in range(20):
        ts = ts_base + confirm_ms + 100 + i * 100
        boll = FakeBollSnapshot(lower=2800.0, middle=3000.0, upper=3200.0, ts_ms=ts)
        cvd = FakeCvdSnapshot(ts_ms=ts)
        strategy.on_tick(price=2850.0, ts_ms=ts, boll=boll, cvd=cvd)

    confirmed_count = sum(
        1 for r in caplog.records if "LOWER_RECLAIM_CONFIRMED" in r.getMessage()
    )
    assert confirmed_count == 1, (
        f"LOWER_RECLAIM_CONFIRMED should log exactly once per cycle, "
        f"got {confirmed_count}"
    )

    # State flag should be set
    assert strategy.state.lower_reclaim_confirmed_logged is True


def test_upper_reclaim_confirmed_logged_once_per_cycle(caplog):
    """UPPER_RECLAIM_CONFIRMED must only log once per reclaim cycle."""
    strategy = _make_strategy()
    cfg = strategy.config
    ts_base = 1000000
    confirm_ms = int(strategy.config.entry_reclaim_confirm_seconds * 1000)

    _arm_upper_reclaim(strategy, price=3300.0, ts_ms=ts_base)

    caplog.set_level(logging.INFO)
    for i in range(20):
        ts = ts_base + confirm_ms + 100 + i * 100
        boll = FakeBollSnapshot(lower=2800.0, middle=3000.0, upper=3200.0, ts_ms=ts)
        cvd = FakeCvdSnapshot(ts_ms=ts)
        strategy.on_tick(price=3100.0, ts_ms=ts, boll=boll, cvd=cvd)

    confirmed_count = sum(
        1 for r in caplog.records if "UPPER_RECLAIM_CONFIRMED" in r.getMessage()
    )
    assert confirmed_count == 1, (
        f"UPPER_RECLAIM_CONFIRMED should log exactly once per cycle, "
        f"got {confirmed_count}"
    )

    assert strategy.state.upper_reclaim_confirmed_logged is True


# ======================================================================
# 2. Reset allows re-printing in next cycle
# ======================================================================


def test_lower_reclaim_reset_allows_re_log(caplog):
    """After _reset_lower_armed, next reclaim cycle logs CONFIRMED again."""
    strategy = _make_strategy()
    cfg = strategy.config
    ts_base = 1000000
    confirm_ms = int(strategy.config.entry_reclaim_confirm_seconds * 1000)

    # -- First cycle --
    _arm_lower_reclaim(strategy, price=2700.0, ts_ms=ts_base)
    caplog.set_level(logging.INFO)
    for i in range(3):
        ts = ts_base + confirm_ms + 100 + i * 100
        boll = FakeBollSnapshot(lower=2800.0, middle=3000.0, upper=3200.0, ts_ms=ts)
        cvd = FakeCvdSnapshot(ts_ms=ts)
        strategy.on_tick(price=2850.0, ts_ms=ts, boll=boll, cvd=cvd)

    first_round = sum(
        1 for r in caplog.records if "LOWER_RECLAIM_CONFIRMED" in r.getMessage()
    )
    assert first_round == 1, f"First round should have 1 CONFIRMED, got {first_round}"
    caplog.clear()

    # -- Reset and re-arm --
    strategy._reset_lower_armed()
    assert strategy.state.lower_reclaim_confirmed_logged is False
    ts_base2 = ts_base + 100000
    _arm_lower_reclaim(strategy, price=2700.0, ts_ms=ts_base2)

    for i in range(3):
        ts = ts_base2 + confirm_ms + 100 + i * 100
        boll = FakeBollSnapshot(lower=2800.0, middle=3000.0, upper=3200.0, ts_ms=ts)
        cvd = FakeCvdSnapshot(ts_ms=ts)
        strategy.on_tick(price=2850.0, ts_ms=ts, boll=boll, cvd=cvd)

    second_round = sum(
        1 for r in caplog.records if "LOWER_RECLAIM_CONFIRMED" in r.getMessage()
    )
    assert second_round == 1, f"Second round should have 1 CONFIRMED, got {second_round}"
    assert strategy.state.lower_reclaim_confirmed_logged is True


def test_upper_reclaim_reset_allows_re_log(caplog):
    """After _reset_upper_armed, next reclaim cycle logs CONFIRMED again."""
    strategy = _make_strategy()
    cfg = strategy.config
    ts_base = 1000000
    confirm_ms = int(strategy.config.entry_reclaim_confirm_seconds * 1000)

    # -- First cycle --
    _arm_upper_reclaim(strategy, price=3300.0, ts_ms=ts_base)
    caplog.set_level(logging.INFO)
    for i in range(3):
        ts = ts_base + confirm_ms + 100 + i * 100
        boll = FakeBollSnapshot(lower=2800.0, middle=3000.0, upper=3200.0, ts_ms=ts)
        cvd = FakeCvdSnapshot(ts_ms=ts)
        strategy.on_tick(price=3100.0, ts_ms=ts, boll=boll, cvd=cvd)

    first_round = sum(
        1 for r in caplog.records if "UPPER_RECLAIM_CONFIRMED" in r.getMessage()
    )
    assert first_round == 1, f"First round should have 1 CONFIRMED, got {first_round}"
    caplog.clear()

    # -- Reset and re-arm --
    strategy._reset_upper_armed()
    assert strategy.state.upper_reclaim_confirmed_logged is False
    ts_base2 = ts_base + 100000
    _arm_upper_reclaim(strategy, price=3300.0, ts_ms=ts_base2)

    for i in range(3):
        ts = ts_base2 + confirm_ms + 100 + i * 100
        boll = FakeBollSnapshot(lower=2800.0, middle=3000.0, upper=3200.0, ts_ms=ts)
        cvd = FakeCvdSnapshot(ts_ms=ts)
        strategy.on_tick(price=3100.0, ts_ms=ts, boll=boll, cvd=cvd)

    second_round = sum(
        1 for r in caplog.records if "UPPER_RECLAIM_CONFIRMED" in r.getMessage()
    )
    assert second_round == 1, f"Second round should have 1 CONFIRMED, got {second_round}"
    assert strategy.state.upper_reclaim_confirmed_logged is True


# ======================================================================
# 3. TREND_METRICS_MISSING throttling (tested via _log_info_throttled)
# ======================================================================


def test_log_info_throttled_limits_same_key(caplog):
    """Same key within interval_ms only logs once."""
    strategy = _make_strategy()
    caplog.set_level(logging.INFO)

    for i in range(5):
        ts = 1000000 + i * 5000  # 5 ticks within 20s
        strategy._log_info_throttled(
            "TREND_METRICS_MISSING:UP:episode_volume_cvd_not_accumulated",
            30_000,  # 30s interval
            ts,
            "TREND_METRICS_MISSING | reason=episode_volume_cvd_not_accumulated direction=%s price=%.4f ts_ms=%s",
            "UP", 3100.0 + i, ts,
        )

    missing_count = sum(
        1 for r in caplog.records if "TREND_METRICS_MISSING" in r.getMessage()
    )
    assert missing_count == 1, (
        f"Same key should log once in 30s window, got {missing_count}"
    )


def test_log_info_throttled_different_keys_log_separately(caplog):
    """Different keys log independently."""
    strategy = _make_strategy()
    caplog.set_level(logging.INFO)

    # UP direction
    strategy._log_info_throttled(
        "TREND_METRICS_MISSING:UP:episode_volume_cvd_not_accumulated",
        30_000, 1000000,
        "TREND_METRICS_MISSING direction=%s", "UP",
    )
    # DOWN direction (different key)
    strategy._log_info_throttled(
        "TREND_METRICS_MISSING:DOWN:episode_volume_cvd_not_accumulated",
        30_000, 1001000,
        "TREND_METRICS_MISSING direction=%s", "DOWN",
    )

    missing_count = sum(
        1 for r in caplog.records if "TREND_METRICS_MISSING" in r.getMessage()
    )
    assert missing_count == 2, (
        f"Different keys should each log once, got {missing_count}"
    )


def test_log_info_throttled_allows_after_interval(caplog):
    """Same key logs again after interval_ms has passed."""
    strategy = _make_strategy()
    caplog.set_level(logging.INFO)

    # First log at t=1000000
    strategy._log_info_throttled("TEST_KEY", 30_000, 1000000, "TEST | first")
    # Second at t=1030000 (30s + 1ms later) → should log again
    strategy._log_info_throttled("TEST_KEY", 30_000, 1030001, "TEST | second")

    test_count = sum(1 for r in caplog.records if "TEST" in r.getMessage())
    assert test_count == 2, (
        f"Should log again after interval, got {test_count}"
    )
