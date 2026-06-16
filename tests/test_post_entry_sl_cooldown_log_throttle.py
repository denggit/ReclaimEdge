"""Tests for post-entry SL cooldown log throttling.

Covers:
1. POST_ENTRY_SL_COOLDOWN_ACTIVE — throttled to 60s per unique key
2. POST_ENTRY_SL_COOLDOWN_ACTIVE — allows re-log after 60s
3. Different side/reason/until use different throttle keys (no cross-suppression)
4. _log_info_throttled prints first call even with small ts_ms
5. TREND_ENTRY_SKIPPED_POST_ENTRY_SL_COOLDOWN — throttled to 60s per unique key
"""

from __future__ import annotations

import logging

import pytest

from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
)
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig


# ======================================================================
# Helpers
# ======================================================================


def _make_strategy(**config_overrides) -> BollCvdReclaimStrategy:
    """Build a minimal BollCvdReclaimStrategy for log testing."""
    cfg = BollCvdReclaimStrategyConfig(**config_overrides)
    sizer_cfg = SimplePositionSizerConfig()
    sizer = SimplePositionSizer(sizer_cfg)
    return BollCvdReclaimStrategy(cfg, sizer)


def _arm_cooldown(
    strategy: BollCvdReclaimStrategy,
    ts_ms: int = 100_000,
    side: str = "LONG",
    reason: str = "negative_flat_before_partial_tp",
) -> None:
    """Arm cooldown with given parameters."""
    strategy.arm_post_entry_sl_cooldown(ts_ms, side, reason)


# ======================================================================
# 1. POST_ENTRY_SL_COOLDOWN_ACTIVE throttling — same key only one log per 60s
# ======================================================================


def test_cooldown_active_throttled_same_key_one_log_per_60s(caplog):
    """Same side/scope/until/reason should only log ACTIVE once within 60s."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )
    caplog.set_level(logging.INFO)

    # Arm cooldown at ts_ms=100000
    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="negative_flat_before_partial_tp")

    # Call 100 times within 60s window
    for i in range(100):
        result = strategy._post_entry_sl_cooldown_ok("LONG", 100_000 + i)
        # Return value must always be False (cooldown active)
        assert result is False, f"Iteration {i}: expected False, got {result}"

    active_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_ACTIVE" in r.getMessage()
    )
    assert active_count == 1, (
        f"Same key should log ACTIVE once in 60s window, got {active_count}"
    )


def test_cooldown_active_global_scope_throttled(caplog):
    """GLOBAL scope cooldown ACTIVE log should also be throttled."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="GLOBAL",
    )
    caplog.set_level(logging.INFO)

    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    # GLOBAL blocks both sides — call both sides multiple times
    for i in range(50):
        result_l = strategy._post_entry_sl_cooldown_ok("LONG", 100_000 + i)
        result_s = strategy._post_entry_sl_cooldown_ok("SHORT", 100_000 + i)
        assert result_l is False
        assert result_s is False

    # LONG and SHORT have different keys → each should log once
    active_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_ACTIVE" in r.getMessage()
    )
    assert active_count == 2, (
        f"GLOBAL: LONG and SHORT are different keys, expected 2 ACTIVE logs, got {active_count}"
    )


# ======================================================================
# 2. POST_ENTRY_SL_COOLDOWN_ACTIVE — allows re-log after 60s
# ======================================================================


def test_cooldown_active_allows_re_log_after_60s(caplog):
    """Same key should log ACTIVE again after 60s interval."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )
    caplog.set_level(logging.INFO)

    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="negative_flat_before_partial_tp")

    # First call at t=10000 (within cooldown)
    strategy._post_entry_sl_cooldown_ok("LONG", 10_000)
    # Second call at t=70001 (60s + 1ms after first)
    strategy._post_entry_sl_cooldown_ok("LONG", 70_001)

    active_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_ACTIVE" in r.getMessage()
    )
    assert active_count == 2, (
        f"Should log ACTIVE again after 60s interval, got {active_count}"
    )


# ======================================================================
# 3. Different keys don't cross-suppress
# ======================================================================


def test_cooldown_active_different_side_independent_keys(caplog):
    """LONG and SHORT cooldown ACTIVE logs use different keys — no cross-suppression."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )
    caplog.set_level(logging.INFO)

    # Arm LONG cooldown, test both sides
    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    # LONG is blocked (SIDE scope, same side)
    result_l = strategy._post_entry_sl_cooldown_ok("LONG", 100_001)
    assert result_l is False

    # SHORT is not blocked (SIDE scope, opposite side) — should not log ACTIVE
    result_s = strategy._post_entry_sl_cooldown_ok("SHORT", 100_001)
    assert result_s is True  # Allowed

    active_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_ACTIVE" in r.getMessage()
    )
    assert active_count == 1, (
        f"Only LONG should trigger ACTIVE, got {active_count}"
    )


def test_cooldown_active_different_until_ts_ms_independent_keys(caplog):
    """Different until_ts_ms values produce different keys — each logs independently."""
    strategy1 = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )
    strategy2 = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=3600,  # different cooldown → different until_ts_ms
        post_entry_sl_cooldown_scope="SIDE",
    )
    caplog.set_level(logging.INFO)

    # Arm at same ts, different durations → different until_ts_ms
    _arm_cooldown(strategy1, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    _arm_cooldown(strategy2, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    # Both strategies have different until_ts_ms → different keys → each logs
    strategy1._post_entry_sl_cooldown_ok("LONG", 100_001)
    strategy2._post_entry_sl_cooldown_ok("LONG", 100_001)

    active_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_ACTIVE" in r.getMessage()
    )
    # Two different strategies with different until_ts_ms → 2 ACTIVE logs
    assert active_count == 2, (
        f"Different until_ts_ms should log independently, got {active_count}"
    )


def test_cooldown_active_different_reason_independent_keys(caplog):
    """Different reason values produce different keys — each logs independently."""
    strategy1 = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )
    strategy2 = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )
    caplog.set_level(logging.INFO)

    _arm_cooldown(strategy1, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    _arm_cooldown(strategy2, ts_ms=100_000, side="LONG", reason="negative_flat_before_partial_tp")

    strategy1._post_entry_sl_cooldown_ok("LONG", 100_001)
    strategy2._post_entry_sl_cooldown_ok("LONG", 100_001)

    active_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_ACTIVE" in r.getMessage()
    )
    # Different reasons → different keys → 2 ACTIVE logs
    assert active_count == 2, (
        f"Different reasons should log independently, got {active_count}"
    )


# ======================================================================
# 4. _log_info_throttled prints first call even with small ts_ms
# ======================================================================


def test_log_info_throttled_first_call_small_ts_ms(caplog):
    """First call to _log_info_throttled should always print, even with very small ts_ms."""
    strategy = _make_strategy()
    caplog.set_level(logging.INFO)

    strategy._log_info_throttled("test_key", 60_000, 1_000, "TEST_LOG_FIRST_SMALL_TS")

    test_count = sum(
        1 for r in caplog.records if "TEST_LOG_FIRST_SMALL_TS" in r.getMessage()
    )
    assert test_count == 1, (
        f"First call with small ts_ms=1000 should print, got {test_count}"
    )


def test_log_info_throttled_first_call_zero_ts_ms(caplog):
    """First call to _log_info_throttled with ts_ms=0 should still print."""
    strategy = _make_strategy()
    caplog.set_level(logging.INFO)

    strategy._log_info_throttled("test_key_zero", 60_000, 0, "TEST_LOG_FIRST_ZERO_TS")

    test_count = sum(
        1 for r in caplog.records if "TEST_LOG_FIRST_ZERO_TS" in r.getMessage()
    )
    assert test_count == 1, (
        f"First call with ts_ms=0 should print, got {test_count}"
    )


# ======================================================================
# 5. TREND_ENTRY_SKIPPED_POST_ENTRY_SL_COOLDOWN throttling
# ======================================================================


def test_trend_entry_skipped_log_throttled(caplog):
    """TREND_ENTRY_SKIPPED_POST_ENTRY_SL_COOLDOWN should be throttled to 60s."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )
    caplog.set_level(logging.INFO)

    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    # _post_entry_sl_cooldown_ok returns False → simulates what _maybe_trend_entry sees
    # But _maybe_trend_entry has its own throttled log.
    # We simulate the same pattern: call _post_entry_sl_cooldown_ok + log the skip
    for i in range(10):
        if not strategy._post_entry_sl_cooldown_ok("LONG", 100_000 + i * 1000):
            strategy._log_info_throttled(
                "TREND_ENTRY_SKIPPED_POST_ENTRY_SL_COOLDOWN:"
                f"LONG:"
                f"{strategy.state.post_entry_sl_cooldown_side}:"
                f"{strategy.state.post_entry_sl_cooldown_until_ts_ms}:"
                f"{strategy.config.post_entry_sl_cooldown_scope}",
                60_000,
                100_000 + i * 1000,
                "TREND_ENTRY_SKIPPED_POST_ENTRY_SL_COOLDOWN | side=%s "
                "cooldown_side=%s cooldown_until_ts_ms=%s scope=%s",
                "LONG",
                strategy.state.post_entry_sl_cooldown_side,
                strategy.state.post_entry_sl_cooldown_until_ts_ms,
                strategy.config.post_entry_sl_cooldown_scope,
            )

    skip_count = sum(
        1 for r in caplog.records if "TREND_ENTRY_SKIPPED_POST_ENTRY_SL_COOLDOWN" in r.getMessage()
    )
    assert skip_count == 1, (
        f"TREND_ENTRY_SKIPPED should be throttled to 1 in 60s, got {skip_count}"
    )


# ======================================================================
# 6. Cooldown return logic is unchanged
# ======================================================================


def test_cooldown_return_logic_unchanged():
    """Verify cooldown return values are correct after throttling changes."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )

    # No cooldown → returns True
    assert strategy._post_entry_sl_cooldown_ok("LONG", 100_000) is True
    assert strategy._post_entry_sl_cooldown_ok("SHORT", 100_000) is True

    # Arm cooldown
    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    # SIDE scope: LONG blocked, SHORT allowed
    assert strategy._post_entry_sl_cooldown_ok("LONG", 100_001) is False
    assert strategy._post_entry_sl_cooldown_ok("SHORT", 100_001) is True

    # After expiry → both allowed, state cleared
    assert strategy._post_entry_sl_cooldown_ok("LONG", 100_000 + 1_800_001) is True
    assert strategy.state.post_entry_sl_cooldown_until_ts_ms == 0
    assert strategy.state.post_entry_sl_cooldown_side is None
    assert strategy.state.post_entry_sl_cooldown_reason is None


def test_cooldown_return_logic_global_unchanged():
    """Verify GLOBAL scope return values are correct after throttling changes."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="GLOBAL",
    )

    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    # GLOBAL: both sides blocked
    assert strategy._post_entry_sl_cooldown_ok("LONG", 100_001) is False
    assert strategy._post_entry_sl_cooldown_ok("SHORT", 100_001) is False

    # After expiry → both allowed
    assert strategy._post_entry_sl_cooldown_ok("LONG", 100_000 + 1_800_001) is True


def test_cooldown_disabled_always_returns_true():
    """When cooldown is disabled, _post_entry_sl_cooldown_ok always returns True."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=False,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )

    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    # Disabled → always True regardless of state
    assert strategy._post_entry_sl_cooldown_ok("LONG", 100_001) is True
    assert strategy._post_entry_sl_cooldown_ok("SHORT", 100_001) is True
