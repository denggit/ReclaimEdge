"""Tests for post-entry SL cooldown gate purity and log throttling.

Covers:
1. _post_entry_sl_cooldown_ok is a pure gate — no ACTIVE logging
2. _log_info_throttled prints first call even with small ts_ms
3. Cooldown gate return values are correct
4. Different keys don't cross-suppress in throttle helper
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
# 1. _post_entry_sl_cooldown_ok is a pure gate — no ACTIVE logging
# ======================================================================


def test_cooldown_ok_is_pure_gate_no_active_log(caplog):
    """_post_entry_sl_cooldown_ok must be a pure predicate — no ACTIVE logging."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )
    caplog.set_level(logging.INFO)

    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="negative_flat_before_partial_tp")

    # Call 100 times — gate must return correct values but never log ACTIVE
    for i in range(100):
        result = strategy._post_entry_sl_cooldown_ok("LONG", 100_000 + i)
        assert result is False, f"Iteration {i}: expected False, got {result}"

    active_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_ACTIVE" in r.getMessage()
    )
    assert active_count == 0, (
        f"_post_entry_sl_cooldown_ok must not log ACTIVE, got {active_count}"
    )


def test_cooldown_ok_global_pure_gate_no_active_log(caplog):
    """GLOBAL scope gate must also be pure — no ACTIVE logging."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="GLOBAL",
    )
    caplog.set_level(logging.INFO)

    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    for i in range(50):
        result_l = strategy._post_entry_sl_cooldown_ok("LONG", 100_000 + i)
        result_s = strategy._post_entry_sl_cooldown_ok("SHORT", 100_000 + i)
        assert result_l is False
        assert result_s is False

    active_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_ACTIVE" in r.getMessage()
    )
    assert active_count == 0, (
        f"GLOBAL gate must not log ACTIVE, got {active_count}"
    )


# ======================================================================
# 2. blocks_side correctly distinguishes sides
# ======================================================================


def test_blocks_side_distinguishes_sides():
    """SIDE scope: blocks_side returns True only for matching side."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )
    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    # LONG is blocked (same side)
    assert strategy._post_entry_sl_cooldown_blocks_side("LONG", 100_001) is True
    # SHORT is not blocked (opposite side, SIDE scope)
    assert strategy._post_entry_sl_cooldown_blocks_side("SHORT", 100_001) is False


def test_blocks_side_global_blocks_both():
    """GLOBAL scope: blocks_side returns True for both sides."""
    strategy = _make_strategy(
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="GLOBAL",
    )
    _arm_cooldown(strategy, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    assert strategy._post_entry_sl_cooldown_blocks_side("LONG", 100_001) is True
    assert strategy._post_entry_sl_cooldown_blocks_side("SHORT", 100_001) is True


# ======================================================================
# 3. _log_info_throttled prints first call even with small ts_ms
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


def test_log_info_throttled_limits_same_key(caplog):
    """Same key within interval_ms only logs once."""
    strategy = _make_strategy()
    caplog.set_level(logging.INFO)

    for i in range(10):
        strategy._log_info_throttled(
            "THROTTLE_TEST_KEY",
            30_000,
            1000000 + i * 1000,
            "THROTTLE_TEST | iteration=%s ts_ms=%s",
            i, 1000000 + i * 1000,
        )

    log_count = sum(
        1 for r in caplog.records if "THROTTLE_TEST" in r.getMessage()
    )
    assert log_count == 1, f"Same key should log once in 30s window, got {log_count}"


def test_log_info_throttled_allows_after_interval(caplog):
    """Same key logs again after interval_ms has passed."""
    strategy = _make_strategy()
    caplog.set_level(logging.INFO)

    strategy._log_info_throttled("TEST_KEY", 30_000, 1000000, "TEST | first")
    strategy._log_info_throttled("TEST_KEY", 30_000, 1030001, "TEST | second")

    test_count = sum(1 for r in caplog.records if "TEST" in r.getMessage())
    assert test_count == 2, f"Should log again after interval, got {test_count}"


def test_log_info_throttled_different_keys_log_separately(caplog):
    """Different keys log independently."""
    strategy = _make_strategy()
    caplog.set_level(logging.INFO)

    strategy._log_info_throttled("KEY_A", 30_000, 1000000, "LOGGED | key=A")
    strategy._log_info_throttled("KEY_B", 30_000, 1001000, "LOGGED | key=B")

    log_count = sum(
        1 for r in caplog.records if "LOGGED" in r.getMessage()
    )
    assert log_count == 2, f"Different keys should each log once, got {log_count}"


# ======================================================================
# 4. Cooldown return logic is unchanged
# ======================================================================


def test_cooldown_return_logic_unchanged():
    """Verify cooldown return values are correct after refactoring."""
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
    """Verify GLOBAL scope return values are correct after refactoring."""
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
