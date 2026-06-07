"""Tests for TP Update Coordinator middle bucket split fallback control flow.

Verifies:
- Three-Stage: fast insufficient, slow OK → UNSPLIT_SLOW_MIDDLE uses BOLL20 middle
- Middle Runner: fast insufficient, slow OK → UNSPLIT_SLOW_MIDDLE uses BOLL20 middle
- No fallback to outer when UNSPLIT_SLOW_MIDDLE
- Actions: SPLIT, UNSPLIT_SLOW_MIDDLE, FALLBACK_OUTER, DISABLED, INVALID
"""

from __future__ import annotations

import copy
from unittest import mock

import pytest

from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)
from src.strategies.tp_update_coordinator import (
    MiddleBucketSplitApplyResult,
    TpUpdateCoordinator,
)


# ── reusable fixtures ──────────────────────────────────────────────────

def _boll_with_tp(
    middle: float = 100.0,
    upper: float = 110.0,
    lower: float = 90.0,
    tp_middle: float | None = 101.0,
    tp_upper: float | None = 108.0,
    tp_lower: float | None = 92.0,
    tp_window: int | None = 15,
    candle_ts_ms: int = 1000,
) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=candle_ts_ms,
        close=middle,
        middle=middle,
        upper=upper,
        lower=lower,
        upper_distance_pct=0.01,
        lower_distance_pct=0.01,
        alert_switch_on=True,
        live_mode=True,
        tp_lower=tp_lower,
        tp_middle=tp_middle,
        tp_upper=tp_upper,
        tp_window=tp_window,
    )


def _strategy_with_three_stage(side="LONG", **overrides) -> BollCvdReclaimStrategy:
    """Create a strategy configured for Three-Stage runner."""
    config = BollCvdReclaimStrategyConfig(
        three_stage_runner_enabled=True,
        middle_bucket_split_enabled=True,
        middle_bucket_split_fast_ratio=0.70,
        tp_min_net_profit_pct=0.002,
    )

    sizer_config = SimplePositionSizerConfig()
    sizer = SimplePositionSizer(sizer_config)

    strategy = BollCvdReclaimStrategy(config, sizer)
    strategy.state.side = side
    strategy.state.layers = 1
    strategy.state.three_stage_runner_enabled_for_position = True
    strategy.state.three_stage_tp1_consumed = False
    strategy.state.three_stage_tp1_ratio = 0.70
    strategy.state.three_stage_tp2_ratio = 0.20
    strategy.state.three_stage_runner_ratio = 0.10
    strategy.state.avg_entry_price = 1600.0
    strategy.state.breakeven_price = 1600.0
    strategy.state.last_tp_update_candle_ts_ms = 0

    for k, v in overrides.items():
        setattr(strategy.state, k, v)

    return strategy


def _strategy_with_middle_runner(side="LONG", **overrides) -> BollCvdReclaimStrategy:
    """Create a strategy configured for Middle Runner."""
    config = BollCvdReclaimStrategyConfig(
        middle_bucket_split_enabled=True,
        middle_bucket_split_fast_ratio=0.70,
        tp_min_net_profit_pct=0.002,
        middle_runner_first_close_ratio=0.80,
    )

    sizer_config = SimplePositionSizerConfig()
    sizer = SimplePositionSizer(sizer_config)

    strategy = BollCvdReclaimStrategy(config, sizer)
    strategy.state.side = side
    strategy.state.layers = 1
    strategy.state.middle_runner_pending = True
    strategy.state.middle_runner_active = False
    strategy.state.middle_runner_first_close_ratio = 0.80
    strategy.state.avg_entry_price = 1600.0
    strategy.state.breakeven_price = 1600.0
    strategy.state.last_tp_update_candle_ts_ms = 0

    for k, v in overrides.items():
        setattr(strategy.state, k, v)

    return strategy


# ── Three-Stage Middle Bucket Split tests ──────────────────────────────

class TestThreeStageMiddleBucketSplit:
    """Three-Stage branch with middle bucket split."""

    def test_split_enabled_action(self):
        """BOLL15 and BOLL20 both above required → SPLIT."""
        s = _strategy_with_three_stage(side="LONG")
        boll = _boll_with_tp(middle=1640.0, tp_middle=1650.0)
        coordinator = TpUpdateCoordinator(s)

        result = coordinator._apply_middle_bucket_split_for_three_stage(boll)
        assert result.action == "SPLIT"
        assert result.split_active is True
        assert result.tp_plan == "THREE_STAGE_RUNNER"
        assert result.partial_tp_price is not None
        assert s.state.middle_bucket_split_active is True
        assert s.state.three_stage_tp1_price == result.partial_tp_price

    def test_unsplit_slow_middle_action_long(self):
        """BOLL15 insufficient, BOLL20 sufficient → UNSPLIT_SLOW_MIDDLE.
        Must use BOLL20 middle, NOT fallback to outer."""
        s = _strategy_with_three_stage(side="LONG")
        # required = 1600 * 1.002 = 1603.2
        # fast (BOLL15) = 1601 < 1603.2 → insufficient
        # slow (BOLL20) = 1610 >= 1603.2 → sufficient
        boll = _boll_with_tp(middle=1610.0, tp_middle=1601.0)
        coordinator = TpUpdateCoordinator(s)

        result = coordinator._apply_middle_bucket_split_for_three_stage(boll)
        assert result.action == "UNSPLIT_SLOW_MIDDLE"
        assert result.split_active is False
        assert result.tp_plan == "THREE_STAGE_RUNNER"
        assert result.partial_tp_price == pytest.approx(1610.0)  # boll.middle
        # State must be reset
        assert s.state.middle_bucket_split_active is False
        # tp1_price must be BOLL20 middle
        assert s.state.three_stage_tp1_price == pytest.approx(1610.0)

    def test_unsplit_slow_middle_does_not_fallback_to_outer(self):
        """Three-Stage enabled branch with UNSPLIT_SLOW_MIDDLE result:
        must return THREE_STAGE_RUNNER with tp1_price = boll.middle."""
        s = _strategy_with_three_stage(side="LONG")
        boll = _boll_with_tp(middle=1610.0, tp_middle=1601.0)
        coordinator = TpUpdateCoordinator(s)

        # Mock _select_valid_tp_outer_with_profit_fallback
        with mock.patch.object(s, "_select_valid_tp_outer_with_profit_fallback",
                               return_value=(1700.0, "UPPER")):
            tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason = \
                coordinator._apply_three_stage_enabled_branch(
                    1700.0, boll, 1000, None,
                )
            assert tp_plan == "THREE_STAGE_RUNNER"
            assert partial_tp_price == pytest.approx(1610.0)  # boll.middle, NOT outer
            assert s.state.three_stage_tp1_price == pytest.approx(1610.0)

    def test_fallback_outer_action(self):
        """Both BOLL15 and BOLL20 insufficient → FALLBACK_OUTER."""
        s = _strategy_with_three_stage(side="LONG")
        # required = 1603.2, both are below
        boll = _boll_with_tp(middle=1601.0, tp_middle=1601.0)
        coordinator = TpUpdateCoordinator(s)

        result = coordinator._apply_middle_bucket_split_for_three_stage(boll)
        assert result.action == "FALLBACK_OUTER"
        assert result.split_active is False
        assert s.state.middle_bucket_split_active is False


# ── Middle Runner Middle Bucket Split tests ────────────────────────────

class TestMiddleRunnerMiddleBucketSplit:
    """Middle Runner branch with middle bucket split."""

    def test_split_enabled_action(self):
        """BOLL15 and BOLL20 both above required → SPLIT."""
        s = _strategy_with_middle_runner(side="LONG")
        boll = _boll_with_tp(middle=1640.0, tp_middle=1650.0)
        coordinator = TpUpdateCoordinator(s)

        result = coordinator._apply_middle_bucket_split_for_middle_runner(boll)
        assert result.action == "SPLIT"
        assert result.split_active is True
        assert result.tp_plan == "MIDDLE_RUNNER"
        assert result.partial_tp_price is not None
        assert s.state.middle_bucket_split_active is True
        assert s.state.middle_runner_first_tp_price == result.partial_tp_price

    def test_unsplit_slow_middle_action(self):
        """BOLL15 insufficient, BOLL20 sufficient → UNSPLIT_SLOW_MIDDLE."""
        s = _strategy_with_middle_runner(side="LONG")
        boll = _boll_with_tp(middle=1610.0, tp_middle=1601.0)
        coordinator = TpUpdateCoordinator(s)

        result = coordinator._apply_middle_bucket_split_for_middle_runner(boll)
        assert result.action == "UNSPLIT_SLOW_MIDDLE"
        assert result.split_active is False
        assert result.tp_plan == "MIDDLE_RUNNER"
        assert result.partial_tp_price == pytest.approx(1610.0)
        assert s.state.middle_bucket_split_active is False
        assert s.state.middle_runner_first_tp_price == pytest.approx(1610.0)

    def test_unsplit_slow_middle_pending_branch_no_fallback_outer(self):
        """Middle Runner pending branch with UNSPLIT_SLOW_MIDDLE:
        must return MIDDLE_RUNNER with BOLL20 middle, not fallback to outer."""
        s = _strategy_with_middle_runner(side="LONG")
        boll = _boll_with_tp(middle=1610.0, tp_middle=1601.0)
        coordinator = TpUpdateCoordinator(s)

        with mock.patch.object(s, "_select_valid_tp_outer_with_profit_fallback",
                               return_value=(1700.0, "UPPER")):
            tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason = \
                coordinator._apply_middle_runner_pending_branch(
                    1700.0, boll, 1000, None,
                )
            assert tp_plan == "MIDDLE_RUNNER"
            assert partial_tp_price == pytest.approx(1610.0)  # boll.middle
            assert s.state.middle_runner_first_tp_price == pytest.approx(1610.0)

    def test_fallback_outer_action(self):
        """Both insufficient → FALLBACK_OUTER."""
        s = _strategy_with_middle_runner(side="LONG")
        boll = _boll_with_tp(middle=1601.0, tp_middle=1601.0)
        coordinator = TpUpdateCoordinator(s)

        result = coordinator._apply_middle_bucket_split_for_middle_runner(boll)
        assert result.action == "FALLBACK_OUTER"
        assert result.split_active is False
        assert s.state.middle_bucket_split_active is False


# ── Disabled / Invalid tests ───────────────────────────────────────────

class TestDisabledAndInvalid:
    """Tests for DISABLED and INVALID actions."""

    def test_config_disabled_returns_disabled_action(self):
        """When middle_bucket_split_enabled=False → DISABLED."""
        s = _strategy_with_three_stage(side="LONG")
        # Use object.__setattr__ to bypass frozen dataclass
        object.__setattr__(s.config, "middle_bucket_split_enabled", False)
        boll = _boll_with_tp(middle=1640.0, tp_middle=1650.0)
        coordinator = TpUpdateCoordinator(s)

        result = coordinator._apply_middle_bucket_split_for_three_stage(boll)
        assert result.action == "DISABLED"
        assert result.split_active is False

    def test_side_none_returns_invalid_action(self):
        """When side is None → INVALID."""
        s = _strategy_with_three_stage(side="LONG")
        s.state.side = None
        boll = _boll_with_tp(middle=1640.0, tp_middle=1650.0)
        coordinator = TpUpdateCoordinator(s)

        result = coordinator._apply_middle_bucket_split_for_three_stage(boll)
        assert result.action == "INVALID"
        assert result.split_active is False
