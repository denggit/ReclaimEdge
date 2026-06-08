"""Tests for TP Update Coordinator Three-Stage TP2 using structure BOLL20.

Verifies:
- _maybe_update_three_stage_waiting_tp2 uses structure BOLL20 outer.
- _apply_three_stage_enabled_branch uses structure BOLL20 outer.
- startup_force_tp_reconcile waiting-TP2 update uses structure BOLL20 outer.
- TP1成交后动态更新TP2不会把TP2更新回TP_BOLL15 outer.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pytest

from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
)


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


def _cvd():
    from src.indicators.cvd_tracker import CvdSnapshot
    return CvdSnapshot(
        ts_ms=1000,
        price=100.0,
        side="buy",
        size=1.0,
        signed_delta=0.1,
        total_cvd=0.5,
        fast_cvd=0.1,
        previous_fast_cvd=0.09,
        buy_volume=60.0,
        sell_volume=40.0,
        buy_ratio=0.6,
        sell_ratio=0.4,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=False,
        window_low=99.0,
        window_high=101.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.0,
        baseline_range_pct=0.0,
        burst_move_ratio=0.0,
        burst_volume=0.0,
        baseline_volume=0.0,
        burst_volume_ratio=0.0,
        up_burst=False,
        down_burst=False,
    )


def _strategy(**kwargs) -> BollCvdReclaimStrategy:
    config = BollCvdReclaimStrategyConfig(**kwargs)
    sizer_config = SimplePositionSizerConfig()
    sizer = SimplePositionSizer(sizer_config)
    return BollCvdReclaimStrategy(config, sizer)


# ── Test: _maybe_update_three_stage_waiting_tp2 uses structure BOLL20 ──

class TestWaitingTp2UsesStructureBoll(unittest.TestCase):
    """_maybe_update_three_stage_waiting_tp2 must select TP2 from structure BOLL20."""

    def _setup_waiting_tp2_state(self, strategy: BollCvdReclaimStrategy) -> None:
        s = strategy.state
        s.side = "LONG"
        s.layers = 1
        s.avg_entry_price = 1640.0
        s.breakeven_price = 1649.0
        s.net_remaining_breakeven_price = 1649.0
        s.three_stage_runner_enabled_for_position = True
        s.three_stage_tp1_consumed = True
        s.three_stage_tp2_consumed = False
        s.trend_runner_active = False
        s.three_stage_tp1_price = 1660.0
        s.three_stage_tp2_price = 1670.0  # old TP_BOLL15 value
        s.tp_price = 1670.0
        s.tp_plan = "THREE_STAGE_RUNNER"

    def test_waiting_tp2_updates_to_structure_upper(self) -> None:
        boll = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1660.0,
            tp_upper=1670.0,  # TP_BOLL15 upper
            tp_lower=1610.0,
            tp_window=15,
            candle_ts_ms=2000,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
        )
        self._setup_waiting_tp2_state(strategy)
        cvd = _cvd()

        # When the coordinator runs for waiting TP2, TP2 must be updated to
        # structure BOLL20 upper (1700), not TP_BOLL15 upper (1670).
        intent = strategy._maybe_update_tp(1650.0, 2000, boll, cvd)

        # The intent may be None if TP2 didn't change enough, so check state directly
        # TP2 should now be 1700 (structure upper), NOT 1670 (TP_BOLL15 upper)
        self.assertIsNotNone(strategy.state.three_stage_tp2_price)
        self.assertAlmostEqual(strategy.state.three_stage_tp2_price, 1700.0, places=4,
                               msg="waiting TP2 must be structure BOLL20 upper (1700)")
        self.assertNotAlmostEqual(strategy.state.three_stage_tp2_price, 1670.0, places=4,
                                  msg="waiting TP2 must NOT be TP_BOLL15 upper (1670)")

    def test_waiting_tp2_short_uses_structure_lower(self) -> None:
        boll = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1640.0,
            tp_upper=1670.0,
            tp_lower=1620.0,  # TP_BOLL15 lower
            tp_window=15,
            candle_ts_ms=2000,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
        )
        s = strategy.state
        s.side = "SHORT"
        s.layers = 1
        s.avg_entry_price = 1700.0
        s.breakeven_price = 1691.0
        s.net_remaining_breakeven_price = 1691.0
        s.three_stage_runner_enabled_for_position = True
        s.three_stage_tp1_consumed = True
        s.three_stage_tp2_consumed = False
        s.trend_runner_active = False
        s.three_stage_tp1_price = 1640.0
        s.three_stage_tp2_price = 1620.0
        s.tp_price = 1620.0
        s.tp_plan = "THREE_STAGE_RUNNER"
        cvd = _cvd()

        strategy._maybe_update_tp(1650.0, 2000, boll, cvd)

        self.assertIsNotNone(strategy.state.three_stage_tp2_price)
        self.assertAlmostEqual(strategy.state.three_stage_tp2_price, 1600.0, places=4,
                               msg="SHORT waiting TP2 must be structure BOLL20 lower (1600)")
        self.assertNotAlmostEqual(strategy.state.three_stage_tp2_price, 1620.0, places=4,
                                  msg="SHORT waiting TP2 must NOT be TP_BOLL15 lower (1620)")


# ── Test: _apply_three_stage_enabled_branch uses structure BOLL20 ──────

class TestThreeStageEnabledBranchStructureBoll(unittest.TestCase):
    """_apply_three_stage_enabled_branch uses structure BOLL20 outer for TP2."""

    def _setup_three_stage_enabled_state(self, strategy: BollCvdReclaimStrategy) -> None:
        s = strategy.state
        s.side = "LONG"
        s.layers = 1
        s.avg_entry_price = 1640.0
        s.breakeven_price = 1649.0
        s.net_remaining_breakeven_price = 1649.0
        s.three_stage_runner_enabled_for_position = True
        s.three_stage_tp1_consumed = False
        s.three_stage_tp2_consumed = False
        s.trend_runner_active = False
        s.tp_plan = "THREE_STAGE_RUNNER"

    def test_enabled_branch_tp2_is_structure_upper(self) -> None:
        boll = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1660.0,
            tp_upper=1670.0,
            tp_lower=1610.0,
            tp_window=15,
            candle_ts_ms=2000,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
            tp_min_net_profit_pct=0.002,
        )
        self._setup_three_stage_enabled_state(strategy)
        cvd = _cvd()

        strategy._maybe_update_tp(1650.0, 2000, boll, cvd)

        # After update, TP2 should be structure BOLL20 upper (1700)
        self.assertIsNotNone(strategy.state.three_stage_tp2_price)
        self.assertAlmostEqual(strategy.state.three_stage_tp2_price, 1700.0, places=4,
                               msg="Three-Stage enabled TP2 must be structure BOLL20 upper")
        self.assertNotAlmostEqual(strategy.state.three_stage_tp2_price, 1670.0, places=4,
                                  msg="TP2 must NOT be TP_BOLL15 upper")


# ── Test: startup_force_tp_reconcile uses structure BOLL20 ─────────────

class TestStartupForceReconcileStructureBoll(unittest.TestCase):
    """startup_force_tp_reconcile with waiting-TP2 must use structure BOLL20."""

    def test_force_reconcile_waiting_tp2_uses_structure_upper(self) -> None:
        boll = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1660.0,
            tp_upper=1670.0,
            tp_lower=1610.0,
            tp_window=15,
            candle_ts_ms=2000,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
        )
        s = strategy.state
        s.side = "LONG"
        s.layers = 1
        s.avg_entry_price = 1640.0
        s.breakeven_price = 1649.0
        s.net_remaining_breakeven_price = 1649.0
        s.three_stage_runner_enabled_for_position = True
        s.three_stage_tp1_consumed = True
        s.three_stage_tp2_consumed = False
        s.trend_runner_active = False
        s.three_stage_tp1_price = 1660.0
        s.three_stage_tp2_price = 1670.0
        s.tp_price = 1670.0
        s.tp_plan = "THREE_STAGE_RUNNER"
        s.startup_force_tp_reconcile = True
        s.last_tp_update_candle_ts_ms = 0  # different candle → force reconcile fires
        cvd = _cvd()

        strategy._maybe_update_tp(1650.0, 2000, boll, cvd)

        self.assertAlmostEqual(strategy.state.three_stage_tp2_price, 1700.0, places=4,
                               msg="force_reconcile waiting TP2 must be structure BOLL20 upper")


# ── Test: TP1成交后动态更新TP2不会回退到TP_BOLL15 ─────────────────────

class TestTp2NotRevertedToTpBoll15(unittest.TestCase):
    """After TP1 fills, dynamic TP2 update must not revert TP2 to TP_BOLL15."""

    def test_dynamic_update_keeps_structure_outer(self) -> None:
        boll_initial = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1660.0,
            tp_upper=1670.0,
            tp_lower=1610.0,
            tp_window=15,
            candle_ts_ms=1000,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
            tp_min_net_profit_pct=0.002,
        )
        s = strategy.state
        s.side = "LONG"
        s.layers = 1
        s.avg_entry_price = 1640.0
        s.breakeven_price = 1649.0
        s.net_remaining_breakeven_price = 1649.0
        s.three_stage_runner_enabled_for_position = True
        s.three_stage_tp1_consumed = False
        s.three_stage_tp2_consumed = False
        s.trend_runner_active = False
        s.tp_plan = "THREE_STAGE_RUNNER"
        cvd = _cvd()

        # First update: TP2 = structure upper 1700
        strategy._maybe_update_tp(1650.0, 1000, boll_initial, cvd)
        self.assertAlmostEqual(strategy.state.three_stage_tp2_price, 1700.0, places=4)

        # Second candle with same BOLL values but different candle_ts
        boll_next = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1660.0,
            tp_upper=1670.0,
            tp_lower=1610.0,
            tp_window=15,
            candle_ts_ms=2000,
        )
        strategy._maybe_update_tp(1650.0, 2000, boll_next, cvd)

        # TP2 must still be structure BOLL20 upper (1700), not TP_BOLL15 (1670)
        self.assertAlmostEqual(strategy.state.three_stage_tp2_price, 1700.0, places=4,
                               msg="dynamic update must not revert TP2 to TP_BOLL15")
        self.assertNotAlmostEqual(strategy.state.three_stage_tp2_price, 1670.0, places=4,
                                  msg="TP2 must NOT revert to TP_BOLL15 upper")


# ── Test: _select_three_stage_tp2_outer is called in the right places ──

class TestSelectThreeStageTp2OuterCalled(unittest.TestCase):
    """Verify _select_three_stage_tp2_outer is called in the right call sites."""

    def test_waiting_tp2_calls_three_stage_tp2_outer(self) -> None:
        boll = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1660.0,
            tp_upper=1670.0,
            tp_lower=1610.0,
            tp_window=15,
            candle_ts_ms=2000,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
        )
        s = strategy.state
        s.side = "LONG"
        s.layers = 1
        s.avg_entry_price = 1640.0
        s.breakeven_price = 1649.0
        s.net_remaining_breakeven_price = 1649.0
        s.three_stage_runner_enabled_for_position = True
        s.three_stage_tp1_consumed = True
        s.three_stage_tp2_consumed = False
        s.trend_runner_active = False
        s.three_stage_tp1_price = 1660.0
        s.three_stage_tp2_price = 1670.0
        s.tp_price = 1670.0
        s.tp_plan = "THREE_STAGE_RUNNER"
        cvd = _cvd()

        with mock.patch.object(
            strategy, "_select_three_stage_tp2_outer",
            wraps=strategy._select_three_stage_tp2_outer,
        ) as mock_selector:
            strategy._maybe_update_tp(1650.0, 2000, boll, cvd)
            # The new selector must be called for waiting TP2 updates
            mock_selector.assert_called()

    def test_enabled_branch_calls_three_stage_tp2_outer(self) -> None:
        boll = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1660.0,
            tp_upper=1670.0,
            tp_lower=1610.0,
            tp_window=15,
            candle_ts_ms=2000,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
            tp_min_net_profit_pct=0.002,
        )
        s = strategy.state
        s.side = "LONG"
        s.layers = 1
        s.avg_entry_price = 1640.0
        s.breakeven_price = 1649.0
        s.net_remaining_breakeven_price = 1649.0
        s.three_stage_runner_enabled_for_position = True
        s.three_stage_tp1_consumed = False
        s.three_stage_tp2_consumed = False
        s.trend_runner_active = False
        s.tp_plan = "THREE_STAGE_RUNNER"
        cvd = _cvd()

        with mock.patch.object(
            strategy, "_select_three_stage_tp2_outer",
            wraps=strategy._select_three_stage_tp2_outer,
        ) as mock_selector:
            strategy._maybe_update_tp(1650.0, 2000, boll, cvd)
            mock_selector.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
