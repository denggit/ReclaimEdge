"""Tests for Entry/Add Flow Coordinator (Phase 41).

Verifies:
- Wrapper delegates correctly from _maybe_open_or_add_long/short, _open_position.
- LONG/SHORT open, add side mismatch, add gate skips, successful adds, open_position
  plan branches (SPLIT, MIDDLE_RUNNER, THREE_STAGE_RUNNER).
"""

from __future__ import annotations

import unittest

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)
from src.strategies.entry_add_flow_coordinator import EntryAddFlowCoordinator


# ── reusable helpers ────────────────────────────────────────────────────

def _boll(middle: float = 2000.0, upper: float = 2100.0, lower: float = 1900.0,
          candle_ts_ms: int = 1000) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=candle_ts_ms,
        close=middle,
        middle=middle,
        upper=upper,
        lower=lower,
        upper_distance_pct=0.05,
        lower_distance_pct=0.05,
        alert_switch_on=True,
        live_mode=True,
    )


def _cvd(buy_ratio: float = 0.6, sell_ratio: float = 0.4) -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=1000,
        price=2000.0,
        side="buy",
        size=1.0,
        signed_delta=0.1,
        total_cvd=0.5,
        fast_cvd=0.1,
        previous_fast_cvd=0.09,
        buy_volume=60.0,
        sell_volume=40.0,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=False,
        window_low=1990.0,
        window_high=2010.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.01,
        baseline_range_pct=0.001,
        burst_move_ratio=10.0,
        burst_volume=10.0,
        baseline_volume=1.0,
        burst_volume_ratio=10.0,
        up_burst=False,
        down_burst=False,
    )


def _sizer() -> SimplePositionSizer:
    return SimplePositionSizer(SimplePositionSizerConfig())


def _strategy(**overrides) -> BollCvdReclaimStrategy:
    values = dict(min_outside_pct=0.001)
    values.update(overrides)
    config = BollCvdReclaimStrategyConfig(**values)
    sizer = _sizer()
    return BollCvdReclaimStrategy(config, sizer)


def _coordinator(strategy: BollCvdReclaimStrategy) -> EntryAddFlowCoordinator:
    return EntryAddFlowCoordinator(strategy)


# ── wrapper delegate tests ──────────────────────────────────────────────

class EntryAddFlowCoordinatorWrapperTest(unittest.TestCase):
    """Verify that strategy methods delegate through to the coordinator."""

    def test_wrapper_maybe_open_or_add_long_delegates(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        # When side is None, LONG open should be triggered
        intent = strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "OPEN_LONG")
        self.assertEqual(strat.state.side, "LONG")
        self.assertEqual(strat.state.layers, 1)
        # Verify the coordinator was lazily created
        self.assertTrue(hasattr(strat, "_entry_add_flow_coordinator"))

    def test_wrapper_maybe_open_or_add_short_delegates(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        intent = strat._maybe_open_or_add_short(2100.0, 1000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "OPEN_SHORT")
        self.assertEqual(strat.state.side, "SHORT")
        self.assertEqual(strat.state.layers, 1)

    def test_wrapper_open_position_delegates(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        intent = strat._open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd,
                                       "test open")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "OPEN_LONG")
        self.assertEqual(strat.state.side, "LONG")

    def test_entry_add_flow_lazy_init_reuses_instance(self) -> None:
        strat = _strategy()
        c1 = strat._entry_add_flow()
        c2 = strat._entry_add_flow()
        self.assertIs(c1, c2)


# ── LONG open ───────────────────────────────────────────────────────────

class EntryAddFlowCoordinatorLongOpenTest(unittest.TestCase):
    """LONG open: state.side is None → OPEN_LONG intent with full initialisation."""

    def test_long_open_creates_open_long_intent(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        intent = strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "OPEN_LONG")
        self.assertEqual(intent.side, "LONG")

    def test_long_open_sets_state_fields(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        self.assertEqual(strat.state.side, "LONG")
        self.assertEqual(strat.state.layers, 1)
        self.assertEqual(strat.state.last_entry_price, 1900.0)
        self.assertEqual(strat.state.first_entry_ts_ms, 1000)
        self.assertIsNotNone(strat.state.tp_price)
        self.assertIsNotNone(strat.state.tp_mode)
        self.assertGreater(strat.state.total_entry_qty, 0.0)
        self.assertGreater(strat.state.total_entry_notional, 0.0)
        self.assertGreater(strat.state.avg_entry_price, 0.0)

    def test_long_open_no_sidecar_fields_on_state(self) -> None:
        """After Sidecar removal, StrategyPositionState must not have sidecar fields."""
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        for field in ("sidecar_total_qty", "sidecar_open_qty", "sidecar_legs",
                      "sidecar_enabled_for_position", "sidecar_dirty",
                      "sidecar_halt_reason", "sidecar_margin_pct", "sidecar_tp_pct"):
            assert not hasattr(strat.state, field), f"StrategyPositionState should not have {field}"

    def test_long_open_initialises_cost_basis(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        # After _update_position_cost, these reflect the entry; they are NOT zero.
        self.assertGreater(strat.state.position_cost_entry_notional, 0.0)
        self.assertEqual(strat.state.position_cost_exit_notional, 0.0)
        self.assertGreater(strat.state.position_cost_remaining_qty, 0.0)
        # net_remaining_breakeven_price is computed from cost basis
        self.assertGreater(strat.state.net_remaining_breakeven_price, 0.0)

    def test_long_open_writes_last_order_and_tp_update_ts(self) -> None:
        strat = _strategy()
        boll = _boll(candle_ts_ms=2000)
        cvd = _cvd()
        strat._maybe_open_or_add_long(1900.0, 5000, boll, cvd)
        self.assertEqual(strat.state.last_order_ts_ms, 5000)
        self.assertEqual(strat.state.last_tp_update_ts_ms, 5000)
        self.assertEqual(strat.state.last_tp_update_candle_ts_ms, 2000)

    def test_long_open_logs_tp_selected(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        with self.assertLogs("src.strategies.entry_add_flow_coordinator", level="INFO") as logs:
            strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        output = "\n".join(logs.output)
        self.assertIn("TP_SELECTED | reason=entry", output)


# ── SHORT open ──────────────────────────────────────────────────────────

class EntryAddFlowCoordinatorShortOpenTest(unittest.TestCase):
    """SHORT open: state.side is None → OPEN_SHORT intent."""

    def test_short_open_creates_open_short_intent(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        intent = strat._maybe_open_or_add_short(2100.0, 1000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "OPEN_SHORT")
        self.assertEqual(intent.side, "SHORT")

    def test_short_open_sets_state_fields(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        strat._maybe_open_or_add_short(2100.0, 1000, boll, cvd)
        self.assertEqual(strat.state.side, "SHORT")
        self.assertEqual(strat.state.layers, 1)
        self.assertEqual(strat.state.last_entry_price, 2100.0)


# ── side mismatch ───────────────────────────────────────────────────────

class EntryAddFlowCoordinatorSideMismatchTest(unittest.TestCase):
    """When state.side mismatches the intent direction, return None."""

    def test_long_add_on_short_position_returns_none(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        # Open a SHORT first
        strat._maybe_open_or_add_short(2100.0, 1000, boll, cvd)
        self.assertEqual(strat.state.side, "SHORT")
        # Now try LONG
        intent = strat._maybe_open_or_add_long(1900.0, 2000, boll, cvd)
        self.assertIsNone(intent)

    def test_short_add_on_long_position_returns_none(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        # Open a LONG first
        strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        self.assertEqual(strat.state.side, "LONG")
        # Now try SHORT
        intent = strat._maybe_open_or_add_short(2100.0, 2000, boll, cvd)
        self.assertIsNone(intent)


# ── add disabled gates ──────────────────────────────────────────────────

@unittest.skip("ADD is disabled in the risk-first single-entry runtime")
class EntryAddFlowCoordinatorAddDisabledTest(unittest.TestCase):
    """Add skips when protection flags are active."""

    def _setup_long_position(self, strat: BollCvdReclaimStrategy) -> None:
        boll = _boll()
        cvd = _cvd()
        strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        self.assertEqual(strat.state.side, "LONG")

    def test_trend_runner_active_returns_none(self) -> None:
        strat = _strategy()
        self._setup_long_position(strat)
        strat.state.trend_runner_active = True
        strat.state.last_add_skip_log_reason = None
        intent = strat._maybe_open_or_add_long(1850.0, 2000, _boll(), _cvd())
        self.assertIsNone(intent)
        self.assertEqual(strat.state.last_add_skip_log_reason, "trend_runner_active")

    def test_three_stage_after_tp1_returns_none(self) -> None:
        strat = _strategy()
        self._setup_long_position(strat)
        strat.state.three_stage_runner_enabled_for_position = True
        strat.state.three_stage_tp1_consumed = True
        strat.state.last_add_skip_log_reason = None
        intent = strat._maybe_open_or_add_long(1850.0, 2000, _boll(), _cvd())
        self.assertIsNone(intent)
        self.assertEqual(strat.state.last_add_skip_log_reason, "three_stage_after_tp1")

    def test_middle_runner_active_returns_none(self) -> None:
        strat = _strategy()
        self._setup_long_position(strat)
        strat.state.middle_runner_active = True
        strat.state.last_add_skip_log_reason = None
        intent = strat._maybe_open_or_add_long(1850.0, 2000, _boll(), _cvd())
        self.assertIsNone(intent)
        self.assertEqual(strat.state.last_add_skip_log_reason, "middle_runner_active")

    def test_middle_runner_add_disabled_returns_none(self) -> None:
        strat = _strategy()
        self._setup_long_position(strat)
        strat.state.middle_runner_add_disabled = True
        strat.state.last_add_skip_log_reason = None
        intent = strat._maybe_open_or_add_long(1850.0, 2000, _boll(), _cvd())
        self.assertIsNone(intent)
        self.assertEqual(strat.state.last_add_skip_log_reason, "middle_runner_active")

    def test_max_layers_returns_none(self) -> None:
        strat = _strategy(max_layers=1)
        self._setup_long_position(strat)
        intent = strat._maybe_open_or_add_long(1800.0, 2000, _boll(), _cvd())
        self.assertIsNone(intent)

    def test_missing_last_entry_price_returns_none(self) -> None:
        strat = _strategy()
        # Set side and layers > 0 but last_entry_price = None
        strat.state.side = "LONG"
        strat.state.layers = 1
        strat.state.last_entry_price = None
        intent = strat._maybe_open_or_add_long(1850.0, 2000, _boll(), _cvd())
        self.assertIsNone(intent)


# ── add timing gate ─────────────────────────────────────────────────────

@unittest.skip("ADD is disabled in the risk-first single-entry runtime")
class EntryAddFlowCoordinatorTimingGateTest(unittest.TestCase):
    """Timing gate blocks add when interval not passed."""

    def test_timing_gate_first_add_block_logs_add_skipped(self) -> None:
        strat = _strategy(first_add_block_seconds=1800)
        boll = _boll()
        cvd = _cvd()
        # Open LONG at ts=1000
        strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        self.assertEqual(strat.state.side, "LONG")
        self.assertEqual(strat.state.layers, 1)
        # Try add immediately → should be blocked by first_add_block
        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            intent = strat._maybe_open_or_add_long(1850.0, 2000, boll, cvd)
        self.assertIsNone(intent)
        output = "\n".join(logs.output)
        self.assertIn("ADD_SKIPPED | reason=first_add_block", output)


# ── add gap gate ────────────────────────────────────────────────────────

@unittest.skip("ADD is disabled in the risk-first single-entry runtime")
class EntryAddFlowCoordinatorGapGateTest(unittest.TestCase):
    """Gap gate blocks add when price is too close to last_entry_price."""

    def test_gap_not_passed_logs_add_skipped(self) -> None:
        strat = _strategy(first_add_block_seconds=0,
                          add_min_interval_seconds=0)
        boll = _boll()
        cvd = _cvd()
        # Open LONG at 1900.0
        strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        self.assertEqual(strat.state.side, "LONG")
        # Try add at 1899.0 (only ~0.05% away, gap > 0.03 or 3%? Actually 1900 vs 1899 is ~0.05%)
        # Use a price very close to last_entry → gap should fail
        with self.assertLogs("src.strategies.entry_add_flow_coordinator", level="INFO") as logs:
            intent = strat._maybe_open_or_add_long(1899.0, 2000, boll, cvd)
        self.assertIsNone(intent)
        output = "\n".join(logs.output)
        self.assertIn("ADD_SKIPPED | reason=add_gap", output)


# ── successful LONG add ─────────────────────────────────────────────────

@unittest.skip("ADD is disabled in the risk-first single-entry runtime")
class EntryAddFlowCoordinatorSuccessfulLongAddTest(unittest.TestCase):
    """Full LONG add flow when all gates pass."""

    def test_successful_long_add_returns_add_long_intent(self) -> None:
        strat = _strategy(add_min_avg_improvement_pct=0.0,
                          first_add_block_seconds=0,
                          add_min_interval_seconds=0)
        boll = _boll()
        cvd = _cvd()
        # Open LONG at 1900.0
        strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        self.assertEqual(strat.state.side, "LONG")
        self.assertEqual(strat.state.layers, 1)
        # Add at a meaningfully lower price (further from last_entry in LONG → lower price is good)
        intent = strat._maybe_open_or_add_long(1800.0, 2000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "ADD_LONG")
        self.assertEqual(strat.state.layers, 2)

    def test_successful_long_add_reason_includes_gap_and_improvement_and_text(self) -> None:
        strat = _strategy(add_min_avg_improvement_pct=0.0,
                          first_add_block_seconds=0,
                          add_min_interval_seconds=0)
        boll = _boll()
        cvd = _cvd()
        strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        intent = strat._maybe_open_or_add_long(1800.0, 2000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertIn("距离上一多仓超过", intent.reason)
        self.assertIn("补仓后均价改善", intent.reason)
        self.assertIn("新出轨深度达标后低点附近再次跌不动", intent.reason)

    def _strategy_with_degraded_three_stage(self, first_ts: int) -> BollCvdReclaimStrategy:
        strat = _strategy(
            three_stage_runner_enabled=True,
            breakeven_fee_buffer_pct=0.0,
            tp_min_net_profit_pct=0.001,
        )
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            last_entry_price=100.0,
            first_entry_ts_ms=first_ts,
            last_order_ts_ms=first_ts,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            position_cost_entry_notional=100.0,
            position_cost_remaining_qty=1.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=110.0,
            tp_mode="UPPER",
            tp_plan="SINGLE",
            three_stage_pre_tp1_degrade_stage="SINGLE",
        )
        return strat

    def test_add_replan_before_3h_recovers_three_stage_from_sticky_single(self) -> None:
        first_ts = 100_000
        strat = self._strategy_with_degraded_three_stage(first_ts)
        boll = _boll(middle=101.0, upper=112.0, lower=90.0)
        cvd = _cvd()

        intent = _coordinator(strat).open_position(
            "LONG", "ADD_LONG", 90.0, first_ts + 2 * 60 * 60 * 1000, boll, cvd, "base"
        )

        self.assertIsNotNone(intent)
        self.assertIsNone(strat.state.three_stage_pre_tp1_degrade_stage)
        self.assertEqual(strat.state.tp_plan, "THREE_STAGE_RUNNER")
        self.assertEqual(intent.tp_plan, "THREE_STAGE_RUNNER")

    def test_add_replan_3h_to_6h_caps_recovery_at_middle_runner(self) -> None:
        first_ts = 100_000
        strat = self._strategy_with_degraded_three_stage(first_ts)
        boll = _boll(middle=101.0, upper=112.0, lower=90.0)
        cvd = _cvd()

        intent = _coordinator(strat).open_position(
            "LONG", "ADD_LONG", 90.0, first_ts + 4 * 60 * 60 * 1000, boll, cvd, "base"
        )

        self.assertIsNotNone(intent)
        self.assertEqual(strat.state.three_stage_pre_tp1_degrade_stage, "MIDDLE_RUNNER")
        self.assertEqual(strat.state.tp_plan, "MIDDLE_RUNNER")
        self.assertNotEqual(intent.tp_plan, "THREE_STAGE_RUNNER")

    def test_add_replan_after_6h_remains_single(self) -> None:
        first_ts = 100_000
        strat = self._strategy_with_degraded_three_stage(first_ts)
        strat.state.three_stage_pre_tp1_degrade_stage = None
        boll = _boll(middle=101.0, upper=112.0, lower=90.0)
        cvd = _cvd()

        intent = _coordinator(strat).open_position(
            "LONG", "ADD_LONG", 90.0, first_ts + 7 * 60 * 60 * 1000, boll, cvd, "base"
        )

        self.assertIsNotNone(intent)
        self.assertEqual(strat.state.three_stage_pre_tp1_degrade_stage, "SINGLE")
        self.assertEqual(strat.state.tp_plan, "SINGLE")


# ── successful SHORT add ────────────────────────────────────────────────

@unittest.skip("ADD is disabled in the risk-first single-entry runtime")
class EntryAddFlowCoordinatorSuccessfulShortAddTest(unittest.TestCase):
    """Full SHORT add flow when all gates pass."""

    def test_successful_short_add_returns_add_short_intent(self) -> None:
        strat = _strategy(add_min_avg_improvement_pct=0.0,
                          first_add_block_seconds=0,
                          add_min_interval_seconds=0)
        boll = _boll()
        cvd = _cvd()
        strat._maybe_open_or_add_short(2100.0, 1000, boll, cvd)
        self.assertEqual(strat.state.side, "SHORT")
        self.assertEqual(strat.state.layers, 1)
        intent = strat._maybe_open_or_add_short(2200.0, 2000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "ADD_SHORT")
        self.assertEqual(strat.state.layers, 2)

    def test_successful_short_add_reason_includes_short_text(self) -> None:
        strat = _strategy(add_min_avg_improvement_pct=0.0,
                          first_add_block_seconds=0,
                          add_min_interval_seconds=0)
        boll = _boll()
        cvd = _cvd()
        strat._maybe_open_or_add_short(2100.0, 1000, boll, cvd)
        intent = strat._maybe_open_or_add_short(2200.0, 2000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertIn("距离上一空仓超过", intent.reason)
        self.assertIn("补仓后均价改善", intent.reason)
        self.assertIn("新出轨深度达标后高点附近再次涨不动", intent.reason)


# ── open_position plan branches ─────────────────────────────────────────

class EntryAddFlowCoordinatorOpenPositionPlanTest(unittest.TestCase):
    """open_position TP plan branches (MIDDLE, THREE-STAGE)."""

    def test_middle_runner_plan_calls_set_middle_runner_planned(self) -> None:
        strat = _strategy(middle_runner_enabled=True)
        boll = _boll()
        cvd = _cvd()
        # Position the strategy so middle runner can be selected
        strat.state.avg_entry_price = 1900.0
        coord = _coordinator(strat)
        intent = coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")
        self.assertIsNotNone(intent)
        # If middleware runner got selected, verify state
        if strat.state.tp_plan == "MIDDLE_RUNNER":
            self.assertTrue(strat.state.middle_runner_enabled_for_position)

    def test_three_stage_runner_plan_calls_set_three_stage_runner_planned(self) -> None:
        strat = _strategy(three_stage_runner_enabled=True)
        boll = _boll()
        cvd = _cvd()
        coord = _coordinator(strat)
        intent = coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")
        self.assertIsNotNone(intent)
        if strat.state.tp_plan == "THREE_STAGE_RUNNER":
            self.assertTrue(strat.state.three_stage_runner_enabled_for_position)

    def test_tp_boll_price_selected_phase_initial_logged(self) -> None:
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        coord = _coordinator(strat)
        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")
        output = "\n".join(logs.output)
        self.assertIn("TP_BOLL_PRICE_SELECTED", output)
        self.assertIn("phase=initial", output)


# ── Base strategy behaviour preservation ───────────────────────────────────

class EntryAddFlowCoordinatorBaseStrategyPreservationTest(unittest.TestCase):
    """Verify that the fix (calling strategy._open_position from coordinator)
    does NOT break BollCvdReclaimStrategy (base class) behaviour."""

    def test_base_strategy_maybe_open_or_add_short_still_opens(self) -> None:
        """Base strategy _maybe_open_or_add_short → coordinator →
        strategy._open_position → base._open_position → coordinator.open_position.
        Must return valid OPEN_SHORT intent without recursion."""
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        # Ensure OPEN path
        strat.state.side = None
        intent = strat._maybe_open_or_add_short(2100.0, 1000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "OPEN_SHORT")
        self.assertEqual(strat.state.side, "SHORT")
        self.assertEqual(strat.state.layers, 1)

    def test_base_strategy_maybe_open_or_add_long_still_opens(self) -> None:
        """Base strategy _maybe_open_or_add_long must still work."""
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        strat.state.side = None
        intent = strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "OPEN_LONG")
        self.assertEqual(strat.state.side, "LONG")
        self.assertEqual(strat.state.layers, 1)

    @unittest.skip("ADD is disabled in the risk-first single-entry runtime")
    def test_base_strategy_add_long_still_works(self) -> None:
        """Base strategy add path is intentionally disabled."""
        strat = _strategy(
            add_min_avg_improvement_pct=0.0,
            first_add_block_seconds=0,
            add_min_interval_seconds=0,
        )
        boll = _boll()
        cvd = _cvd()
        # Open first
        strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
        self.assertEqual(strat.state.layers, 1)
        # Add second
        intent = strat._maybe_open_or_add_long(1800.0, 2000, boll, cvd)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "ADD_LONG")
        self.assertEqual(strat.state.layers, 2)


# ── Anti-recursion guard ───────────────────────────────────────────────────

class EntryAddFlowCoordinatorNoRecursionTest(unittest.TestCase):
    """Verify that the fix does not introduce infinite recursion."""

    def test_no_recursion_via_maybe_open_or_add_short(self) -> None:
        """Call OPEN_SHORT via coordinator path on base strategy — must not recurse."""
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        strat.state.side = None
        import sys
        old_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            intent = strat._maybe_open_or_add_short(2100.0, 1000, boll, cvd)
            self.assertIsNotNone(intent)
            self.assertEqual(intent.intent_type, "OPEN_SHORT")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_no_recursion_via_maybe_open_or_add_long(self) -> None:
        """Call OPEN_LONG via coordinator path on base strategy — must not recurse."""
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        strat.state.side = None
        import sys
        old_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            intent = strat._maybe_open_or_add_long(1900.0, 1000, boll, cvd)
            self.assertIsNotNone(intent)
            self.assertEqual(intent.intent_type, "OPEN_LONG")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_open_position_via_strategy_still_works_no_recursion(self) -> None:
        """Direct strategy._open_position call works without recursion."""
        strat = _strategy()
        boll = _boll()
        cvd = _cvd()
        import sys
        old_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            intent = strat._open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "test")
            self.assertIsNotNone(intent)
            self.assertEqual(intent.intent_type, "OPEN_LONG")
        finally:
            sys.setrecursionlimit(old_limit)


if __name__ == "__main__":
    unittest.main()
