from __future__ import annotations

import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import MethodType
from unittest.mock import patch

from src.execution.trader import PositionSnapshot, Trader
from tests.conftest import FakeOkxClient
from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.position_management.runner_live_helpers import middle_runner_size_mismatch_needs_degraded_protection
from src.position_management.tp_progress import (
    mark_middle_runner_active_if_position_reduced,
)
from src.reporting.trade_journal import LiveTradeJournal
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig
from src.execution.okx_trading_client import OkxTradingClient
from src.execution.tp_sl_execution_manager import TpSlExecutionManager
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
)


def strategy(**config_overrides) -> BollCvdReclaimStrategy:
    config_values = dict(
        entry_rr_target="FINAL_TP",
        entry_max_stop_distance_pct=0.0,
    )
    config_values.update(config_overrides)
    return BollCvdReclaimStrategy(
        BollCvdReclaimStrategyConfig(**config_values),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )


def boll(middle: float = 112.0, upper: float = 120.0, lower: float = 91.0, candle_ts_ms: int = 1_000) -> BollSnapshot:
    return BollSnapshot("ETH-USDT-SWAP", candle_ts_ms, 100.0, middle, upper, lower, 0.1, 0.1, True, True)


def cvd() -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=1_000,
        price=100.0,
        side="buy",
        size=1.0,
        signed_delta=1.0,
        total_cvd=1.0,
        fast_cvd=1.0,
        previous_fast_cvd=0.0,
        buy_volume=1.0,
        sell_volume=0.0,
        buy_ratio=1.0,
        sell_ratio=0.0,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=True,
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


def intent(**overrides) -> TradeIntent:
    values = dict(
        intent_type="UPDATE_TP",
        side="LONG",
        price=100.0,
        layer_index=4,
        tp_price=110.0,
        reason="test",
        size=PositionSize(1.0, 50.0, 0.5, 4, 1.45),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        boll_upper=110.0,
        boll_middle=105.0,
        boll_lower=95.0,
        ts_ms=1_000,
        avg_entry_price=100.0,
        breakeven_price=100.1,
        tp_mode="MIDDLE",
        partial_tp_price=None,
        partial_tp_ratio=0.0,
        tp_plan="SINGLE",
        partial_tp_consumed=False,
    )
    values.update(overrides)
    return TradeIntent(**values)  # type: ignore[arg-type]


class SplitTakeProfitStrategyTest(unittest.TestCase):
    def test_middle_runner_config_defaults_and_env_clamp(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = BollCvdReclaimStrategyConfig.from_env()
        self.assertFalse(config.middle_runner_enabled)
        self.assertEqual(config.middle_runner_first_close_ratio, 0.8)

        with patch.dict(
                os.environ,
                {
                    "MIDDLE_RUNNER_ENABLED": "true",
                    "MIDDLE_RUNNER_FIRST_CLOSE_RATIO": "0.99",
                    "MIDDLE_RUNNER_EXTENSION_TRIGGER_RATIO": "0.6",
                },
                clear=True,
        ):
            config = BollCvdReclaimStrategyConfig.from_env()
        self.assertTrue(config.middle_runner_enabled)
        self.assertEqual(config.middle_runner_first_close_ratio, 0.95)
        self.assertEqual(config.middle_runner_extension_trigger_ratio, 0.6)

        with patch.dict(os.environ, {"MIDDLE_RUNNER_FIRST_CLOSE_RATIO": "0.01"}, clear=True):
            config = BollCvdReclaimStrategyConfig.from_env()
        self.assertEqual(config.middle_runner_first_close_ratio, 0.1)

    def test_long_tp_switches_to_upper_when_middle_net_profit_below_threshold(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state.avg_entry_price = 100.0

        tp_price, mode = strat._select_tp_price("LONG", boll(middle=100.25, upper=110.0))

        self.assertEqual(mode, "UPPER")
        self.assertEqual(tp_price, 110.0)
        self.assertAlmostEqual(strat.state.breakeven_price, 100.1)

    def test_long_tp_uses_middle_at_or_above_min_net_profit_threshold(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state.avg_entry_price = 100.0

        tp_price, mode = strat._select_tp_price("LONG", boll(middle=100.31, upper=110.0))
        self.assertEqual(mode, "MIDDLE")
        self.assertAlmostEqual(tp_price, 100.31)

        tp_price, mode = strat._select_tp_price("LONG", boll(middle=100.40, upper=110.0))
        self.assertEqual(mode, "MIDDLE")
        self.assertAlmostEqual(tp_price, 100.40)

    def test_short_tp_switches_to_lower_when_middle_net_profit_below_threshold(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state.avg_entry_price = 100.0

        tp_price, mode = strat._select_tp_price("SHORT", boll(middle=99.75, lower=90.0))

        self.assertEqual(mode, "LOWER")
        self.assertEqual(tp_price, 90.0)
        self.assertAlmostEqual(strat.state.breakeven_price, 99.9)

    def test_short_tp_uses_middle_at_or_below_min_net_profit_threshold(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state.avg_entry_price = 100.0

        tp_price, mode = strat._select_tp_price("SHORT", boll(middle=99.70, lower=90.0))
        self.assertEqual(mode, "MIDDLE")
        self.assertAlmostEqual(tp_price, 99.70)

        tp_price, mode = strat._select_tp_price("SHORT", boll(middle=99.69, lower=90.0))
        self.assertEqual(mode, "MIDDLE")
        self.assertAlmostEqual(tp_price, 99.69)


    def test_open_position_initializes_net_remaining_breakeven_long(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001)

        strat._open_position("LONG", "OPEN_LONG", 100.0, 2_000, boll(), cvd(), "test")

        self.assertGreater(strat.state.position_cost_entry_notional, 0)
        self.assertGreater(strat.state.position_cost_remaining_qty, 0)
        self.assertGreater(strat.state.net_remaining_breakeven_price, 0)
        self.assertAlmostEqual(strat.state.net_remaining_breakeven_price, strat.state.avg_entry_price * 1.001)

    def test_open_position_initializes_net_remaining_breakeven_short(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001)

        strat._open_position("SHORT", "OPEN_SHORT", 100.0, 2_000, boll(middle=95.0, upper=104.0, lower=90.0), cvd(),
                             "test")

        self.assertGreater(strat.state.position_cost_entry_notional, 0)
        self.assertGreater(strat.state.position_cost_remaining_qty, 0)
        self.assertGreater(strat.state.net_remaining_breakeven_price, 0)
        self.assertAlmostEqual(strat.state.net_remaining_breakeven_price, strat.state.avg_entry_price * 0.999)

    def test_middle_runner_long_uses_middle_first_and_upper_runner(self) -> None:
        strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state.avg_entry_price = 100.0
        bands = boll(middle=101.0, upper=110.0, lower=91.0)

        intent_ = strat._open_position("LONG", "OPEN_LONG", 100.0, 2_000, bands, cvd(), "test")

        self.assertEqual(intent_.tp_mode, "MIDDLE")
        self.assertEqual(intent_.tp_plan, "MIDDLE_RUNNER")
        self.assertEqual(intent_.partial_tp_price, 101.0)
        self.assertEqual(intent_.partial_tp_ratio, 0.8)
        self.assertEqual(intent_.tp_price, 110.0)
        self.assertTrue(strat.state.middle_runner_pending)
        self.assertAlmostEqual(strat.state.middle_runner_keep_ratio, 0.2)

    def test_middle_runner_short_uses_middle_first_and_lower_runner(self) -> None:
        strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state.avg_entry_price = 100.0
        bands = boll(middle=99.0, upper=108.0, lower=90.0)

        intent_ = strat._open_position("SHORT", "OPEN_SHORT", 100.0, 2_000, bands, cvd(), "test")

        self.assertEqual(intent_.tp_mode, "MIDDLE")
        self.assertEqual(intent_.tp_plan, "MIDDLE_RUNNER")
        self.assertEqual(intent_.partial_tp_price, 99.0)
        self.assertEqual(intent_.partial_tp_ratio, 0.8)
        self.assertEqual(intent_.tp_price, 90.0)

    def test_middle_runner_not_used_when_final_tp_switches_outer_or_disabled(self) -> None:
        enabled = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        enabled.state.avg_entry_price = 100.0
        tp_price, mode = enabled._select_tp_price("LONG", boll(middle=100.1, upper=110.0))
        partial_tp, partial_ratio, plan = enabled._select_tp_plan("LONG", tp_price, 1, tp_mode=mode,
                                                                  boll=boll(middle=100.1, upper=110.0))
        self.assertEqual(mode, "UPPER")
        self.assertEqual(plan, "SINGLE")
        self.assertIsNone(partial_tp)
        self.assertEqual(partial_ratio, 0.0)

        disabled = strategy(middle_runner_enabled=False)
        disabled.state.avg_entry_price = 100.0
        partial_tp, partial_ratio, plan = disabled._select_tp_plan("LONG", 101.0, 1, tp_mode="MIDDLE",
                                                                   boll=boll(middle=101.0, upper=110.0))
        self.assertEqual(plan, "SINGLE")
        self.assertIsNone(partial_tp)
        self.assertEqual(partial_ratio, 0.0)

    def test_middle_runner_sl_calculation_and_tightening(self) -> None:
        long_strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        long_strat.state.avg_entry_price = 100.0
        long_sl = long_strat._calculate_middle_runner_protective_sl("LONG", 103.0, boll(middle=102.0, lower=96.0))
        # New relaxed: cost=100.1, structure=96 → max(100.1, 96) = 100.1
        self.assertAlmostEqual(long_sl or 0, max(100.1, 96.0))
        self.assertEqual(long_strat._tighten_middle_runner_sl("LONG", 101.5, 100.5), 101.5)

        short_strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        short_strat.state.avg_entry_price = 100.0
        short_sl = short_strat._calculate_middle_runner_protective_sl("SHORT", 97.0, boll(middle=98.0, upper=104.0))
        # New relaxed: cost=99.9, structure=104 → min(99.9, 104) = 99.9
        self.assertAlmostEqual(short_sl or 0, min(99.9, 104.0))
        self.assertEqual(short_strat._tighten_middle_runner_sl("SHORT", 98.5, 99.5), 98.5)

    def test_middle_runner_sl_uses_net_remaining_breakeven_when_present(self) -> None:
        # LONG protective SL = max(net_remaining_breakeven, boll_lower)
        long_strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        long_strat.state = StrategyPositionState(side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=95.0)

        # Scenario A: structure candidate wins (lower > net_breakeven)
        long_sl = long_strat._calculate_middle_runner_protective_sl("LONG", 103.0, boll(middle=102.0, lower=96.0))
        self.assertEqual(long_sl, max(95.0, 96.0))  # = 96.0 (structure wins)

        # Scenario B: breakeven candidate wins (lower is far below breakeven)
        long_sl_b = long_strat._calculate_middle_runner_protective_sl("LONG", 103.0, boll(middle=102.0, lower=90.0))
        self.assertEqual(long_sl_b, max(95.0, 90.0))  # = 95.0 (breakeven wins)

        # SHORT protective SL = min(net_remaining_breakeven, boll_upper)
        short_strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        short_strat.state = StrategyPositionState(side="SHORT", avg_entry_price=100.0,
                                                  net_remaining_breakeven_price=105.0)

        # Scenario C: structure candidate wins (upper < net_breakeven)
        short_sl = short_strat._calculate_middle_runner_protective_sl("SHORT", 97.0, boll(middle=98.0, upper=104.0))
        self.assertEqual(short_sl, min(105.0, 104.0))  # = 104.0 (structure wins)

        # Scenario D: breakeven candidate wins (upper is far above breakeven)
        short_sl_d = short_strat._calculate_middle_runner_protective_sl("SHORT", 97.0, boll(middle=98.0, upper=106.0))
        self.assertEqual(short_sl_d, min(105.0, 106.0))  # = 105.0 (breakeven wins)

    def test_middle_runner_uses_same_time_tighten_formula(self) -> None:
        long_strat = strategy(middle_runner_enabled=True)
        long_strat.state = StrategyPositionState(
            side="LONG",
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            middle_runner_active=True,
            middle_runner_sl_time_tighten_candle_count=1,
        )
        long_sl = long_strat._calculate_middle_runner_protective_sl("LONG", 250.0,
                                                                    boll(middle=200.0, upper=220.0, lower=100.0))

        short_strat = strategy(middle_runner_enabled=True)
        short_strat.state = StrategyPositionState(
            side="SHORT",
            avg_entry_price=200.0,
            net_remaining_breakeven_price=200.0,
            middle_runner_active=True,
            middle_runner_sl_time_tighten_candle_count=1,
        )
        short_sl = short_strat._calculate_middle_runner_protective_sl("SHORT", 50.0,
                                                                      boll(middle=100.0, upper=200.0, lower=80.0))

        self.assertEqual(long_strat._runner_sl_time_tighten_ratio(1), 0.55)
        # New relaxed logic: time tighten ratio ignored. cost=100, structure=100 → max=100
        self.assertEqual(long_sl, 100.0)
        # New relaxed logic: cost=200, structure=200 → min=200
        self.assertEqual(short_sl, 200.0)

    def test_candle_count_starts_at_zero_on_activation(self) -> None:
        strat = strategy(middle_runner_enabled=True)

        first_count = strat._advance_runner_sl_time_tighten_candle_count(target="middle_runner", candle_ts_ms=1_000)
        same_count = strat._advance_runner_sl_time_tighten_candle_count(target="middle_runner", candle_ts_ms=1_000)
        next_count = strat._advance_runner_sl_time_tighten_candle_count(target="middle_runner", candle_ts_ms=2_000)

        self.assertEqual(first_count, 0)
        self.assertEqual(same_count, 0)
        self.assertEqual(strat._runner_sl_time_tighten_ratio(first_count), 0.50)
        self.assertEqual(next_count, 1)
        self.assertEqual(strat._runner_sl_time_tighten_ratio(next_count), 0.55)

    def test_middle_runner_time_tighten_seeded_activation_next_real_candle_advances_to_55pct(self) -> None:
        strat = strategy(middle_runner_enabled=True)
        strat.state = StrategyPositionState(last_tp_update_candle_ts_ms=1_000)

        strat._seed_runner_sl_time_tighten_activation_candle(target="middle_runner", candle_ts_ms=0)

        self.assertEqual(strat.state.middle_runner_sl_time_tighten_candle_count, 0)
        self.assertEqual(strat.state.middle_runner_sl_time_tighten_last_candle_ts_ms, 1_000)
        self.assertEqual(strat._runner_sl_time_tighten_ratio(0), 0.50)

        count = strat._advance_runner_sl_time_tighten_candle_count(target="middle_runner", candle_ts_ms=2_000)

        self.assertEqual(count, 1)
        self.assertEqual(strat.state.middle_runner_sl_time_tighten_candle_count, 1)
        self.assertEqual(strat._runner_sl_time_tighten_ratio(count), 0.55)

    def test_middle_runner_sl_diag_logs_once_per_signature(self) -> None:
        strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        strat.state = StrategyPositionState(
            side="LONG",
            avg_entry_price=100.0,
            net_remaining_breakeven_price=95.0,
            position_cost_entry_notional=100.0,
            position_cost_exit_notional=76.0,
            position_cost_remaining_qty=0.2,
        )
        snapshot = boll(middle=102.0, lower=90.0, upper=110.0, candle_ts_ms=1_000)

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="WARNING") as logs:
            first = strat._calculate_middle_runner_protective_sl("LONG", 103.0, snapshot)
            second = strat._calculate_middle_runner_protective_sl("LONG", 103.0, snapshot)

        self.assertEqual(first, second)
        joined = "\n".join(logs.output)
        self.assertEqual(joined.count("MIDDLE_RUNNER_PROTECTIVE_SL_DIAG"), 1)
        self.assertIn("net_remaining_breakeven=95.0000", joined)
        self.assertIn("breakeven_source=net_remaining_breakeven", joined)
        self.assertIn("candidate_cost=95.0000", joined)
        self.assertIn("candidate_structure=90.0000", joined)
        self.assertIn("protective_sl=95.0000", joined)

    def test_middle_runner_sl_diag_logs_again_when_signature_changes(self) -> None:
        strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        strat.state = StrategyPositionState(side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=95.0)

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="WARNING") as logs:
            strat._calculate_middle_runner_protective_sl("LONG", 103.0, boll(middle=102.0, lower=90.0, upper=110.0,
                                                                             candle_ts_ms=1_000))
            strat._calculate_middle_runner_protective_sl("LONG", 103.0, boll(middle=102.0, lower=90.0, upper=110.0,
                                                                             candle_ts_ms=2_000))

        self.assertEqual("\n".join(logs.output).count("MIDDLE_RUNNER_PROTECTIVE_SL_DIAG"), 2)

    def test_middle_runner_sl_diag_logs_avg_entry_fallback(self) -> None:
        strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        strat.state = StrategyPositionState(side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=0.0)

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="WARNING") as logs:
            strat._calculate_middle_runner_protective_sl("LONG", 103.0, boll(middle=102.0, lower=90.0, upper=110.0,
                                                                             candle_ts_ms=1_000))

        joined = "\n".join(logs.output)
        self.assertIn("MIDDLE_RUNNER_PROTECTIVE_SL_DIAG", joined)
        self.assertIn("breakeven_source=avg_entry_fallback", joined)

    def test_middle_runner_extension_trigger_moves_sl_to_middle(self) -> None:
        long_strat = strategy(middle_runner_enabled=True, middle_runner_extension_trigger_ratio=0.6)
        long_strat.state.middle_runner_active = True
        new_sl = long_strat._apply_middle_runner_extension_trigger("LONG", 106.0, boll(middle=100.0, upper=110.0), 99.0)
        self.assertEqual(new_sl, 100.0)
        self.assertTrue(long_strat.state.middle_runner_extension_triggered)

        short_strat = strategy(middle_runner_enabled=True, middle_runner_extension_trigger_ratio=0.6)
        short_strat.state.middle_runner_active = True
        new_sl = short_strat._apply_middle_runner_extension_trigger("SHORT", 94.0, boll(middle=100.0, lower=90.0),
                                                                    101.0)
        self.assertEqual(new_sl, 100.0)
        self.assertTrue(short_strat.state.middle_runner_extension_triggered)

    def test_middle_runner_pending_and_active_new_candle_updates(self) -> None:
        strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            last_entry_price=100.0,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_mode="MIDDLE",
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=101.0,
            partial_tp_ratio=0.8,
            middle_runner_pending=True,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
            last_tp_update_candle_ts_ms=1_000,
        )

        pending_intent = strat._maybe_update_tp(100.0, 2_000,
                                                boll(middle=102.0, upper=112.0, lower=92.0, candle_ts_ms=2_000), cvd())
        self.assertIsNotNone(pending_intent)
        self.assertEqual(pending_intent.tp_plan, "MIDDLE_RUNNER")
        self.assertEqual(pending_intent.partial_tp_price, 102.0)
        self.assertEqual(pending_intent.tp_price, 112.0)
        self.assertEqual(strat.state.partial_tp_price, 102.0)
        self.assertEqual(strat.state.tp_price, 112.0)
        self.assertTrue(strat.state.middle_runner_pending)

        strat.state.middle_runner_pending = False
        strat.state.middle_runner_active = True
        strat.state.middle_runner_protective_sl_price = 101.5
        active_intent = strat._maybe_update_tp(106.0, 3_000,
                                               boll(middle=103.0, upper=113.0, lower=93.0, candle_ts_ms=3_000), cvd())
        self.assertIsNotNone(active_intent)
        self.assertEqual(active_intent.tp_price, 113.0)
        # New relaxed logic: cost=100.1, structure=93 → max=100.1 (looser than old 101.5, allowed)
        self.assertAlmostEqual(active_intent.middle_runner_protective_sl_price or 0, 100.1, places=3)

    def test_middle_runner_pending_does_not_migrate_to_three_stage_after_env_change(self) -> None:
        strat = strategy(middle_runner_enabled=False, three_stage_runner_enabled=True, breakeven_fee_buffer_pct=0.001,
                         tp_min_net_profit_pct=0.002)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            last_entry_price=100.0,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_mode="MIDDLE",
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=101.0,
            partial_tp_ratio=0.8,
            middle_runner_enabled_for_position=True,
            middle_runner_pending=True,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
            middle_runner_first_tp_price=101.0,
            middle_runner_final_tp_price=110.0,
            last_tp_update_candle_ts_ms=1_000,
        )

        # middle=100.5 is sufficient: effective_be=100.1, required=100.3002 < 100.5 → MIDDLE
        update_intent = strat._maybe_update_tp(100.0, 2_000,
                                               boll(middle=100.5, upper=111.0, lower=90.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(update_intent)
        self.assertEqual(update_intent.tp_plan, "MIDDLE_RUNNER")
        self.assertAlmostEqual(update_intent.partial_tp_price, 100.5)
        self.assertAlmostEqual(update_intent.tp_price, 111.0)
        self.assertTrue(strat.state.middle_runner_pending)
        self.assertTrue(strat.state.middle_runner_enabled_for_position)
        self.assertAlmostEqual(strat.state.middle_runner_keep_ratio, 0.2)
        self.assertEqual(strat.state.tp_plan, "MIDDLE_RUNNER")
        self.assertFalse(strat._three_stage_runner_plan_allowed("MIDDLE", boll(middle=100.5, upper=111.0, lower=90.0)))

    def test_middle_runner_active_keeps_old_sl_when_new_sl_calculation_returns_none(self) -> None:
        strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            last_entry_price=100.0,
            total_entry_qty=1.0,
            total_entry_notional=0.0,
            avg_entry_price=0.0,  # No cost basis → calculated_sl returns None
            tp_price=110.0,
            tp_mode="MIDDLE",
            partial_tp_consumed=True,
            middle_runner_enabled_for_position=True,
            middle_runner_active=True,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
            middle_runner_final_tp_price=110.0,
            middle_runner_protective_sl_price=101.5,
            middle_runner_protective_sl_order_id="algo-old",
            last_tp_update_candle_ts_ms=1_000,
        )

        update_intent = strat._maybe_update_tp(100.5, 2_000,
                                               boll(middle=103.0, upper=113.0, lower=97.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(update_intent)
        # calculated_sl returns None → keep old = 101.5
        self.assertEqual(strat.state.middle_runner_protective_sl_price, 101.5)
        self.assertEqual(update_intent.middle_runner_protective_sl_price, 101.5)
        self.assertEqual(strat.state.middle_runner_protective_sl_order_id, "algo-old")
        self.assertEqual(update_intent.middle_runner_protective_sl_order_id, "algo-old")

    # ── Middle-profit eligibility enforcement ──

    def test_middle_runner_pending_disables_when_middle_profit_insufficient(self) -> None:
        """Middle Runner pending (first close not done) must disable when middle profit insufficient."""
        strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=110.0,
            tp_mode="MIDDLE",
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=101.0,
            partial_tp_ratio=0.8,
            middle_runner_enabled_for_position=True,
            middle_runner_pending=True,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
            middle_runner_first_tp_price=101.0,
            middle_runner_final_tp_price=110.0,
            last_tp_update_candle_ts_ms=1_000,
        )

        # middle=100.1: required_middle = 100.0 * 1.002 = 100.2 > 100.1 → insufficient
        got = strat._maybe_update_tp(99.0, 2_000, boll(middle=100.1, upper=103.0, lower=97.0, candle_ts_ms=2_000),
                                     cvd())

        self.assertIsNotNone(got, "Must return UPDATE_TP when middle profit insufficient")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertEqual(got.tp_plan, "SINGLE")
        self.assertAlmostEqual(got.tp_price, 103.0)
        self.assertIsNone(got.partial_tp_price)
        self.assertEqual(got.partial_tp_ratio, 0.0)
        self.assertFalse(strat.state.middle_runner_pending)
        self.assertFalse(strat.state.middle_runner_enabled_for_position)

    def test_middle_runner_active_does_not_reset_when_middle_profit_insufficient(self) -> None:
        """Middle Runner active must NOT be reset when middle profit insufficient."""
        strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=110.0,
            tp_mode="MIDDLE",
            tp_plan="SINGLE",
            partial_tp_consumed=True,
            middle_runner_enabled_for_position=True,
            middle_runner_active=True,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
            middle_runner_final_tp_price=110.0,
            middle_runner_protective_sl_price=101.5,
            middle_runner_protective_sl_order_id="algo-old",
            last_tp_update_candle_ts_ms=1_000,
        )

        # middle insufficient but middle_runner_active → must NOT reset
        got = strat._maybe_update_tp(100.5, 2_000, boll(middle=100.1, upper=103.0, lower=97.0, candle_ts_ms=2_000),
                                     cvd())

        self.assertIsNotNone(got, "Must return UPDATE_TP for active middle runner")
        self.assertTrue(strat.state.middle_runner_active, "middle_runner_active must be preserved")
        self.assertIsNotNone(strat.state.middle_runner_protective_sl_price)


class SplitTakeProfitTraderTest(unittest.IsolatedAsyncioTestCase):
    def make_trader(self) -> Trader:
        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.position_contracts = Decimal("4")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.tp_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.contract_multiplier = Decimal("0.1")
        trader._client = FakeOkxClient(trader)
        trader.trading_client = OkxTradingClient(trader, private_client=trader._client)  # type: ignore[assignment]
        trader._tp_sl_manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)  # type: ignore[arg-type]
        return trader

    def test_build_middle_runner_order_specs_uses_first_and_runner_labels(self) -> None:
        trader = self.make_trader()
        specs = trader._build_take_profit_order_specs(
            intent(partial_tp_price=105.0, partial_tp_ratio=0.8, tp_plan="MIDDLE_RUNNER", tp_price=110.0)
        )

        self.assertEqual(specs, [("middle", Decimal("3.20"), 105.0), ("runner", Decimal("0.80"), 110.0)])


class SplitTakeProfitLifecycleTest(unittest.TestCase):
    def test_mark_middle_runner_active_when_position_reduced_to_keep_ratio(self) -> None:
        strat = strategy(middle_runner_enabled=True)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=105.0,
            partial_tp_ratio=0.8,
            middle_runner_pending=True,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
            middle_runner_first_tp_price=105.0,
            middle_runner_final_tp_price=110.0,
            last_tp_update_candle_ts_ms=1_000,
        )
        position = PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2"))

        activated = mark_middle_runner_active_if_position_reduced(strat, position)

        self.assertTrue(activated)
        self.assertTrue(strat.state.middle_runner_active)
        self.assertFalse(strat.state.middle_runner_pending)
        self.assertTrue(strat.state.middle_runner_add_disabled)
        self.assertTrue(strat.state.partial_tp_consumed)
        self.assertEqual(strat.state.tp_plan, "SINGLE")
        self.assertEqual(strat.state.middle_runner_sl_time_tighten_candle_count, 0)
        self.assertEqual(strat.state.middle_runner_sl_time_tighten_last_candle_ts_ms, 1_000)
        self.assertEqual(strat._advance_runner_sl_time_tighten_candle_count(target="middle_runner", candle_ts_ms=2_000),
                         1)

    def test_middle_runner_partial_size_mismatch_disables_add_without_activation(self) -> None:
        strat = strategy(middle_runner_enabled=True)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_plan="MIDDLE_RUNNER",
            middle_runner_pending=True,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
        )
        position = PositionSnapshot("LONG", Decimal("5"), 100.0, 0.5, Decimal("5"))

        activated = mark_middle_runner_active_if_position_reduced(strat, position)

        self.assertFalse(activated)
        self.assertFalse(strat.state.middle_runner_active)
        self.assertTrue(strat.state.middle_runner_pending)
        self.assertTrue(strat.state.middle_runner_add_disabled)
        self.assertTrue(middle_runner_size_mismatch_needs_degraded_protection(strat, position))


if __name__ == "__main__":
    unittest.main()
