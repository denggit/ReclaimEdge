from __future__ import annotations

import tempfile
import unittest
import os
from decimal import Decimal
from pathlib import Path
from types import MethodType
from unittest.mock import patch

from scripts.run_boll_cvd_live import (
    mark_middle_runner_active_if_position_reduced,
    mark_partial_tp_consumed_if_position_reduced,
    middle_runner_size_mismatch_needs_degraded_protection,
)
from src.execution.trader import PositionSnapshot, Trader
from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.reporting.trade_journal import LiveTradeJournal
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
)


def strategy(**config_overrides) -> BollCvdReclaimStrategy:
    config_values = dict(
        split_tp_min_layers=4,
        split_tp_path_ratio=0.8,
        split_tp_partial_ratio=0.5,
        split_tp_min_profit_pct=0.004,
    )
    config_values.update(config_overrides)
    return BollCvdReclaimStrategy(
        BollCvdReclaimStrategyConfig(**config_values),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )


def boll(middle: float = 110.0, upper: float = 120.0, lower: float = 90.0, candle_ts_ms: int = 1_000) -> BollSnapshot:
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
        self.assertTrue(config.split_tp_enabled)

        with patch.dict(
            os.environ,
            {
                "MIDDLE_RUNNER_ENABLED": "true",
                "MIDDLE_RUNNER_FIRST_CLOSE_RATIO": "0.99",
                "MIDDLE_RUNNER_EXTENSION_TRIGGER_RATIO": "0.6",
                "NEAR_TP_ENABLED": "false",
                "SPLIT_TP_ENABLED": "false",
            },
            clear=True,
        ):
            config = BollCvdReclaimStrategyConfig.from_env()
        self.assertTrue(config.middle_runner_enabled)
        self.assertEqual(config.middle_runner_first_close_ratio, 0.95)
        self.assertEqual(config.middle_runner_extension_trigger_ratio, 0.6)
        self.assertFalse(config.split_tp_enabled)

        with patch.dict(os.environ, {"MIDDLE_RUNNER_FIRST_CLOSE_RATIO": "0.01"}, clear=True):
            config = BollCvdReclaimStrategyConfig.from_env()
        self.assertEqual(config.middle_runner_first_close_ratio, 0.1)

    def test_middle_runner_and_near_tp_env_conflict_raises(self) -> None:
        with patch.dict(os.environ, {"MIDDLE_RUNNER_ENABLED": "true", "NEAR_TP_ENABLED": "true"}, clear=True):
            with self.assertRaises(RuntimeError):
                BollCvdReclaimStrategyConfig.from_env()

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

    def test_long_split_tp_uses_upper_when_final_tp_switches_to_upper(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state.avg_entry_price = 100.0

        tp_price, mode = strat._select_tp_price("LONG", boll(middle=100.25, upper=110.0))
        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", tp_price, 4)

        self.assertEqual(mode, "UPPER")
        # Middle profit insufficient → SPLIT is blocked; SINGLE only
        self.assertEqual(plan, "SINGLE")
        self.assertIsNone(partial_tp)
        self.assertEqual(partial_ratio, 0.0)

    def test_short_split_tp_uses_lower_when_final_tp_switches_to_lower(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state.avg_entry_price = 100.0

        tp_price, mode = strat._select_tp_price("SHORT", boll(middle=99.75, lower=90.0))
        partial_tp, partial_ratio, plan = strat._select_tp_plan("SHORT", tp_price, 4)

        self.assertEqual(mode, "LOWER")
        # Middle profit insufficient → SPLIT is blocked; SINGLE only
        self.assertEqual(plan, "SINGLE")
        self.assertIsNone(partial_tp)
        self.assertEqual(partial_ratio, 0.0)

    def test_long_split_tp_uses_80_pct_path_when_min_profit_is_inside_final_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", 110.0, 4, tp_mode="MIDDLE", boll=boll(middle=101.0, upper=110.0))

        self.assertEqual(plan, "SPLIT_PARTIAL_FINAL")
        self.assertEqual(partial_ratio, 0.5)
        self.assertAlmostEqual(partial_tp or 0, 108.0)

    def test_long_does_not_split_when_min_profit_would_exceed_final_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", 100.3, 4)

        self.assertEqual(plan, "SINGLE")
        self.assertEqual(partial_ratio, 0.0)
        self.assertIsNone(partial_tp)

    def test_short_split_tp_uses_80_pct_path_when_min_profit_is_inside_final_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("SHORT", 90.0, 4, tp_mode="MIDDLE", boll=boll(middle=99.0, upper=110.0, lower=90.0))

        self.assertEqual(plan, "SPLIT_PARTIAL_FINAL")
        self.assertEqual(partial_ratio, 0.5)
        self.assertAlmostEqual(partial_tp or 0, 92.0)

    def test_short_does_not_split_when_min_profit_would_exceed_final_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("SHORT", 99.7, 4)

        self.assertEqual(plan, "SINGLE")
        self.assertEqual(partial_ratio, 0.0)
        self.assertIsNone(partial_tp)

    def test_partial_tp_consumed_prevents_repeated_split(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0
        strat.state.partial_tp_consumed = True

        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", 110.0, 4)

        self.assertEqual(plan, "SINGLE")
        self.assertEqual(partial_ratio, 0.0)
        self.assertIsNone(partial_tp)

    def test_new_add_rearms_split_after_partial_consumed(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=4,
            last_entry_price=100.0,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            partial_tp_consumed=True,
            tp_plan="SINGLE",
        )

        new_intent = strat._open_position("LONG", "ADD_LONG", 90.0, 2_000, boll(), cvd(), "test add")

        self.assertFalse(new_intent.partial_tp_consumed)
        self.assertFalse(strat.state.partial_tp_consumed)
        self.assertEqual(new_intent.tp_plan, "SPLIT_PARTIAL_FINAL")
        self.assertIsNotNone(new_intent.partial_tp_price)

    def test_layers_below_threshold_keep_single_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", 110.0, 3)

        self.assertEqual(plan, "SINGLE")
        self.assertEqual(partial_ratio, 0.0)
        self.assertIsNone(partial_tp)

    def test_split_tp_enabled_false_keeps_single_tp(self) -> None:
        strat = strategy(split_tp_enabled=False)
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", 110.0, 4)

        self.assertEqual(plan, "SINGLE")
        self.assertEqual(partial_ratio, 0.0)
        self.assertIsNone(partial_tp)

    def test_open_position_initializes_net_remaining_breakeven_long(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001)

        strat._open_position("LONG", "OPEN_LONG", 100.0, 2_000, boll(), cvd(), "test")

        self.assertGreater(strat.state.position_cost_entry_notional, 0)
        self.assertGreater(strat.state.position_cost_remaining_qty, 0)
        self.assertGreater(strat.state.net_remaining_breakeven_price, 0)
        self.assertAlmostEqual(strat.state.net_remaining_breakeven_price, strat.state.avg_entry_price * 1.001)

    def test_open_position_initializes_net_remaining_breakeven_short(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001)

        strat._open_position("SHORT", "OPEN_SHORT", 100.0, 2_000, boll(middle=99.0, upper=110.0, lower=90.0), cvd(), "test")

        self.assertGreater(strat.state.position_cost_entry_notional, 0)
        self.assertGreater(strat.state.position_cost_remaining_qty, 0)
        self.assertGreater(strat.state.net_remaining_breakeven_price, 0)
        self.assertAlmostEqual(strat.state.net_remaining_breakeven_price, strat.state.avg_entry_price * 0.999)

    def test_middle_runner_long_uses_middle_first_and_upper_runner(self) -> None:
        strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state.avg_entry_price = 100.0
        bands = boll(middle=101.0, upper=110.0, lower=90.0)

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
        bands = boll(middle=99.0, upper=110.0, lower=90.0)

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
        partial_tp, partial_ratio, plan = enabled._select_tp_plan("LONG", tp_price, 1, tp_mode=mode, boll=boll(middle=100.1, upper=110.0))
        self.assertEqual(mode, "UPPER")
        self.assertEqual(plan, "SINGLE")
        self.assertIsNone(partial_tp)
        self.assertEqual(partial_ratio, 0.0)

        disabled = strategy(middle_runner_enabled=False)
        disabled.state.avg_entry_price = 100.0
        partial_tp, partial_ratio, plan = disabled._select_tp_plan("LONG", 101.0, 1, tp_mode="MIDDLE", boll=boll(middle=101.0, upper=110.0))
        self.assertEqual(plan, "SINGLE")
        self.assertIsNone(partial_tp)
        self.assertEqual(partial_ratio, 0.0)

    def test_middle_runner_sl_calculation_and_tightening(self) -> None:
        long_strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        long_strat.state.avg_entry_price = 100.0
        long_sl = long_strat._calculate_middle_runner_protective_sl("LONG", 103.0, boll(middle=102.0, lower=96.0))
        self.assertAlmostEqual(long_sl or 0, max((100.1 + 102.0) / 2, (96.0 + 102.0) / 2))
        self.assertEqual(long_strat._tighten_middle_runner_sl("LONG", 101.5, 100.5), 101.5)

        short_strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        short_strat.state.avg_entry_price = 100.0
        short_sl = short_strat._calculate_middle_runner_protective_sl("SHORT", 97.0, boll(middle=98.0, upper=104.0))
        self.assertAlmostEqual(short_sl or 0, min((99.9 + 98.0) / 2, (104.0 + 98.0) / 2))
        self.assertEqual(short_strat._tighten_middle_runner_sl("SHORT", 98.5, 99.5), 98.5)

    def test_middle_runner_sl_uses_net_remaining_breakeven_when_present(self) -> None:
        # LONG protective SL = max(
        #     (net_remaining_breakeven + middle) / 2,   ← candidate_1 (breakeven)
        #     (lower + middle) / 2,                     ← candidate_2 (structure)
        # )
        long_strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        long_strat.state = StrategyPositionState(side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=95.0)

        # Scenario A: structure candidate wins (lower is tight enough)
        long_sl = long_strat._calculate_middle_runner_protective_sl("LONG", 103.0, boll(middle=102.0, lower=96.0))
        candidate_breakeven = (95.0 + 102.0) / 2  # 98.5
        candidate_structure = (96.0 + 102.0) / 2  # 99.0
        self.assertEqual(long_sl, max(candidate_breakeven, candidate_structure))

        # Scenario B: breakeven candidate wins (lower is far away → structure candidate too loose)
        long_sl_b = long_strat._calculate_middle_runner_protective_sl("LONG", 103.0, boll(middle=102.0, lower=90.0))
        candidate_breakeven_b = (95.0 + 102.0) / 2   # 98.5
        candidate_structure_b = (90.0 + 102.0) / 2   # 96.0
        self.assertEqual(long_sl_b, max(candidate_breakeven_b, candidate_structure_b))
        # breakeven candidate wins when structure is looser
        self.assertEqual(long_sl_b, candidate_breakeven_b)

        # SHORT protective SL = min(
        #     (net_remaining_breakeven + middle) / 2,   ← candidate_1 (breakeven)
        #     (upper + middle) / 2,                     ← candidate_2 (structure)
        # )
        short_strat = strategy(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001)
        short_strat.state = StrategyPositionState(side="SHORT", avg_entry_price=100.0, net_remaining_breakeven_price=105.0)

        # Scenario C: structure candidate wins (upper is tight enough)
        short_sl = short_strat._calculate_middle_runner_protective_sl("SHORT", 97.0, boll(middle=98.0, upper=104.0))
        s_candidate_breakeven = (105.0 + 98.0) / 2  # 101.5
        s_candidate_structure = (104.0 + 98.0) / 2  # 101.0
        self.assertEqual(short_sl, min(s_candidate_breakeven, s_candidate_structure))

        # Scenario D: breakeven candidate wins (upper is far away → structure candidate too loose)
        short_sl_d = short_strat._calculate_middle_runner_protective_sl("SHORT", 97.0, boll(middle=98.0, upper=106.0))
        s_candidate_breakeven_d = (105.0 + 98.0) / 2  # 101.5
        s_candidate_structure_d = (106.0 + 98.0) / 2  # 102.0
        self.assertEqual(short_sl_d, min(s_candidate_breakeven_d, s_candidate_structure_d))
        # breakeven candidate wins when structure is looser
        self.assertEqual(short_sl_d, s_candidate_breakeven_d)

    def test_middle_runner_extension_trigger_moves_sl_to_middle(self) -> None:
        long_strat = strategy(middle_runner_enabled=True, middle_runner_extension_trigger_ratio=0.6)
        long_strat.state.middle_runner_active = True
        new_sl = long_strat._apply_middle_runner_extension_trigger("LONG", 106.0, boll(middle=100.0, upper=110.0), 99.0)
        self.assertEqual(new_sl, 100.0)
        self.assertTrue(long_strat.state.middle_runner_extension_triggered)

        short_strat = strategy(middle_runner_enabled=True, middle_runner_extension_trigger_ratio=0.6)
        short_strat.state.middle_runner_active = True
        new_sl = short_strat._apply_middle_runner_extension_trigger("SHORT", 94.0, boll(middle=100.0, lower=90.0), 101.0)
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

        pending_intent = strat._maybe_update_tp(100.0, 2_000, boll(middle=102.0, upper=112.0, lower=92.0, candle_ts_ms=2_000), cvd())
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
        active_intent = strat._maybe_update_tp(106.0, 3_000, boll(middle=103.0, upper=113.0, lower=93.0, candle_ts_ms=3_000), cvd())
        self.assertIsNotNone(active_intent)
        self.assertEqual(active_intent.tp_price, 113.0)
        self.assertGreaterEqual(active_intent.middle_runner_protective_sl_price or 0, 101.5)

    def test_middle_runner_pending_does_not_migrate_to_three_stage_after_env_change(self) -> None:
        strat = strategy(middle_runner_enabled=False, three_stage_runner_enabled=True, breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
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
        update_intent = strat._maybe_update_tp(100.0, 2_000, boll(middle=100.5, upper=111.0, lower=90.0, candle_ts_ms=2_000), cvd())

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
            total_entry_notional=100.0,
            avg_entry_price=100.0,
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

        update_intent = strat._maybe_update_tp(100.5, 2_000, boll(middle=103.0, upper=113.0, lower=97.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(update_intent)
        self.assertEqual(strat.state.middle_runner_protective_sl_price, 101.5)
        self.assertEqual(update_intent.middle_runner_protective_sl_price, 101.5)
        self.assertEqual(strat.state.middle_runner_protective_sl_order_id, "algo-old")
        self.assertEqual(update_intent.middle_runner_protective_sl_order_id, "algo-old")

    def test_near_tp_skips_middle_runner_pending_and_active(self) -> None:
        strat = strategy(
            near_tp_enabled=True,
            middle_runner_enabled=True,
            near_tp_min_progress_ratio=0.1,
            near_tp_min_profit_pct=0.0,
            near_tp_min_reduce_profit_pct=0.0,
        )
        base_state = dict(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_plan="MIDDLE_RUNNER",
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
        )

        strat.state = StrategyPositionState(**base_state, middle_runner_pending=True)
        self.assertIsNone(strat._maybe_near_tp_reduce(109.0, 2_000, boll(), cvd()))
        self.assertFalse(strat.state.near_tp_armed)

        strat.state = StrategyPositionState(**base_state, middle_runner_active=True)
        self.assertIsNone(strat._maybe_near_tp_reduce(109.0, 3_000, boll(), cvd()))
        self.assertFalse(strat.state.near_tp_armed)

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
        got = strat._maybe_update_tp(99.0, 2_000, boll(middle=100.1, upper=103.0, lower=97.0, candle_ts_ms=2_000), cvd())

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
        got = strat._maybe_update_tp(100.5, 2_000, boll(middle=100.1, upper=103.0, lower=97.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got, "Must return UPDATE_TP for active middle runner")
        self.assertTrue(strat.state.middle_runner_active, "middle_runner_active must be preserved")
        self.assertIsNotNone(strat.state.middle_runner_protective_sl_price)

    def test_split_partial_disabled_when_middle_profit_insufficient(self) -> None:
        """SPLIT with partial not consumed must fall back to SINGLE outer when middle profit insufficient."""
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002, split_tp_enabled=True, split_tp_min_layers=2)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=2,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=110.0,
            tp_mode="MIDDLE",
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_price=108.0,
            partial_tp_ratio=0.5,
            partial_tp_consumed=False,
            last_tp_update_candle_ts_ms=1_000,
        )

        # middle=100.1: required_middle = 100.0 * 1.002 = 100.2 > 100.1 → insufficient
        got = strat._maybe_update_tp(99.0, 2_000, boll(middle=100.1, upper=103.0, lower=97.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got, "Must return UPDATE_TP when middle profit insufficient")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertEqual(got.tp_plan, "SINGLE")
        self.assertAlmostEqual(got.tp_price, 103.0)
        self.assertIsNone(got.partial_tp_price)
        self.assertEqual(got.partial_tp_ratio, 0.0)


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
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        return trader

    def test_build_split_order_specs_rounds_half_position(self) -> None:
        trader = self.make_trader()
        specs = trader._build_take_profit_order_specs(
            intent(partial_tp_price=108.0, partial_tp_ratio=0.5, tp_plan="SPLIT_PARTIAL_FINAL")
        )

        self.assertEqual(specs, [("partial", Decimal("2.00"), 108.0), ("final", Decimal("2.00"), 110.0)])

    def test_build_middle_runner_order_specs_uses_first_and_runner_labels(self) -> None:
        trader = self.make_trader()
        specs = trader._build_take_profit_order_specs(
            intent(partial_tp_price=105.0, partial_tp_ratio=0.8, tp_plan="MIDDLE_RUNNER", tp_price=110.0)
        )

        self.assertEqual(specs, [("middle", Decimal("3.20"), 105.0), ("runner", Decimal("0.80"), 110.0)])

    async def test_replace_take_profit_places_two_reduce_only_orders_for_split_plan(self) -> None:
        trader = self.make_trader()
        posted: list[dict] = []

        async def fetch_position_snapshot(self) -> PositionSnapshot:  # type: ignore[no-untyped-def]
            return PositionSnapshot("LONG", Decimal("4"), 100.0, 0.4, Decimal("4"))

        async def fetch_pending_orders(self) -> list[dict]:  # type: ignore[no-untyped-def]
            return []

        async def request(self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
            posted.append(dict(payload or {}))
            return {"code": "0", "data": [{"ordId": f"tp-{len(posted)}"}]}

        trader.fetch_position_snapshot = MethodType(fetch_position_snapshot, trader)  # type: ignore[method-assign]
        trader.fetch_pending_orders = MethodType(fetch_pending_orders, trader)  # type: ignore[method-assign]
        trader.request = MethodType(request, trader)  # type: ignore[method-assign]

        result = await trader.replace_take_profit(
            intent(partial_tp_price=108.0, partial_tp_ratio=0.5, tp_plan="SPLIT_PARTIAL_FINAL")
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.tp_order_ids, ("tp-1", "tp-2"))
        self.assertEqual(result.tp_order_id, "tp-1,tp-2")
        self.assertEqual(result.tp_price, "partial:108.00,final:110.00")
        self.assertEqual([item["sz"] for item in posted], ["2", "2"])
        self.assertEqual([item["px"] for item in posted], ["108.00", "110.00"])
        self.assertTrue(all(item["reduceOnly"] == "true" for item in posted))


class SplitTakeProfitLifecycleTest(unittest.TestCase):
    def test_mark_partial_tp_consumed_when_position_reduced_enough(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=4,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            partial_tp_price=108.0,
            partial_tp_ratio=0.5,
            tp_plan="SPLIT_PARTIAL_FINAL",
        )
        position = PositionSnapshot("LONG", Decimal("5"), 100.0, 0.5, Decimal("5"))

        consumed = mark_partial_tp_consumed_if_position_reduced(strat, position)

        self.assertTrue(consumed)
        self.assertTrue(strat.state.partial_tp_consumed)
        self.assertEqual(strat.state.tp_plan, "SINGLE")
        self.assertIsNone(strat.state.partial_tp_price)
        self.assertEqual(strat.state.partial_tp_ratio, 0.0)

    def test_does_not_mark_partial_consumed_for_small_rounding_difference(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=4,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            partial_tp_price=108.0,
            partial_tp_ratio=0.5,
            tp_plan="SPLIT_PARTIAL_FINAL",
        )
        position = PositionSnapshot("LONG", Decimal("9.98"), 100.0, 0.998, Decimal("9.98"))

        consumed = mark_partial_tp_consumed_if_position_reduced(strat, position)

        self.assertFalse(consumed)
        self.assertFalse(strat.state.partial_tp_consumed)
        self.assertEqual(strat.state.tp_plan, "SPLIT_PARTIAL_FINAL")

    def test_record_flat_accepts_split_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveTradeJournal(Path(tmp) / "live_trade_events.jsonl")
            journal.record_flat(
                position_id="pos-1",
                symbol="ETH-USDT-SWAP",
                side="LONG",
                cash_before_position=100.0,
                cash_after=101.0,
                equity_after=101.0,
                reason="test",
                layers=4,
                avg_entry_price=100.0,
                last_tp_price=110.0,
                last_partial_tp_price=108.0,
                last_tp_plan="SPLIT_PARTIAL_FINAL",
                partial_tp_consumed=True,
            )

            events = journal.load_events()

        self.assertEqual(len(events), 1)
        payload = events[0].payload
        self.assertEqual(payload["last_partial_tp_price"], 108.0)
        self.assertEqual(payload["last_tp_plan"], "SPLIT_PARTIAL_FINAL")
        self.assertTrue(payload["partial_tp_consumed"])

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
        )
        position = PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2"))

        activated = mark_middle_runner_active_if_position_reduced(strat, position)

        self.assertTrue(activated)
        self.assertTrue(strat.state.middle_runner_active)
        self.assertFalse(strat.state.middle_runner_pending)
        self.assertTrue(strat.state.middle_runner_add_disabled)
        self.assertTrue(strat.state.partial_tp_consumed)
        self.assertEqual(strat.state.tp_plan, "SINGLE")

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
