from __future__ import annotations

import os
import unittest
from decimal import Decimal
from unittest.mock import patch

from src.execution.trader import PositionSnapshot, Trader
from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.position_management.tp_progress import append_three_stage_progress_journal_events, mark_three_stage_progress_if_position_reduced
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy, BollCvdReclaimStrategyConfig, StrategyPositionState, TradeIntent


def strategy(**overrides) -> BollCvdReclaimStrategy:
    values = dict(
        three_stage_runner_enabled=True,
        split_tp_enabled=True,
        split_tp_min_layers=4,
        breakeven_fee_buffer_pct=0.001,
        tp_min_net_profit_pct=0.002,
    )
    values.update(overrides)
    return BollCvdReclaimStrategy(BollCvdReclaimStrategyConfig(**values), SimplePositionSizer(SimplePositionSizerConfig()))


def boll(middle: float = 101.0, upper: float = 110.0, lower: float = 90.0, candle_ts_ms: int = 1_000) -> BollSnapshot:
    return BollSnapshot("ETH-USDT-SWAP", candle_ts_ms, 100.0, middle, upper, lower, 0.1, 0.1, True, True)


def cvd(**overrides) -> CvdSnapshot:
    values = dict(
        ts_ms=1_000,
        price=100.0,
        side="sell",
        size=1.0,
        signed_delta=-1.0,
        total_cvd=-1.0,
        fast_cvd=-1.0,
        previous_fast_cvd=0.0,
        buy_volume=0.0,
        sell_volume=1.0,
        buy_ratio=0.0,
        sell_ratio=1.0,
        cross_positive=False,
        cross_negative=True,
        cvd_increasing=False,
        cvd_decreasing=True,
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
    values.update(overrides)
    return CvdSnapshot(**values)


def intent(**overrides) -> TradeIntent:
    values = dict(
        intent_type="OPEN_LONG",
        side="LONG",
        price=100.0,
        layer_index=1,
        tp_price=111.1,
        reason="test",
        size=PositionSize(1.0, 50.0, 0.5, 1, 1.0),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        boll_upper=110.0,
        boll_middle=101.0,
        boll_lower=90.0,
        ts_ms=1_000,
        avg_entry_price=100.0,
        breakeven_price=100.1,
        tp_mode="MIDDLE",
        tp_plan="THREE_STAGE_RUNNER",
        three_stage_tp1_price=101.0,
        three_stage_tp1_ratio=0.6,
        three_stage_tp2_price=110.0,
        three_stage_tp2_ratio=0.2,
        three_stage_runner_ratio=0.2,
    )
    values.update(overrides)
    return TradeIntent(**values)  # type: ignore[arg-type]


def three_stage_pre_tp1_state(**overrides) -> StrategyPositionState:
    first_ts = int(overrides.pop("first_entry_ts_ms", 100_000))
    values = dict(
        side="LONG",
        layers=1,
        last_entry_price=100.0,
        first_entry_ts_ms=first_ts,
        last_order_ts_ms=first_ts,
        total_entry_qty=1.0,
        total_entry_notional=100.0,
        avg_entry_price=100.0,
        net_remaining_breakeven_price=100.0,
        tp_price=110.0,
        tp_mode="UPPER",
        tp_plan="THREE_STAGE_RUNNER",
        partial_tp_price=101.0,
        partial_tp_ratio=0.6,
        three_stage_runner_enabled_for_position=True,
        three_stage_tp1_price=101.0,
        three_stage_tp2_price=110.0,
        three_stage_tp1_ratio=0.6,
        three_stage_tp2_ratio=0.2,
        three_stage_runner_ratio=0.2,
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        trend_runner_active=False,
        last_tp_update_candle_ts_ms=1_000,
    )
    values.update(overrides)
    return StrategyPositionState(**values)


class RecordingJournal:
    def __init__(self) -> None:
        self.events = []

    def append(self, event_name: str, payload: dict, position_id: str | None = None) -> None:
        self.events.append((event_name, dict(payload), position_id))


class ThreeStageTrendRunnerStrategyTest(unittest.TestCase):
    def _three_stage_add_state(self, side: str) -> StrategyPositionState:
        last_entry = 1743.57
        return StrategyPositionState(
            side=side,
            layers=1,
            last_entry_price=last_entry,
            last_order_ts_ms=0,
            total_entry_qty=1.0,
            total_entry_notional=last_entry,
            avg_entry_price=last_entry,
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=1760.0 if side == "LONG" else 1720.0,
            three_stage_tp2_price=1800.0 if side == "LONG" else 1680.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            trend_runner_active=False,
            lower_armed=True,
            lower_deep_enough=True,
            lower_extreme_price=1728.0,
            upper_armed=True,
            upper_deep_enough=True,
            upper_extreme_price=1760.0,
        )

    def test_middle_tp_mode_enables_three_stage_runner_long(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        got = strat._open_position("LONG", "OPEN_LONG", 100.0, 2_000, boll(), cvd(), "test")

        self.assertEqual(got.tp_mode, "MIDDLE")
        self.assertEqual(got.tp_plan, "THREE_STAGE_RUNNER")
        self.assertEqual(got.three_stage_tp1_price, 101.0)
        self.assertEqual(got.three_stage_tp1_ratio, 0.6)
        self.assertEqual(got.three_stage_tp2_price, 110.0)
        self.assertEqual(got.three_stage_tp2_ratio, 0.2)
        self.assertIsNone(got.three_stage_runner_tp_price)
        self.assertIsNone(got.three_stage_runner_sl_price)
        self.assertEqual(got.three_stage_runner_ratio, 0.2)
        self.assertFalse(got.trend_runner_active)
        self.assertFalse(strat.state.trend_runner_active)
        self.assertIsNone(strat.state.trend_runner_tp_price)
        self.assertIsNone(strat.state.trend_runner_sl_price)
        self.assertIsNone(strat.state.three_stage_post_tp1_protective_sl_price)
        self.assertIsNone(strat.state.three_stage_post_tp1_protective_sl_order_id)
        self.assertFalse(strat.state.three_stage_post_tp1_protected)
        self.assertEqual(got.tp_price, 110.0)

    def test_middle_tp_mode_enables_three_stage_runner_short(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        got = strat._open_position("SHORT", "OPEN_SHORT", 100.0, 2_000, boll(middle=99.0, upper=110.0, lower=90.0), cvd(), "test")

        self.assertEqual(got.tp_mode, "MIDDLE")
        self.assertEqual(got.tp_plan, "THREE_STAGE_RUNNER")
        self.assertEqual(got.three_stage_tp1_price, 99.0)
        self.assertEqual(got.three_stage_tp2_price, 90.0)
        self.assertIsNone(got.three_stage_runner_tp_price)
        self.assertIsNone(got.three_stage_runner_sl_price)
        self.assertFalse(got.trend_runner_active)
        self.assertEqual(got.tp_price, 90.0)

    def test_three_stage_allows_add_before_tp1(self) -> None:
        strat = strategy()
        strat.state = self._three_stage_add_state("LONG")

        got = strat._maybe_open_or_add_long(
            1728.0,
            2_000_000,
            boll(middle=1750.0, upper=1810.0, lower=1700.0),
            cvd(side="buy", buy_volume=1.0, sell_volume=0.0, buy_ratio=1.0, sell_ratio=0.0, cross_positive=True, cvd_increasing=True, no_new_low=True),
        )

        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.intent_type, "ADD_LONG")
        self.assertEqual(got.tp_plan, "THREE_STAGE_RUNNER")
        self.assertEqual(strat.state.layers, 2)
        self.assertFalse(strat.state.three_stage_tp1_consumed)
        self.assertFalse(strat.state.three_stage_tp2_consumed)

    def test_three_stage_blocks_add_after_tp1_consumed(self) -> None:
        strat = strategy()
        strat.state = self._three_stage_add_state("LONG")
        strat.state.three_stage_tp1_consumed = True

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            got = strat._maybe_open_or_add_long(
                1728.0,
                2_000_000,
                boll(middle=1750.0, upper=1810.0, lower=1700.0),
                cvd(side="buy", buy_volume=1.0, sell_volume=0.0, buy_ratio=1.0, sell_ratio=0.0, cross_positive=True, cvd_increasing=True, no_new_low=True),
            )

        self.assertIsNone(got)
        self.assertTrue(any("reason=three_stage_after_tp1" in line for line in logs.output))

    def test_three_stage_blocks_add_when_trend_runner_active(self) -> None:
        strat = strategy()
        strat.state = self._three_stage_add_state("LONG")
        strat.state.trend_runner_active = True

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            got = strat._maybe_open_or_add_long(
                1728.0,
                2_000_000,
                boll(middle=1750.0, upper=1810.0, lower=1700.0),
                cvd(side="buy", buy_volume=1.0, sell_volume=0.0, buy_ratio=1.0, sell_ratio=0.0, cross_positive=True, cvd_increasing=True, no_new_low=True),
            )

        self.assertIsNone(got)
        self.assertTrue(any("reason=trend_runner_active" in line for line in logs.output))

    def test_three_stage_short_allows_add_before_tp1(self) -> None:
        strat = strategy()
        strat.state = self._three_stage_add_state("SHORT")

        got = strat._maybe_open_or_add_short(
            1760.0,
            2_000_000,
            boll(middle=1730.0, upper=1780.0, lower=1680.0),
            cvd(side="sell", buy_volume=0.0, sell_volume=1.0, buy_ratio=0.0, sell_ratio=1.0, cross_negative=True, cvd_decreasing=True, no_new_high=True),
        )

        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.intent_type, "ADD_SHORT")
        self.assertEqual(got.tp_plan, "THREE_STAGE_RUNNER")
        self.assertEqual(strat.state.layers, 2)
        self.assertFalse(strat.state.three_stage_tp1_consumed)
        self.assertFalse(strat.state.three_stage_tp2_consumed)

    def test_add_skip_log_throttled_for_three_stage_after_tp1(self) -> None:
        strat = strategy()
        strat.state = self._three_stage_add_state("LONG")
        strat.state.three_stage_tp1_consumed = True

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            for ts_ms in (2_000_000, 2_010_000, 2_020_000):
                got = strat._maybe_open_or_add_long(
                    1728.0,
                    ts_ms,
                    boll(middle=1750.0, upper=1810.0, lower=1700.0),
                    cvd(side="buy", buy_volume=1.0, sell_volume=0.0, buy_ratio=1.0, sell_ratio=0.0, cross_positive=True, cvd_increasing=True, no_new_low=True),
                )
                self.assertIsNone(got)

        add_skip_logs = [line for line in logs.output if "ADD_SKIPPED" in line and "reason=three_stage_after_tp1" in line]
        self.assertEqual(len(add_skip_logs), 1)

    def test_outer_tp_mode_does_not_enable_three_stage_or_split(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        tp_price, mode = strat._select_tp_price("LONG", boll(middle=100.1, upper=110.0))
        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", tp_price, 4, tp_mode=mode, boll=boll(middle=100.1, upper=110.0))

        self.assertEqual(mode, "UPPER")
        self.assertEqual(plan, "SINGLE")
        self.assertIsNone(partial_tp)
        self.assertEqual(partial_ratio, 0.0)

    def test_three_stage_and_near_tp_env_conflict_raises(self) -> None:
        with patch.dict(os.environ, {"THREE_STAGE_RUNNER_ENABLED": "true", "NEAR_TP_ENABLED": "true"}, clear=True):
            with self.assertRaises(RuntimeError):
                BollCvdReclaimStrategyConfig.from_env()

    def test_dynamic_runner_orders_long_and_short(self) -> None:
        strat = strategy()
        bands = boll(middle=100.0, upper=110.0, lower=90.0)

        tp, sl, _, _ = strat._calculate_trend_runner_dynamic_orders("LONG", bands, 0, None)
        self.assertAlmostEqual(tp, 111.1)
        self.assertAlmostEqual(sl, 100.0)
        tp, sl, _, _ = strat._calculate_trend_runner_dynamic_orders("LONG", bands, 1, 100.0)
        self.assertAlmostEqual(tp, 110.99)
        self.assertAlmostEqual(sl, 101.0)
        tp, sl, _, _ = strat._calculate_trend_runner_dynamic_orders("LONG", bands, 6, 101.0)
        self.assertAlmostEqual(tp, 110.44)
        self.assertAlmostEqual(sl, 105.0)

        tp, sl, _, _ = strat._calculate_trend_runner_dynamic_orders("SHORT", bands, 1, 100.0)
        self.assertAlmostEqual(tp, 89.19)
        self.assertAlmostEqual(sl, 99.0)

    def test_runner_sl_only_tightens(self) -> None:
        strat = strategy()
        tp, sl, _, _ = strat._calculate_trend_runner_dynamic_orders("LONG", boll(middle=100.0, upper=110.0), 1, 105.0)
        self.assertAlmostEqual(tp, 110.99)
        self.assertEqual(sl, 105.0)

        _tp, sl, _, _ = strat._calculate_trend_runner_dynamic_orders("SHORT", boll(middle=100.0, lower=90.0), 1, 95.0)
        self.assertEqual(sl, 95.0)

    def test_reverse_burst_single_tick_only_arms(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            trend_runner_active=True,
            trend_runner_trend_start_ts_ms=1_000,
            trend_runner_tp_price=112.0,
            trend_runner_sl_price=100.0,
        )

        got = strat._maybe_trend_runner_market_exit(110.0, 62_000, boll(middle=101.0, upper=110.0), cvd(down_burst=True, sell_ratio=0.7, fast_cvd=-1.0))

        self.assertIsNone(got)
        self.assertTrue(strat.state.trend_runner_reverse_candidate)

    def test_reverse_burst_confirmed_after_five_seconds(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            trend_runner_active=True,
            trend_runner_trend_start_ts_ms=1_000,
            trend_runner_tp_price=112.0,
            trend_runner_sl_price=100.0,
        )
        strat._maybe_trend_runner_market_exit(110.0, 62_000, boll(middle=101.0), cvd(down_burst=True, sell_ratio=0.7, fast_cvd=-1.0))

        got = strat._maybe_trend_runner_market_exit(109.7, 67_000, boll(middle=101.0), cvd(sell_ratio=0.7, fast_cvd=-2.0))

        self.assertIsNotNone(got)
        self.assertEqual(got.reason, "trend_runner_reverse_burst_confirmed")

    def test_reverse_burst_recovery_cancels(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            trend_runner_active=True,
            trend_runner_trend_start_ts_ms=1_000,
            trend_runner_tp_price=112.0,
            trend_runner_sl_price=100.0,
        )
        strat._maybe_trend_runner_market_exit(110.0, 62_000, boll(middle=101.0), cvd(down_burst=True, sell_ratio=0.7, fast_cvd=-1.0))
        strat._maybe_trend_runner_market_exit(109.7, 64_000, boll(middle=101.0), cvd(sell_ratio=0.7, fast_cvd=-1.5))

        got = strat._maybe_trend_runner_market_exit(109.9, 67_000, boll(middle=101.0), cvd(sell_ratio=0.7, fast_cvd=-2.0))

        self.assertIsNone(got)
        self.assertFalse(strat.state.trend_runner_reverse_candidate)

    def test_reverse_burst_arm_delay_blocks_candidate(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(side="LONG", layers=1, trend_runner_active=True, trend_runner_trend_start_ts_ms=10_000)

        got = strat._maybe_trend_runner_market_exit(110.0, 20_000, boll(middle=101.0), cvd(down_burst=True, sell_ratio=0.7, fast_cvd=-1.0))

        self.assertIsNone(got)
        self.assertFalse(strat.state.trend_runner_reverse_candidate)

    def test_max_trend_time_exit(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            trend_runner_active=True,
            trend_runner_trend_start_ts_ms=1_000,
            trend_runner_tp_price=112.0,
            trend_runner_sl_price=100.0,
        )

        got = strat._maybe_trend_runner_market_exit(110.0, 18_001_000, boll(middle=101.0), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.reason, "trend_runner_max_time_after_second_tp")

    def test_on_tick_runner_market_exit_preempts_tp_update_on_new_candle(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            trend_runner_active=True,
            trend_runner_trend_start_ts_ms=1_000,
            trend_runner_tp_price=112.0,
            trend_runner_sl_price=100.5,
            last_tp_update_candle_ts_ms=1_000,
        )

        got = strat.on_tick(100.0, 20_000, boll(middle=101.0, upper=110.0, lower=90.0, candle_ts_ms=2_000), cvd())

        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].intent_type, "MARKET_EXIT_RUNNER")
        self.assertEqual(got[0].reason, "trend_runner_sl_failsafe")
        self.assertNotIn("UPDATE_TP", [item.intent_type for item in got])

    def test_trend_runner_tp_crossed_is_market_exit_signal(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            trend_runner_active=True,
            trend_runner_trend_start_ts_ms=1_000,
            trend_runner_tp_price=112.0,
            trend_runner_sl_price=100.0,
        )

        got = strat._maybe_trend_runner_market_exit(112.1, 20_000, boll(middle=101.0), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.intent_type, "MARKET_EXIT_RUNNER")
        self.assertEqual(got.reason, "trend_runner_tp_crossed")

    def test_tp1_and_tp2_position_sync_activation(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
        )

        event = mark_three_stage_progress_if_position_reduced(strat, PositionSnapshot("LONG", Decimal("4"), 100.0, 0.4, Decimal("4")), 10_000)
        self.assertEqual(event, "TP1")
        self.assertTrue(strat.state.three_stage_tp1_consumed)
        self.assertFalse(strat.state.trend_runner_active)
        self.assertIsNone(strat.state.trend_runner_tp_price)
        self.assertIsNone(strat.state.trend_runner_sl_price)

        event = mark_three_stage_progress_if_position_reduced(strat, PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2")), 20_000)
        self.assertEqual(event, "TP2")
        self.assertTrue(strat.state.three_stage_tp2_consumed)
        self.assertTrue(strat.state.trend_runner_active)
        self.assertEqual(strat.state.trend_runner_trend_start_ts_ms, 20_000)
        self.assertIsNone(strat.state.trend_runner_tp_price)
        self.assertIsNone(strat.state.trend_runner_sl_price)

        strat.state.three_stage_post_tp1_protective_sl_order_id = "algo-post"
        strat.state.three_stage_post_tp1_protective_sl_price = 100.2
        strat.state.three_stage_post_tp1_protected = True

        update = strat._maybe_update_tp(110.0, 20_100, boll(middle=101.0, upper=110.0, lower=90.0), cvd())

        self.assertIsNotNone(update)
        self.assertEqual(update.intent_type, "UPDATE_TP")
        self.assertTrue(update.trend_runner_active)
        self.assertAlmostEqual(update.trend_runner_tp_price or 0, 111.1)
        self.assertEqual(update.trend_runner_sl_price, 101.0)
        self.assertEqual(update.tp_price, 111.1)

    def test_tp1_and_tp2_same_position_sync_activates_runner_immediately(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
        )

        event = mark_three_stage_progress_if_position_reduced(
            strat,
            PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2")),
            30_000,
        )

        self.assertEqual(event, "TP1_TP2")
        self.assertTrue(strat.state.three_stage_tp1_consumed)
        self.assertTrue(strat.state.three_stage_tp2_consumed)
        self.assertTrue(strat.state.trend_runner_active)
        self.assertEqual(strat.state.trend_runner_trend_start_ts_ms, 30_000)
        self.assertIsNone(strat.state.trend_runner_tp_price)
        self.assertIsNone(strat.state.trend_runner_sl_price)

    def test_tp1_tp2_journal_event_records_both_tp_legs_and_activation(self) -> None:
        journal = RecordingJournal()
        payload = {
            "event": "TP1_TP2",
            "position_id": "pos-1",
            "tp_plan": "THREE_STAGE_RUNNER",
            "trend_runner_active": True,
        }

        append_three_stage_progress_journal_events(journal, payload)

        self.assertEqual(
            [event_name for event_name, _payload, _position_id in journal.events],
            ["THREE_STAGE_TP1_FILLED", "THREE_STAGE_TP2_FILLED", "TREND_RUNNER_ACTIVATED"],
        )
        self.assertEqual([position_id for _event_name, _payload, position_id in journal.events], ["pos-1", "pos-1", "pos-1"])

    def test_tp2_sync_does_not_activate_when_remaining_above_tight_tolerance(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_consumed=True,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=True,
        )

        event = mark_three_stage_progress_if_position_reduced(
            strat,
            PositionSnapshot("LONG", Decimal("2.5"), 100.0, 0.25, Decimal("2.5")),
            30_000,
        )

        self.assertIsNone(event)
        self.assertFalse(strat.state.three_stage_tp2_consumed)
        self.assertFalse(strat.state.trend_runner_active)

    def test_tp2_sync_activates_at_runner_ratio_plus_tight_tolerance(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_consumed=True,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=True,
        )

        event = mark_three_stage_progress_if_position_reduced(
            strat,
            PositionSnapshot("LONG", Decimal("2.2"), 100.0, 0.22, Decimal("2.2")),
            30_000,
        )

        self.assertEqual(event, "TP2")
        self.assertTrue(strat.state.three_stage_tp2_consumed)
        self.assertTrue(strat.state.trend_runner_active)

    def test_waiting_tp2_new_candle_does_not_reset_three_stage_state(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_mode="MIDDLE",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_consumed=True,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
            trend_runner_active=False,
            last_tp_update_candle_ts_ms=1_000,
        )

        bands = boll(middle=102.0, upper=112.0, lower=92.0, candle_ts_ms=2_000)
        got = strat._maybe_update_tp(105.0, 2_000, bands, cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertEqual(got.reason, "three_stage_post_tp1_dynamic_tp_sl_update")
        self.assertEqual(got.tp_plan, "THREE_STAGE_RUNNER")
        self.assertTrue(got.three_stage_tp1_consumed)
        self.assertFalse(got.trend_runner_active)
        self.assertIsNotNone(got.three_stage_post_tp1_protective_sl_price)
        self.assertEqual(got.three_stage_tp2_price, 112.0)
        self.assertEqual(got.tp_price, 112.0)
        self.assertEqual(strat.state.last_tp_update_ts_ms, 2_000)
        self.assertEqual(strat.state.last_tp_update_candle_ts_ms, 2_000)
        self.assertTrue(strat.state.three_stage_runner_enabled_for_position)
        self.assertTrue(strat.state.three_stage_tp1_consumed)
        self.assertFalse(strat.state.three_stage_tp2_consumed)
        self.assertFalse(strat.state.trend_runner_active)
        self.assertEqual(strat.state.three_stage_tp2_price, 112.0)

        with self.assertNoLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO"):
            repeated = strat._maybe_update_tp(105.5, 2_500, bands, cvd())
        self.assertIsNone(repeated)

    def test_three_stage_dynamic_update_does_not_reset_consumed_flags(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_mode="MIDDLE",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_consumed=True,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
            trend_runner_active=False,
            last_tp_update_candle_ts_ms=1_000,
        )

        def fail_if_reset_called(*_args, **_kwargs):
            raise AssertionError("_set_three_stage_runner_planned must not be called during dynamic update")

        strat._set_three_stage_runner_planned = fail_if_reset_called  # type: ignore[method-assign]

        got = strat._maybe_update_tp(105.0, 2_000, boll(middle=102.0, upper=112.0, lower=92.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got)
        self.assertTrue(strat.state.three_stage_tp1_consumed)
        self.assertFalse(strat.state.three_stage_tp2_consumed)
        self.assertFalse(strat.state.trend_runner_active)
        self.assertTrue(got.three_stage_tp1_consumed)
        self.assertFalse(got.three_stage_tp2_consumed)
        self.assertIsNone(got.partial_tp_price)
        self.assertEqual(got.three_stage_tp2_price, 112.0)

    def test_post_tp1_protective_sl_calculation_tightening_and_extension(self) -> None:
        long_strat = strategy(three_stage_post_tp1_sl_extension_trigger_ratio=0.6)
        long_strat.state = StrategyPositionState(
            side="LONG",
            avg_entry_price=100.0,
            three_stage_tp1_price=102.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp1_consumed=True,
        )
        long_sl = long_strat._calculate_three_stage_post_tp1_protective_sl("LONG", 105.0, boll(middle=102.0, upper=112.0, lower=92.0))
        self.assertIsNotNone(long_sl)
        self.assertGreater(long_strat._tighten_three_stage_post_tp1_sl("LONG", 100.5, 99.5), 100.0)
        extension = long_strat._apply_three_stage_post_tp1_extension_trigger("LONG", 108.0, boll(middle=102.0, upper=112.0, lower=92.0), long_sl)
        self.assertGreaterEqual(extension or 0, 102.0)
        self.assertTrue(long_strat.state.three_stage_post_tp1_sl_extension_triggered)

        short_strat = strategy(three_stage_post_tp1_sl_extension_trigger_ratio=0.6)
        short_strat.state = StrategyPositionState(
            side="SHORT",
            avg_entry_price=100.0,
            three_stage_tp1_price=98.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp1_consumed=True,
        )
        short_sl = short_strat._calculate_three_stage_post_tp1_protective_sl("SHORT", 95.0, boll(middle=98.0, upper=108.0, lower=88.0))
        self.assertIsNotNone(short_sl)
        self.assertLess(short_strat._tighten_three_stage_post_tp1_sl("SHORT", 99.5, 100.5), 100.0)
        extension = short_strat._apply_three_stage_post_tp1_extension_trigger("SHORT", 92.0, boll(middle=98.0, upper=108.0, lower=88.0), short_sl)
        self.assertLessEqual(extension or 999, 98.0)
        self.assertTrue(short_strat.state.three_stage_post_tp1_sl_extension_triggered)

    def test_post_tp1_protective_sl_uses_net_remaining_breakeven_when_present(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            avg_entry_price=100.0,
            net_remaining_breakeven_price=95.0,
            three_stage_tp1_price=102.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp1_consumed=True,
        )

        got = strat._calculate_three_stage_post_tp1_protective_sl("LONG", 105.0, boll(middle=102.0, upper=112.0, lower=92.0))

        self.assertEqual(got, ((95.0 + 102.0) / 2))

    def test_three_stage_time_tighten_long_moves_both_candidates_to_55pct(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            net_remaining_breakeven_price=100.0,
            three_stage_tp1_consumed=True,
            three_stage_post_tp1_sl_time_tighten_candle_count=1,
        )

        got = strat._calculate_three_stage_post_tp1_protective_sl("LONG", 250.0, boll(middle=200.0, upper=220.0, lower=100.0))

        self.assertEqual(strat._runner_sl_time_tighten_ratio(1), 0.55)
        self.assertEqual(got, 155.0)

    def test_three_stage_time_tighten_long_uses_more_conservative_of_two_candidates(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            net_remaining_breakeven_price=120.0,
            three_stage_tp1_consumed=True,
            three_stage_post_tp1_sl_time_tighten_candle_count=1,
        )

        got = strat._calculate_three_stage_post_tp1_protective_sl("LONG", 250.0, boll(middle=200.0, upper=220.0, lower=100.0))

        self.assertEqual(got, 164.0)

    def test_three_stage_time_tighten_short_moves_both_candidates_to_55pct(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="SHORT",
            net_remaining_breakeven_price=200.0,
            three_stage_tp1_consumed=True,
            three_stage_post_tp1_sl_time_tighten_candle_count=1,
        )

        got = strat._calculate_three_stage_post_tp1_protective_sl("SHORT", 50.0, boll(middle=100.0, upper=200.0, lower=80.0))

        self.assertEqual(got, 145.0)

    def test_three_stage_time_tighten_short_uses_more_conservative_of_two_candidates(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="SHORT",
            net_remaining_breakeven_price=180.0,
            three_stage_tp1_consumed=True,
            three_stage_post_tp1_sl_time_tighten_candle_count=1,
        )

        got = strat._calculate_three_stage_post_tp1_protective_sl("SHORT", 50.0, boll(middle=100.0, upper=200.0, lower=80.0))

        self.assertEqual(got, 136.0)

    def test_time_tighten_reaches_middle_after_10_candles(self) -> None:
        long_strat = strategy()
        long_strat.state = StrategyPositionState(
            side="LONG",
            net_remaining_breakeven_price=100.0,
            three_stage_tp1_consumed=True,
            three_stage_post_tp1_sl_time_tighten_candle_count=10,
        )
        long_sl = long_strat._calculate_three_stage_post_tp1_protective_sl("LONG", 250.0, boll(middle=200.0, upper=220.0, lower=100.0))

        short_strat = strategy()
        short_strat.state = StrategyPositionState(
            side="SHORT",
            net_remaining_breakeven_price=200.0,
            three_stage_tp1_consumed=True,
            three_stage_post_tp1_sl_time_tighten_candle_count=10,
        )
        short_sl = short_strat._calculate_three_stage_post_tp1_protective_sl("SHORT", 50.0, boll(middle=100.0, upper=200.0, lower=80.0))

        self.assertEqual(long_strat._runner_sl_time_tighten_ratio(10), 1.0)
        self.assertEqual(long_sl, 200.0)
        self.assertLessEqual(long_sl or 0.0, 200.0)
        self.assertEqual(short_sl, 100.0)
        self.assertGreaterEqual(short_sl or 0.0, 100.0)

    def test_extension_trigger_still_overrides_to_middle(self) -> None:
        strat = strategy(three_stage_post_tp1_sl_extension_trigger_ratio=0.6)
        strat.state = StrategyPositionState(
            side="LONG",
            net_remaining_breakeven_price=100.0,
            three_stage_tp1_consumed=True,
            three_stage_post_tp1_sl_time_tighten_candle_count=1,
        )
        bands = boll(middle=200.0, upper=220.0, lower=100.0)

        base_sl = strat._calculate_three_stage_post_tp1_protective_sl("LONG", 213.0, bands)
        final_sl = strat._apply_three_stage_post_tp1_extension_trigger("LONG", 213.0, bands, base_sl)

        self.assertEqual(final_sl, 200.0)

    def test_tighten_only_prevents_sl_from_moving_back_when_boll_changes(self) -> None:
        long_strat = strategy()
        long_strat.state = StrategyPositionState(
            side="LONG",
            net_remaining_breakeven_price=100.0,
            three_stage_tp1_consumed=True,
            three_stage_post_tp1_sl_time_tighten_candle_count=1,
        )
        long_calculated = long_strat._calculate_three_stage_post_tp1_protective_sl(
            "LONG",
            250.0,
            boll(middle=200.0, upper=220.0, lower=100.0),
        )

        short_strat = strategy()
        short_strat.state = StrategyPositionState(
            side="SHORT",
            net_remaining_breakeven_price=200.0,
            three_stage_tp1_consumed=True,
            three_stage_post_tp1_sl_time_tighten_candle_count=1,
        )
        short_calculated = short_strat._calculate_three_stage_post_tp1_protective_sl(
            "SHORT",
            50.0,
            boll(middle=100.0, upper=200.0, lower=80.0),
        )

        self.assertEqual(long_calculated, 155.0)
        self.assertEqual(long_strat._tighten_optional_three_stage_post_tp1_sl("LONG", 160.0, long_calculated), 160.0)
        self.assertEqual(short_calculated, 145.0)
        self.assertEqual(short_strat._tighten_optional_three_stage_post_tp1_sl("SHORT", 140.0, short_calculated), 140.0)

    def test_three_stage_post_tp1_sl_diag_logs_once_per_signature(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            avg_entry_price=100.0,
            net_remaining_breakeven_price=95.0,
            three_stage_tp1_price=102.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp1_consumed=True,
            position_cost_entry_notional=100.0,
            position_cost_exit_notional=61.2,
            position_cost_remaining_qty=0.4,
        )
        snapshot = boll(middle=102.0, upper=110.0, lower=90.0, candle_ts_ms=1_000)

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="WARNING") as logs:
            first = strat._calculate_three_stage_post_tp1_protective_sl("LONG", 103.0, snapshot)
            second = strat._calculate_three_stage_post_tp1_protective_sl("LONG", 103.0, snapshot)

        self.assertEqual(first, second)
        joined = "\n".join(logs.output)
        self.assertEqual(joined.count("THREE_STAGE_POST_TP1_PROTECTIVE_SL_DIAG"), 1)
        self.assertIn("candidate_cost=98.5000", joined)
        self.assertIn("candidate_structure=96.0000", joined)
        self.assertIn("net_remaining_breakeven=95.0000", joined)

    def test_three_stage_post_tp1_sl_diag_logs_warning_reason_when_invalid(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            avg_entry_price=100.0,
            net_remaining_breakeven_price=95.0,
            three_stage_tp1_price=102.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp1_consumed=True,
        )
        snapshot = boll(middle=102.0, upper=110.0, lower=90.0, candle_ts_ms=1_000)

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="WARNING") as logs:
            first = strat._calculate_three_stage_post_tp1_protective_sl("LONG", 98.0, snapshot)
            second = strat._calculate_three_stage_post_tp1_protective_sl("LONG", 98.0, snapshot)

        self.assertIsNone(first)
        self.assertIsNone(second)
        joined = "\n".join(logs.output)
        self.assertEqual(joined.count("THREE_STAGE_POST_TP1_PROTECTIVE_SL_DIAG"), 1)
        self.assertIn("reason=long_sl_not_below_current", joined)

    def test_post_tp1_protective_sl_falls_back_to_core_avg_logic_without_net_breakeven(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.001)
        strat.state = StrategyPositionState(
            side="LONG",
            avg_entry_price=100.0,
            net_remaining_breakeven_price=0.0,
            three_stage_tp1_price=102.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp1_consumed=True,
        )

        got = strat._calculate_three_stage_post_tp1_protective_sl("LONG", 105.0, boll(middle=102.0, upper=112.0, lower=92.0))

        fallback_breakeven = (100.0 - 0.6 * (102.0 - 100.0) / 0.4) * 1.001
        self.assertEqual(got, max((fallback_breakeven + 102.0) / 2, (92.0 + 102.0) / 2))

    def test_post_tp1_protective_sl_long_uses_max_candidate_with_net_breakeven(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(side="LONG", net_remaining_breakeven_price=85.0)

        got = strat._calculate_three_stage_post_tp1_protective_sl("LONG", 105.0, boll(middle=102.0, upper=112.0, lower=92.0))

        self.assertEqual(got, (92.0 + 102.0) / 2)

    def test_post_tp1_protective_sl_short_uses_min_candidate_with_net_breakeven(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(side="SHORT", net_remaining_breakeven_price=110.0)

        got = strat._calculate_three_stage_post_tp1_protective_sl("SHORT", 95.0, boll(middle=98.0, upper=108.0, lower=88.0))

        self.assertEqual(got, (108.0 + 98.0) / 2)

    def test_three_stage_position_blocks_middle_runner_plan_after_env_change(self) -> None:
        strat = strategy(three_stage_runner_enabled=False, middle_runner_enabled=True)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            last_tp_update_candle_ts_ms=1_000,
        )

        got = strat._maybe_update_tp(100.0, 2_000, boll(middle=102.0, upper=112.0, lower=92.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.tp_plan, "THREE_STAGE_RUNNER")
        self.assertEqual(got.three_stage_tp1_price, 102.0)
        self.assertEqual(got.three_stage_tp2_price, 112.0)
        self.assertEqual(strat.state.tp_plan, "THREE_STAGE_RUNNER")
        self.assertTrue(strat.state.three_stage_runner_enabled_for_position)
        self.assertFalse(strat._middle_runner_plan_allowed("MIDDLE", boll()))

    # ── startup force TP reconcile ──────────────────────────────────────

    def test_force_tp_reconcile_bypasses_same_candle_guard(self) -> None:
        """When startup_force_tp_reconcile=True, same candle does NOT skip TP update."""
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_mode="UPPER",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_consumed=False,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            trend_runner_active=False,
            last_tp_update_candle_ts_ms=1_000,
            startup_force_tp_reconcile=True,
        )

        bands = boll(middle=102.0, upper=112.0, lower=92.0, candle_ts_ms=1_000)  # same candle as last_tp_update
        got = strat._maybe_update_tp(105.0, 2_000, bands, cvd())

        self.assertIsNotNone(got, "Force reconcile must return UPDATE_TP even on same candle")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertIn("startup_force_tp_reconcile", got.reason)
        self.assertEqual(got.tp_plan, "THREE_STAGE_RUNNER")
        self.assertFalse(strat.state.startup_force_tp_reconcile, "Flag must be cleared after reconcile")
        self.assertEqual(strat.state.last_tp_update_candle_ts_ms, 1_000, "candle_ts must update to current candle")
        self.assertEqual(strat.state.three_stage_tp1_price, 102.0, "TP1 must update to latest boll middle")
        self.assertEqual(strat.state.three_stage_tp2_price, 112.0, "TP2 must update to latest boll upper")

    def test_force_tp_reconcile_returns_update_even_when_plan_unchanged(self) -> None:
        """Even when TP plan/prices match current state, force reconcile returns UPDATE_TP."""
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=112.0,
            tp_mode="UPPER",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_consumed=False,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=102.0,
            three_stage_tp2_price=112.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            trend_runner_active=False,
            last_tp_update_candle_ts_ms=1_000,
            startup_force_tp_reconcile=True,
        )

        bands = boll(middle=102.0, upper=112.0, lower=92.0, candle_ts_ms=1_000)
        got = strat._maybe_update_tp(105.0, 2_000, bands, cvd())

        self.assertIsNotNone(got, "Force reconcile must return UPDATE_TP even when plan unchanged")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertIn("startup_force_tp_reconcile", got.reason)
        self.assertFalse(strat.state.startup_force_tp_reconcile)

    def test_force_tp_reconcile_three_stage_waiting_tp2_updates_tp2_and_sl(self) -> None:
        """After TP1 consumed, force reconcile must update TP2 and recalculate post-TP1 SL."""
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_mode="UPPER",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_consumed=True,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
            trend_runner_active=False,
            last_tp_update_candle_ts_ms=1_000,
            startup_force_tp_reconcile=True,
        )

        bands = boll(middle=102.0, upper=112.0, lower=92.0, candle_ts_ms=1_000)
        got = strat._maybe_update_tp(105.0, 2_000, bands, cvd())

        self.assertIsNotNone(got, "Force reconcile must return UPDATE_TP for waiting_tp2")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertIn("startup_force_tp_reconcile", got.reason)
        self.assertEqual(got.three_stage_tp2_price, 112.0, "TP2 must update to latest outer band")
        self.assertEqual(got.tp_price, 112.0)
        self.assertTrue(got.three_stage_tp1_consumed, "TP1 consumed flag must be preserved")
        self.assertFalse(got.three_stage_tp2_consumed, "TP2 consumed flag must remain False")
        self.assertFalse(got.trend_runner_active, "Trend runner must not activate on reconcile")
        self.assertIsNotNone(got.three_stage_post_tp1_protective_sl_price, "Post-TP1 SL must be recalculated")
        self.assertFalse(strat.state.startup_force_tp_reconcile, "Flag must be cleared")
        self.assertEqual(strat.state.last_tp_update_candle_ts_ms, 1_000)

    def test_force_tp_reconcile_trend_runner_generates_initial_orders(self) -> None:
        """When trend_runner_active=True with no TP/SL orders, force reconcile generates them."""
        strat = strategy(runner_dynamic_enabled=True)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=0.2,
            total_entry_notional=20.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_mode="UPPER",
            tp_plan="SINGLE",
            partial_tp_consumed=True,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=True,
            trend_runner_active=True,
            trend_runner_trend_start_ts_ms=1_000,
            trend_runner_adjust_count=0,
            trend_runner_last_update_candle_ts_ms=500,
            trend_runner_tp_price=None,
            trend_runner_sl_price=None,
            last_tp_update_candle_ts_ms=1_000,
            startup_force_tp_reconcile=True,
        )

        bands = boll(middle=102.0, upper=112.0, lower=92.0, candle_ts_ms=1_000)
        got = strat._maybe_update_tp(108.0, 2_000, bands, cvd())

        self.assertIsNotNone(got, "Force reconcile must return UPDATE_TP for trend runner with initial orders")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertIn("startup_force_tp_reconcile", got.reason)
        self.assertIsNotNone(strat.state.trend_runner_tp_price, "Trend runner TP must be generated")
        self.assertIsNotNone(strat.state.trend_runner_sl_price, "Trend runner SL must be generated")
        self.assertFalse(strat.state.startup_force_tp_reconcile)

    def test_force_tp_reconcile_single_tp_mode_works(self) -> None:
        """Force reconcile works with SINGLE tp_plan (no multi-stage)."""
        strat = strategy(three_stage_runner_enabled=False, split_tp_enabled=False)
        strat.state = StrategyPositionState(
            side="SHORT",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=96.0,
            tp_mode="LOWER",
            tp_plan="SINGLE",
            partial_tp_consumed=False,
            last_tp_update_candle_ts_ms=1_000,
            startup_force_tp_reconcile=True,
        )

        bands = boll(middle=98.0, upper=108.0, lower=88.0, candle_ts_ms=1_000)
        got = strat._maybe_update_tp(97.0, 2_000, bands, cvd())

        self.assertIsNotNone(got, "Force reconcile must return UPDATE_TP for SINGLE TP plan")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertIn("startup_force_tp_reconcile", got.reason)
        self.assertEqual(got.tp_price, 98.0, "TP must update to latest middle band (sufficient profit)")
        self.assertFalse(strat.state.startup_force_tp_reconcile)

    # ── Middle-profit eligibility enforcement ──

    def test_three_stage_before_tp1_disables_when_middle_profit_insufficient_on_update(self) -> None:
        """Three-Stage before TP1 consumed must disable when middle profit insufficient."""
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=110.0,
            tp_mode="UPPER",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_price=101.0,
            partial_tp_ratio=0.6,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            trend_runner_active=False,
            last_tp_update_candle_ts_ms=1_000,
        )

        # middle=100.1 is too low for net_remaining_be=100.0 with min_profit=0.002
        # required_middle = 100.0 * 1.002 = 100.2 > 100.1 → middle insufficient
        got = strat._maybe_update_tp(99.0, 2_000, boll(middle=100.1, upper=103.0, lower=97.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got, "Must return UPDATE_TP when middle profit insufficient")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertEqual(got.tp_plan, "SINGLE")
        self.assertAlmostEqual(got.tp_price, 103.0)
        self.assertIsNone(got.partial_tp_price)
        self.assertEqual(got.partial_tp_ratio, 0.0)
        self.assertFalse(strat.state.three_stage_runner_enabled_for_position)
        self.assertIsNone(strat.state.three_stage_tp1_price)
        self.assertIsNone(strat.state.three_stage_tp2_price)

    def test_three_stage_before_tp1_keeps_when_middle_profit_sufficient(self) -> None:
        """Three-Stage before TP1 consumed keeps running when middle profit sufficient."""
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=110.0,
            tp_mode="UPPER",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_price=101.0,
            partial_tp_ratio=0.6,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            trend_runner_active=False,
            last_tp_update_candle_ts_ms=1_000,
        )

        # middle=100.3 is high enough: required_middle = 100.0 * 1.002 = 100.2 < 100.3 → middle sufficient
        got = strat._maybe_update_tp(100.0, 2_000, boll(middle=100.3, upper=103.0, lower=97.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got, "Must return UPDATE_TP when middle profit sufficient")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertEqual(got.tp_plan, "THREE_STAGE_RUNNER")
        self.assertAlmostEqual(got.partial_tp_price, 100.3)
        self.assertAlmostEqual(got.tp_price, 103.0)
        self.assertTrue(strat.state.three_stage_runner_enabled_for_position)

    def test_three_stage_after_tp1_does_not_disable_when_middle_profit_insufficient(self) -> None:
        """Three-Stage after TP1 consumed must NOT be reset when middle profit insufficient."""
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=110.0,
            tp_mode="UPPER",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_consumed=True,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
            trend_runner_active=False,
            last_tp_update_candle_ts_ms=1_000,
        )

        # middle insufficient but TP1 already consumed → must NOT reset
        got = strat._maybe_update_tp(105.0, 2_000, boll(middle=100.1, upper=103.0, lower=97.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got, "Must return UPDATE_TP for waiting_tp2 even when middle insufficient")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertTrue(got.three_stage_tp1_consumed, "TP1 consumed flag must be preserved")
        self.assertEqual(got.tp_plan, "THREE_STAGE_RUNNER")
        self.assertAlmostEqual(got.three_stage_tp2_price, 103.0, "TP2 must update to latest outer band")
        self.assertTrue(strat.state.three_stage_runner_enabled_for_position)
        self.assertTrue(strat.state.three_stage_tp1_consumed)

    def test_three_stage_middle_profit_fallback_is_locked(self) -> None:
        """Middle-profit fallback must be SINGLE now, without recording time degrade."""
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=110.0,
            tp_mode="UPPER",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_price=101.0,
            partial_tp_ratio=0.6,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            trend_runner_active=False,
            last_tp_update_candle_ts_ms=1_000,
        )

        # middle=100.1 is too low: required_middle = 100.0*1.002 = 100.2 > 100.1 → insufficient
        got = strat._maybe_update_tp(99.0, 2_000, boll(middle=100.1, upper=103.0, lower=97.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got, "Must return UPDATE_TP when middle profit insufficient")
        self.assertEqual(got.intent_type, "UPDATE_TP")
        self.assertEqual(got.tp_plan, "SINGLE")
        self.assertIn("middle_profit_insufficient", got.reason)
        self.assertNotIn("degraded_to_single", got.reason)
        self.assertAlmostEqual(got.tp_price, 103.0)
        self.assertIsNone(got.partial_tp_price, "Partial TP must be cleared when fallback locked")
        self.assertEqual(got.partial_tp_ratio, 0.0)
        self.assertFalse(strat.state.three_stage_runner_enabled_for_position, "Three-Stage runner must be disabled")
        self.assertIsNone(strat.state.three_stage_tp1_price)
        self.assertIsNone(strat.state.three_stage_tp2_price)
        self.assertEqual(strat.state.tp_plan, "SINGLE")
        self.assertIsNone(strat.state.partial_tp_price)
        self.assertEqual(strat.state.partial_tp_ratio, 0.0)
        self.assertIsNone(strat.state.three_stage_pre_tp1_degrade_stage)

    def test_middle_profit_insufficient_before_3h_does_not_permanently_degrade_to_single(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        first_ts = 100_000
        strat.state = three_stage_pre_tp1_state(first_entry_ts_ms=first_ts)

        got = strat._maybe_update_tp(99.0, first_ts + 60_000, boll(middle=100.1, upper=110.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.tp_plan, "SINGLE")
        self.assertEqual(got.tp_mode, "UPPER")
        self.assertAlmostEqual(got.tp_price, 110.0)
        self.assertIn("middle_profit_insufficient", got.reason)
        self.assertNotIn("degraded_to_single", got.reason)
        self.assertFalse(strat.state.three_stage_runner_enabled_for_position)
        self.assertFalse(strat.state.middle_runner_pending)
        self.assertIsNone(strat.state.three_stage_pre_tp1_degrade_stage)

    def test_middle_profit_recovered_before_3h_can_reenter_three_stage(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        first_ts = 100_000
        strat.state = three_stage_pre_tp1_state(first_entry_ts_ms=first_ts)

        first = strat._maybe_update_tp(99.0, first_ts + 60_000, boll(middle=100.1, upper=110.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(first)
        self.assertEqual(first.tp_plan, "SINGLE")
        self.assertIsNone(strat.state.three_stage_pre_tp1_degrade_stage)

        strat.state.net_remaining_breakeven_price = 95.0
        got = strat._maybe_update_tp(99.0, first_ts + 2 * 60 * 60 * 1000, boll(middle=101.0, upper=110.0, candle_ts_ms=3_000), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.tp_plan, "THREE_STAGE_RUNNER")
        self.assertTrue(strat.state.three_stage_runner_enabled_for_position)
        self.assertIsNone(strat.state.three_stage_pre_tp1_degrade_stage)

    def test_middle_profit_insufficient_after_3h_does_not_skip_future_middle_runner(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        first_ts = 100_000
        strat.state = three_stage_pre_tp1_state(first_entry_ts_ms=first_ts)

        first = strat._maybe_update_tp(99.0, first_ts + 10_801_000, boll(middle=100.1, upper=110.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(first)
        self.assertEqual(first.tp_plan, "SINGLE")
        self.assertFalse(strat.state.middle_runner_pending)
        self.assertIsNone(strat.state.three_stage_pre_tp1_degrade_stage)

        strat.state.net_remaining_breakeven_price = 95.0
        got = strat._maybe_update_tp(99.0, first_ts + 10_901_000, boll(middle=101.0, upper=110.0, candle_ts_ms=3_000), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.tp_plan, "MIDDLE_RUNNER")
        self.assertEqual(got.reason, "three_stage_pre_tp1_degraded_to_middle_runner")
        self.assertTrue(strat.state.middle_runner_pending)
        self.assertEqual(strat.state.three_stage_pre_tp1_degrade_stage, "MIDDLE_RUNNER")

    def test_middle_profit_insufficient_after_middle_runner_degrade_does_not_force_single_stage(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        first_ts = 100_000
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            first_entry_ts_ms=first_ts,
            last_order_ts_ms=first_ts,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=110.0,
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=101.0,
            partial_tp_ratio=0.8,
            middle_runner_enabled_for_position=True,
            middle_runner_pending=True,
            middle_runner_active=False,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
            middle_runner_first_tp_price=101.0,
            middle_runner_final_tp_price=110.0,
            three_stage_pre_tp1_degrade_stage="MIDDLE_RUNNER",
            last_tp_update_candle_ts_ms=1_000,
        )

        first = strat._maybe_update_tp(99.0, first_ts + 14_400_000, boll(middle=100.1, upper=110.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(first)
        self.assertEqual(first.tp_plan, "SINGLE")
        self.assertEqual(first.reason, "middle_runner_middle_profit_insufficient_single_outer")
        self.assertEqual(strat.state.three_stage_pre_tp1_degrade_stage, "MIDDLE_RUNNER")

        strat.state.net_remaining_breakeven_price = 95.0
        got = strat._maybe_update_tp(99.0, first_ts + 14_500_000, boll(middle=101.0, upper=110.0, candle_ts_ms=3_000), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.tp_plan, "MIDDLE_RUNNER")
        self.assertTrue(strat.state.middle_runner_pending)
        self.assertEqual(strat.state.three_stage_pre_tp1_degrade_stage, "MIDDLE_RUNNER")

    def test_three_stage_pre_tp1_degrades_to_middle_runner_after_3h_when_middle_profit_sufficient(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        first_ts = 100_000
        strat.state = three_stage_pre_tp1_state(first_entry_ts_ms=first_ts)

        got = strat._maybe_update_tp(99.0, first_ts + 10_801_000, boll(middle=101.0, upper=110.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.reason, "three_stage_pre_tp1_degraded_to_middle_runner")
        self.assertEqual(got.tp_plan, "MIDDLE_RUNNER")
        self.assertEqual(strat.state.tp_plan, "MIDDLE_RUNNER")
        self.assertTrue(strat.state.middle_runner_pending)
        self.assertFalse(strat.state.three_stage_runner_enabled_for_position)
        self.assertEqual(strat.state.three_stage_pre_tp1_degrade_stage, "MIDDLE_RUNNER")

    def test_three_stage_pre_tp1_degrades_to_single_after_6h(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        first_ts = 100_000
        strat.state = three_stage_pre_tp1_state(first_entry_ts_ms=first_ts)

        got = strat._maybe_update_tp(99.0, first_ts + 21_601_000, boll(middle=101.0, upper=110.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.reason, "three_stage_pre_tp1_degraded_to_single")
        self.assertEqual(got.tp_plan, "SINGLE")
        self.assertAlmostEqual(got.tp_price, 101.0)
        self.assertEqual(strat.state.tp_plan, "SINGLE")
        self.assertFalse(strat.state.middle_runner_pending)
        self.assertFalse(strat.state.three_stage_runner_enabled_for_position)
        self.assertIsNone(strat.state.partial_tp_price)

    def test_time_degrade_to_single_still_sets_degrade_stage(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        first_ts = 100_000
        strat.state = three_stage_pre_tp1_state(first_entry_ts_ms=first_ts)

        got = strat._maybe_update_tp(99.0, first_ts + 21_601_000, boll(middle=101.0, upper=110.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.reason, "three_stage_pre_tp1_degraded_to_single")
        self.assertEqual(strat.state.three_stage_pre_tp1_degrade_stage, "SINGLE")

    def test_degraded_middle_runner_degrades_to_single_after_6h_if_no_first_close(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        first_ts = 100_000
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            first_entry_ts_ms=first_ts,
            last_order_ts_ms=first_ts + 7_200_000,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=110.0,
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=101.0,
            partial_tp_ratio=0.8,
            middle_runner_enabled_for_position=True,
            middle_runner_pending=True,
            middle_runner_active=False,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
            middle_runner_first_tp_price=101.0,
            middle_runner_final_tp_price=110.0,
            three_stage_pre_tp1_degrade_stage="MIDDLE_RUNNER",
            last_tp_update_candle_ts_ms=1_000,
        )

        got = strat._maybe_update_tp(99.0, first_ts + 21_601_000, boll(middle=101.0, upper=110.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.reason, "three_stage_pre_tp1_degraded_to_single")
        self.assertEqual(strat.state.tp_plan, "SINGLE")
        self.assertFalse(strat.state.middle_runner_pending)

    def test_three_stage_pre_tp1_timeout_does_not_reset_after_add(self) -> None:
        strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        first_ts = 100_000
        strat.state = three_stage_pre_tp1_state(first_entry_ts_ms=first_ts, layers=2, last_order_ts_ms=first_ts + 7_200_000)

        got = strat._maybe_update_tp(99.0, first_ts + 10_801_000, boll(middle=101.0, upper=110.0, candle_ts_ms=2_000), cvd())

        self.assertIsNotNone(got)
        self.assertEqual(got.reason, "three_stage_pre_tp1_degraded_to_middle_runner")

    def test_three_stage_after_tp1_consumed_does_not_degrade(self) -> None:
        strat = strategy()
        strat.state = three_stage_pre_tp1_state(three_stage_tp1_consumed=True, partial_tp_consumed=True)

        self.assertIsNone(strat._three_stage_pre_tp1_degrade_target(100_000 + 21_601_000))

    def test_middle_runner_active_does_not_degrade_to_single(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            first_entry_ts_ms=100_000,
            middle_runner_active=True,
            three_stage_pre_tp1_degrade_stage="MIDDLE_RUNNER",
        )

        self.assertIsNone(strat._three_stage_pre_tp1_degrade_target(100_000 + 21_601_000))

    def test_single_degrade_uses_outer_when_middle_profit_insufficient(self) -> None:
        long_strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        long_strat.state = StrategyPositionState(side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        long_tp, long_mode = long_strat._degrade_three_stage_pre_tp1_to_single(21_601_000, boll(middle=100.1, upper=110.0, lower=90.0))
        self.assertAlmostEqual(long_tp, 110.0)
        self.assertEqual(long_mode, "UPPER")

        short_strat = strategy(breakeven_fee_buffer_pct=0.0, tp_min_net_profit_pct=0.002)
        short_strat.state = StrategyPositionState(side="SHORT", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        short_tp, short_mode = short_strat._degrade_three_stage_pre_tp1_to_single(21_601_000, boll(middle=99.9, upper=110.0, lower=90.0))
        self.assertAlmostEqual(short_tp, 90.0)
        self.assertEqual(short_mode, "LOWER")

    def test_tp_selection_uses_net_remaining_breakeven(self) -> None:
        """_select_tp_price uses net_remaining_breakeven_price over avg_entry_price."""
        strat = strategy(breakeven_fee_buffer_pct=0.001, tp_min_net_profit_pct=0.002)
        strat.state.avg_entry_price = 100.0
        strat.state.net_remaining_breakeven_price = 98.0

        # With avg_entry=100: required_middle=100*1.002=100.2 > 98.3 → would be UPPER
        # With net_remaining_be=98: required_middle=98*1.002=98.196 < 98.3 → MIDDLE
        tp_price, tp_mode = strat._select_tp_price("LONG", boll(middle=98.3, upper=105.0))

        self.assertEqual(tp_mode, "MIDDLE")
        self.assertAlmostEqual(tp_price, 98.3)


class RecordingTrader(Trader):
    def __init__(self, side: str = "LONG") -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.td_mode = "isolated"
        self.pos_side_mode = "net"
        self.position_contracts = Decimal("10")
        self.contract_precision = Decimal("0.01")
        self.min_contracts = Decimal("0.01")
        self.tp_order_id = None
        self.near_tp_protective_sl_order_id = None
        self.middle_runner_protective_sl_order_id = None
        self.three_stage_post_tp1_protective_sl_order_id = None
        self.trend_runner_sl_order_id = None
        self.side = side
        self.placed_specs = []
        self.trend_stop_calls = 0
        self.post_tp1_stop_calls = 0
        self.cancel_reduce_only_calls = 0
        self.cancelled_trend_runner_stop_ids = []
        self.cancelled_post_tp1_stop_ids = []

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(self.side, self.position_contracts, 100.0, float(self.position_contracts * Decimal("0.1")), self.position_contracts)

    async def cancel_existing_reduce_only_orders(self) -> None:
        self.cancel_reduce_only_calls += 1
        return None

    async def market_exit_remaining_position_with_retries(self, side, retry_count):  # type: ignore[no-untyped-def]
        self.position_contracts = Decimal("0")
        await self._cleanup_after_near_tp_market_exit()
        return True, "market_exit_order_id=runner-exit"

    async def _place_reduce_only_take_profit_orders(self, intent_: TradeIntent, specs):  # type: ignore[no-untyped-def]
        self.placed_specs = specs
        return [f"ord-{label}" for label, _contracts, _price in specs]

    async def place_trend_runner_protective_stop_with_retries(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.trend_stop_calls += 1
        return True, "algo-runner", "protective_sl_placed"

    async def place_three_stage_post_tp1_protective_stop_with_retries(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.post_tp1_stop_calls += 1
        self.three_stage_post_tp1_protective_sl_order_id = "algo-post"
        return True, "algo-post", "protective_sl_placed"

    async def cancel_trend_runner_protective_stop(self, order_id: str | None) -> bool:
        self.cancelled_trend_runner_stop_ids.append(order_id)
        if self.trend_runner_sl_order_id == order_id:
            self.trend_runner_sl_order_id = None
        return True

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id: str | None) -> bool:
        self.cancelled_post_tp1_stop_ids.append(order_id)
        if self.three_stage_post_tp1_protective_sl_order_id == order_id:
            self.three_stage_post_tp1_protective_sl_order_id = None
        return True


class NoPositionRecordingTrader(RecordingTrader):
    async def fetch_position_snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))


class ThreeStageTrendRunnerTraderTest(unittest.IsolatedAsyncioTestCase):
    def test_build_three_stage_order_specs(self) -> None:
        trader = Trader.__new__(Trader)
        trader.position_contracts = Decimal("10")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")

        specs = trader._build_take_profit_order_specs(intent())

        self.assertEqual(
            specs,
            [
                ("tp1_middle", Decimal("6.00"), 101.0),
                ("tp2_outer", Decimal("2.00"), 110.0),
            ],
        )

    async def test_initial_three_stage_long_does_not_place_middle_sell_stop_below_middle(self) -> None:
        trader = RecordingTrader("LONG")

        result = await trader.replace_take_profit(
            intent(price=90.0, three_stage_runner_sl_price=101.0, trend_runner_active=False)
        )

        self.assertTrue(result.ok)
        self.assertEqual(trader.trend_stop_calls, 0)
        self.assertFalse(result.protective_sl_ok)
        self.assertEqual([label for label, _contracts, _price in trader.placed_specs], ["tp1_middle", "tp2_outer"])

    async def test_initial_three_stage_short_does_not_place_middle_buy_stop_above_middle(self) -> None:
        trader = RecordingTrader("SHORT")

        result = await trader.replace_take_profit(
            intent(
                side="SHORT",
                price=110.0,
                tp_price=90.0,
                three_stage_tp1_price=99.0,
                three_stage_tp2_price=90.0,
                three_stage_runner_sl_price=99.0,
                trend_runner_active=False,
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(trader.trend_stop_calls, 0)
        self.assertFalse(result.protective_sl_ok)
        self.assertEqual([label for label, _contracts, _price in trader.placed_specs], ["tp1_middle", "tp2_outer"])

    async def test_active_trend_runner_places_runner_tp_and_sl(self) -> None:
        trader = RecordingTrader("LONG")
        trader.position_contracts = Decimal("2")

        result = await trader.replace_take_profit(
            intent(
                intent_type="UPDATE_TP",
                tp_plan="SINGLE",
                tp_price=111.1,
                trend_runner_active=True,
                trend_runner_tp_price=111.1,
                trend_runner_sl_price=101.0,
            )
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.protective_sl_ok)
        self.assertEqual(trader.trend_stop_calls, 1)
        self.assertEqual(trader.placed_specs, [("final", Decimal("2"), 111.1)])

    async def test_post_tp1_update_rebuilds_only_tp2_and_global_sl(self) -> None:
        trader = RecordingTrader("LONG")
        trader.position_contracts = Decimal("4")

        result = await trader.replace_take_profit(
            intent(
                intent_type="UPDATE_TP",
                tp_plan="THREE_STAGE_RUNNER",
                tp_price=110.0,
                three_stage_tp1_consumed=True,
                three_stage_tp2_consumed=False,
                three_stage_post_tp1_protective_sl_price=100.5,
                three_stage_post_tp1_protective_sl_order_id="old-post",
                trend_runner_active=False,
            )
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.protective_sl_ok)
        self.assertEqual(trader.post_tp1_stop_calls, 1)
        self.assertEqual(trader.cancel_reduce_only_calls, 1)
        self.assertEqual(trader.placed_specs, [("tp2_outer", Decimal("2.00"), 110.0)])
        self.assertEqual(trader.cancelled_post_tp1_stop_ids, ["old-post"])

    def test_after_tp1_build_specs_keeps_runner_contracts(self) -> None:
        trader = RecordingTrader("LONG")
        trader.position_contracts = Decimal("0.40")

        specs = trader._build_three_stage_order_specs(
            intent(
                tp_plan="THREE_STAGE_RUNNER",
                tp_price=110.0,
                three_stage_tp2_price=110.0,
                three_stage_tp2_ratio=0.20,
                three_stage_runner_ratio=0.20,
                three_stage_tp1_consumed=True,
                three_stage_tp2_consumed=False,
            )
        )

        self.assertEqual(specs, [("tp2_outer", Decimal("0.20"), 110.0)])
        self.assertEqual(trader.position_contracts - specs[0][1], Decimal("0.20"))

        trader.position_contracts = Decimal("4.00")
        specs = trader._build_three_stage_order_specs(
            intent(
                tp_plan="THREE_STAGE_RUNNER",
                tp_price=110.0,
                three_stage_tp2_price=110.0,
                three_stage_tp2_ratio=0.20,
                three_stage_runner_ratio=0.20,
                three_stage_tp1_consumed=True,
                three_stage_tp2_consumed=False,
            )
        )

        self.assertEqual(specs, [("tp2_outer", Decimal("2.00"), 110.0)])
        self.assertEqual(trader.position_contracts - specs[0][1], Decimal("2.00"))

    async def test_active_trend_runner_cancels_restored_sl_order_id_from_intent(self) -> None:
        trader = RecordingTrader("LONG")
        trader.position_contracts = Decimal("2")
        trader.trend_runner_sl_order_id = None

        result = await trader.replace_take_profit(
            intent(
                intent_type="UPDATE_TP",
                tp_plan="SINGLE",
                tp_price=111.1,
                trend_runner_active=True,
                trend_runner_tp_price=111.1,
                trend_runner_sl_price=101.0,
                trend_runner_sl_order_id="old-algo",
            )
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.protective_sl_ok)
        self.assertEqual(trader.trend_stop_calls, 1)
        self.assertEqual(trader.cancelled_trend_runner_stop_ids, ["old-algo"])
        self.assertEqual(trader.trend_runner_sl_order_id, "algo-runner")

    async def test_market_exit_runner_cancels_restored_sl_order_id_from_intent(self) -> None:
        trader = RecordingTrader("LONG")
        trader.position_contracts = Decimal("2")
        trader.trend_runner_sl_order_id = None

        result = await trader.execute_market_exit_runner(
            intent(
                intent_type="MARKET_EXIT_RUNNER",
                tp_plan="SINGLE",
                tp_price=111.1,
                trend_runner_active=True,
                trend_runner_tp_price=111.1,
                trend_runner_sl_price=101.0,
                trend_runner_sl_order_id="old-algo",
                reason="trend_runner_middle_lost",
            )
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.near_tp_exit_all)
        self.assertEqual(trader.cancelled_trend_runner_stop_ids, ["old-algo"])
        self.assertIsNone(trader.trend_runner_sl_order_id)

    async def test_market_exit_runner_already_flat_cleans_restored_sl_order_id(self) -> None:
        trader = NoPositionRecordingTrader("LONG")
        trader.trend_runner_sl_order_id = None

        result = await trader.execute_market_exit_runner(
            intent(
                intent_type="MARKET_EXIT_RUNNER",
                tp_plan="SINGLE",
                tp_price=111.1,
                trend_runner_active=True,
                trend_runner_tp_price=111.1,
                trend_runner_sl_price=101.0,
                trend_runner_sl_order_id="old-algo",
                reason="trend_runner_middle_lost",
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.message, "runner_already_flat")
        self.assertEqual(trader.cancelled_trend_runner_stop_ids, ["old-algo"])
        self.assertIsNone(trader.trend_runner_sl_order_id)


if __name__ == "__main__":
    unittest.main()
