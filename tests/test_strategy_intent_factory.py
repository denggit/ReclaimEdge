from __future__ import annotations

import importlib.util
import sys
import types
import unittest

if importlib.util.find_spec("aiohttp") is None:
    sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
)
from src.strategies.strategy_intent_factory import StrategyIntentFactory


# ── helpers ────────────────────────────────────────────────────────────────


def _boll(**overrides) -> BollSnapshot:
    values = dict(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1_000,
        close=100.0,
        middle=110.0,
        upper=120.0,
        lower=90.0,
        upper_distance_pct=0.1,
        lower_distance_pct=0.1,
        alert_switch_on=True,
        live_mode=True,
    )
    values.update(overrides)
    return BollSnapshot(**values)


def _cvd(**overrides) -> CvdSnapshot:
    values = dict(
        ts_ms=1_000,
        price=100.0,
        side="buy",
        size=1.0,
        signed_delta=1.0,
        total_cvd=1.0,
        fast_cvd=0.5,
        previous_fast_cvd=0.3,
        buy_volume=1.0,
        sell_volume=0.0,
        buy_ratio=0.65,
        sell_ratio=0.35,
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
    values.update(overrides)
    return CvdSnapshot(**values)


def _size(**overrides) -> PositionSize:
    kwargs = dict(margin_usdt=100.0, notional_usdt=100.0, eth_qty=1.0, layer_index=1, layer_multiplier=1.0)
    kwargs.update(overrides)
    return PositionSize(**kwargs)


def _strategy(**config_overrides) -> BollCvdReclaimStrategy:
    values = dict()
    values.update(config_overrides)
    config = BollCvdReclaimStrategyConfig(**values)
    sizer = SimplePositionSizer(SimplePositionSizerConfig())
    return BollCvdReclaimStrategy(config, sizer)


def _setup_long_position(strat: BollCvdReclaimStrategy) -> None:
    """Set up a basic LONG position state on the strategy."""
    s = strat.state
    s.side = "LONG"
    s.layers = 2
    s.last_entry_price = 95.0
    s.tp_price = 115.0
    s.tp_mode = "UPPER"
    s.avg_entry_price = 96.0
    s.breakeven_price = 96.5
    s.total_entry_qty = 2.0
    s.total_entry_notional = 192.0
    s.partial_tp_price = None
    s.partial_tp_ratio = 0.0
    s.tp_plan = "SINGLE"
    s.partial_tp_consumed = False


# ── tests ───────────────────────────────────────────────────────────────────


class StrategyIntentFactoryInitTest(unittest.TestCase):
    def test_factory_stores_strategy_reference(self) -> None:
        strat = _strategy()
        factory = StrategyIntentFactory(strat)
        self.assertIs(factory.strategy, strat)


class WrapperDelegateTest(unittest.TestCase):
    """Verify that strategy wrapper methods delegate to the factory."""

    def setUp(self) -> None:
        self.strat = _strategy()
        _setup_long_position(self.strat)

    def test_intent_delegates_to_factory_build_intent(self) -> None:
        b = _boll()
        c = _cvd()
        sz = _size()
        intent = self.strat._intent(
            intent_type="ADD_LONG",
            side="LONG",
            price=100.0,
            layer_index=2,
            tp_price=115.0,
            reason="test",
            size=sz,
            boll=b,
            cvd=c,
            ts_ms=2_000,
        )
        self.assertIsInstance(intent, TradeIntent)
        self.assertEqual(intent.intent_type, "ADD_LONG")
        self.assertEqual(intent.side, "LONG")
        self.assertEqual(intent.price, 100.0)
        self.assertEqual(intent.layer_index, 2)
        self.assertEqual(intent.tp_price, 115.0)
        self.assertEqual(intent.reason, "test")
        self.assertEqual(intent.size, sz)
        self.assertEqual(intent.fast_cvd, c.fast_cvd)
        self.assertEqual(intent.previous_fast_cvd, c.previous_fast_cvd)
        self.assertEqual(intent.buy_ratio, c.buy_ratio)
        self.assertEqual(intent.sell_ratio, c.sell_ratio)
        self.assertEqual(intent.boll_upper, b.upper)
        self.assertEqual(intent.boll_middle, b.middle)
        self.assertEqual(intent.boll_lower, b.lower)
        self.assertEqual(intent.ts_ms, 2_000)
        self.assertEqual(intent.avg_entry_price, 96.0)
        self.assertEqual(intent.breakeven_price, 96.5)
        self.assertEqual(intent.tp_mode, "UPPER")

    def test_managed_core_contracts_for_intent_delegates(self) -> None:
        result = self.strat._managed_core_contracts_for_intent("OPEN_LONG")
        self.assertIsNone(result)

    def test_managed_core_eth_qty_for_intent_delegates(self) -> None:
        result = self.strat._managed_core_eth_qty_for_intent("OPEN_LONG")
        self.assertEqual(result, 0.0)

    def test_protected_order_ids_delegates(self) -> None:
        result = self.strat._protected_order_ids()
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, ())


class BuildIntentFullPayloadTest(unittest.TestCase):
    """Verify build_intent returns a TradeIntent with all fields populated correctly."""

    def setUp(self) -> None:
        self.strat = _strategy()
        _setup_long_position(self.strat)
        s = self.strat.state
        # Set additional state fields to verify they are copied into the intent.
        s.middle_runner_enabled_for_position = True
        s.middle_runner_pending = False
        s.middle_runner_active = False
        s.middle_runner_first_close_ratio = 0.8
        s.middle_runner_keep_ratio = 0.2
        s.middle_runner_first_tp_price = 108.0
        s.middle_runner_final_tp_price = 118.0
        s.middle_runner_protective_sl_price = 97.0
        s.middle_runner_protective_sl_order_id = "mid-sl-1"
        s.middle_runner_extension_triggered = False
        s.middle_runner_add_disabled = False
        s.three_stage_tp1_price = 108.0
        s.three_stage_tp1_ratio = 0.6
        s.three_stage_tp2_price = 118.0
        s.three_stage_tp2_ratio = 0.2
        s.trend_runner_tp_price = 125.0
        s.three_stage_runner_ratio = 0.2
        s.trend_runner_sl_price = 94.0
        s.three_stage_tp1_consumed = False
        s.three_stage_tp2_consumed = False
        s.three_stage_post_tp1_protective_sl_price = 96.0
        s.three_stage_post_tp1_protective_sl_order_id = "ts-sl-1"
        s.three_stage_post_tp1_sl_extension_triggered = False
        s.three_stage_post_tp1_protected = False
        s.trend_runner_active = False
        s.trend_runner_tp_order_id = None
        s.trend_runner_sl_order_id = None
        s.trend_runner_exit_reason = None
        s.trend_runner_adjust_count = 0
        self.factory = StrategyIntentFactory(self.strat)

    def test_build_intent_populates_all_state_fields(self) -> None:
        b = _boll()
        c = _cvd()
        sz = _size()
        intent = self.factory.build_intent(
            intent_type="OPEN_LONG",
            side="LONG",
            price=100.0,
            layer_index=1,
            tp_price=115.0,
            reason="test open",
            size=sz,
            boll=b,
            cvd=c,
            ts_ms=2_000,
        )
        # Basic fields
        self.assertEqual(intent.intent_type, "OPEN_LONG")
        self.assertEqual(intent.side, "LONG")
        self.assertEqual(intent.price, 100.0)
        self.assertEqual(intent.layer_index, 1)
        self.assertEqual(intent.tp_price, 115.0)
        self.assertEqual(intent.reason, "test open")
        self.assertEqual(intent.size, sz)
        self.assertEqual(intent.ts_ms, 2_000)
        # CVD fields
        self.assertEqual(intent.fast_cvd, c.fast_cvd)
        self.assertEqual(intent.previous_fast_cvd, c.previous_fast_cvd)
        self.assertEqual(intent.buy_ratio, c.buy_ratio)
        self.assertEqual(intent.sell_ratio, c.sell_ratio)
        # Boll fields
        self.assertEqual(intent.boll_upper, b.upper)
        self.assertEqual(intent.boll_middle, b.middle)
        self.assertEqual(intent.boll_lower, b.lower)
        # State-derived fields
        self.assertEqual(intent.avg_entry_price, 96.0)
        self.assertEqual(intent.breakeven_price, 96.5)
        self.assertEqual(intent.tp_mode, "UPPER")
        self.assertEqual(intent.tp_plan, "SINGLE")
        # Middle runner fields
        self.assertTrue(intent.middle_runner_enabled_for_position)
        self.assertEqual(intent.middle_runner_first_close_ratio, 0.8)
        self.assertEqual(intent.middle_runner_keep_ratio, 0.2)
        self.assertEqual(intent.middle_runner_first_tp_price, 108.0)
        self.assertEqual(intent.middle_runner_final_tp_price, 118.0)
        self.assertEqual(intent.middle_runner_protective_sl_price, 97.0)
        self.assertEqual(intent.middle_runner_protective_sl_order_id, "mid-sl-1")
        # Three-stage fields
        self.assertEqual(intent.three_stage_tp1_price, 108.0)
        self.assertEqual(intent.three_stage_tp1_ratio, 0.6)
        self.assertEqual(intent.three_stage_tp2_price, 118.0)
        self.assertEqual(intent.three_stage_tp2_ratio, 0.2)
        self.assertEqual(intent.three_stage_runner_tp_price, 125.0)
        self.assertEqual(intent.three_stage_runner_ratio, 0.2)
        self.assertEqual(intent.three_stage_runner_sl_price, 94.0)
        self.assertEqual(intent.three_stage_post_tp1_protective_sl_price, 96.0)
        self.assertEqual(intent.three_stage_post_tp1_protective_sl_order_id, "ts-sl-1")
        # Trend runner fields
        self.assertFalse(intent.trend_runner_active)
        self.assertEqual(intent.trend_runner_adjust_count, 0)

    def test_build_intent_includes_protected_order_ids(self) -> None:
        b = _boll()
        c = _cvd()
        sz = _size()
        intent = self.factory.build_intent(
            intent_type="OPEN_LONG",
            side="LONG",
            price=100.0,
            layer_index=1,
            tp_price=115.0,
            reason="test",
            size=sz,
            boll=b,
            cvd=c,
            ts_ms=2_000,
        )
        self.assertIsInstance(intent.protected_order_ids, tuple)

    def test_build_intent_includes_managed_core_fields(self) -> None:
        b = _boll()
        c = _cvd()
        sz = _size()
        intent = self.factory.build_intent(
            intent_type="OPEN_LONG",
            side="LONG",
            price=100.0,
            layer_index=1,
            tp_price=115.0,
            reason="test",
            size=sz,
            boll=b,
            cvd=c,
            ts_ms=2_000,
        )
        self.assertIsNone(intent.managed_core_contracts)
        self.assertEqual(intent.managed_core_eth_qty, 0.0)


class ProtectedOrderIdsTest(unittest.TestCase):
    """Verify protected_order_ids behavior (no-sidecar runtime)."""

    def setUp(self) -> None:
        self.strat = _strategy()
        _setup_long_position(self.strat)
        self.factory = StrategyIntentFactory(self.strat)

    def test_empty_when_no_protective_ids(self) -> None:
        result = self.factory.protected_order_ids()
        self.assertEqual(result, ())

    def test_includes_core_protective_order_ids(self) -> None:
        """protected_order_ids includes entry, middle, three-stage, and trend runner SL IDs."""
        self.strat.state.entry_protective_sl_order_id = "entry-sl-1"
        self.strat.state.middle_runner_protective_sl_order_id = "mid-sl-1"
        self.strat.state.three_stage_post_tp1_protective_sl_order_id = "ts-sl-1"
        self.strat.state.trend_runner_sl_order_id = "tr-sl-1"
        result = self.factory.protected_order_ids()
        self.assertIn("entry-sl-1", result)
        self.assertIn("mid-sl-1", result)
        self.assertIn("ts-sl-1", result)
        self.assertIn("tr-sl-1", result)
        self.assertEqual(len(result), 4)

    def test_no_sidecar_legs_in_protected_order_ids(self) -> None:
        """With Sidecar removed, protected_order_ids only returns core order IDs."""
        result = self.factory.protected_order_ids()
        # No sidecar-derived entries
        self.assertEqual(result, ())

    def test_includes_middle_runner_protective_sl_order_id(self) -> None:
        self.strat.state.middle_runner_protective_sl_order_id = "mid-sl-1"
        result = self.factory.protected_order_ids()
        self.assertIn("mid-sl-1", result)

    def test_includes_three_stage_post_tp1_protective_sl_order_id(self) -> None:
        self.strat.state.three_stage_post_tp1_protective_sl_order_id = "ts-sl-1"
        result = self.factory.protected_order_ids()
        self.assertIn("ts-sl-1", result)

    def test_includes_trend_runner_sl_order_id(self) -> None:
        self.strat.state.trend_runner_sl_order_id = "tr-sl-1"
        result = self.factory.protected_order_ids()
        self.assertIn("tr-sl-1", result)

    def test_deduplicates_preserving_order(self) -> None:
        self.strat.state.trend_runner_sl_order_id = "dup-id"
        self.strat.state.middle_runner_protective_sl_order_id = "unique-id"
        result = self.factory.protected_order_ids()
        # "dup-id" should appear only once, preserving first occurrence order
        ids_list = list(result)
        self.assertEqual(ids_list.count("dup-id"), 1)
        self.assertEqual(len(result), 2)

    def test_none_order_ids_are_skipped(self) -> None:
        self.strat.state.middle_runner_protective_sl_order_id = None
        self.strat.state.three_stage_post_tp1_protective_sl_order_id = None
        self.strat.state.trend_runner_sl_order_id = None
        result = self.factory.protected_order_ids()
        self.assertEqual(result, ())




class RunnerMarketExitIntentFactoryTest(unittest.TestCase):
    """Verify build_runner_market_exit_intent constructs correct TradeIntent."""

    def setUp(self) -> None:
        self.strat = _strategy()
        _setup_long_position(self.strat)
        s = self.strat.state
        s.layers = 2
        s.tp_price = 115.0
        s.tp_mode = "UPPER"
        s.avg_entry_price = 96.0
        s.breakeven_price = 96.5
        s.trend_runner_tp_price = 125.0
        s.trend_runner_sl_price = 94.0
        s.trend_runner_tp_order_id = "tr-tp-1"
        s.trend_runner_sl_order_id = "tr-sl-1"
        s.trend_runner_adjust_count = 2
        self.factory = StrategyIntentFactory(self.strat)

    def test_build_runner_market_exit_intent_fields(self) -> None:
        b = _boll()
        c = _cvd()
        sz = _size()
        intent = self.factory.build_runner_market_exit_intent(
            side="LONG",
            price=93.0,
            layer_index=2,
            tp_price=125.0,
            reason="trend_runner_sl_hit",
            size=sz,
            boll=b,
            cvd=c,
            ts_ms=4_000,
        )
        self.assertEqual(intent.intent_type, "MARKET_EXIT_RUNNER")
        self.assertEqual(intent.side, "LONG")
        self.assertEqual(intent.price, 93.0)
        self.assertEqual(intent.layer_index, 2)
        self.assertEqual(intent.tp_price, 125.0)
        self.assertEqual(intent.reason, "trend_runner_sl_hit")
        self.assertEqual(intent.size, sz)
        self.assertEqual(intent.ts_ms, 4_000)
        # Standard market exit fields
        self.assertEqual(intent.tp_plan, "SINGLE")
        self.assertTrue(intent.partial_tp_consumed)
        self.assertIsNone(intent.partial_tp_price)
        self.assertEqual(intent.partial_tp_ratio, 0.0)
        # Trend runner fields
        self.assertTrue(intent.trend_runner_active)
        self.assertEqual(intent.trend_runner_tp_price, 125.0)
        self.assertEqual(intent.trend_runner_sl_price, 94.0)
        self.assertEqual(intent.trend_runner_tp_order_id, "tr-tp-1")
        self.assertEqual(intent.trend_runner_sl_order_id, "tr-sl-1")
        self.assertEqual(intent.trend_runner_exit_reason, "trend_runner_sl_hit")
        self.assertEqual(intent.trend_runner_adjust_count, 2)

    def test_build_runner_market_exit_intent_cvd_boll(self) -> None:
        b = _boll(upper=120.0, middle=110.0, lower=90.0)
        c = _cvd(fast_cvd=-0.3, previous_fast_cvd=-0.1, buy_ratio=0.3, sell_ratio=0.7)
        sz = _size()
        intent = self.factory.build_runner_market_exit_intent(
            side="LONG",
            price=93.0,
            layer_index=2,
            tp_price=125.0,
            reason="trend_runner_sl_hit",
            size=sz,
            boll=b,
            cvd=c,
            ts_ms=4_000,
        )
        self.assertEqual(intent.fast_cvd, -0.3)
        self.assertEqual(intent.previous_fast_cvd, -0.1)
        self.assertEqual(intent.buy_ratio, 0.3)
        self.assertEqual(intent.sell_ratio, 0.7)
        self.assertEqual(intent.boll_upper, 120.0)
        self.assertEqual(intent.boll_middle, 110.0)
        self.assertEqual(intent.boll_lower, 90.0)
        self.assertEqual(intent.avg_entry_price, 96.0)
        self.assertEqual(intent.breakeven_price, 96.5)
        self.assertEqual(intent.tp_mode, "UPPER")


class RunnerMarketExitIntentWrapperTest(unittest.TestCase):
    """Verify _runner_market_exit_intent wrapper preserves behavior
    (state writes, logs) and delegates intent construction."""

    def test_returns_none_when_side_is_none(self) -> None:
        strat = _strategy()
        strat.state.side = None
        intent = strat._runner_market_exit_intent(100.0, 1_000, _boll(), _cvd(), "test")
        self.assertIsNone(intent)

    def test_returns_intent_when_side_set(self) -> None:
        strat = _strategy()
        _setup_long_position(strat)
        strat.state.layers = 2
        strat.state.trend_runner_tp_price = 125.0
        strat.state.trend_runner_sl_price = 94.0
        intent = strat._runner_market_exit_intent(93.0, 4_000, _boll(), _cvd(), "trend_runner_sl_hit")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "MARKET_EXIT_RUNNER")
        self.assertEqual(strat.state.trend_runner_exit_reason, "trend_runner_sl_hit")

    def test_writes_trend_runner_exit_reason(self) -> None:
        strat = _strategy()
        _setup_long_position(strat)
        strat.state.trend_runner_exit_reason = None
        strat._runner_market_exit_intent(93.0, 4_000, _boll(), _cvd(), "trend_runner_sl_hit")
        self.assertEqual(strat.state.trend_runner_exit_reason, "trend_runner_sl_hit")

    def test_tp_price_fallback_order(self) -> None:
        """tp_price should use trend_runner_tp_price first, then state.tp_price, then price."""
        strat = _strategy()
        _setup_long_position(strat)
        strat.state.trend_runner_tp_price = None
        strat.state.tp_price = 115.0
        intent = strat._runner_market_exit_intent(90.0, 4_000, _boll(), _cvd(), "test")
        self.assertEqual(intent.tp_price, 115.0)

    def test_tp_price_fallback_to_price(self) -> None:
        strat = _strategy()
        _setup_long_position(strat)
        strat.state.trend_runner_tp_price = None
        strat.state.tp_price = None
        intent = strat._runner_market_exit_intent(90.0, 4_000, _boll(), _cvd(), "test")
        self.assertEqual(intent.tp_price, 90.0)
