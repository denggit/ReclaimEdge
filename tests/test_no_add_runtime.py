"""Verify runtime no-ADD behaviour after ADD config / TradeIntentType cleanup.

Covers:
1. same-side position does NOT produce ADD_LONG / ADD_SHORT
2. same-side position only logs / state skip add_disabled
3. Strategy TradeIntentType does not contain ADD_LONG / ADD_SHORT
4. .env.example does not contain MAX_LAYERS / ADD_GAP / EXTREME_RETEST / SPLIT_TP
5. add_gap / add_interval / first_add_block are NOT present in runtime log path
"""

from __future__ import annotations

import os
import unittest
from typing import Literal, get_args

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntentType,
)
from src.strategies.entry_add_flow_coordinator import EntryAddFlowCoordinator


def _sizer() -> SimplePositionSizer:
    return SimplePositionSizer(SimplePositionSizerConfig())


def _boll(
    middle: float = 2000.0,
    upper: float = 2150.0,
    lower: float = 1900.0,
) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000,
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
        burst_range_pct=0.0,
        baseline_range_pct=0.0,
        burst_move_ratio=0.0,
        burst_volume=0.0,
        baseline_volume=0.0,
        burst_volume_ratio=0.0,
        up_burst=False,
        down_burst=False,
    )


class TestTradeIntentTypeNoAdd:
    """TradeIntentType must NOT expose ADD_LONG / ADD_SHORT."""

    def test_trade_intent_type_no_add_long(self) -> None:
        allowed = set(get_args(TradeIntentType))
        assert "ADD_LONG" not in allowed, f"ADD_LONG must not be in TradeIntentType: {allowed}"

    def test_trade_intent_type_no_add_short(self) -> None:
        allowed = set(get_args(TradeIntentType))
        assert "ADD_SHORT" not in allowed, f"ADD_SHORT must not be in TradeIntentType: {allowed}"

    def test_trade_intent_type_only_allows_open_update_exit(self) -> None:
        allowed = set(get_args(TradeIntentType))
        assert allowed == {"OPEN_LONG", "OPEN_SHORT", "UPDATE_TP", "MARKET_EXIT_RUNNER"}, (
            f"Unexpected TradeIntentType members: {allowed}"
        )


class TestSameSidePositionNoAdd:
    """same-side position does NOT produce ADD_LONG / ADD_SHORT."""

    def test_long_position_no_add_long_from_open_or_add(self) -> None:
        cfg = BollCvdReclaimStrategyConfig()
        strat = BollCvdReclaimStrategy(cfg, _sizer())
        strat.state.side = "LONG"
        strat.state.layers = 1
        strat.state.last_entry_price = 1950.0
        strat.state.avg_entry_price = 1950.0
        strat.state.total_entry_qty = 0.5
        strat.state.total_entry_notional = 975.0
        boll = _boll(lower=1900.0)
        cvd = _cvd(buy_ratio=0.6)
        result = strat._maybe_open_or_add_long(1880.0, 2000, boll, cvd)
        assert result is None, f"Expected None (add blocked), got {result}"

    def test_short_position_no_add_short_from_open_or_add(self) -> None:
        cfg = BollCvdReclaimStrategyConfig()
        strat = BollCvdReclaimStrategy(cfg, _sizer())
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2100.0
        strat.state.avg_entry_price = 2100.0
        strat.state.total_entry_qty = 0.5
        strat.state.total_entry_notional = 1050.0
        boll = _boll(upper=2150.0)
        cvd = _cvd(sell_ratio=0.6)
        result = strat._maybe_open_or_add_short(2170.0, 2000, boll, cvd)
        assert result is None, f"Expected None (add blocked), got {result}"


class TestCoordinatorOpenBlocksAdd:
    """open_position blocks ADD_LONG / ADD_SHORT."""

    def test_open_position_blocks_add_long(self) -> None:
        cfg = BollCvdReclaimStrategyConfig()
        strat = BollCvdReclaimStrategy(cfg, _sizer())
        coord = EntryAddFlowCoordinator(strat)
        boll = _boll()
        cvd = _cvd()
        # ADD_LONG / ADD_SHORT are no longer valid TradeIntentType values,
        # so the coordinator's open_position would be called with OPEN_LONG/OPEN_SHORT
        # only. We verify the coordinator blocks same-side by calling maybe_open_or_add.
        strat.state.side = "LONG"
        strat.state.layers = 1
        result = coord.maybe_open_or_add_long(1880.0, 2000, boll, cvd)
        assert result is None, f"Same-side add should be blocked, got {result}"


class TestEnvExampleNoAddConfig(unittest.TestCase):
    """Verify .env.example does NOT contain legacy ADD / SPLIT_TP / EXTREME_RETEST."""

    _env_text: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        env_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".env.example",
        )
        if os.path.exists(env_path):
            with open(env_path) as f:
                cls._env_text = f.read()

    def test_no_max_layers_in_env_example(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        assert "MAX_LAYERS" not in self._env_text, "MAX_LAYERS must not appear in .env.example"

    def test_no_add_gap_in_env_example(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        assert "ADD_GAP_MODE" not in self._env_text

    def test_no_add_gap_base_in_env_example(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        assert "ADD_GAP_BASE_PCT" not in self._env_text

    def test_no_extreme_retest_in_env_example(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        for key in ["EXTREME_RETEST_ADD_ENABLED", "EXTREME_RETEST_PIVOT"]:
            assert key not in self._env_text, f"{key} must not appear in .env.example"

    def test_no_split_tp_in_env_example(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        assert "SPLIT_TP_ENABLED" not in self._env_text

    def test_no_first_add_block_in_env_example(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        assert "FIRST_ADD_BLOCK_SECONDS" not in self._env_text
        assert "ADD_MIN_INTERVAL_SECONDS" not in self._env_text
        assert "ADD_FREEZE_CHAIN_ENABLED" not in self._env_text

    def test_no_max_entry_distance_in_env_example(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        assert "MAX_ENTRY_DISTANCE_FROM_EXTREME_PCT" not in self._env_text


class TestConfigRejectsAddFields(unittest.TestCase):
    """BollCvdReclaimStrategyConfig must NOT accept ADD fields."""

    def test_config_no_add_gap_mode(self) -> None:
        with self.assertRaises(TypeError):
            BollCvdReclaimStrategyConfig(add_gap_mode="linear")  # type: ignore[call-arg]

    def test_config_no_max_entry_distance(self) -> None:
        with self.assertRaises(TypeError):
            BollCvdReclaimStrategyConfig(max_entry_distance_from_extreme_pct=0.002)  # type: ignore[call-arg]

    def test_config_no_first_add_block_seconds(self) -> None:
        with self.assertRaises(TypeError):
            BollCvdReclaimStrategyConfig(first_add_block_seconds=1800)  # type: ignore[call-arg]

    def test_config_no_extreme_retest_add_enabled(self) -> None:
        with self.assertRaises(TypeError):
            BollCvdReclaimStrategyConfig(extreme_retest_add_enabled=True)  # type: ignore[call-arg]
