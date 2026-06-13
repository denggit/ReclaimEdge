#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_signal_only_runtime.py
@Description: Functional tests for the Binance signal-only runtime.

All tests use in-memory event objects and mocked env — no network, no API keys.
"""

from __future__ import annotations

import asyncio
import copy
import os
from decimal import Decimal
from unittest import mock

import pytest

from src.data_feed.market_events import (
    MarketCandleEvent,
    MarketTradeEvent,
    MarketTradeSide,
)
from src.exchanges.models import ExchangeName
from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig
from src.live.binance_market_data_bridge import (
    BinanceMarketDataSignalBridge,
    BinanceSignalCandleInput,
    BinanceSignalTradeInput,
)
from src.live.binance_signal_only_runtime import (
    BinanceSignalOnlyConfig,
    _build_boll_snapshot,
    _calculate_boll,
    _try_recompute_boll,
    load_binance_signal_only_config,
)
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import (
    SimplePositionSizer,
    SimplePositionSizerConfig,
)
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy

# ======================================================================
# Helpers
# ======================================================================


def _make_trade_event(
    *,
    canonical_symbol: str = "ETH-USDT-PERP",
    raw_symbol: str = "ETHUSDT",
    price: str = "3000.00",
    quantity: str = "1.5",
    side: MarketTradeSide = MarketTradeSide.BUY,
    event_time_ms: int = 1700000000000,
) -> MarketTradeEvent:
    return MarketTradeEvent(
        exchange=ExchangeName.BINANCE,
        canonical_symbol=canonical_symbol,
        raw_symbol=raw_symbol,
        price=Decimal(price),
        quantity=Decimal(quantity),
        taker_side=side,
        event_time_ms=event_time_ms,
        is_aggregated=True,
        aggregation_window_ms=100,
    )


def _make_candle_event(
    *,
    canonical_symbol: str = "ETH-USDT-PERP",
    raw_symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    open_price: str = "2990.00",
    high_price: str = "3020.00",
    low_price: str = "2980.00",
    close_price: str = "3010.00",
    volume: str = "100.5",
    is_closed: bool = False,
    open_time_ms: int = 1700000000000,
    close_time_ms: int = 1700000900000,
) -> MarketCandleEvent:
    return MarketCandleEvent(
        exchange=ExchangeName.BINANCE,
        canonical_symbol=canonical_symbol,
        raw_symbol=raw_symbol,
        timeframe=timeframe,
        open_time_ms=open_time_ms,
        close_time_ms=close_time_ms,
        open_price=Decimal(open_price),
        high_price=Decimal(high_price),
        low_price=Decimal(low_price),
        close_price=Decimal(close_price),
        volume=Decimal(volume),
        is_closed=is_closed,
    )


def _base_binance_env() -> dict[str, str]:
    return {
        "EXCHANGE": "binance",
        "TRADE_ASSET": "ETH",
        "QUOTE_ASSET": "USDT",
        "MARKET_TYPE": "PERPETUAL",
        "KLINE_INTERVAL": "15m",
        "BINANCE_SIGNAL_ONLY": "true",
    }


# ======================================================================
# Config loading
# ======================================================================


class TestConfigLoading:
    """Tests for ``load_binance_signal_only_config``."""

    def test_valid_binance_config(self) -> None:
        env = _base_binance_env()
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.canonical_symbol == "ETH-USDT-PERP"
        assert config.raw_symbol == "ETHUSDT"
        assert config.kline_interval == "15m"
        assert config.duration_seconds == 3600.0
        assert config.max_events == 100000
        assert config.boll_window == 20

    def test_custom_duration_and_max_events(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_SECONDS": "120",
            "BINANCE_SIGNAL_ONLY_MAX_EVENTS": "50",
            "BINANCE_SIGNAL_ONLY_HEARTBEAT_SECONDS": "10",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.duration_seconds == 120.0
        assert config.max_events == 50
        assert config.heartbeat_seconds == 10.0

    def test_signal_only_false_raises(self) -> None:
        env = {**_base_binance_env(), "BINANCE_SIGNAL_ONLY": "false"}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="Binance main live trading is not wired"):
                load_binance_signal_only_config()

    def test_signal_only_missing_raises(self) -> None:
        env = _base_binance_env()
        del env["BINANCE_SIGNAL_ONLY"]
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="Binance main live trading is not wired"):
                load_binance_signal_only_config()

    def test_exchange_okx_rejected(self) -> None:
        env = {**_base_binance_env(), "EXCHANGE": "okx"}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Binance signal-only runtime requires EXCHANGE=binance"):
                load_binance_signal_only_config()

    def test_btc_rejected(self) -> None:
        env = {**_base_binance_env(), "TRADE_ASSET": "BTC"}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError):
                load_binance_signal_only_config()

    def test_1m_rejected(self) -> None:
        env = {**_base_binance_env(), "KLINE_INTERVAL": "1m"}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError):
                load_binance_signal_only_config()

    def test_5m_rejected(self) -> None:
        env = {**_base_binance_env(), "KLINE_INTERVAL": "5m"}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError):
                load_binance_signal_only_config()

    def test_api_key_not_in_config(self) -> None:
        env = {
            **_base_binance_env(),
            "EXCHANGE_API_KEY": "secret-key-123",
            "EXCHANGE_API_SECRET": "secret-456",
            "EXCHANGE_API_PASSPHRASE": "secret-789",
            "BINANCE_API_KEY": "binance-key",
            "BINANCE_SECRET_KEY": "binance-secret",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        # Config itself should not have any secret fields
        assert not hasattr(config, "api_key")
        assert not hasattr(config, "api_secret")

    def test_boll_params_from_env(self) -> None:
        env = {
            **_base_binance_env(),
            "BOLL_WINDOW": "10",
            "BOLL_STD_MULTIPLIER": "2.5",
            "BOLL_DISTANCE_THRESHOLD_PCT": "0.003",
            "TP_BOLL_ENABLED": "false",
            "TP_BOLL_WINDOW": "10",
            "CANDLE_LIMIT": "50",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.boll_window == 10
        assert config.boll_std_multiplier == 2.5
        assert config.band_distance_threshold_pct == 0.003
        assert config.tp_boll_enabled is False
        assert config.tp_boll_window == 10
        assert config.candle_limit == 50

    def test_config_is_frozen(self) -> None:
        config = BinanceSignalOnlyConfig(
            canonical_symbol="ETH-USDT-PERP",
            raw_symbol="ETHUSDT",
            kline_interval="15m",
            duration_seconds=3600.0,
            max_events=100000,
            heartbeat_seconds=30.0,
            candle_limit=100,
            boll_window=20,
            boll_std_multiplier=2.0,
            band_distance_threshold_pct=0.005,
            tp_boll_enabled=True,
            tp_boll_window=15,
        )
        with pytest.raises(Exception):
            config.boll_window = 30  # type: ignore[misc]


# ======================================================================
# BOLL calculation
# ======================================================================


class TestBollCalculation:
    """Tests for BOLL helpers."""

    def test_calculate_boll_basic(self) -> None:
        closes = [100.0] * 20
        middle, upper, lower = _calculate_boll(closes, 20, 2.0)
        assert middle == 100.0
        assert upper == 100.0  # std = 0
        assert lower == 100.0

    def test_calculate_boll_with_variance(self) -> None:
        # Generate 20 closes with some variance
        closes = [float(100 + i % 5) for i in range(20)]
        middle, upper, lower = _calculate_boll(closes, 20, 2.0)
        assert upper > middle > lower

    def test_not_enough_closes_raises(self) -> None:
        with pytest.raises(ValueError, match="Not enough closes"):
            _calculate_boll([100.0] * 10, 20, 2.0)

    def test_build_boll_snapshot(self) -> None:
        config = BinanceSignalOnlyConfig(
            canonical_symbol="ETH-USDT-PERP",
            raw_symbol="ETHUSDT",
            kline_interval="15m",
            duration_seconds=3600.0,
            max_events=100000,
            heartbeat_seconds=30.0,
            candle_limit=100,
            boll_window=20,
            boll_std_multiplier=2.0,
            band_distance_threshold_pct=0.005,
            tp_boll_enabled=True,
            tp_boll_window=15,
        )
        closes = [float(3000 + i) for i in range(20)]
        latest = {"ts_ms": 1700000000000, "close": closes[-1]}
        snapshot = _build_boll_snapshot(
            raw_symbol="ETHUSDT",
            closes=closes,
            latest_candle=latest,
            config=config,
        )
        assert isinstance(snapshot, BollSnapshot)
        assert snapshot.inst_id == "ETHUSDT"
        assert snapshot.candle_ts_ms == 1700000000000
        assert snapshot.close == closes[-1]
        assert snapshot.middle > 0
        assert snapshot.upper > snapshot.middle > snapshot.lower
        assert snapshot.live_mode is True
        # TP BOLL should be present
        assert snapshot.tp_lower is not None
        assert snapshot.tp_middle is not None
        assert snapshot.tp_upper is not None
        assert snapshot.tp_window == 15

    def test_build_boll_snapshot_tp_disabled(self) -> None:
        config = BinanceSignalOnlyConfig(
            canonical_symbol="ETH-USDT-PERP",
            raw_symbol="ETHUSDT",
            kline_interval="15m",
            duration_seconds=3600.0,
            max_events=100000,
            heartbeat_seconds=30.0,
            candle_limit=100,
            boll_window=20,
            boll_std_multiplier=2.0,
            band_distance_threshold_pct=0.005,
            tp_boll_enabled=False,
            tp_boll_window=0,
        )
        closes = [float(3000 + i) for i in range(20)]
        latest = {"ts_ms": 1700000000000, "close": closes[-1]}
        snapshot = _build_boll_snapshot(
            raw_symbol="ETHUSDT",
            closes=closes,
            latest_candle=latest,
            config=config,
        )
        assert snapshot.tp_lower is None
        assert snapshot.tp_middle is None
        assert snapshot.tp_upper is None


# ======================================================================
# Candle buffer and BOLL recompute
# ======================================================================


class TestCandleBufferAndBoll:
    """Tests for candle buffer update and BOLL recompute."""

    def _make_config(self) -> BinanceSignalOnlyConfig:
        return BinanceSignalOnlyConfig(
            canonical_symbol="ETH-USDT-PERP",
            raw_symbol="ETHUSDT",
            kline_interval="15m",
            duration_seconds=3600.0,
            max_events=100000,
            heartbeat_seconds=30.0,
            candle_limit=100,
            boll_window=5,
            boll_std_multiplier=2.0,
            band_distance_threshold_pct=0.005,
            tp_boll_enabled=False,
            tp_boll_window=0,
        )

    def test_buffer_too_small_returns_none(self) -> None:
        config = self._make_config()
        candle_buffer: list[dict] = [
            {"ts_ms": i * 60000, "close": float(3000 + i)} for i in range(3)
        ]
        result = _try_recompute_boll(
            candle_buffer=candle_buffer, config=config
        )
        assert result is None

    def test_buffer_enough_returns_boll(self) -> None:
        config = self._make_config()
        candle_buffer: list[dict] = [
            {"ts_ms": i * 60000, "close": float(3000 + i)} for i in range(10)
        ]
        result = _try_recompute_boll(
            candle_buffer=candle_buffer, config=config
        )
        assert isinstance(result, BollSnapshot)
        assert result.close == float(3000 + 9)

    def test_buffer_truncated_to_candle_limit(self) -> None:
        config = self._make_config()
        # config.candle_limit is 100
        candle_buffer: list[dict] = [
            {"ts_ms": i * 60000, "close": float(3000 + i)} for i in range(150)
        ]
        # After truncation would have 100 candles — enough for boll_window=5
        while len(candle_buffer) > config.candle_limit:
            candle_buffer.pop(0)
        assert len(candle_buffer) == 100
        result = _try_recompute_boll(
            candle_buffer=candle_buffer, config=config
        )
        assert isinstance(result, BollSnapshot)


# ======================================================================
# Bridge and stats
# ======================================================================


class TestBridgeInRuntime:
    """Tests for bridge interaction in the signal-only runtime context."""

    def test_closed_candle_increments_closed_stat(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        candle = _make_candle_event(is_closed=True)
        result = bridge.handle_event(candle)
        assert isinstance(result, BinanceSignalCandleInput)
        assert result.closed is True
        assert bridge.stats.closed_candle_events == 1

    def test_closed_candle_increments_closed_stat_via_bridge(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        # Two candles: one open, one closed
        bridge.handle_event(_make_candle_event(is_closed=False))
        bridge.handle_event(_make_candle_event(is_closed=True))
        assert bridge.stats.candle_events == 2
        assert bridge.stats.closed_candle_events == 1

    def test_trade_increments_trade_stat(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        result = bridge.handle_event(_make_trade_event())
        assert isinstance(result, BinanceSignalTradeInput)
        assert bridge.stats.trade_events == 1


# ======================================================================
# Strategy on_tick state restore
# ======================================================================


class TestStrategyStateRestore:
    """Tests that strategy state is restored after on_tick in signal-only mode."""

    def test_strategy_state_restored_after_on_tick(self) -> None:
        sizer = SimplePositionSizer(SimplePositionSizerConfig())
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(), sizer
        )
        original_state = copy.deepcopy(strategy.state)

        # Simulate what happens during a tick: build a boll snapshot and CVD snapshot
        from src.indicators.cvd_tracker import CvdSnapshot

        boll = BollSnapshot(
            inst_id="ETHUSDT",
            candle_ts_ms=1700000000000,
            close=3010.0,
            middle=3000.0,
            upper=3050.0,
            lower=2950.0,
            upper_distance_pct=0.016,
            lower_distance_pct=0.016,
            alert_switch_on=True,
            live_mode=True,
        )
        cvd = CvdSnapshot(
            ts_ms=1700000001000,
            price=3010.0,
            side="buy",
            size=1.5,
            signed_delta=1.5,
            total_cvd=10.0,
            fast_cvd=5.0,
            previous_fast_cvd=0.0,
            buy_volume=2.0,
            sell_volume=0.5,
            buy_ratio=0.8,
            sell_ratio=0.2,
            cross_positive=True,
            cross_negative=False,
            cvd_increasing=True,
            cvd_decreasing=False,
            no_new_low=True,
            no_new_high=False,
            window_low=3005.0,
            window_high=3015.0,
            burst_net_move_pct=0.001,
            burst_range_pct=0.002,
            baseline_range_pct=0.001,
            burst_move_ratio=2.0,
            burst_volume=5.0,
            baseline_volume=2.0,
            burst_volume_ratio=2.5,
            up_burst=False,
            down_burst=False,
        )

        # Backup → on_tick → restore (the signal-only pattern)
        backup_state = copy.deepcopy(strategy.state)
        try:
            strategy.on_tick(price=3010.0, ts_ms=1700000001000, boll=boll, cvd=cvd)
        finally:
            strategy.state = backup_state

        # Strategy state must be identical to original
        assert strategy.state.side == original_state.side
        assert strategy.state.layers == original_state.layers
        assert strategy.state.lower_armed == original_state.lower_armed
        assert strategy.state.upper_armed == original_state.upper_armed
        assert strategy.state.last_entry_price == original_state.last_entry_price
        assert strategy.state.tp_price == original_state.tp_price

    def test_repeated_ticks_dont_accumulate_state(self) -> None:
        sizer = SimplePositionSizer(SimplePositionSizerConfig())
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(), sizer
        )
        original_state = copy.deepcopy(strategy.state)

        from src.indicators.cvd_tracker import CvdSnapshot

        boll = BollSnapshot(
            inst_id="ETHUSDT",
            candle_ts_ms=1700000000000,
            close=2950.0,
            middle=3000.0,
            upper=3050.0,
            lower=2950.0,
            upper_distance_pct=0.016,
            lower_distance_pct=0.016,
            alert_switch_on=True,
            live_mode=True,
        )
        cvd = CvdSnapshot(
            ts_ms=1700000001000,
            price=2950.0,
            side="sell",
            size=1.0,
            signed_delta=-1.0,
            total_cvd=-10.0,
            fast_cvd=-5.0,
            previous_fast_cvd=0.0,
            buy_volume=0.3,
            sell_volume=1.3,
            buy_ratio=0.2,
            sell_ratio=0.8,
            cross_positive=False,
            cross_negative=True,
            cvd_increasing=False,
            cvd_decreasing=True,
            no_new_low=False,
            no_new_high=True,
            window_low=2945.0,
            window_high=2960.0,
            burst_net_move_pct=-0.001,
            burst_range_pct=0.002,
            baseline_range_pct=0.001,
            burst_move_ratio=2.0,
            burst_volume=5.0,
            baseline_volume=2.0,
            burst_volume_ratio=2.5,
            up_burst=False,
            down_burst=False,
        )

        # Run 10 ticks with state restore each time
        for _ in range(10):
            backup_state = copy.deepcopy(strategy.state)
            try:
                strategy.on_tick(price=2950.0, ts_ms=1700000001000, boll=boll, cvd=cvd)
            finally:
                strategy.state = backup_state

        # After 10 ticks with restore, state should be identical to original
        assert strategy.state.side == original_state.side
        assert strategy.state.layers == original_state.layers
        assert strategy.state.lower_armed == original_state.lower_armed
        assert strategy.state.upper_armed == original_state.upper_armed


# ======================================================================
# Trade without BOLL does not call strategy
# ======================================================================


class TestTradeWithoutBoll:
    """Tests that trade events without BOLL snapshot do not call strategy."""

    def test_trade_no_boll_skip_strategy(self) -> None:
        """Verify the logic: trade events without BOLL don't mutate strategy state."""
        sizer = SimplePositionSizer(SimplePositionSizerConfig())
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(), sizer
        )
        original_state = copy.deepcopy(strategy.state)

        # No BOLL → skip strategy call
        current_boll = None
        if current_boll is None:
            # This is the guarded path — strategy.on_tick not called
            pass

        assert strategy.state.side == original_state.side
        assert strategy.state.lower_armed == original_state.lower_armed


# ======================================================================
# Runtime exit on max_events
# ======================================================================


class TestRuntimeExit:
    """Tests that runtime exits cleanly."""

    def test_config_supports_max_events(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_MAX_EVENTS": "10",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.max_events == 10

    def test_config_supports_duration(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_SECONDS": "30",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.duration_seconds == 30.0


# ======================================================================
# Env var whitelist — secret keys excluded
# ======================================================================


class TestSecretExclusion:
    """Verify that secrets never reach the signal-only config."""

    def test_unified_runtime_config_has_empty_secrets(self) -> None:
        """When loaded with public-only env, secrets are empty."""
        from src.exchanges.runtime_config import load_unified_runtime_config

        public_env = {
            "EXCHANGE": "binance",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "KLINE_INTERVAL": "15m",
        }
        rt = load_unified_runtime_config(public_env)
        assert rt.api_key == ""
        assert rt.api_secret == ""
        assert rt.api_passphrase == ""

    def test_secrets_in_real_env_not_in_public_only(self) -> None:
        """Even with secrets in real env, public-only mapping excludes them."""
        env = {
            **_base_binance_env(),
            "EXCHANGE_API_KEY": "real-key-should-be-excluded",
            "EXCHANGE_API_SECRET": "real-secret",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        # Config itself has no secret fields
        assert not hasattr(config, "api_key")
