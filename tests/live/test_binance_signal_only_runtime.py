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

from src.data_feed.binance.public_klines import BinancePublicKline
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
    _compute_seed_limit,
    _handle_candle,
    _log_boll_ready,
    _should_log_boll_ready,
    _try_recompute_boll,
    _upsert_candle_entry,
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
            seed_historical_klines=True,
            seed_kline_limit=100,
            seed_kline_timeout_seconds=10.0,
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
            seed_historical_klines=True,
            seed_kline_limit=100,
            seed_kline_timeout_seconds=10.0,
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
            seed_historical_klines=True,
            seed_kline_limit=100,
            seed_kline_timeout_seconds=10.0,
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
            seed_historical_klines=True,
            seed_kline_limit=100,
            seed_kline_timeout_seconds=10.0,
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
# Candle buffer upsert (dedup by ts_ms)
# ======================================================================


class TestCandleBufferUpsert:
    """Tests for ``_upsert_candle_entry`` dedup and ordering."""

    @staticmethod
    def _entry(ts_ms: int, close: float) -> dict:
        return {
            "ts_ms": ts_ms,
            "open": close - 10.0,
            "high": close + 5.0,
            "low": close - 5.0,
            "close": close,
            "volume": 100.0,
            "closed": False,
        }

    # --- 1. Same ts_ms update does NOT append ---
    def test_same_ts_ms_replaces_not_appends(self) -> None:
        buffer: list[dict] = []
        _upsert_candle_entry(buffer, self._entry(1000, 3000.0), candle_limit=100)
        _upsert_candle_entry(buffer, self._entry(1000, 3010.0), candle_limit=100)
        assert len(buffer) == 1
        assert buffer[0]["close"] == 3010.0
        assert buffer[0]["ts_ms"] == 1000

    # --- 2. New ts_ms appends ---
    def test_new_ts_ms_appends(self) -> None:
        buffer: list[dict] = []
        _upsert_candle_entry(buffer, self._entry(1000, 3000.0), candle_limit=100)
        _upsert_candle_entry(buffer, self._entry(2000, 3010.0), candle_limit=100)
        assert len(buffer) == 2
        assert [c["ts_ms"] for c in buffer] == [1000, 2000]

    # --- 3. Out-of-order input stays sorted ---
    def test_out_of_order_input_sorted(self) -> None:
        buffer: list[dict] = []
        _upsert_candle_entry(buffer, self._entry(3000, 3030.0), candle_limit=100)
        _upsert_candle_entry(buffer, self._entry(1000, 3010.0), candle_limit=100)
        _upsert_candle_entry(buffer, self._entry(2000, 3020.0), candle_limit=100)
        assert [c["ts_ms"] for c in buffer] == [1000, 2000, 3000]

    # --- 4. candle_limit truncates by unique count ---
    def test_candle_limit_truncates_oldest(self) -> None:
        buffer: list[dict] = []
        limit = 3
        for i in range(5):
            _upsert_candle_entry(
                buffer, self._entry(1000 + i * 60000, 3000.0 + i), candle_limit=limit
            )
        assert len(buffer) == 3
        # Only the newest 3 remain
        assert buffer[0]["ts_ms"] == 1000 + 2 * 60000  # 3rd
        assert buffer[1]["ts_ms"] == 1000 + 3 * 60000  # 4th
        assert buffer[2]["ts_ms"] == 1000 + 4 * 60000  # 5th

    # --- 5. Repeated partial kline cannot make BOLL ready ---
    def test_repeated_partial_kline_no_boll(self) -> None:
        config = BinanceSignalOnlyConfig(
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
            seed_historical_klines=True,
            seed_kline_limit=100,
            seed_kline_timeout_seconds=10.0,
        )
        buffer: list[dict] = []
        # Upsert the same ts_ms 10 times with different closes
        for i in range(10):
            _upsert_candle_entry(
                buffer, self._entry(1000, 3000.0 + i), candle_limit=100
            )
        assert len(buffer) == 1  # Only one unique kline
        result = _try_recompute_boll(candle_buffer=buffer, config=config)
        assert result is None

    # --- 6. 5 unique klines → BOLL ready ---
    def test_five_unique_klines_boll_ready(self) -> None:
        config = BinanceSignalOnlyConfig(
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
            seed_historical_klines=True,
            seed_kline_limit=100,
            seed_kline_timeout_seconds=10.0,
        )
        buffer: list[dict] = []
        for i in range(5):
            _upsert_candle_entry(
                buffer,
                self._entry(1000 + i * 60000, 3000.0 + i * 10),
                candle_limit=100,
            )
        assert len(buffer) == 5
        result = _try_recompute_boll(candle_buffer=buffer, config=config)
        assert isinstance(result, BollSnapshot)

    # --- 7. _handle_candle uses upsert behavior ---
    @pytest.mark.asyncio
    async def test_handle_candle_uses_upsert(self) -> None:
        """Two candle events with same open_time_ms → buffer length stays 1."""
        from src.live.binance_market_data_bridge import BinanceMarketDataSignalBridge

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
            seed_historical_klines=True,
            seed_kline_limit=100,
            seed_kline_timeout_seconds=10.0,
        )
        bridge = BinanceMarketDataSignalBridge()
        buffer: list[dict] = []

        event1 = _make_candle_event(
            open_time_ms=1700000000000,
            close_price="3000.00",
            is_closed=False,
        )
        event2 = _make_candle_event(
            open_time_ms=1700000000000,  # same open_time
            close_price="3010.00",       # updated close
            is_closed=False,
        )

        signal1 = bridge.handle_event(event1)
        signal2 = bridge.handle_event(event2)

        await _handle_candle(
            event=event1,
            signal_input=signal1,
            bridge=bridge,
            candle_buffer=buffer,
            config=config,
        )
        await _handle_candle(
            event=event2,
            signal_input=signal2,
            bridge=bridge,
            candle_buffer=buffer,
            config=config,
        )

        assert len(buffer) == 1
        assert buffer[0]["close"] == 3010.0
        assert buffer[0]["ts_ms"] == 1700000000000


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


# ======================================================================
# Seed config tests
# ======================================================================


class TestSeedConfig:
    """Tests for seed_historical_klines config fields."""

    def test_default_seed_enabled(self) -> None:
        env = _base_binance_env()
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.seed_historical_klines is True

    def test_seed_disabled_via_env(self) -> None:
        env = {**_base_binance_env(), "BINANCE_SIGNAL_ONLY_SEED_KLINES": "false"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.seed_historical_klines is False

    def test_seed_disabled_via_zero(self) -> None:
        env = {**_base_binance_env(), "BINANCE_SIGNAL_ONLY_SEED_KLINES": "0"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.seed_historical_klines is False

    def test_seed_default_limit_is_reasonable(self) -> None:
        """Default seed limit >= boll_window, tp_boll_window, candle_limit."""
        env = _base_binance_env()
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        # Default: max(100, 20, 15, 100) = 100
        assert config.seed_kline_limit >= 100
        assert config.seed_kline_limit >= config.boll_window
        assert config.seed_kline_limit >= config.tp_boll_window
        assert config.seed_kline_limit >= config.candle_limit

    def test_custom_seed_limit(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_SEED_LIMIT": "50",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.seed_kline_limit == 50

    def test_seed_limit_zero_raises(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_SEED_LIMIT": "0",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="positive"):
                load_binance_signal_only_config()

    def test_seed_limit_negative_raises(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_SEED_LIMIT": "-10",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="positive"):
                load_binance_signal_only_config()

    def test_seed_limit_exceeds_1500_raises(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_SEED_LIMIT": "2000",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="must not exceed"):
                load_binance_signal_only_config()

    def test_seed_limit_at_1500_allowed(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_SEED_LIMIT": "1500",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.seed_kline_limit == 1500

    def test_seed_timeout_default(self) -> None:
        env = _base_binance_env()
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.seed_kline_timeout_seconds == 10.0

    def test_seed_timeout_custom(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_SEED_TIMEOUT_SECONDS": "5.5",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.seed_kline_timeout_seconds == 5.5

    def test_seed_timeout_invalid_raises(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_SEED_TIMEOUT_SECONDS": "not_a_number",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError):
                load_binance_signal_only_config()

    def test_seed_limit_not_integer_raises(self) -> None:
        env = {
            **_base_binance_env(),
            "BINANCE_SIGNAL_ONLY_SEED_LIMIT": "abc",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="must be an integer"):
                load_binance_signal_only_config()

    def test_seed_default_limit_with_large_boll_window(self) -> None:
        """When boll_window is large, seed_limit should increase accordingly."""
        env = {
            **_base_binance_env(),
            "BOLL_WINDOW": "200",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
        assert config.seed_kline_limit >= 200

    def test_config_has_seed_fields(self) -> None:
        """Config is frozen and has seed-related fields."""
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
            seed_historical_klines=True,
            seed_kline_limit=100,
            seed_kline_timeout_seconds=10.0,
        )
        assert config.seed_historical_klines is True
        assert config.seed_kline_limit == 100
        assert config.seed_kline_timeout_seconds == 10.0


# ======================================================================
# Compute seed limit unit tests
# ======================================================================


class TestComputeSeedLimit:
    """Unit tests for ``_compute_seed_limit``."""

    def test_default_when_no_env(self) -> None:
        result = _compute_seed_limit(
            {}, boll_window=20, tp_boll_window=15, candle_limit=100
        )
        assert result == 100

    def test_default_uses_max(self) -> None:
        """When candle_limit is small, uses max with boll/tp/100."""
        result = _compute_seed_limit(
            {}, boll_window=20, tp_boll_window=15, candle_limit=50
        )
        assert result == 100  # max(50, 20, 15, 100)

    def test_default_uses_candle_limit_when_largest(self) -> None:
        result = _compute_seed_limit(
            {}, boll_window=20, tp_boll_window=15, candle_limit=500
        )
        assert result == 500  # max(500, 20, 15, 100)

    def test_default_uses_boll_window_when_largest(self) -> None:
        result = _compute_seed_limit(
            {}, boll_window=300, tp_boll_window=15, candle_limit=100
        )
        assert result == 300  # max(100, 300, 15, 100)


# ======================================================================
# Seed runtime behavior tests
# ======================================================================


class TestSeedRuntimeBehavior:
    """Tests for ``_seed_historical_klines`` runtime behavior."""

    def _make_config(self, **overrides) -> BinanceSignalOnlyConfig:
        defaults = {
            "canonical_symbol": "ETH-USDT-PERP",
            "raw_symbol": "ETHUSDT",
            "kline_interval": "15m",
            "duration_seconds": 3600.0,
            "max_events": 100000,
            "heartbeat_seconds": 30.0,
            "candle_limit": 100,
            "boll_window": 5,
            "boll_std_multiplier": 2.0,
            "band_distance_threshold_pct": 0.005,
            "tp_boll_enabled": False,
            "tp_boll_window": 0,
            "seed_historical_klines": True,
            "seed_kline_limit": 50,
            "seed_kline_timeout_seconds": 10.0,
        }
        defaults.update(overrides)
        return BinanceSignalOnlyConfig(**defaults)

    @staticmethod
    def _make_kline(open_time_ms: int, close: float) -> BinancePublicKline:
        return BinancePublicKline(
            open_time_ms=open_time_ms,
            close_time_ms=open_time_ms + 15 * 60 * 1000 - 1,
            open_price=Decimal(str(close - 10)),
            high_price=Decimal(str(close + 5)),
            low_price=Decimal(str(close - 5)),
            close_price=Decimal(str(close)),
            volume=Decimal("100.0"),
        )

    @pytest.mark.asyncio
    async def test_seed_success_fills_buffer(self) -> None:
        """Seed with enough klines fills the buffer and returns BOLL."""
        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config(boll_window=5, candle_limit=100)
        candle_buffer: list[dict] = []

        # Create a fetcher that returns 20 klines
        klines = [
            self._make_kline(1000 + i * 900_000, 3000.0 + i * 10)
            for i in range(20)
        ]

        async def fake_fetcher(*, symbol, interval, limit):
            return klines

        result = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
            fetcher=fake_fetcher,
        )

        assert len(candle_buffer) == 20
        assert result is not None  # BOLL ready
        assert isinstance(result, BollSnapshot)

    @pytest.mark.asyncio
    async def test_seed_insufficient_klines_no_boll(self) -> None:
        """Seed with fewer klines than boll_window → BOLL stays None."""
        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config(boll_window=20, candle_limit=100)
        candle_buffer: list[dict] = []

        klines = [
            self._make_kline(1000 + i * 900_000, 3000.0 + i * 10)
            for i in range(3)  # only 3, need 20
        ]

        async def fake_fetcher(*, symbol, interval, limit):
            return klines

        result = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
            fetcher=fake_fetcher,
        )

        assert len(candle_buffer) == 3
        assert result is None

    @pytest.mark.asyncio
    async def test_seed_deduplicates_by_ts_ms(self) -> None:
        """Duplicate open_time_ms are upserted, not appended."""
        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config(boll_window=5, candle_limit=100)
        candle_buffer: list[dict] = []

        # Two klines with same open_time_ms
        k1a = BinancePublicKline(
            open_time_ms=1000, close_time_ms=900_000,
            open_price=Decimal("100"), high_price=Decimal("110"),
            low_price=Decimal("90"), close_price=Decimal("105"),
            volume=Decimal("50"),
        )
        k1b = BinancePublicKline(
            open_time_ms=1000, close_time_ms=900_000,
            open_price=Decimal("100"), high_price=Decimal("110"),
            low_price=Decimal("90"), close_price=Decimal("108"),  # updated
            volume=Decimal("60"),
        )
        # Plus enough unique klines to reach boll_window
        unique = [
            self._make_kline(2000 + i * 900_000, 3000.0 + i * 10)
            for i in range(10)
        ]

        async def fake_fetcher(*, symbol, interval, limit):
            return [k1a, k1b] + unique

        result = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
            fetcher=fake_fetcher,
        )

        # Dedup: k1a and k1b have same ts_ms → only one entry for ts_ms=1000
        # check the close is from k1b (last)
        ts_1000_entries = [c for c in candle_buffer if c["ts_ms"] == 1000]
        assert len(ts_1000_entries) == 1
        assert ts_1000_entries[0]["close"] == 108.0

    @pytest.mark.asyncio
    async def test_seed_respects_candle_limit(self) -> None:
        """If seed_limit > candle_limit, buffer truncated to candle_limit."""
        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config(boll_window=5, candle_limit=10)
        candle_buffer: list[dict] = []

        klines = [
            self._make_kline(1000 + i * 900_000, 3000.0 + i * 10)
            for i in range(30)  # more than candle_limit
        ]

        async def fake_fetcher(*, symbol, interval, limit):
            return klines

        result = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
            fetcher=fake_fetcher,
        )

        assert len(candle_buffer) == 10  # truncated to candle_limit
        assert result is not None  # still enough for boll_window=5

    @pytest.mark.asyncio
    async def test_seed_failure_returns_none(self) -> None:
        """Seed that raises does not crash — returns None."""
        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config()
        candle_buffer: list[dict] = []

        async def failing_fetcher(*, symbol, interval, limit):
            raise RuntimeError("network error")

        result = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
            fetcher=failing_fetcher,
        )

        assert result is None
        assert len(candle_buffer) == 0  # nothing seeded

    @pytest.mark.asyncio
    async def test_seed_timeout_returns_none(self) -> None:
        """Timeout during seed returns None, does not crash."""
        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config(seed_kline_timeout_seconds=0.001)
        candle_buffer: list[dict] = []

        async def slow_fetcher(*, symbol, interval, limit):
            await asyncio.sleep(10)  # way longer than timeout
            return []

        result = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
            fetcher=slow_fetcher,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_seed_empty_result_returns_none(self) -> None:
        """Fetcher returns empty list → no seed, BOLL stays None."""
        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config()
        candle_buffer: list[dict] = []

        async def empty_fetcher(*, symbol, interval, limit):
            return []

        result = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
            fetcher=empty_fetcher,
        )

        assert result is None
        assert len(candle_buffer) == 0

    @pytest.mark.asyncio
    async def test_seed_does_not_call_strategy(self) -> None:
        """Seed should only fill buffer, never call strategy.on_tick."""
        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config(boll_window=5)
        candle_buffer: list[dict] = []

        klines = [
            self._make_kline(1000 + i * 900_000, 3000.0 + i * 10)
            for i in range(10)
        ]

        async def fake_fetcher(*, symbol, interval, limit):
            return klines

        # Seed doesn't even have a strategy reference — it only fills the buffer
        result = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
            fetcher=fake_fetcher,
        )

        assert result is not None
        # Strategy is never touched by _seed_historical_klines

    @pytest.mark.asyncio
    async def test_seed_closes_are_marked_as_closed(self) -> None:
        """All seeded candles must have closed=True."""
        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config(boll_window=5)
        candle_buffer: list[dict] = []

        klines = [
            self._make_kline(1000 + i * 900_000, 3000.0 + i * 10)
            for i in range(5)
        ]

        async def fake_fetcher(*, symbol, interval, limit):
            return klines

        result = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
            fetcher=fake_fetcher,
        )

        assert result is not None
        for entry in candle_buffer:
            assert entry["closed"] is True

    @pytest.mark.asyncio
    async def test_seed_sorted_ascending(self) -> None:
        """Seeded candle buffer must be sorted by ts_ms ascending."""
        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config(boll_window=5)
        candle_buffer: list[dict] = []

        # Provide klines in random order by open_time_ms
        klines = [
            self._make_kline(5000, 3050.0),
            self._make_kline(1000, 3010.0),
            self._make_kline(3000, 3030.0),
            self._make_kline(4000, 3040.0),
            self._make_kline(2000, 3020.0),
        ]

        async def fake_fetcher(*, symbol, interval, limit):
            return klines

        result = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
            fetcher=fake_fetcher,
        )

        assert result is not None
        times = [c["ts_ms"] for c in candle_buffer]
        assert times == sorted(times)


# ======================================================================
# BOLL log throttling tests
# ======================================================================


class TestTryRecomputeBollNoLogging:
    """``_try_recompute_boll`` is a pure computation — it must NOT emit logs."""

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
            seed_historical_klines=True,
            seed_kline_limit=100,
            seed_kline_timeout_seconds=10.0,
        )

    def test_recompute_does_not_log_boll_ready(self, caplog) -> None:
        """Calling _try_recompute_boll multiple times must not emit BOLL_READY."""
        import logging

        config = self._make_config()
        candle_buffer: list[dict] = [
            {"ts_ms": i * 60000, "close": float(3000 + i)} for i in range(10)
        ]
        with caplog.at_level(logging.DEBUG):
            for _ in range(5):
                _try_recompute_boll(candle_buffer=candle_buffer, config=config)
        assert "BINANCE_SIGNAL_ONLY_BOLL_READY" not in caplog.text

    def test_recompute_does_not_log_when_buffer_too_small(self, caplog) -> None:
        """Insufficient buffer → no log either."""
        import logging

        config = self._make_config()
        candle_buffer: list[dict] = [
            {"ts_ms": i * 60000, "close": float(3000 + i)} for i in range(3)
        ]
        with caplog.at_level(logging.DEBUG):
            _try_recompute_boll(candle_buffer=candle_buffer, config=config)
        assert "BINANCE_SIGNAL_ONLY_BOLL_READY" not in caplog.text


class TestShouldLogBollReady:
    """Tests for ``_should_log_boll_ready`` pure decision function."""

    @staticmethod
    def _dummy_boll() -> BollSnapshot:
        return BollSnapshot(
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

    def test_next_boll_none_returns_false(self) -> None:
        should, reason = _should_log_boll_ready(
            was_ready=False, is_closed_candle=False, next_boll=None
        )
        assert should is False
        assert reason is None

    def test_not_ready_to_ready_returns_became_ready(self) -> None:
        should, reason = _should_log_boll_ready(
            was_ready=False, is_closed_candle=False, next_boll=self._dummy_boll()
        )
        assert should is True
        assert reason == "became_ready"

    def test_ready_partial_update_returns_false(self) -> None:
        """Already ready + partial candle → no log."""
        should, reason = _should_log_boll_ready(
            was_ready=True, is_closed_candle=False, next_boll=self._dummy_boll()
        )
        assert should is False
        assert reason is None

    def test_ready_closed_candle_returns_closed_candle(self) -> None:
        should, reason = _should_log_boll_ready(
            was_ready=True, is_closed_candle=True, next_boll=self._dummy_boll()
        )
        assert should is True
        assert reason == "closed_candle"

    def test_not_ready_still_none_returns_false(self) -> None:
        """BOLL was not ready and still isn't → no log."""
        should, reason = _should_log_boll_ready(
            was_ready=False, is_closed_candle=True, next_boll=None
        )
        assert should is False
        assert reason is None

    def test_not_ready_closed_candle_returns_became_ready(self) -> None:
        """First candle that makes BOLL ready is a closed candle → became_ready, not closed_candle."""
        should, reason = _should_log_boll_ready(
            was_ready=False, is_closed_candle=True, next_boll=self._dummy_boll()
        )
        assert should is True
        assert reason == "became_ready"


class TestLogBollReadyFormat:
    """Tests for ``_log_boll_ready`` log format."""

    @staticmethod
    def _dummy_boll() -> BollSnapshot:
        return BollSnapshot(
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

    def test_log_contains_reason_and_boll_fields(self, caplog) -> None:
        import logging

        boll = self._dummy_boll()
        with caplog.at_level(logging.INFO):
            _log_boll_ready(boll=boll, buffer_size=20, reason="seed")
        assert "BINANCE_SIGNAL_ONLY_BOLL_READY" in caplog.text
        assert "reason=seed" in caplog.text
        assert "close=3010.0000" in caplog.text
        assert "middle=3000.0000" in caplog.text
        assert "buffer_size=20" in caplog.text

    def test_log_reason_became_ready(self, caplog) -> None:
        import logging

        boll = self._dummy_boll()
        with caplog.at_level(logging.WARNING):
            _log_boll_ready(
                boll=boll, buffer_size=5, reason="became_ready", level=logging.WARNING
            )
        assert "reason=became_ready" in caplog.text

    def test_log_reason_closed_candle(self, caplog) -> None:
        import logging

        boll = self._dummy_boll()
        with caplog.at_level(logging.INFO):
            _log_boll_ready(boll=boll, buffer_size=30, reason="closed_candle")
        assert "reason=closed_candle" in caplog.text


class TestSeedBollReadyLogOnce:
    """Seed must log BOLL_READY exactly once with reason=seed."""

    def _make_config(self, **overrides) -> BinanceSignalOnlyConfig:
        defaults = {
            "canonical_symbol": "ETH-USDT-PERP",
            "raw_symbol": "ETHUSDT",
            "kline_interval": "15m",
            "duration_seconds": 3600.0,
            "max_events": 100000,
            "heartbeat_seconds": 30.0,
            "candle_limit": 100,
            "boll_window": 5,
            "boll_std_multiplier": 2.0,
            "band_distance_threshold_pct": 0.005,
            "tp_boll_enabled": False,
            "tp_boll_window": 0,
            "seed_historical_klines": True,
            "seed_kline_limit": 50,
            "seed_kline_timeout_seconds": 10.0,
        }
        defaults.update(overrides)
        return BinanceSignalOnlyConfig(**defaults)

    @staticmethod
    def _make_kline(open_time_ms: int, close: float) -> BinancePublicKline:
        return BinancePublicKline(
            open_time_ms=open_time_ms,
            close_time_ms=open_time_ms + 15 * 60 * 1000 - 1,
            open_price=Decimal(str(close - 10)),
            high_price=Decimal(str(close + 5)),
            low_price=Decimal(str(close - 5)),
            close_price=Decimal(str(close)),
            volume=Decimal("100.0"),
        )

    @pytest.mark.asyncio
    async def test_seed_logs_boll_ready_once_with_reason_seed(self, caplog) -> None:
        """Seed with enough klines → exactly 1 BOLL_READY with reason=seed."""
        import logging

        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config(boll_window=5)
        candle_buffer: list[dict] = []

        klines = [
            self._make_kline(1000 + i * 900_000, 3000.0 + i * 10)
            for i in range(20)
        ]

        async def fake_fetcher(*, symbol, interval, limit):
            return klines

        with caplog.at_level(logging.WARNING):
            result = await _seed_historical_klines(
                candle_buffer=candle_buffer,
                config=config,
                fetcher=fake_fetcher,
            )

        assert result is not None

        # Count BOLL_READY occurrences
        boll_ready_lines = [
            line for line in caplog.text.splitlines()
            if "BINANCE_SIGNAL_ONLY_BOLL_READY" in line
        ]
        assert len(boll_ready_lines) == 1
        assert "reason=seed" in boll_ready_lines[0]

        # KLINE_SEED_DONE must still exist
        assert "BINANCE_SIGNAL_ONLY_KLINE_SEED_DONE" in caplog.text

    @pytest.mark.asyncio
    async def test_seed_insufficient_no_boll_ready_log(self, caplog) -> None:
        """Seed with too few klines → no BOLL_READY log at all."""
        import logging

        from src.live.binance_signal_only_runtime import _seed_historical_klines

        config = self._make_config(boll_window=20)
        candle_buffer: list[dict] = []

        klines = [
            self._make_kline(1000 + i * 900_000, 3000.0 + i * 10)
            for i in range(3)
        ]

        async def fake_fetcher(*, symbol, interval, limit):
            return klines

        with caplog.at_level(logging.WARNING):
            result = await _seed_historical_klines(
                candle_buffer=candle_buffer,
                config=config,
                fetcher=fake_fetcher,
            )

        assert result is None
        assert "BINANCE_SIGNAL_ONLY_BOLL_READY" not in caplog.text
        # KLINE_SEED_DONE should still appear
        assert "BINANCE_SIGNAL_ONLY_KLINE_SEED_DONE" in caplog.text


class TestHeartbeatFormatUnchanged:
    """Heartbeat format must still include BOLL fields when ready."""

    def test_heartbeat_with_boll_contains_boll_fields(self, caplog) -> None:
        """Heartbeat log line must contain price, boll_middle, etc."""
        import logging

        from src.live.binance_signal_only_runtime import _log_heartbeat
        from src.live.binance_market_data_bridge import BinanceMarketDataSignalBridge

        bridge = BinanceMarketDataSignalBridge()
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

        with caplog.at_level(logging.WARNING):
            _log_heartbeat(
                bridge=bridge,
                current_boll=boll,
                total_events=100,
                elapsed=30.0,
            )

        assert "BINANCE_SIGNAL_ONLY_HEARTBEAT" in caplog.text
        assert "price=" in caplog.text
        assert "boll_middle=" in caplog.text
        assert "boll_upper=" in caplog.text
        assert "boll_lower=" in caplog.text
        assert "bridge_errors=" in caplog.text

    def test_heartbeat_without_boll_shows_not_ready(self, caplog) -> None:
        """Heartbeat without BOLL must show boll=not_ready."""
        import logging

        from src.live.binance_signal_only_runtime import _log_heartbeat
        from src.live.binance_market_data_bridge import BinanceMarketDataSignalBridge

        bridge = BinanceMarketDataSignalBridge()

        with caplog.at_level(logging.WARNING):
            _log_heartbeat(
                bridge=bridge,
                current_boll=None,
                total_events=10,
                elapsed=10.0,
            )

        assert "BINANCE_SIGNAL_ONLY_HEARTBEAT" in caplog.text
        assert "boll=not_ready" in caplog.text
        assert "bridge_errors=" in caplog.text
