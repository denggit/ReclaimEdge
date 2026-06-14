#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_okx_market_data_client.py
@Description: Functional tests for OkxMarketDataClient using FakeMonitor / FakeClient.

No real API calls.  No env reads.  No production wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pytest

from src.data_feed.okx_market_data_client import OkxMarketDataClient
from src.data_feed.market_data_client_port import (
    CandleSnapshot,
    MarketTradeSnapshot,
)


# ======================================================================
# Fake objects
# ======================================================================


@dataclass(frozen=True)
class FakeConfig:
    inst_id: str = "ETH-USDT-SWAP"
    bar: str = "15m"
    use_live_candle: bool = True


@dataclass(frozen=True)
class FakeCandle:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirmed: bool


@dataclass(frozen=True)
class FakeTick:
    inst_id: str
    price: float
    size: float
    side: str
    ts_ms: int


@dataclass(frozen=True)
class FakeMarketTickEvent:
    tick: FakeTick
    boll: object | None = None


class FakeClient:
    """Fake OKX REST client that returns canned candles."""

    def __init__(self, candles: list[FakeCandle] | None = None) -> None:
        self.candles: list[FakeCandle] = candles or []
        self.closed: bool = False
        self.last_include_live: bool | None = None

    async def fetch_candles(self, include_live: bool) -> list[FakeCandle]:
        self.last_include_live = include_live
        return list(self.candles)

    async def close(self) -> None:
        self.closed = True


class FakeMonitor:
    """Fake BollBandBreakoutMonitor that delegates to FakeClient."""

    def __init__(
        self,
        candles: list[FakeCandle] | None = None,
        *,
        bar_interval_ms: int = 15 * 60 * 1000,
        use_live_candle: bool = True,
    ) -> None:
        self.config = FakeConfig(use_live_candle=use_live_candle)
        self.client = FakeClient(candles or [])
        self._bar_interval_ms = bar_interval_ms
        self._running: bool = False
        self.tick_handlers: list[Any] = []
        self.run_forever_called: bool = False

    def add_tick_handler(self, handler: Any) -> None:
        self.tick_handlers.append(handler)

    async def run_forever(self) -> None:
        self.run_forever_called = True
        self._running = True
        # Simulate firing one tick through each handler
        for handler in self.tick_handlers:
            await handler(
                FakeMarketTickEvent(
                    tick=FakeTick(
                        inst_id=self.config.inst_id,
                        price=3000.5,
                        size=1.2,
                        side="buy",
                        ts_ms=123456,
                    )
                )
            )


# ======================================================================
# Helpers
# ======================================================================


def _make_candle(
    ts_ms: int = 1000000,
    open_p: float = 3000.0,
    high: float = 3100.0,
    low: float = 2900.0,
    close: float = 3050.0,
    volume: float = 100.5,
    confirmed: bool = True,
) -> FakeCandle:
    return FakeCandle(ts_ms, open_p, high, low, close, volume, confirmed)


# ======================================================================
# Tests: fetch_recent_klines
# ======================================================================


class TestFetchRecentKlines:
    @pytest.mark.asyncio
    async def test_returns_last_n_candles(self) -> None:
        candles = [_make_candle(ts_ms=1000 + i * 1000) for i in range(10)]
        monitor = FakeMonitor(candles)
        client = OkxMarketDataClient(monitor)

        result = await client.fetch_recent_klines(limit=2)

        assert len(result) == 2
        assert result[0].open_time_ms == 9000
        assert result[1].open_time_ms == 10000

    @pytest.mark.asyncio
    async def test_candle_field_mapping(self) -> None:
        candles = [_make_candle(ts_ms=5000, open_p=3000.0, high=3100.0, low=2900.0, close=3050.0, volume=100.5,
                                confirmed=True)]
        monitor = FakeMonitor(candles, bar_interval_ms=15 * 60 * 1000)
        client = OkxMarketDataClient(monitor)

        result = await client.fetch_recent_klines(limit=1)

        assert len(result) == 1
        c = result[0]
        assert isinstance(c, CandleSnapshot)
        assert c.open_time_ms == 5000
        assert c.close_time_ms == 5000 + 15 * 60 * 1000
        assert c.open_price == Decimal("3000.0")
        assert c.high_price == Decimal("3100.0")
        assert c.low_price == Decimal("2900.0")
        assert c.close_price == Decimal("3050.0")
        assert c.volume == Decimal("100.5")
        assert c.is_closed is True

    @pytest.mark.asyncio
    async def test_raw_contains_inst_id_and_bar(self) -> None:
        candles = [_make_candle()]
        monitor = FakeMonitor(candles)
        client = OkxMarketDataClient(monitor)

        result = await client.fetch_recent_klines(limit=1)

        assert result[0].raw["inst_id"] == "ETH-USDT-SWAP"
        assert result[0].raw["bar"] == "15m"

    @pytest.mark.asyncio
    async def test_include_live_uses_monitor_config(self) -> None:
        candles = [_make_candle()]
        monitor = FakeMonitor(candles, use_live_candle=True)
        client = OkxMarketDataClient(monitor)

        await client.fetch_recent_klines(limit=1)

        assert monitor.client.last_include_live is True

    @pytest.mark.asyncio
    async def test_include_live_false(self) -> None:
        candles = [_make_candle()]
        monitor = FakeMonitor(candles, use_live_candle=False)
        client = OkxMarketDataClient(monitor)

        await client.fetch_recent_klines(limit=1)

        assert monitor.client.last_include_live is False

    @pytest.mark.asyncio
    async def test_close_time_ms_equals_ts_plus_bar_interval(self) -> None:
        candles = [_make_candle(ts_ms=100000)]
        monitor = FakeMonitor(candles, bar_interval_ms=900000)  # 15 min
        client = OkxMarketDataClient(monitor)

        result = await client.fetch_recent_klines(limit=1)

        assert result[0].close_time_ms == 100000 + 900000

    @pytest.mark.asyncio
    async def test_bar_interval_ms_zero_falls_back_to_ts(self) -> None:
        candles = [_make_candle(ts_ms=100000)]
        monitor = FakeMonitor(candles, bar_interval_ms=0)
        client = OkxMarketDataClient(monitor)

        result = await client.fetch_recent_klines(limit=1)

        assert result[0].close_time_ms == 100000

    @pytest.mark.asyncio
    async def test_bar_interval_ms_missing_falls_back_to_ts(self) -> None:
        candles = [_make_candle(ts_ms=200000)]
        monitor = FakeMonitor(candles)
        # Remove _bar_interval_ms to simulate missing attribute
        del monitor._bar_interval_ms  # type: ignore[attr-defined]
        client = OkxMarketDataClient(monitor)

        result = await client.fetch_recent_klines(limit=1)

        # close_time_ms should equal ts_ms (no bar_interval_ms available)
        assert result[0].close_time_ms == 200000

    @pytest.mark.asyncio
    async def test_limit_zero_raises(self) -> None:
        monitor = FakeMonitor()
        client = OkxMarketDataClient(monitor)

        with pytest.raises(ValueError, match="limit must be positive"):
            await client.fetch_recent_klines(limit=0)

    @pytest.mark.asyncio
    async def test_limit_negative_raises(self) -> None:
        monitor = FakeMonitor()
        client = OkxMarketDataClient(monitor)

        with pytest.raises(ValueError, match="limit must be positive"):
            await client.fetch_recent_klines(limit=-5)

    @pytest.mark.asyncio
    async def test_unconfirmed_candle(self) -> None:
        candles = [_make_candle(ts_ms=5000, confirmed=False)]
        monitor = FakeMonitor(candles)
        client = OkxMarketDataClient(monitor)

        result = await client.fetch_recent_klines(limit=1)

        assert result[0].is_closed is False

    @pytest.mark.asyncio
    async def test_empty_candles_returns_empty_list(self) -> None:
        monitor = FakeMonitor([])
        client = OkxMarketDataClient(monitor)

        result = await client.fetch_recent_klines(limit=5)

        assert result == []


# ======================================================================
# Tests: stream_market_events
# ======================================================================


class TestStreamMarketEvents:
    @pytest.mark.asyncio
    async def test_registers_tick_handler(self) -> None:
        monitor = FakeMonitor()
        client = OkxMarketDataClient(monitor)

        received: list[MarketTradeSnapshot] = []

        async def on_event(event: MarketTradeSnapshot) -> None:
            received.append(event)

        await client.stream_market_events(on_event)

        assert len(monitor.tick_handlers) == 1

    @pytest.mark.asyncio
    async def test_calls_monitor_run_forever(self) -> None:
        monitor = FakeMonitor()
        client = OkxMarketDataClient(monitor)

        async def on_event(event: MarketTradeSnapshot) -> None:
            pass

        await client.stream_market_events(on_event)

        assert monitor.run_forever_called is True

    @pytest.mark.asyncio
    async def test_tick_mapped_to_market_trade_snapshot(self) -> None:
        monitor = FakeMonitor()
        client = OkxMarketDataClient(monitor)

        received: list[MarketTradeSnapshot] = []

        async def on_event(event: MarketTradeSnapshot) -> None:
            received.append(event)

        await client.stream_market_events(on_event)

        assert len(received) == 1
        t = received[0]
        assert isinstance(t, MarketTradeSnapshot)
        assert t.event_time_ms == 123456
        assert t.price == Decimal("3000.5")
        assert t.qty == Decimal("1.2")
        assert t.side == "buy"
        assert t.raw == {"inst_id": "ETH-USDT-SWAP"}


# ======================================================================
# Tests: close
# ======================================================================


class TestClose:
    @pytest.mark.asyncio
    async def test_sets_running_false(self) -> None:
        monitor = FakeMonitor()
        monitor._running = True
        client = OkxMarketDataClient(monitor)

        await client.close()

        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_calls_client_close(self) -> None:
        monitor = FakeMonitor()
        client = OkxMarketDataClient(monitor)

        await client.close()

        assert monitor.client.closed is True

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        monitor = FakeMonitor()
        client = OkxMarketDataClient(monitor)

        await client.close()
        await client.close()

        assert monitor.client.closed is True


# ======================================================================
# Tests: no env / no real construction
# ======================================================================


class TestNoSideEffects:
    def test_does_not_read_env(self) -> None:
        """Verify the source file does not import os.getenv or load_dotenv."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[2] / "src" / "data_feed" / "okx_market_data_client.py").read_text()
        assert "os.getenv" not in source
        assert "load_dotenv" not in source

    def test_does_not_create_monitor(self) -> None:
        """Verify the source file does not instantiate BollBandBreakoutMonitor."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[2] / "src" / "data_feed" / "okx_market_data_client.py").read_text()
        assert "BollBandBreakoutMonitor(" not in source
        assert "BollBandBreakoutMonitorConfig" not in source

    def test_does_not_import_binance(self) -> None:
        """Verify the source file does not reference Binance."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[2] / "src" / "data_feed" / "okx_market_data_client.py").read_text()
        assert "binance" not in source
        assert "Binance" not in source
