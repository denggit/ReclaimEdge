#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_okx_market_data_client.py
@Description: Functional tests for OkxMarketDataClient using FakeRestClient.

No real API calls.  No env reads.  No production wiring.
OkxMarketDataClient now takes OkxMarketDataClientConfig, not a monitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.data_feed.okx_market_data_client import (
    OkxMarketDataClient,
    OkxMarketDataClientConfig,
    _OkxPublicRestClient,
    _OkxPublicRestConfig,
    _RawCandle,
    _parse_raw_candle,
)
from src.data_feed.market_data_client_port import (
    CandleSnapshot,
    MarketTradeSnapshot,
)


# ======================================================================
# Fake REST client that returns canned candle data
# ======================================================================


class FakeRestClient:
    """Fake OKX REST client that returns canned raw candle rows."""

    def __init__(self, raw_rows: list[list] | None = None) -> None:
        self.raw_rows: list[list] = raw_rows or []
        self.closed: bool = False

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def fetch_candles(
        self, *, inst_id: str, bar: str, limit: int
    ) -> list[dict[str, Any]]:
        # Return each row as a dict (mimicking the raw row format)
        # The actual implementation returns list[list], but for testing we use
        # what _parse_raw_candle expects — it expects list rows, so we return
        # them directly
        return self.raw_rows


# ======================================================================
# Helpers
# ======================================================================


def _make_raw_row(
    ts_ms: int = 1000000,
    open_p: float = 3000.0,
    high: float = 3100.0,
    low: float = 2900.0,
    close: float = 3050.0,
    volume: float = 100.5,
    confirmed: bool = True,
) -> list:
    """Build a raw OKX candle row matching the API format.

    OKX format: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    """
    return [
        str(ts_ms),
        str(open_p),
        str(high),
        str(low),
        str(close),
        str(volume),
        "0",      # volCcy
        "0",      # volCcyQuote
        "1" if confirmed else "0",
    ]


def _make_config(**kwargs) -> OkxMarketDataClientConfig:
    defaults = dict(
        inst_id="ETH-USDT-SWAP",
        bar="15m",
        rest_base_url="https://www.okx.com",
        ws_public_url="wss://ws.okx.com:8443/ws/v5/public",
        candle_limit=100,
    )
    defaults.update(kwargs)
    return OkxMarketDataClientConfig(**defaults)


# ======================================================================
# Tests: _parse_raw_candle
# ======================================================================


class TestParseRawCandle:
    def test_parse_confirmed(self) -> None:
        row = _make_raw_row(ts_ms=5000, open_p=3000.0, high=3100.0, low=2900.0, close=3050.0, volume=100.5, confirmed=True)
        result = _parse_raw_candle(row, include_live=True)
        assert result is not None
        assert result.ts_ms == 5000
        assert result.open == 3000.0
        assert result.high == 3100.0
        assert result.low == 2900.0
        assert result.close == 3050.0
        assert result.volume == 100.5
        assert result.confirmed is True

    def test_parse_unconfirmed(self) -> None:
        row = _make_raw_row(ts_ms=5000, confirmed=False)
        result = _parse_raw_candle(row, include_live=True)
        assert result is not None
        assert result.confirmed is False

    def test_skip_unconfirmed_when_include_live_false(self) -> None:
        row = _make_raw_row(ts_ms=5000, confirmed=False)
        result = _parse_raw_candle(row, include_live=False)
        assert result is None

    def test_short_row_returns_none(self) -> None:
        result = _parse_raw_candle(["1", "2"], include_live=True)
        assert result is None

    def test_missing_confirm_field_defaults_true(self) -> None:
        # Row with exactly 8 fields (no confirm field)
        row = ["1000", "3000", "3100", "2900", "3050", "100", "0", "0"]
        result = _parse_raw_candle(row, include_live=True)
        assert result is not None
        assert result.confirmed is True


# ======================================================================
# Tests: fetch_recent_klines
# ======================================================================


class TestFetchRecentKlines:
    @pytest.mark.asyncio
    async def test_returns_last_n_candles(self) -> None:
        rows = [_make_raw_row(ts_ms=1000 + i * 1000) for i in range(10)]
        fake_rest = FakeRestClient(rows)
        config = _make_config()
        client = OkxMarketDataClient.__new__(OkxMarketDataClient)
        client._config = config
        client._rest = fake_rest
        client._bar_interval_ms = 15 * 60 * 1000
        client._ws_running = False

        result = await client.fetch_recent_klines(limit=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_candle_field_mapping(self) -> None:
        rows = [_make_raw_row(ts_ms=5000, open_p=3000.0, high=3100.0, low=2900.0, close=3050.0, volume=100.5,
                              confirmed=True)]
        fake_rest = FakeRestClient(rows)
        config = _make_config(bar="15m")
        client = OkxMarketDataClient.__new__(OkxMarketDataClient)
        client._config = config
        client._rest = fake_rest
        client._bar_interval_ms = 15 * 60 * 1000

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
        rows = [_make_raw_row()]
        fake_rest = FakeRestClient(rows)
        config = _make_config(inst_id="ETH-USDT-SWAP", bar="15m")
        client = OkxMarketDataClient.__new__(OkxMarketDataClient)
        client._config = config
        client._rest = fake_rest
        client._bar_interval_ms = 15 * 60 * 1000

        result = await client.fetch_recent_klines(limit=1)

        assert result[0].raw["inst_id"] == "ETH-USDT-SWAP"
        assert result[0].raw["bar"] == "15m"

    @pytest.mark.asyncio
    async def test_limit_zero_raises(self) -> None:
        config = _make_config()
        client = OkxMarketDataClient(config)

        with pytest.raises(ValueError, match="limit must be positive"):
            await client.fetch_recent_klines(limit=0)

    @pytest.mark.asyncio
    async def test_limit_negative_raises(self) -> None:
        config = _make_config()
        client = OkxMarketDataClient(config)

        with pytest.raises(ValueError, match="limit must be positive"):
            await client.fetch_recent_klines(limit=-5)

    @pytest.mark.asyncio
    async def test_unconfirmed_candle(self) -> None:
        rows = [_make_raw_row(ts_ms=5000, confirmed=False)]
        fake_rest = FakeRestClient(rows)
        config = _make_config()
        client = OkxMarketDataClient.__new__(OkxMarketDataClient)
        client._config = config
        client._rest = fake_rest
        client._bar_interval_ms = 15 * 60 * 1000

        result = await client.fetch_recent_klines(limit=1)

        assert result[0].is_closed is False

    @pytest.mark.asyncio
    async def test_empty_candles_returns_empty_list(self) -> None:
        fake_rest = FakeRestClient([])
        config = _make_config()
        client = OkxMarketDataClient.__new__(OkxMarketDataClient)
        client._config = config
        client._rest = fake_rest
        client._bar_interval_ms = 15 * 60 * 1000

        result = await client.fetch_recent_klines(limit=5)

        assert result == []


# ======================================================================
# Tests: close
# ======================================================================


class TestClose:
    @pytest.mark.asyncio
    async def test_closes_rest_client(self) -> None:
        fake_rest = FakeRestClient()
        config = _make_config()
        client = OkxMarketDataClient.__new__(OkxMarketDataClient)
        client._config = config
        client._rest = fake_rest
        client._ws_running = False
        client._ws_session = None

        await client.close()

        assert fake_rest.closed is True

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        fake_rest = FakeRestClient()
        config = _make_config()
        client = OkxMarketDataClient.__new__(OkxMarketDataClient)
        client._config = config
        client._rest = fake_rest
        client._ws_running = False
        client._ws_session = None

        await client.close()
        await client.close()

        assert fake_rest.closed is True


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


# ======================================================================
# Tests: config object
# ======================================================================


class TestConfig:
    def test_default_config(self) -> None:
        config = OkxMarketDataClientConfig()
        assert config.inst_id == "ETH-USDT-SWAP"
        assert config.bar == "15m"
        assert config.rest_base_url == "https://www.okx.com"
        assert config.ws_public_url == "wss://ws.okx.com:8443/ws/v5/public"
        assert config.candle_limit == 100

    def test_custom_config(self) -> None:
        config = OkxMarketDataClientConfig(
            inst_id="BTC-USDT-SWAP",
            bar="1h",
            rest_base_url="https://www.okx.cab",
            ws_public_url="wss://wsp.okx.com:8443/ws/v5/public",
            candle_limit=50,
        )
        assert config.inst_id == "BTC-USDT-SWAP"
        assert config.bar == "1h"
        assert config.rest_base_url == "https://www.okx.cab"
        assert config.ws_public_url == "wss://wsp.okx.com:8443/ws/v5/public"
        assert config.candle_limit == 50
