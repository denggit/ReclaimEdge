#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_binance_market_data_client_port.py
@Description: Functional tests for BinanceMarketDataClient.
              No real API calls.  No env reads.  No production wiring.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any

import pytest

from src.data_feed.binance.market_data_client import (
    BinanceMarketDataClient,
    _BinancePublicRestClient,
    _BinancePublicRestConfig,
)
from src.data_feed.market_data_client_port import (
    CandleSnapshot,
    MarketTradeSnapshot,
)


# ======================================================================
# Fake REST client
# ======================================================================


class FakeRestClient:
    """Fake Binance REST client returning canned kline rows."""

    def __init__(self, raw_rows: list[list] | None = None) -> None:
        self.raw_rows: list[list] = raw_rows or []
        self.closed: bool = False
        self.started: bool = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def fetch_klines(
        self, *, symbol: str, interval: str, limit: int
    ) -> list[list[Any]]:
        return self.raw_rows


# ======================================================================
# Fake WS connector
# ======================================================================


class FakeWsConnection:
    """Fake WebSocket connection yielding canned messages."""

    def __init__(self, messages: list[str | bytes]) -> None:
        self._messages = messages
        self.closed: bool = False
        self._pos = 0

    def __aiter__(self) -> "FakeWsConnection":
        return self

    async def __anext__(self) -> str | bytes:
        if self._pos >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._pos]
        self._pos += 1
        return msg

    async def close(self) -> None:
        self.closed = True


class FakeWsConnector:
    """Fake WebSocket connector that returns FakeWsConnection instances."""

    def __init__(self, messages: list[str | bytes] | None = None) -> None:
        self.messages: list[str | bytes] = messages or []
        self.urls: list[str] = []

    async def __call__(self, url: str) -> FakeWsConnection:
        self.urls.append(url)
        return FakeWsConnection(list(self.messages))


# ======================================================================
# Helpers
# ======================================================================


def _make_kline_row(
    open_time: int = 1710000000000,
    open_p: str = "3000.00",
    high: str = "3100.00",
    low: str = "2900.00",
    close: str = "3050.00",
    volume: str = "100.50",
    close_time: int = 1710000899999,
) -> list:
    return [
        str(open_time),
        open_p,
        high,
        low,
        close,
        volume,
        str(close_time),
    ]


def _make_agg_trade_msg(
    event_time: int = 1710000000123,
    price: str = "3100.50",
    qty: str = "1.25",
    m: bool = True,
) -> str:
    payload = {
        "e": "aggTrade",
        "E": event_time,
        "s": "ETHUSDT",
        "a": 5933014,
        "p": price,
        "q": qty,
        "f": 100,
        "l": 105,
        "T": 1710000000111,
        "m": m,
    }
    return json.dumps(payload)


def _make_client(
    *,
    symbol: str = "ETHUSDT",
    interval: str = "15m",
    rest_client: Any = None,
    ws_connector: Any = None,
    request_timeout_seconds: float = 10.0,
) -> BinanceMarketDataClient:
    """Construct a BinanceMarketDataClient using the real constructor.

    Passes fake implementations for rest_client and ws_connector so that
    the constructor injection path is exercised in every test.
    """
    return BinanceMarketDataClient(
        symbol=symbol,
        interval=interval,
        rest_client=rest_client or FakeRestClient(),
        ws_connector=ws_connector or FakeWsConnector(),
        request_timeout_seconds=request_timeout_seconds,
    )


def _inject_deps(
    client: BinanceMarketDataClient,
    *,
    symbol: str = "ETHUSDT",
    interval: str = "15m",
    rest: Any = None,
    ws_connector: Any = None,
) -> None:
    """Inject internal dependencies into a bare BinanceMarketDataClient instance.

    Reserved for rare internal-state tests that cannot use the real
    constructor.  Most tests should use ``_make_client(...)`` instead.
    """
    client._symbol = symbol
    client._interval = interval
    client._rest = rest or FakeRestClient()
    client._ws_connector = ws_connector or FakeWsConnector()
    client._ws_connection = None
    client._ws_running = False


# ======================================================================
# Tests: fetch_recent_klines
# ======================================================================


class TestFetchRecentKlines:
    @pytest.mark.asyncio
    async def test_returns_candle_snapshots(self) -> None:
        rows = [_make_kline_row(open_time=1000 + i * 60000) for i in range(5)]
        fake_rest = FakeRestClient(rows)
        client = _make_client(rest_client=fake_rest)

        result = await client.fetch_recent_klines(limit=3)

        assert len(result) == 3
        for c in result:
            assert isinstance(c, CandleSnapshot)

    @pytest.mark.asyncio
    async def test_candle_field_mapping(self) -> None:
        rows = [_make_kline_row(
            open_time=5000, open_p="3000.00", high="3100.00",
            low="2900.00", close="3050.00", volume="100.50",
            close_time=5000 + 15 * 60 * 1000,
        )]
        fake_rest = FakeRestClient(rows)
        client = _make_client(rest_client=fake_rest)

        result = await client.fetch_recent_klines(limit=1)

        assert len(result) == 1
        c = result[0]
        assert c.open_time_ms == 5000
        assert c.open_price == Decimal("3000.00")
        assert c.high_price == Decimal("3100.00")
        assert c.low_price == Decimal("2900.00")
        assert c.close_price == Decimal("3050.00")
        assert c.volume == Decimal("100.50")

    @pytest.mark.asyncio
    async def test_is_closed_based_on_close_time(self) -> None:
        # close_time_ms = 1000, now_ms will be much larger → closed
        rows = [_make_kline_row(open_time=500, close_time=1000)]
        fake_rest = FakeRestClient(rows)
        client = _make_client(rest_client=fake_rest)

        result = await client.fetch_recent_klines(limit=1)

        assert result[0].is_closed is True

    @pytest.mark.asyncio
    async def test_raw_contains_symbol(self) -> None:
        rows = [_make_kline_row()]
        fake_rest = FakeRestClient(rows)
        client = _make_client(symbol="ETHUSDT", interval="15m", rest_client=fake_rest)

        result = await client.fetch_recent_klines(limit=1)

        assert result[0].raw["symbol"] == "ETHUSDT"
        assert result[0].raw["interval"] == "15m"

    @pytest.mark.asyncio
    async def test_limit_zero_raises_value_error(self) -> None:
        client = _make_client(rest_client=FakeRestClient())

        with pytest.raises(ValueError, match="limit must be positive"):
            await client.fetch_recent_klines(limit=0)

    @pytest.mark.asyncio
    async def test_limit_negative_raises_value_error(self) -> None:
        client = _make_client(rest_client=FakeRestClient())

        with pytest.raises(ValueError, match="limit must be positive"):
            await client.fetch_recent_klines(limit=-5)

    @pytest.mark.asyncio
    async def test_limit_exceeds_max_raises(self) -> None:
        client = _make_client(rest_client=FakeRestClient())

        with pytest.raises(ValueError, match="limit must not exceed"):
            await client.fetch_recent_klines(limit=1501)

    @pytest.mark.asyncio
    async def test_empty_klines_returns_empty_list(self) -> None:
        fake_rest = FakeRestClient([])
        client = _make_client(rest_client=fake_rest)

        result = await client.fetch_recent_klines(limit=5)

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_oldest_first(self) -> None:
        """Input rows are unordered (3000, 1000, 2000).
        Output must be sorted oldest -> newest: 1000, 2000, 3000."""
        rows = [
            _make_kline_row(open_time=3000),
            _make_kline_row(open_time=1000),
            _make_kline_row(open_time=2000),
        ]
        fake_rest = FakeRestClient(rows)
        client = _make_client(rest_client=fake_rest)

        result = await client.fetch_recent_klines(limit=3)

        assert result[0].open_time_ms == 1000
        assert result[1].open_time_ms == 2000
        assert result[2].open_time_ms == 3000

    @pytest.mark.asyncio
    async def test_oldest_first_with_limit(self) -> None:
        """5 unordered rows, limit=3 → returns the 3 most recent, oldest -> newest."""
        rows = [
            _make_kline_row(open_time=5000),
            _make_kline_row(open_time=1000),
            _make_kline_row(open_time=3000),
            _make_kline_row(open_time=2000),
            _make_kline_row(open_time=4000),
        ]
        fake_rest = FakeRestClient(rows)
        client = _make_client(rest_client=fake_rest)

        result = await client.fetch_recent_klines(limit=3)

        assert len(result) == 3
        assert result[0].open_time_ms == 3000
        assert result[1].open_time_ms == 4000
        assert result[2].open_time_ms == 5000

    @pytest.mark.asyncio
    async def test_malformed_kline_row_raises_value_error(self) -> None:
        """REST kline row malformed must raise ValueError — no silent skip."""
        rows = [
            _make_kline_row(open_time=2000),
            ["just", "two"],  # too short → must raise
            _make_kline_row(open_time=3000),
        ]
        fake_rest = FakeRestClient(rows)
        client = _make_client(rest_client=fake_rest)

        with pytest.raises(ValueError):
            await client.fetch_recent_klines(limit=3)


# ======================================================================
# Tests: close
# ======================================================================


class TestClose:
    @pytest.mark.asyncio
    async def test_closes_rest_client(self) -> None:
        fake_rest = FakeRestClient()
        client = _make_client(rest_client=fake_rest)

        await client.close()

        assert fake_rest.closed is True

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        fake_rest = FakeRestClient()
        client = _make_client(rest_client=fake_rest)

        await client.close()
        await client.close()

        assert fake_rest.closed is True

    @pytest.mark.asyncio
    async def test_close_after_real_constructor_is_idempotent(self) -> None:
        """close() must be idempotent when client is built via real constructor."""
        fake_rest = FakeRestClient()
        client = _make_client(rest_client=fake_rest)

        await client.close()
        await client.close()

        assert fake_rest.closed is True

    @pytest.mark.asyncio
    async def test_stops_ws_running_flag(self) -> None:
        fake_rest = FakeRestClient()
        client = _make_client(rest_client=fake_rest)
        client._ws_running = True

        await client.close()

        assert client._ws_running is False

    @pytest.mark.asyncio
    async def test_closes_ws_connection(self) -> None:
        fake_rest = FakeRestClient()
        fake_ws_conn = FakeWsConnection([])
        client = _make_client(rest_client=fake_rest)
        client._ws_running = True
        client._ws_connection = fake_ws_conn

        await client.close()

        assert fake_ws_conn.closed is True


# ======================================================================
# Tests: stream_market_events
# ======================================================================


class TestStreamMarketEvents:
    @pytest.mark.asyncio
    async def test_calls_on_event_with_market_trade_snapshot(self) -> None:
        messages = [_make_agg_trade_msg()]
        connector = FakeWsConnector(messages)
        client = _make_client(rest_client=FakeRestClient(), ws_connector=connector)
        client._ws_running = True

        events: list = []

        async def _on_event(event: Any) -> None:
            events.append(event)
            # Stop after first event to exit the loop
            client._ws_running = False

        await client.stream_market_events(on_event=_on_event)

        assert len(events) == 1
        assert isinstance(events[0], MarketTradeSnapshot)

    @pytest.mark.asyncio
    async def test_trade_snapshot_fields(self) -> None:
        messages = [_make_agg_trade_msg(price="3100.50", qty="1.25", event_time=1710000000123, m=True)]
        connector = FakeWsConnector(messages)
        client = _make_client(rest_client=FakeRestClient(), ws_connector=connector)
        client._ws_running = True

        events: list = []

        async def _on_event(event: Any) -> None:
            events.append(event)
            client._ws_running = False

        await client.stream_market_events(on_event=_on_event)

        assert len(events) == 1
        t = events[0]
        assert t.price == Decimal("3100.50")
        assert t.qty == Decimal("1.25")
        assert t.event_time_ms == 1710000000123
        assert t.side == "SELL"

    @pytest.mark.asyncio
    async def test_ignores_non_agg_trade_messages(self) -> None:
        messages = [
            json.dumps({"e": "kline", "E": 1, "k": {"t": 1, "T": 2, "o": "1", "c": "1", "h": "1", "l": "1", "v": "1", "x": True, "s": "ETHUSDT", "i": "15m"}}),
            json.dumps({"e": "bookTicker", "s": "ETHUSDT"}),
            _make_agg_trade_msg(),
        ]
        connector = FakeWsConnector(messages)
        client = _make_client(rest_client=FakeRestClient(), ws_connector=connector)
        client._ws_running = True

        events: list = []

        async def _on_event(event: Any) -> None:
            events.append(event)
            client._ws_running = False

        await client.stream_market_events(on_event=_on_event)

        assert len(events) == 1
        assert isinstance(events[0], MarketTradeSnapshot)

    @pytest.mark.asyncio
    async def test_handles_invalid_json_gracefully(self) -> None:
        messages = [
            "not valid json",
            _make_agg_trade_msg(),
        ]
        connector = FakeWsConnector(messages)
        client = _make_client(rest_client=FakeRestClient(), ws_connector=connector)
        client._ws_running = True

        events: list = []

        async def _on_event(event: Any) -> None:
            events.append(event)
            client._ws_running = False

        await client.stream_market_events(on_event=_on_event)

        assert len(events) == 1
        assert isinstance(events[0], MarketTradeSnapshot)

    @pytest.mark.asyncio
    async def test_handles_invalid_trade_payload(self) -> None:
        # Missing required fields → skipped
        messages = [
            json.dumps({"e": "aggTrade", "E": 123}),  # no p, no q
            _make_agg_trade_msg(),
        ]
        connector = FakeWsConnector(messages)
        client = _make_client(rest_client=FakeRestClient(), ws_connector=connector)
        client._ws_running = True

        events: list = []

        async def _on_event(event: Any) -> None:
            events.append(event)
            client._ws_running = False

        await client.stream_market_events(on_event=_on_event)

        assert len(events) == 1
        assert isinstance(events[0], MarketTradeSnapshot)

    @pytest.mark.asyncio
    async def test_uses_correct_ws_url(self) -> None:
        msg = _make_agg_trade_msg()
        connector = FakeWsConnector([msg])
        client = _make_client(symbol="ETHUSDT", rest_client=FakeRestClient(), ws_connector=connector)
        client._ws_running = True

        async def _on_event(event: Any) -> None:
            client._ws_running = False  # Stop after first event

        try:
            await asyncio.wait_for(client.stream_market_events(on_event=_on_event), timeout=2.0)
        except asyncio.TimeoutError:
            pass

        assert len(connector.urls) >= 1
        assert "ethusdt@aggTrade" in connector.urls[0]

    @pytest.mark.asyncio
    async def test_does_not_call_strategy_directly(self) -> None:
        """Verify that stream_market_events only passes events to on_event,
        and does not call any strategy module."""
        messages = [_make_agg_trade_msg()]
        connector = FakeWsConnector(messages)
        client = _make_client(rest_client=FakeRestClient(), ws_connector=connector)
        client._ws_running = True

        events: list = []

        async def _on_event(event: Any) -> None:
            events.append(event)
            client._ws_running = False

        await client.stream_market_events(on_event=_on_event)

        # The on_event callback received exactly one event
        assert len(events) == 1
        # The event is ONLY a MarketTradeSnapshot — no strategy wrapping
        assert isinstance(events[0], MarketTradeSnapshot)

    @pytest.mark.asyncio
    async def test_side_none_when_m_missing(self) -> None:
        """When aggTrade has no m field, side should be None."""
        payload = {
            "e": "aggTrade",
            "E": 1710000000123,
            "p": "3100.50",
            "q": "1.25",
        }
        messages = [json.dumps(payload)]
        connector = FakeWsConnector(messages)
        client = _make_client(rest_client=FakeRestClient(), ws_connector=connector)
        client._ws_running = True

        events: list = []

        async def _on_event(event: Any) -> None:
            events.append(event)
            client._ws_running = False

        await client.stream_market_events(on_event=_on_event)

        assert len(events) == 1
        assert events[0].side is None


# ======================================================================
# Tests: real constructor
# ======================================================================


class TestRealConstructor:
    """Verify that the real __init__ injection path works correctly."""

    def test_constructor_accepts_fake_rest_and_ws_connector(self) -> None:
        fake_rest = FakeRestClient()
        fake_ws = FakeWsConnector()
        client = BinanceMarketDataClient(
            symbol="ETHUSDT",
            interval="15m",
            rest_client=fake_rest,
            ws_connector=fake_ws,
        )
        assert client._rest is fake_rest
        assert client._ws_connector is fake_ws

    def test_constructor_sets_symbol_and_interval(self) -> None:
        client = BinanceMarketDataClient(
            symbol="BTCUSDT",
            interval="1h",
            rest_client=FakeRestClient(),
            ws_connector=FakeWsConnector(),
        )
        assert client._symbol == "BTCUSDT"
        assert client._interval == "1h"


# ======================================================================
# Tests: no side effects
# ======================================================================


class TestNoSideEffects:
    def test_does_not_import_env(self) -> None:
        from pathlib import Path

        source_path = (Path(__file__).resolve().parents[3] / "src" / "data_feed" / "binance" / "market_data_client.py")
        text = source_path.read_text(encoding="utf-8")
        assert "os.getenv" not in text
        assert "load_dotenv" not in text

    def test_does_not_import_strategy(self) -> None:
        from pathlib import Path

        source_path = (Path(__file__).resolve().parents[3] / "src" / "data_feed" / "binance" / "market_data_client.py")
        text = source_path.read_text(encoding="utf-8")
        assert "src.strategies" not in text
        assert "BollBandBreakoutMonitor" not in text

    def test_does_not_import_execution(self) -> None:
        from pathlib import Path

        source_path = (Path(__file__).resolve().parents[3] / "src" / "data_feed" / "binance" / "market_data_client.py")
        text = source_path.read_text(encoding="utf-8")
        assert "src.execution" not in text

    def test_does_not_import_live(self) -> None:
        from pathlib import Path

        source_path = (Path(__file__).resolve().parents[3] / "src" / "data_feed" / "binance" / "market_data_client.py")
        text = source_path.read_text(encoding="utf-8")
        assert "src.live" not in text

    def test_does_not_import_okx(self) -> None:
        from pathlib import Path

        source_path = (Path(__file__).resolve().parents[3] / "src" / "data_feed" / "binance" / "market_data_client.py")
        text = source_path.read_text(encoding="utf-8")
        assert "okx" not in text
        assert "OKX" not in text
