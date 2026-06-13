#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : binance_market_data_probe.py
@Description: Binance USD-M Futures real WebSocket market data probe — no orders.

This script connects to the Binance USD-M Futures market WebSocket, subscribes
to ethusdt@aggTrade and ethusdt@kline_15m, maps raw payloads through the
existing Binance data_feed mappers into MarketTradeEvent / MarketCandleEvent,
and prints summary statistics.  It does NOT place any orders, read API keys,
import strategy modules, or touch the live trading main loop.

Usage::

    EXCHANGE=binance                      \\
    TRADE_ASSET=ETH                       \\
    QUOTE_ASSET=USDT                      \\
    MARKET_TYPE=PERPETUAL                 \\
    KLINE_INTERVAL=15m                    \\
    PYTHONPATH=. python -m scripts.binance_market_data_probe

Optional env::

    BINANCE_MARKET_DATA_PROBE_SECONDS=60
    BINANCE_MARKET_DATA_PROBE_MAX_EVENTS=200
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import aiohttp

from src.data_feed.market_events import MarketCandleEvent, MarketTradeEvent
from src.data_feed.selector import build_market_data_feed
from src.exchanges.models import ExchangeName
from src.exchanges.runtime_config import (
    SUPPORTED_CANONICAL_SYMBOL,
    SUPPORTED_KLINE_INTERVAL,
    ExchangeRuntimeConfig,
    load_unified_runtime_config,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_SYMBOL: str = "ETH-USDT-PERP"
BINANCE_SYMBOL: str = "ETHUSDT"

ENV_PROBE_SECONDS: str = "BINANCE_MARKET_DATA_PROBE_SECONDS"
ENV_MAX_EVENTS: str = "BINANCE_MARKET_DATA_PROBE_MAX_EVENTS"

DEFAULT_PROBE_SECONDS: int = 60
DEFAULT_MAX_EVENTS: int = 200

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def validate_probe_config(rt: ExchangeRuntimeConfig) -> str:
    """Validate the unified runtime config for Binance market data probe use.

    Returns the validated Binance raw symbol (ETHUSDT).

    Raises ``SystemExit`` on any validation failure.
    """
    if rt.exchange != ExchangeName.BINANCE:
        print(
            f"ERROR: EXCHANGE must be 'binance', got {rt.exchange.value!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if rt.canonical_symbol != SUPPORTED_CANONICAL_SYMBOL:
        print(
            f"ERROR: canonical_symbol must be {SUPPORTED_CANONICAL_SYMBOL!r}, "
            f"got {rt.canonical_symbol!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if rt.binance_symbol != BINANCE_SYMBOL:
        print(
            f"ERROR: binance_symbol must be {BINANCE_SYMBOL!r}, "
            f"got {rt.binance_symbol!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if rt.kline_interval != SUPPORTED_KLINE_INTERVAL:
        print(
            f"ERROR: KLINE_INTERVAL must be {SUPPORTED_KLINE_INTERVAL!r}, "
            f"got {rt.kline_interval!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("[probe] unified config validated OK")
    return rt.binance_symbol


def read_probe_duration() -> int:
    """Read ``BINANCE_MARKET_DATA_PROBE_SECONDS`` from env, returning a positive int.

    Raises ``SystemExit`` on invalid values.
    """
    raw = os.environ.get(ENV_PROBE_SECONDS, "").strip()
    if not raw:
        return DEFAULT_PROBE_SECONDS
    try:
        value = int(raw)
    except ValueError:
        print(
            f"ERROR: {ENV_PROBE_SECONDS} must be an integer, got {raw!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if value <= 0:
        print(
            f"ERROR: {ENV_PROBE_SECONDS} must be positive, got {value}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


def read_max_events() -> int:
    """Read ``BINANCE_MARKET_DATA_PROBE_MAX_EVENTS`` from env, returning a positive int.

    Raises ``SystemExit`` on invalid values.
    """
    raw = os.environ.get(ENV_MAX_EVENTS, "").strip()
    if not raw:
        return DEFAULT_MAX_EVENTS
    try:
        value = int(raw)
    except ValueError:
        print(
            f"ERROR: {ENV_MAX_EVENTS} must be an integer, got {raw!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if value <= 0:
        print(
            f"ERROR: {ENV_MAX_EVENTS} must be positive, got {value}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


# ---------------------------------------------------------------------------
# Aiohttp WebSocket connection
# ---------------------------------------------------------------------------


class AiohttpBinanceWsConnection:
    """Async iterator over Binance WebSocket text/binary messages."""

    def __init__(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        session: aiohttp.ClientSession,
    ) -> None:
        self._ws = ws
        self._session = session

    def __aiter__(self) -> "AiohttpBinanceWsConnection":
        return self

    async def __anext__(self) -> str | bytes:
        msg = await self._ws.receive()
        if msg.type == aiohttp.WSMsgType.TEXT:
            return msg.data
        if msg.type == aiohttp.WSMsgType.BINARY:
            return msg.data
        if msg.type == aiohttp.WSMsgType.CLOSED:
            raise StopAsyncIteration
        if msg.type == aiohttp.WSMsgType.ERROR:
            raise ConnectionError(f"WebSocket error: {self._ws.exception()}")
        # CLOSING / CLOSE / other sentinel — end iteration
        raise StopAsyncIteration

    async def close(self) -> None:
        """Close the underlying WebSocket and HTTP session."""
        await self._ws.close()
        await self._session.close()


async def connect_binance_ws(url: str) -> AiohttpBinanceWsConnection:
    """Open an aiohttp WebSocket connection to *url*.

    Returns an :class:`AiohttpBinanceWsConnection` that can be used as an
    async iterator of raw messages (``str | bytes``).

    Raises ``ConnectionError`` (or aiohttp errors) when the connection fails.
    """
    timeout = aiohttp.ClientTimeout(total=30.0)
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        ws = await session.ws_connect(url)
    except Exception:
        await session.close()
        raise
    return AiohttpBinanceWsConnection(ws, session)


# ---------------------------------------------------------------------------
# Probe loop
# ---------------------------------------------------------------------------


def _format_trade(event: MarketTradeEvent) -> str:
    return (
        f"[trade] t={event.event_time_ms} side={event.taker_side.value} "
        f"price={event.price} qty={event.quantity}"
    )


def _format_candle(event: MarketCandleEvent) -> str:
    return (
        f"[candle] tf={event.timeframe} closed={event.is_closed} "
        f"open={event.open_price} high={event.high_price} "
        f"low={event.low_price} close={event.close_price} vol={event.volume}"
    )


async def run_probe(
    *,
    stream_url: str,
    map_message: Any,
    duration_seconds: int,
    max_events: int,
) -> dict[str, int]:
    """Run the market data probe loop.

    Connects to *stream_url*, iterates over raw WebSocket messages, maps each
    one through *map_message* (a callable ``(str | bytes) -> MarketDataEvent | None``),
    prints a one-line summary per event, and stops when *max_events* or
    *duration_seconds* is reached.

    Returns a dict with keys ``trade_events``, ``candle_events``,
    ``closed_candle_events``, ``errors``.

    Parameters
    ----------
    stream_url:
        The full Binance combined market stream URL.
    map_message:
        A callable that parses a raw WebSocket message into a
        ``MarketTradeEvent | MarketCandleEvent | None``.
    duration_seconds:
        Maximum probe duration in seconds.
    max_events:
        Maximum total events (trades + candles) before early exit.
    """
    trade_count = 0
    candle_count = 0
    closed_candle_count = 0
    error_count = 0

    connection: AiohttpBinanceWsConnection | None = None
    start_time = asyncio.get_event_loop().time()
    deadline = start_time + duration_seconds
    total = 0

    try:
        connection = await connect_binance_ws(stream_url)
    except Exception as exc:
        print(
            f"ERROR: cannot connect to Binance WebSocket: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        async for message in connection:
            # --- Parse ---
            try:
                event = map_message(message)
            except Exception:
                error_count += 1
                total = trade_count + candle_count
                # Check exit conditions even after parse errors
                if total >= max_events:
                    print(f"[probe] reached max_events={max_events}")
                    break
                if asyncio.get_event_loop().time() >= deadline:
                    print(f"[probe] reached duration={duration_seconds}s")
                    break
                continue

            if event is None:
                # Unrecognised event type — skip silently (not an error)
                total = trade_count + candle_count
                if total >= max_events:
                    print(f"[probe] reached max_events={max_events}")
                    break
                if asyncio.get_event_loop().time() >= deadline:
                    print(f"[probe] reached duration={duration_seconds}s")
                    break
                continue

            # --- Count & print ---
            if isinstance(event, MarketTradeEvent):
                trade_count += 1
                print(_format_trade(event))
            elif isinstance(event, MarketCandleEvent):
                candle_count += 1
                if event.is_closed:
                    closed_candle_count += 1
                print(_format_candle(event))
            else:
                # Defensive — shouldn't happen with current mappers
                error_count += 1

            # --- Exit checks ---
            total = trade_count + candle_count
            if total >= max_events:
                print(f"[probe] reached max_events={max_events}")
                break

            now = asyncio.get_event_loop().time()
            if now >= deadline:
                print(f"[probe] reached duration={duration_seconds}s")
                break

    except asyncio.CancelledError:
        # KeyboardInterrupt arrives as CancelledError in asyncio.run()
        pass
    finally:
        if connection is not None:
            try:
                await connection.close()
            except Exception:
                pass

    return {
        "trade_events": trade_count,
        "candle_events": candle_count,
        "closed_candle_events": closed_candle_count,
        "errors": error_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Entry point — validate config, connect, probe, print stats."""

    # --- Config (no network) ---
    rt = load_unified_runtime_config()
    binance_symbol = validate_probe_config(rt)
    duration_seconds = read_probe_duration()
    max_events = read_max_events()

    # --- Build data feed (also validates symbol/interval) ---
    feed = build_market_data_feed(
        exchange=rt.exchange,
        canonical_symbol=rt.canonical_symbol,
        raw_symbol=rt.binance_symbol,
        kline_interval=rt.kline_interval,
        binance_ws_connector=connect_binance_ws,
    )

    stream_url = feed.stream_url()
    stream_names = feed.stream_names()

    # --- Startup banner ---
    print("[probe] Binance market data probe starting")
    print(f"[probe] exchange=binance")
    print(f"[probe] canonical_symbol={CANONICAL_SYMBOL}")
    print(f"[probe] raw_symbol={binance_symbol}")
    print(f"[probe] streams={stream_names[0]},{stream_names[1]}")
    print(f"[probe] duration={duration_seconds}s max_events={max_events}")

    # --- Run ---
    stats = await run_probe(
        stream_url=stream_url,
        map_message=feed.map_message,
        duration_seconds=duration_seconds,
        max_events=max_events,
    )

    # --- Summary ---
    print("[probe] done")
    print(f"[probe] trade_events={stats['trade_events']}")
    print(f"[probe] candle_events={stats['candle_events']}")
    print(f"[probe] closed_candle_events={stats['closed_candle_events']}")
    print(f"[probe] errors={stats['errors']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
