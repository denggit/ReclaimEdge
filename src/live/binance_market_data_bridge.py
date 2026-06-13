#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : binance_market_data_bridge.py
@Description: Signal-only bridge — converts Binance canonical market events
              (MarketTradeEvent / MarketCandleEvent) into lightweight signal-input
              DTOs consumable by the live strategy observation pipeline.

This module does NOT:

- connect to any WebSocket
- read any API key or environment variable
- import or call any broker / trader / execution / order / strategy module
- place or cancel any order
- modify OKX behaviour
- support any symbol other than ETH-USDT-PERP / ETHUSDT / 15m

It is a pure market-event converter with in-memory statistics.  It performs
no network, secret, broker, or order side effects.  The caller (20C-2B) is
responsible for wiring the output into the live-runner tick/candle path.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Union

from src.data_feed.market_events import MarketCandleEvent, MarketTradeEvent

# ---------------------------------------------------------------------------
# Supported values — hard-coded, NOT read from env
# ---------------------------------------------------------------------------

SUPPORTED_CANONICAL_SYMBOL: str = "ETH-USDT-PERP"
SUPPORTED_RAW_SYMBOL: str = "ETHUSDT"
SUPPORTED_INTERVAL: str = "15m"

# ---------------------------------------------------------------------------
# Lightweight signal-input DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinanceSignalTradeInput:
    """Signal-only trade input derived from a Binance aggTrade canonical event.

    All numeric fields use ``Decimal`` — no float conversion.
    """

    canonical_symbol: str
    raw_symbol: str
    timestamp_ms: int
    side: str
    price: Decimal
    quantity: Decimal
    source: str = "binance_agg_trade"


@dataclass(frozen=True)
class BinanceSignalCandleInput:
    """Signal-only candle input derived from a Binance kline canonical event.

    ``closed`` directly mirrors ``MarketCandleEvent.is_closed``.
    """

    canonical_symbol: str
    raw_symbol: str
    interval: str
    timestamp_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    closed: bool
    source: str = "binance_kline"


@dataclass
class BinanceSignalBridgeStats:
    """Accumulated statistics for the signal bridge."""

    trade_events: int = 0
    candle_events: int = 0
    closed_candle_events: int = 0
    ignored_events: int = 0
    error_events: int = 0


# ---------------------------------------------------------------------------
# Typed union
# ---------------------------------------------------------------------------

BinanceSignalInput = Union[BinanceSignalTradeInput, BinanceSignalCandleInput]

# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class BinanceMarketDataSignalBridge:
    """Convert Binance canonical market events into signal-only input DTOs.

    This bridge is deliberately narrow:

    * Only ``ETH-USDT-PERP`` / ``ETHUSDT`` / ``15m``.
    * Only ``MarketTradeEvent`` and ``MarketCandleEvent`` as inputs.
    * Stateless — no network, no secrets, no orders.
    """

    def __init__(
        self,
        *,
        canonical_symbol: str = SUPPORTED_CANONICAL_SYMBOL,
        raw_symbol: str = SUPPORTED_RAW_SYMBOL,
        interval: str = SUPPORTED_INTERVAL,
    ) -> None:
        # --- Validate at construction time ---
        if canonical_symbol != SUPPORTED_CANONICAL_SYMBOL:
            raise ValueError(
                f"Unsupported canonical_symbol: {canonical_symbol!r}. "
                f"Only {SUPPORTED_CANONICAL_SYMBOL!r} is supported."
            )
        if raw_symbol != SUPPORTED_RAW_SYMBOL:
            raise ValueError(
                f"Unsupported raw_symbol: {raw_symbol!r}. "
                f"Only {SUPPORTED_RAW_SYMBOL!r} is supported."
            )
        if interval != SUPPORTED_INTERVAL:
            raise ValueError(
                f"Unsupported interval: {interval!r}. "
                f"Only {SUPPORTED_INTERVAL!r} is supported."
            )

        self._canonical_symbol = canonical_symbol
        self._raw_symbol = raw_symbol
        self._interval = interval
        self._stats = BinanceSignalBridgeStats()

    # -- public read-only properties ---------------------------------------

    @property
    def canonical_symbol(self) -> str:
        return self._canonical_symbol

    @property
    def raw_symbol(self) -> str:
        return self._raw_symbol

    @property
    def interval(self) -> str:
        return self._interval

    @property
    def stats(self) -> BinanceSignalBridgeStats:
        return self._stats

    def get_stats(self) -> BinanceSignalBridgeStats:
        """Return a snapshot copy of the current statistics."""
        return BinanceSignalBridgeStats(
            trade_events=self._stats.trade_events,
            candle_events=self._stats.candle_events,
            closed_candle_events=self._stats.closed_candle_events,
            ignored_events=self._stats.ignored_events,
            error_events=self._stats.error_events,
        )

    # -- event handler -----------------------------------------------------

    def handle_event(
        self,
        event: MarketTradeEvent | MarketCandleEvent,
    ) -> BinanceSignalInput | None:
        """Convert a canonical market event into a signal-input DTO.

        Returns
        -------
        BinanceSignalTradeInput | BinanceSignalCandleInput | None
            The corresponding signal-input DTO, or ``None`` if the event
            is of an unknown type or its fields do not match the bridge config.
        """
        try:
            if isinstance(event, MarketTradeEvent):
                return self._handle_trade(event)

            if isinstance(event, MarketCandleEvent):
                return self._handle_candle(event)

            # Unknown event type
            self._stats.ignored_events += 1
            return None

        except Exception:
            self._stats.error_events += 1
            return None

    # -- private handlers --------------------------------------------------

    def _handle_trade(self, event: MarketTradeEvent) -> BinanceSignalTradeInput | None:
        if event.canonical_symbol != self._canonical_symbol:
            self._stats.ignored_events += 1
            return None
        if event.raw_symbol != self._raw_symbol:
            self._stats.ignored_events += 1
            return None

        self._stats.trade_events += 1
        return BinanceSignalTradeInput(
            canonical_symbol=event.canonical_symbol,
            raw_symbol=event.raw_symbol,
            timestamp_ms=event.event_time_ms,
            side=event.taker_side.value,
            price=event.price,
            quantity=event.quantity,
        )

    def _handle_candle(self, event: MarketCandleEvent) -> BinanceSignalCandleInput | None:
        if event.canonical_symbol != self._canonical_symbol:
            self._stats.ignored_events += 1
            return None
        if event.raw_symbol != self._raw_symbol:
            self._stats.ignored_events += 1
            return None
        if event.timeframe != self._interval:
            self._stats.ignored_events += 1
            return None

        self._stats.candle_events += 1
        if event.is_closed:
            self._stats.closed_candle_events += 1

        return BinanceSignalCandleInput(
            canonical_symbol=event.canonical_symbol,
            raw_symbol=event.raw_symbol,
            interval=event.timeframe,
            timestamp_ms=event.open_time_ms,
            open=event.open_price,
            high=event.high_price,
            low=event.low_price,
            close=event.close_price,
            volume=event.volume,
            closed=event.is_closed,
        )
