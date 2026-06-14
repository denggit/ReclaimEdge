#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : okx_market_data_client.py
@Description: OKX implementation of MarketDataClientPort.

This class wraps an existing BollBandBreakoutMonitor instance.
It is NOT wired into production yet.
It does not create the monitor, read env, or open connections by itself.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.data_feed.market_data_client_port import (
    CandleSnapshot,
    MarketDataClientPort,
    MarketDataEvent,
    MarketTradeSnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.monitors.boll_band_breakout_monitor import (
        BollBandBreakoutMonitor,
        Candle,
        MarketTickEvent,
    )


class OkxMarketDataClient(MarketDataClientPort):
    """OKX implementation of MarketDataClientPort.

    This class wraps an existing BollBandBreakoutMonitor instance.
    It is not wired into production yet.
    It does not create the monitor, read env, or open connections by itself.
    """

    def __init__(self, monitor: BollBandBreakoutMonitor) -> None:
        self._monitor = monitor

    # ------------------------------------------------------------------
    # MarketDataClientPort methods
    # ------------------------------------------------------------------

    async def fetch_recent_klines(self, *, limit: int) -> list[CandleSnapshot]:
        """Fetch recent klines from the wrapped monitor's REST client.

        Returns the last *limit* candles mapped to ``CandleSnapshot`` DTOs.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")

        candles: list[Candle] = await self._monitor.client.fetch_candles(
            include_live=self._monitor.config.use_live_candle,
        )

        candles = candles[-limit:]

        bar_interval_ms: int = getattr(self._monitor, "_bar_interval_ms", 0)
        if bar_interval_ms <= 0:
            bar_interval_ms = 0

        result: list[CandleSnapshot] = []
        for candle in candles:
            result.append(
                CandleSnapshot(
                    open_time_ms=candle.ts_ms,
                    close_time_ms=candle.ts_ms + bar_interval_ms if bar_interval_ms > 0 else candle.ts_ms,
                    open_price=Decimal(str(candle.open)),
                    high_price=Decimal(str(candle.high)),
                    low_price=Decimal(str(candle.low)),
                    close_price=Decimal(str(candle.close)),
                    volume=Decimal(str(candle.volume)),
                    is_closed=candle.confirmed,
                    raw={
                        "inst_id": self._monitor.config.inst_id,
                        "bar": self._monitor.config.bar,
                    },
                )
            )
        return result

    async def stream_market_events(
        self,
        on_event: Callable[[MarketDataEvent], Awaitable[None]],
    ) -> None:
        """Stream market trade events via the wrapped monitor's tick handlers.

        Registers a tick handler that maps legacy ``MarketTickEvent`` to
        ``MarketTradeSnapshot``, then delegates to ``monitor.run_forever()``.
        """

        async def _tick_handler(event: MarketTickEvent) -> None:
            tick = event.tick
            snapshot = MarketTradeSnapshot(
                event_time_ms=tick.ts_ms,
                price=Decimal(str(tick.price)),
                qty=Decimal(str(tick.size)),
                side=tick.side,
                raw={"inst_id": tick.inst_id},
            )
            await on_event(snapshot)

        self._monitor.add_tick_handler(_tick_handler)
        await self._monitor.run_forever()

    async def close(self) -> None:
        """Stop the monitor loop and close the underlying REST client."""
        if hasattr(self._monitor, "_running"):
            self._monitor._running = False

        client: Any = getattr(self._monitor, "client", None)
        if client is not None and hasattr(client, "close"):
            await client.close()
