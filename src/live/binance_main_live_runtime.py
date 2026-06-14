#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : binance_main_live_runtime.py
@Description: Binance main live trading runtime.

Uses Binance WebSocket market data (aggTrade + kline_15m) via the
existing BinanceWebSocketMarketDataFeed and BinanceMarketDataSignalBridge.
Executes real orders through BinanceLiveTrader.

No dry-run.  No shadow.  No OKX monitor.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping
from typing import Any

from src.data_feed.binance.aiohttp_ws_connector import (
    AiohttpBinanceWsConnection,
    connect_binance_market_ws,
)
from src.data_feed.binance.public_klines import (
    BinancePublicKline,
    fetch_public_klines,
)
from src.data_feed.binance.websocket_feed import BinanceWebSocketMarketDataFeed
from src.data_feed.market_events import MarketCandleEvent, MarketTradeEvent
from src.data_feed.selector import (
    SUPPORTED_BINANCE_RAW_SYMBOL,
    SUPPORTED_CANONICAL_SYMBOL,
    SUPPORTED_KLINE_INTERVAL,
)
from src.exchanges.runtime_config import load_unified_runtime_config
from src.execution.live_trader_factory import create_live_trader
from src.execution.trader import LiveTradeResult
from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig
from src.live.binance_market_data_bridge import (
    BinanceMarketDataSignalBridge,
    BinanceSignalCandleInput,
    BinanceSignalTradeInput,
)
from src.live.binance_signal_only_runtime import (
    BinanceSignalOnlyConfig,
    load_binance_signal_only_config,
    _seed_historical_klines,
    _try_recompute_boll,
    _upsert_candle_entry,
    _to_float,
)
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import (
    SimplePositionSizer,
    SimplePositionSizerConfig,
)
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategyConfig,
    TradeIntent,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)


async def run_binance_main_live(
    env: Mapping[str, str] | None = None,
) -> None:
    """Run the Binance main live trading loop.

    Connects to Binance USD-M Futures WebSocket, processes aggTrade and
    kline_15m events, computes BOLL snapshots and CVD, calls the strategy
    on_tick to generate trade intents, and executes them through
    BinanceLiveTrader.
    """
    values = os.environ if env is None else env

    # ── 1. Create and initialize trader (preflight / fail-fast) ────────
    trader = create_live_trader(values)
    await trader.start()
    try:
        await trader.initialize()
    except Exception:
        await trader.close()
        raise

    # ── 2. Load config ─────────────────────────────────────────────────
    # Reuse signal-only config infrastructure for market data params
    config = load_binance_signal_only_config(values)

    logger.warning(
        "BINANCE_MAIN_LIVE_START | exchange=binance "
        "canonical_symbol=%s raw_symbol=%s kline_interval=%s "
        "equity=%.4f position_contracts=%s",
        config.canonical_symbol,
        config.raw_symbol,
        config.kline_interval,
        trader.account_equity_usdt,
        trader.position_contracts,
    )

    # ── 3. Create components ────────────────────────────────────────────
    bridge = BinanceMarketDataSignalBridge(
        canonical_symbol=config.canonical_symbol,
        raw_symbol=config.raw_symbol,
        interval=config.kline_interval,
    )

    feed = BinanceWebSocketMarketDataFeed(
        connector=connect_binance_market_ws,
        canonical_symbol=config.canonical_symbol,
        raw_symbol=config.raw_symbol,
        kline_interval=config.kline_interval,
    )

    cvd_tracker = CvdTracker(CvdTrackerConfig.from_env())
    sizer = SimplePositionSizer(SimplePositionSizerConfig.from_env())
    strategy = BollCvdShockReclaimStrategy(
        BollCvdReclaimStrategyConfig.from_env(), sizer
    )

    # ── 4. Execution queue ─────────────────────────────────────────────
    execution_queue: asyncio.Queue[TradeIntent] = asyncio.Queue(
        maxsize=int(values.get("EXECUTION_QUEUE_MAXSIZE", "1000"))
    )

    # ── 5. State ───────────────────────────────────────────────────────
    candle_buffer: list[dict] = []
    current_boll: BollSnapshot | None = None
    total_events: int = 0
    last_heartbeat_monotonic: float = 0.0
    start_monotonic: float = asyncio.get_event_loop().time()
    trading_halted: bool = False

    # ── 6. Seed historical klines ──────────────────────────────────────
    if config.seed_historical_klines:
        current_boll = await _seed_historical_klines(
            candle_buffer=candle_buffer,
            config=config,
        )

    # ── 7. Execution worker ────────────────────────────────────────────
    async def execution_worker() -> None:
        nonlocal trading_halted
        while True:
            intent = await execution_queue.get()
            if trading_halted:
                logger.warning(
                    "BINANCE_MAIN_EXECUTION_HALTED | intent=%s",
                    intent.intent_type,
                )
                continue

            try:
                result: LiveTradeResult = await trader.execute_intent(intent)
                level = logging.WARNING if result.ok else logging.ERROR
                logger.log(
                    level,
                    "BINANCE_MAIN_EXECUTION | type=%s side=%s ok=%s entry_filled=%s "
                    "tp_ok=%s sl_ok=%s order_id=%s tp_order_id=%s sl_order_id=%s "
                    "contracts=%s message=%s",
                    intent.intent_type,
                    intent.side,
                    result.ok,
                    result.entry_filled,
                    result.tp_ok,
                    result.protective_sl_ok,
                    result.order_id,
                    result.tp_order_id,
                    result.protective_sl_order_id,
                    result.contracts,
                    result.message,
                )

                if not result.ok:
                    # Halt on failure — existing failure handler will manage
                    trading_halted = True
                    logger.error(
                        "BINANCE_MAIN_TRADING_HALTED | intent=%s message=%s",
                        intent.intent_type,
                        result.message,
                    )
            except Exception as exc:
                logger.exception(
                    "BINANCE_MAIN_EXECUTION_EXCEPTION | intent=%s error=%s",
                    intent.intent_type,
                    exc,
                )
                trading_halted = True

    worker_task = asyncio.create_task(execution_worker())

    # ── 8. Main WebSocket loop ─────────────────────────────────────────
    connection: AiohttpBinanceWsConnection | None = None

    try:
        connection = await connect_binance_market_ws(feed.stream_url())

        async for message in connection:
            event = feed.map_message(message)
            if event is None:
                continue

            total_events += 1
            signal_input = bridge.handle_event(event)

            if isinstance(event, MarketCandleEvent):
                await _handle_candle(
                    event=event,
                    signal_input=signal_input,
                    candle_buffer=candle_buffer,
                    config=config,
                )

                next_boll = _try_recompute_boll(
                    candle_buffer=candle_buffer,
                    config=config,
                )

                current_boll = next_boll

            elif isinstance(event, MarketTradeEvent):
                await _handle_trade(
                    event=event,
                    signal_input=signal_input,
                    cvd_tracker=cvd_tracker,
                    strategy=strategy,
                    current_boll=current_boll,
                    execution_queue=execution_queue,
                )

            # ── Heartbeat ───────────────────────────────────────────────
            now_mono = asyncio.get_event_loop().time()
            if now_mono - last_heartbeat_monotonic >= config.heartbeat_seconds:
                last_heartbeat_monotonic = now_mono
                _log_main_heartbeat(
                    bridge=bridge,
                    current_boll=current_boll,
                    total_events=total_events,
                    elapsed=now_mono - start_monotonic,
                    trader=trader,
                    trading_halted=trading_halted,
                )

            # ── Exit checks ─────────────────────────────────────────────
            if total_events >= config.max_events:
                logger.warning(
                    "BINANCE_MAIN_LIVE_DONE | reason=max_events "
                    "total_events=%s max_events=%s",
                    total_events,
                    config.max_events,
                )
                break

            elapsed = now_mono - start_monotonic
            if elapsed >= config.duration_seconds:
                logger.warning(
                    "BINANCE_MAIN_LIVE_DONE | reason=duration "
                    "elapsed=%.1fs duration=%ss total_events=%s",
                    elapsed,
                    config.duration_seconds,
                    total_events,
                )
                break

    except asyncio.CancelledError:
        logger.warning("BINANCE_MAIN_LIVE_DONE | reason=cancelled")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        if connection is not None:
            try:
                await connection.close()
            except Exception:
                pass

        await trader.close()


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


async def _handle_candle(
    *,
    event: MarketCandleEvent,
    signal_input,
    candle_buffer: list[dict],
    config: BinanceSignalOnlyConfig,
) -> None:
    """Process a candle event: update buffer."""
    candle_entry = {
        "ts_ms": event.open_time_ms,
        "open": _to_float(event.open_price),
        "high": _to_float(event.high_price),
        "low": _to_float(event.low_price),
        "close": _to_float(event.close_price),
        "volume": _to_float(event.volume),
        "closed": event.is_closed,
    }
    _upsert_candle_entry(
        candle_buffer=candle_buffer,
        candle_entry=candle_entry,
        candle_limit=config.candle_limit,
    )


async def _handle_trade(
    *,
    event: MarketTradeEvent,
    signal_input,
    cvd_tracker: CvdTracker,
    strategy: BollCvdShockReclaimStrategy,
    current_boll: BollSnapshot | None,
    execution_queue: asyncio.Queue,
) -> None:
    """Process a trade event: update CVD, call strategy, enqueue intents."""
    if not isinstance(signal_input, BinanceSignalTradeInput):
        return

    price_f = _to_float(event.price)
    size_f = _to_float(event.quantity)
    side = event.taker_side.value
    ts_ms = event.event_time_ms

    cvd_snapshot = cvd_tracker.update(
        side=side, size=size_f, price=price_f, ts_ms=ts_ms
    )

    if current_boll is None:
        return

    # Real strategy call — state IS modified
    intents = strategy.on_tick(
        price=price_f,
        ts_ms=ts_ms,
        boll=current_boll,
        cvd=cvd_snapshot,
    )

    for intent in intents:
        logger.warning(
            "BINANCE_MAIN_INTENT | type=%s side=%s price=%.4f "
            "layer=%s reason=%s",
            intent.intent_type,
            intent.side,
            intent.price,
            intent.layer_index,
            intent.reason,
        )
        await execution_queue.put(intent)


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def _log_main_heartbeat(
    *,
    bridge: BinanceMarketDataSignalBridge,
    current_boll: BollSnapshot | None,
    total_events: int,
    elapsed: float,
    trader: Any,
    trading_halted: bool,
) -> None:
    """Log a periodic heartbeat with exchange=binance key fields."""
    stats = bridge.get_stats()
    if current_boll is not None:
        logger.warning(
            "BINANCE_MAIN_HEARTBEAT | exchange=binance "
            "elapsed=%.1fs events=%s "
            "price=%.4f boll_middle=%.4f boll_upper=%.4f boll_lower=%.4f "
            "cvd=%.4f position_contracts=%s "
            "bridge_trades=%s bridge_candles=%s bridge_errors=%s "
            "halted=%s",
            elapsed,
            total_events,
            current_boll.close,
            current_boll.middle,
            current_boll.upper,
            current_boll.lower,
            trader.position_contracts,
            stats.trade_events,
            stats.candle_events,
            stats.error_events,
            trading_halted,
        )
    else:
        logger.warning(
            "BINANCE_MAIN_HEARTBEAT | exchange=binance "
            "elapsed=%.1fs events=%s "
            "boll=not_ready position_contracts=%s "
            "bridge_trades=%s bridge_candles=%s bridge_errors=%s "
            "halted=%s",
            elapsed,
            total_events,
            trader.position_contracts,
            stats.trade_events,
            stats.candle_events,
            stats.error_events,
            trading_halted,
        )
