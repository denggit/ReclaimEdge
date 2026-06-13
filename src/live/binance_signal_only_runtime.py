#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : binance_signal_only_runtime.py
@Description: Binance signal-only live market data observation runtime.

This module provides a lightweight runtime that connects to the Binance
USD-M Futures WebSocket, receives aggTrade / kline_15m events, converts
them through the market-data bridge, maintains a candle buffer, computes
BOLL snapshots, updates CVD, and calls the strategy on_tick to generate
theoretical trade intents — but NEVER places any order.

It is deliberately narrow:

* Only ETH-USDT-PERP / ETHUSDT / 15m.
* Signal-only — no Trader, no execution engine, no broker, no API keys.
* Strategy state is deep-copied before on_tick and restored afterward,
  so running signal-only for hours never produces fake positions.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Mapping

from src.data_feed.binance.aiohttp_ws_connector import (
    AiohttpBinanceWsConnection,
    connect_binance_market_ws,
)
from src.data_feed.binance.websocket_feed import BinanceWebSocketMarketDataFeed
from src.data_feed.market_events import MarketCandleEvent, MarketTradeEvent
from src.data_feed.selector import (
    SUPPORTED_BINANCE_RAW_SYMBOL,
    SUPPORTED_CANONICAL_SYMBOL,
    SUPPORTED_KLINE_INTERVAL,
)
from src.exchanges.models import ExchangeName
from src.exchanges.runtime_config import load_unified_runtime_config
from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig
from src.live.binance_market_data_bridge import (
    BinanceMarketDataSignalBridge,
    BinanceSignalCandleInput,
    BinanceSignalTradeInput,
)
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import (
    SimplePositionSizer,
    SimplePositionSizerConfig,
)
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public runtime env keys (whitelist — no API secrets)
# ---------------------------------------------------------------------------

_PUBLIC_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "EXCHANGE",
    "TRADE_ASSET",
    "QUOTE_ASSET",
    "MARKET_TYPE",
    "MARGIN_MODE",
    "POSITION_MODE",
    "LEVERAGE",
    "KLINE_INTERVAL",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinanceSignalOnlyConfig:
    """Non-sensitive configuration for the Binance signal-only runtime."""

    canonical_symbol: str
    raw_symbol: str
    kline_interval: str
    duration_seconds: float
    max_events: int
    heartbeat_seconds: float
    candle_limit: int
    boll_window: int
    boll_std_multiplier: float
    band_distance_threshold_pct: float
    tp_boll_enabled: bool
    tp_boll_window: int


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _read_positive_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a float, got {raw!r}")
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def _read_positive_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got {raw!r}")
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def _read_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _read_non_negative_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a float, got {raw!r}")
    if value < 0:
        raise ValueError(f"{key} must be non-negative, got {value}")
    return value


def load_binance_signal_only_config(
    env: Mapping[str, str] | None = None,
) -> BinanceSignalOnlyConfig:
    """Load a validated :class:`BinanceSignalOnlyConfig` from environment.

    Only non-sensitive public env keys are forwarded to the unified runtime
    config loader.  API credentials are explicitly excluded.

    Parameters
    ----------
    env:
        Optional env mapping.  Uses ``os.environ`` when omitted.

    Returns
    -------
    BinanceSignalOnlyConfig

    Raises
    ------
    ValueError
        If EXCHANGE is not ``binance``.
    RuntimeError
        If ``BINANCE_SIGNAL_ONLY`` is not truthy.
    """
    values = os.environ if env is None else env

    # Build a public-only env mapping for the unified loader.
    public_env: dict[str, str] = {}
    for key in _PUBLIC_RUNTIME_ENV_KEYS:
        val = values.get(key)
        if val is not None:
            public_env[key] = val

    # The unified loader will get empty strings for api_key/api_secret/api_passphrase
    # because those keys are absent from public_env.
    runtime_config = load_unified_runtime_config(public_env)

    # --- Guard: exchange must be binance ---
    if runtime_config.exchange != ExchangeName.BINANCE:
        raise ValueError(
            f"Binance signal-only runtime requires EXCHANGE=binance, "
            f"got {runtime_config.exchange.value!r}"
        )

    # --- Guard: only ETH-USDT-PERP / ETHUSDT / 15m ---
    if runtime_config.canonical_symbol != SUPPORTED_CANONICAL_SYMBOL:
        raise ValueError(
            f"Binance signal-only runtime only supports "
            f"{SUPPORTED_CANONICAL_SYMBOL!r}, got "
            f"{runtime_config.canonical_symbol!r}"
        )

    binance_symbol = runtime_config.binance_symbol
    if binance_symbol != SUPPORTED_BINANCE_RAW_SYMBOL:
        raise ValueError(
            f"Binance signal-only runtime only supports "
            f"{SUPPORTED_BINANCE_RAW_SYMBOL!r}, got {binance_symbol!r}"
        )

    if runtime_config.kline_interval != SUPPORTED_KLINE_INTERVAL:
        raise ValueError(
            f"Binance signal-only runtime only supports "
            f"{SUPPORTED_KLINE_INTERVAL!r}, got "
            f"{runtime_config.kline_interval!r}"
        )

    # --- Guard: signal-only must be enabled ---
    signal_only = _read_bool(values, "BINANCE_SIGNAL_ONLY", False)
    if not signal_only:
        raise RuntimeError(
            "Binance main live trading is not wired yet. "
            "Set BINANCE_SIGNAL_ONLY=true for signal-only observation."
        )

    # --- Build config from env ---
    return BinanceSignalOnlyConfig(
        canonical_symbol=runtime_config.canonical_symbol,
        raw_symbol=binance_symbol,
        kline_interval=runtime_config.kline_interval,
        duration_seconds=_read_positive_float(
            values, "BINANCE_SIGNAL_ONLY_SECONDS", 3600.0
        ),
        max_events=_read_positive_int(
            values, "BINANCE_SIGNAL_ONLY_MAX_EVENTS", 100000
        ),
        heartbeat_seconds=_read_positive_float(
            values, "BINANCE_SIGNAL_ONLY_HEARTBEAT_SECONDS", 30.0
        ),
        candle_limit=_read_positive_int(values, "CANDLE_LIMIT", 100),
        boll_window=_read_positive_int(values, "BOLL_WINDOW", 20),
        boll_std_multiplier=_read_non_negative_float(
            values, "BOLL_STD_MULTIPLIER", 2.0
        ),
        band_distance_threshold_pct=_read_non_negative_float(
            values, "BOLL_DISTANCE_THRESHOLD_PCT", 0.005
        ),
        tp_boll_enabled=_read_bool(values, "TP_BOLL_ENABLED", True),
        tp_boll_window=_read_positive_int(values, "TP_BOLL_WINDOW", 15),
    )


# ---------------------------------------------------------------------------
# BOLL helpers
# ---------------------------------------------------------------------------


def _calculate_boll(
    closes: list[float],
    window: int,
    std_multiplier: float,
) -> tuple[float, float, float]:
    """Calculate BOLL middle, upper, lower from a list of close prices."""
    if len(closes) < window:
        raise ValueError(f"Not enough closes: {len(closes)} < {window}")
    recent = closes[-window:]
    middle = mean(recent)
    std = pstdev(recent)
    return middle, middle + std_multiplier * std, middle - std_multiplier * std


def _build_boll_snapshot(
    *,
    raw_symbol: str,
    closes: list[float],
    latest_candle: dict,
    config: BinanceSignalOnlyConfig,
) -> BollSnapshot:
    """Build a :class:`BollSnapshot` from the current candle buffer."""
    middle, upper, lower = _calculate_boll(
        closes, config.boll_window, config.boll_std_multiplier
    )
    upper_distance_pct = abs(upper - middle) / middle
    lower_distance_pct = abs(middle - lower) / middle
    alert_switch_on = (
        upper_distance_pct >= config.band_distance_threshold_pct
        or lower_distance_pct >= config.band_distance_threshold_pct
    )

    # TP-only BOLL
    tp_lower: float | None = None
    tp_middle: float | None = None
    tp_upper: float | None = None
    tp_window: int | None = None
    if config.tp_boll_enabled and config.tp_boll_window > 0:
        tp_window = config.tp_boll_window
        if tp_window != config.boll_window and len(closes) >= tp_window:
            try:
                tp_mid, tp_up, tp_lo = _calculate_boll(
                    closes, tp_window, config.boll_std_multiplier
                )
                tp_middle, tp_upper, tp_lower = tp_mid, tp_up, tp_lo
            except ValueError:
                tp_lower = tp_middle = tp_upper = tp_window = None

    return BollSnapshot(
        inst_id=raw_symbol,
        candle_ts_ms=latest_candle["ts_ms"],
        close=latest_candle["close"],
        middle=middle,
        upper=upper,
        lower=lower,
        upper_distance_pct=upper_distance_pct,
        lower_distance_pct=lower_distance_pct,
        alert_switch_on=alert_switch_on,
        live_mode=True,
        tp_lower=tp_lower,
        tp_middle=tp_middle,
        tp_upper=tp_upper,
        tp_window=tp_window,
    )


# ---------------------------------------------------------------------------
# Helpers for float conversion
# ---------------------------------------------------------------------------


def _to_float(value) -> float:
    """Convert a Decimal or float to float."""
    # Decimal supports __float__, but be explicit for safety.
    return float(value)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


async def run_binance_signal_only(
    env: Mapping[str, str] | None = None,
) -> None:
    """Run the Binance signal-only market data observation loop.

    Connects to Binance USD-M Futures WebSocket, processes aggTrade and
    kline_15m events, maintains BOLL snapshots and CVD, and logs
    theoretical trade intents — but NEVER places any order.

    Parameters
    ----------
    env:
        Optional env mapping.  Uses ``os.environ`` when omitted.
    """
    values = os.environ if env is None else env

    # --- Load config ---
    config = load_binance_signal_only_config(values)
    logger.warning(
        "BINANCE_SIGNAL_ONLY_START | exchange=binance "
        "canonical_symbol=%s raw_symbol=%s kline_interval=%s "
        "duration=%ss max_events=%s heartbeat=%ss",
        config.canonical_symbol,
        config.raw_symbol,
        config.kline_interval,
        config.duration_seconds,
        config.max_events,
        config.heartbeat_seconds,
    )
    logger.warning(
        "BINANCE_SIGNAL_ONLY_CONFIG_OK | boll_window=%s boll_std=%.2f "
        "band_threshold=%.4f tp_boll=%s tp_window=%s candle_limit=%s",
        config.boll_window,
        config.boll_std_multiplier,
        config.band_distance_threshold_pct,
        config.tp_boll_enabled,
        config.tp_boll_window,
        config.candle_limit,
    )

    # --- Create components ---
    bridge = BinanceMarketDataSignalBridge(
        canonical_symbol=config.canonical_symbol,
        raw_symbol=config.raw_symbol,
        interval=config.kline_interval,
    )

    # Build the WebSocket market data feed using the shared aiohttp connector.
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

    # --- State ---
    candle_buffer: list[dict] = []  # list of {"ts_ms": int, "close": float, ...}
    current_boll: BollSnapshot | None = None
    total_events: int = 0
    last_heartbeat_monotonic: float = 0.0
    start_monotonic: float = asyncio.get_event_loop().time()

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
                    bridge=bridge,
                    candle_buffer=candle_buffer,
                    config=config,
                )
                # Recompute BOLL after every candle event
                current_boll = _try_recompute_boll(
                    candle_buffer=candle_buffer,
                    config=config,
                )

            elif isinstance(event, MarketTradeEvent):
                await _handle_trade(
                    event=event,
                    signal_input=signal_input,
                    bridge=bridge,
                    cvd_tracker=cvd_tracker,
                    strategy=strategy,
                    current_boll=current_boll,
                )

            # --- Heartbeat ---
            now_mono = asyncio.get_event_loop().time()
            if now_mono - last_heartbeat_monotonic >= config.heartbeat_seconds:
                last_heartbeat_monotonic = now_mono
                _log_heartbeat(
                    bridge=bridge,
                    current_boll=current_boll,
                    total_events=total_events,
                    elapsed=now_mono - start_monotonic,
                )

            # --- Exit checks ---
            if total_events >= config.max_events:
                logger.warning(
                    "BINANCE_SIGNAL_ONLY_DONE | reason=max_events "
                    "total_events=%s max_events=%s",
                    total_events,
                    config.max_events,
                )
                break

            elapsed = now_mono - start_monotonic
            if elapsed >= config.duration_seconds:
                logger.warning(
                    "BINANCE_SIGNAL_ONLY_DONE | reason=duration "
                    "elapsed=%.1fs duration=%ss total_events=%s",
                    elapsed,
                    config.duration_seconds,
                    total_events,
                )
                break

    except asyncio.CancelledError:
        logger.warning("BINANCE_SIGNAL_ONLY_DONE | reason=cancelled")
    finally:
        if connection is not None:
            try:
                await connection.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


async def _handle_candle(
    *,
    event: MarketCandleEvent,
    signal_input,
    bridge: BinanceMarketDataSignalBridge,
    candle_buffer: list[dict],
    config: BinanceSignalOnlyConfig,
) -> None:
    """Process a candle event: update buffer, optionally log."""
    candle_entry = {
        "ts_ms": event.open_time_ms,
        "open": _to_float(event.open_price),
        "high": _to_float(event.high_price),
        "low": _to_float(event.low_price),
        "close": _to_float(event.close_price),
        "volume": _to_float(event.volume),
        "closed": event.is_closed,
    }
    candle_buffer.append(candle_entry)

    # Keep only the most recent candles up to candle_limit
    while len(candle_buffer) > config.candle_limit:
        candle_buffer.pop(0)

    if isinstance(signal_input, BinanceSignalCandleInput):
        level = (
            logging.WARNING if event.is_closed else logging.DEBUG
        )
        if event.is_closed:
            logger.warning(
                "BINANCE_SIGNAL_ONLY_CLOSED_CANDLE | "
                "ts_ms=%s open=%.4f high=%.4f low=%.4f close=%.4f vol=%.4f "
                "buffer_size=%s",
                event.open_time_ms,
                _to_float(event.open_price),
                _to_float(event.high_price),
                _to_float(event.low_price),
                _to_float(event.close_price),
                _to_float(event.volume),
                len(candle_buffer),
            )
        else:
            logger.log(
                level,
                "BINANCE_SIGNAL_ONLY_CANDLE | ts_ms=%s close=%.4f closed=%s",
                event.open_time_ms,
                _to_float(event.close_price),
                event.is_closed,
            )


async def _handle_trade(
    *,
    event: MarketTradeEvent,
    signal_input,
    bridge: BinanceMarketDataSignalBridge,
    cvd_tracker: CvdTracker,
    strategy: BollCvdShockReclaimStrategy,
    current_boll: BollSnapshot | None,
) -> None:
    """Process a trade event: update CVD, optionally call strategy."""
    if not isinstance(signal_input, BinanceSignalTradeInput):
        return

    price_f = _to_float(event.price)
    size_f = _to_float(event.quantity)
    side = event.taker_side.value
    ts_ms = event.event_time_ms

    # Update CVD regardless of BOLL readiness
    cvd_snapshot = cvd_tracker.update(
        side=side, size=size_f, price=price_f, ts_ms=ts_ms
    )

    # Log trade detail (debug level to avoid spam)
    logger.debug(
        "BINANCE_SIGNAL_ONLY_TRADE | price=%.4f side=%s size=%.6f ts_ms=%s",
        price_f,
        side,
        size_f,
        ts_ms,
    )

    if current_boll is None:
        # No BOLL snapshot yet — skip strategy
        return

    # Deep-copy strategy state, call on_tick, restore state.
    # This ensures signal-only mode never produces fake positions.
    backup_state = copy.deepcopy(strategy.state)
    try:
        intents = strategy.on_tick(
            price=price_f,
            ts_ms=ts_ms,
            boll=current_boll,
            cvd=cvd_snapshot,
        )
    finally:
        strategy.state = backup_state

    for intent in intents:
        logger.warning(
            "BINANCE_SIGNAL_ONLY_INTENT | type=%s side=%s price=%.4f "
            "layer=%s reason=%s fast_cvd=%.4f buy_ratio=%.2f sell_ratio=%.2f "
            "boll_upper=%.4f boll_middle=%.4f boll_lower=%.4f",
            intent.intent_type,
            intent.side,
            intent.price,
            intent.layer_index,
            intent.reason,
            intent.fast_cvd,
            intent.buy_ratio,
            intent.sell_ratio,
            intent.boll_upper,
            intent.boll_middle,
            intent.boll_lower,
        )


# ---------------------------------------------------------------------------
# BOLL recompute
# ---------------------------------------------------------------------------


def _try_recompute_boll(
    *,
    candle_buffer: list[dict],
    config: BinanceSignalOnlyConfig,
) -> BollSnapshot | None:
    """Try to recompute the BOLL snapshot from the candle buffer.

    Returns None if there are not enough candles.
    """
    if len(candle_buffer) < config.boll_window:
        logger.debug(
            "BINANCE_SIGNAL_ONLY_CANDLE | buffer_size=%s < boll_window=%s",
            len(candle_buffer),
            config.boll_window,
        )
        return None

    closes = [c["close"] for c in candle_buffer]
    latest = candle_buffer[-1]

    try:
        boll = _build_boll_snapshot(
            raw_symbol=config.raw_symbol,
            closes=closes,
            latest_candle=latest,
            config=config,
        )
    except ValueError:
        return None

    logger.info(
        "BINANCE_SIGNAL_ONLY_BOLL_READY | close=%.4f middle=%.4f "
        "upper=%.4f lower=%.4f upper_dist=%.4f%% lower_dist=%.4f%% "
        "switch=%s buffer_size=%s",
        boll.close,
        boll.middle,
        boll.upper,
        boll.lower,
        boll.upper_distance_pct * 100,
        boll.lower_distance_pct * 100,
        boll.alert_switch_on,
        len(candle_buffer),
    )

    return boll


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def _log_heartbeat(
    *,
    bridge: BinanceMarketDataSignalBridge,
    current_boll: BollSnapshot | None,
    total_events: int,
    elapsed: float,
) -> None:
    """Log a periodic heartbeat with current state."""
    stats = bridge.get_stats()
    if current_boll is not None:
        logger.warning(
            "BINANCE_SIGNAL_ONLY_HEARTBEAT | elapsed=%.1fs events=%s "
            "price=%.4f boll_middle=%.4f boll_upper=%.4f boll_lower=%.4f "
            "bridge_trades=%s bridge_candles=%s bridge_closed=%s "
            "bridge_ignored=%s bridge_errors=%s",
            elapsed,
            total_events,
            current_boll.close,
            current_boll.middle,
            current_boll.upper,
            current_boll.lower,
            stats.trade_events,
            stats.candle_events,
            stats.closed_candle_events,
            stats.ignored_events,
            stats.error_events,
        )
    else:
        logger.warning(
            "BINANCE_SIGNAL_ONLY_HEARTBEAT | elapsed=%.1fs events=%s "
            "boll=not_ready "
            "bridge_trades=%s bridge_candles=%s bridge_closed=%s "
            "bridge_ignored=%s bridge_errors=%s",
            elapsed,
            total_events,
            stats.trade_events,
            stats.candle_events,
            stats.closed_candle_events,
            stats.ignored_events,
            stats.error_events,
        )
