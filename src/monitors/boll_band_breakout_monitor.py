from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from statistics import mean, pstdev
from typing import Awaitable, Callable, Literal, Optional

from src.data_feed.market_data_client_port import (
    CandleSnapshot,
    MarketDataClientPort,
    MarketDataEvent,
    MarketTradeSnapshot,
)
from src.utils.log import get_logger

logger = get_logger(__name__)
PriceZone = Literal["INSIDE", "ABOVE", "BELOW", "UNKNOWN"]
BreakoutDirection = Literal["BREAK_UPPER", "BREAK_LOWER"]


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class BollBandBreakoutMonitorConfig:
    inst_id: str = "ETH-USDT-SWAP"
    bar: str = "15m"
    boll_window: int = 20
    boll_std_multiplier: float = 2.0
    band_distance_threshold_pct: float = 0.005
    alert_freeze_seconds: int = 3600
    use_live_candle: bool = True
    boll_recalc_seconds: float = 1.0
    candle_poll_seconds: int = 30
    candle_limit: int = 100
    rest_base_url: str = "https://www.okx.com"
    ws_public_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    # TP-only BOLL window (15) used exclusively for take-profit prices.
    # BOLL_WINDOW=20 remains the structure window for entry/add/risk/SL/runner.
    tp_boll_enabled: bool = True
    tp_boll_window: int = 15

    @classmethod
    def from_env(cls) -> "BollBandBreakoutMonitorConfig":
        return cls(
            inst_id=os.getenv("OKX_INST_ID", "ETH-USDT-SWAP"),
            bar=os.getenv("OKX_BAR", "15m"),
            boll_window=int(os.getenv("BOLL_WINDOW", "20")),
            boll_std_multiplier=float(os.getenv("BOLL_STD_MULTIPLIER", "2.0")),
            band_distance_threshold_pct=float(os.getenv("BOLL_DISTANCE_THRESHOLD_PCT", "0.005")),
            alert_freeze_seconds=int(os.getenv("ALERT_FREEZE_SECONDS", "3600")),
            use_live_candle=env_bool("BOLL_USE_LIVE_CANDLE", True),
            boll_recalc_seconds=float(os.getenv("BOLL_RECALC_SECONDS", "1")),
            candle_poll_seconds=int(os.getenv("CANDLE_POLL_SECONDS", "30")),
            candle_limit=int(os.getenv("CANDLE_LIMIT", "100")),
            tp_boll_enabled=env_bool("TP_BOLL_ENABLED", True),
            tp_boll_window=int(os.getenv("TP_BOLL_WINDOW", "15")),
        )


@dataclass(frozen=True)
class Candle:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirmed: bool


@dataclass(frozen=True)
class BollSnapshot:
    inst_id: str
    candle_ts_ms: int
    close: float
    middle: float
    upper: float
    lower: float
    upper_distance_pct: float
    lower_distance_pct: float
    alert_switch_on: bool
    live_mode: bool
    # TP-only BOLL bands (TP_BOLL_WINDOW, e.g. 15) used exclusively for take-profit prices.
    # lower/middle/upper always use structure BOLL (BOLL_WINDOW, e.g. 20).
    # When tp_* fields are None, all TP prices fall back to structure BOLL.
    tp_lower: float | None = None
    tp_middle: float | None = None
    tp_upper: float | None = None
    tp_window: int | None = None
    # Latest candle's high and low — used by strategies for pivot detection.
    # These are the confirmed closed-candle values when candle has closed,
    # or the live-candle values when use_live_candle=True.
    high: float | None = None
    low: float | None = None


@dataclass(frozen=True)
class TradeTick:
    inst_id: str
    price: float
    size: float
    side: str
    ts_ms: int


@dataclass(frozen=True)
class MarketTickEvent:
    tick: TradeTick
    boll: Optional[BollSnapshot]


@dataclass(frozen=True)
class BreakoutSignal:
    inst_id: str
    direction: BreakoutDirection
    price: float
    previous_price: float
    tick_ts_ms: int
    candle_ts_ms: int
    middle: float
    upper: float
    lower: float
    upper_distance_pct: float
    lower_distance_pct: float
    freeze_seconds: int


SignalHandler = Callable[[BreakoutSignal], Awaitable[None] | None]
TickHandler = Callable[[MarketTickEvent], Awaitable[None] | None]


class BollCalculator:
    @staticmethod
    def calculate(closes: list[float], window: int, std_multiplier: float) -> tuple[float, float, float]:
        if len(closes) < window:
            raise ValueError(f"Not enough closes: {len(closes)} < {window}")
        recent = closes[-window:]
        middle = mean(recent)
        std = pstdev(recent)
        return middle, middle + std_multiplier * std, middle - std_multiplier * std


def _candle_from_snapshot(snapshot: CandleSnapshot) -> Candle:
    """Convert a MarketDataClientPort CandleSnapshot into a monitor-internal Candle."""
    return Candle(
        ts_ms=snapshot.open_time_ms,
        open=float(snapshot.open_price),
        high=float(snapshot.high_price),
        low=float(snapshot.low_price),
        close=float(snapshot.close_price),
        volume=float(snapshot.volume),
        confirmed=snapshot.is_closed,
    )


class BollBandBreakoutMonitor:
    def __init__(
            self,
            config: BollBandBreakoutMonitorConfig,
            handlers: Optional[list[SignalHandler]] = None,
            tick_handlers: Optional[list[TickHandler]] = None,
            market_data_client: MarketDataClientPort | None = None,
    ):
        if market_data_client is None:
            raise ValueError("BollBandBreakoutMonitor requires market_data_client")
        self.config = config
        self.market_data_client = market_data_client
        self.handlers = handlers or []
        self.tick_handlers = tick_handlers or []
        self._candles: list[Candle] = []
        self._candles_lock = asyncio.Lock()
        self._snapshot: Optional[BollSnapshot] = None
        self._latest_candle_ts_ms: Optional[int] = None
        self._latest_switch_state: Optional[bool] = None
        self._previous_price: Optional[float] = None
        self._previous_zone: PriceZone = "UNKNOWN"
        self._freeze_until_ts: float = 0.0
        self._running = False
        self._bar_interval_ms = self._parse_bar_interval_ms(config.bar)
        self._tick_queue_maxsize = int(os.getenv("TICK_EVENT_QUEUE_MAXSIZE", "10000"))
        self._tick_event_queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue(maxsize=self._tick_queue_maxsize)
        self._last_tick_backlog_log_ts: float = 0.0
        self._tick_backlog_log_seconds = float(os.getenv("TICK_EVENT_QUEUE_BACKLOG_LOG_SECONDS", "30"))
        self._candle_sync_consecutive_failures: int = 0
        self._candle_sync_started_ts_ms: int = self._now_ms()
        self._last_successful_candle_sync_ts_ms: int = 0
        self._last_candle_sync_error_log_ts_ms: int = 0
        self._last_candle_sync_stale_log_ts_ms: int = 0
        self._last_boll_stale_log_ts_ms: int = 0
        self._candle_sync_error_log_interval_seconds = float(os.getenv("CANDLE_SYNC_ERROR_LOG_INTERVAL_SECONDS", "300"))
        self._candle_sync_stale_warn_seconds = float(os.getenv("CANDLE_SYNC_STALE_WARN_SECONDS", "180"))
        self._candle_sync_max_backoff_seconds = float(os.getenv("CANDLE_SYNC_MAX_BACKOFF_SECONDS", "60"))

    def add_handler(self, handler: SignalHandler) -> None:
        self.handlers.append(handler)

    def add_tick_handler(self, handler: TickHandler) -> None:
        self.tick_handlers.append(handler)

    async def run_forever(self) -> None:
        self._running = True
        try:
            await asyncio.gather(
                self._candle_sync_loop(),
                self._boll_recalc_loop(),
                self._tick_loop(),
                self._tick_event_consumer_loop(),
            )
        finally:
            self._running = False
            await self.market_data_client.close()

    async def _candle_sync_loop(self) -> None:
        while self._running:
            sleep_seconds = await self._run_candle_sync_once()
            await asyncio.sleep(sleep_seconds)

    async def _run_candle_sync_once(self) -> float:
        try:
            await self._sync_candles_from_rest()
            self._handle_candle_sync_success()
            return float(self.config.candle_poll_seconds)
        except Exception as exc:
            return self._handle_candle_sync_failure(exc)

    async def _sync_candles_from_rest(self) -> None:
        snapshots = await self.market_data_client.fetch_recent_klines(limit=self.config.candle_limit)
        candles = [_candle_from_snapshot(snapshot) for snapshot in snapshots]
        if len(candles) < self.config.boll_window:
            logger.warning("Not enough candles for BOLL: %s < %s", len(candles), self.config.boll_window)
            return
        async with self._candles_lock:
            self._candles = candles[-self.config.candle_limit:]

    def _handle_candle_sync_success(self) -> None:
        now_ms = self._now_ms()
        failures = self._candle_sync_consecutive_failures
        last_success_age_seconds = self._last_success_age_seconds(now_ms)
        if failures > 0:
            logger.info(
                "CANDLE_SYNC_RECOVERED | failures=%s last_success_age_seconds=%.1f",
                failures,
                last_success_age_seconds,
            )
        self._candle_sync_consecutive_failures = 0
        self._last_successful_candle_sync_ts_ms = now_ms
        self._last_candle_sync_stale_log_ts_ms = 0

    def _handle_candle_sync_failure(self, exc: Exception) -> float:
        now_ms = self._now_ms()
        self._candle_sync_consecutive_failures += 1
        failures = self._candle_sync_consecutive_failures
        last_success_age_seconds = self._last_success_age_seconds(now_ms)
        should_log_error = self._should_log_candle_sync_error(now_ms)
        if should_log_error:
            logger.warning(
                "CANDLE_SYNC_FAILED | failures=%s error_type=%s error=%s last_success_age_seconds=%.1f",
                failures,
                type(exc).__name__,
                exc,
                last_success_age_seconds,
                exc_info=failures == 1 or logger.isEnabledFor(logging.DEBUG),
            )
            self._last_candle_sync_error_log_ts_ms = now_ms
        if last_success_age_seconds >= self._candle_sync_stale_warn_seconds and self._should_log_candle_sync_stale(
                now_ms):
            logger.error(
                "CANDLE_SYNC_STALE | failures=%s last_success_age_seconds=%.1f risk=live_boll_may_be_stale",
                failures,
                last_success_age_seconds,
                exc_info=True,
            )
            self._last_candle_sync_stale_log_ts_ms = now_ms
        return min(
            float(self.config.candle_poll_seconds) + failures * 5,
            self._candle_sync_max_backoff_seconds,
        )

    def _should_log_candle_sync_error(self, now_ms: int) -> bool:
        if self._last_candle_sync_error_log_ts_ms == 0:
            return True
        interval_ms = int(self._candle_sync_error_log_interval_seconds * 1000)
        return now_ms - self._last_candle_sync_error_log_ts_ms >= interval_ms

    def _should_log_candle_sync_stale(self, now_ms: int) -> bool:
        if self._last_candle_sync_stale_log_ts_ms == 0:
            return True
        interval_ms = int(self._candle_sync_error_log_interval_seconds * 1000)
        return now_ms - self._last_candle_sync_stale_log_ts_ms >= interval_ms

    def _last_success_age_seconds(self, now_ms: int) -> float:
        if self._last_successful_candle_sync_ts_ms <= 0:
            return max((now_ms - self._candle_sync_started_ts_ms) / 1000, 0.0)
        return max((now_ms - self._last_successful_candle_sync_ts_ms) / 1000, 0.0)

    async def _boll_recalc_loop(self) -> None:
        while self._running:
            try:
                await self._recalc_boll_snapshot()
            except Exception:
                logger.exception("Failed to recalculate BOLL snapshot")
            await asyncio.sleep(self.config.boll_recalc_seconds)

    async def _recalc_boll_snapshot(self) -> None:
        async with self._candles_lock:
            candles = list(self._candles)
        if len(candles) < self.config.boll_window:
            return
        latest = candles[-1]
        closes = [item.close for item in candles]
        middle, upper, lower = BollCalculator.calculate(closes, self.config.boll_window,
                                                        self.config.boll_std_multiplier)
        upper_distance_pct = abs(upper - middle) / middle
        lower_distance_pct = abs(middle - lower) / middle
        alert_switch_on = upper_distance_pct >= self.config.band_distance_threshold_pct or lower_distance_pct >= self.config.band_distance_threshold_pct
        is_new_candle = latest.ts_ms != self._latest_candle_ts_ms
        switch_changed = alert_switch_on != self._latest_switch_state

        # Compute TP-only BOLL if enabled and we have enough candles.
        tp_lower: float | None = None
        tp_middle: float | None = None
        tp_upper: float | None = None
        tp_window: int | None = None
        if self.config.tp_boll_enabled and self.config.tp_boll_window > 0:
            tp_window = self.config.tp_boll_window
            if tp_window != self.config.boll_window and len(closes) >= tp_window:
                try:
                    tp_middle, tp_upper, tp_lower = BollCalculator.calculate(
                        closes, tp_window, self.config.boll_std_multiplier
                    )
                except ValueError:
                    tp_lower = tp_middle = tp_upper = tp_window = None

        self._snapshot = BollSnapshot(
            self.config.inst_id, latest.ts_ms, latest.close, middle, upper, lower,
            upper_distance_pct, lower_distance_pct, alert_switch_on, self.config.use_live_candle,
            tp_lower=tp_lower, tp_middle=tp_middle, tp_upper=tp_upper, tp_window=tp_window,
            high=latest.high, low=latest.low,
        )
        self._latest_candle_ts_ms = latest.ts_ms
        self._latest_switch_state = alert_switch_on
        if self._previous_price is not None:
            self._previous_zone = self._classify_price_zone(self._previous_price)
        if is_new_candle or switch_changed:
            logger.info(
                "BOLL updated | inst=%s candle_ts=%s close=%.4f middle=%.4f upper=%.4f lower=%.4f upper_dist=%.4f%% lower_dist=%.4f%% switch=%s live_mode=%s tp_boll=%s tp_middle=%s tp_upper=%s tp_lower=%s",
                self.config.inst_id, latest.ts_ms, latest.close, middle, upper, lower,
                upper_distance_pct * 100, lower_distance_pct * 100, alert_switch_on, self.config.use_live_candle,
                f"win{tp_window}" if tp_window is not None else "off",
                f"{tp_middle:.4f}" if tp_middle is not None else "-",
                f"{tp_upper:.4f}" if tp_upper is not None else "-",
                f"{tp_lower:.4f}" if tp_lower is not None else "-",
            )

    async def _tick_loop(self) -> None:
        while self._running:
            try:
                await self.market_data_client.stream_market_events(self._handle_market_data_event)
            except Exception:
                logger.exception("Market data stream disconnected, retrying in 3 seconds")
                await asyncio.sleep(3)

    async def _handle_market_data_event(self, event: MarketDataEvent) -> None:
        if isinstance(event, MarketTradeSnapshot):
            tick = TradeTick(
                inst_id=self.config.inst_id,
                price=float(event.price),
                size=float(event.qty),
                side=event.side if event.side else "unknown",
                ts_ms=event.event_time_ms,
            )
            await self._process_tick(tick)
            return

        if isinstance(event, CandleSnapshot):
            # Upsert the candle into the local candle list for BOLL recalculation
            await self._upsert_candle_from_snapshot(event)
            return

    async def _upsert_candle_from_snapshot(self, snapshot: CandleSnapshot) -> None:
        """Upsert a CandleSnapshot into the local candle list for live BOLL updates."""
        candle = _candle_from_snapshot(snapshot)
        async with self._candles_lock:
            if not self._candles:
                self._candles = [candle]
                return
            # Replace the last candle if same timestamp, otherwise append
            if self._candles[-1].ts_ms == candle.ts_ms:
                self._candles[-1] = candle
            elif candle.ts_ms > self._candles[-1].ts_ms:
                self._candles.append(candle)
                self._candles = self._candles[-self.config.candle_limit:]

    async def _process_tick(self, tick: TradeTick) -> None:
        if self.config.use_live_candle:
            await self._update_live_candle_from_tick(tick.price, tick.ts_ms)
            self._log_stale_rest_candles_if_needed(tick.price)
        snapshot = self._snapshot
        await self._queue_tick_event(MarketTickEvent(tick=tick, boll=snapshot))
        if snapshot is None:
            self._previous_price = tick.price
            self._previous_zone = "UNKNOWN"
            return
        current_zone = self._classify_price_zone(tick.price)
        previous_price = self._previous_price
        previous_zone = self._previous_zone
        self._previous_price = tick.price
        self._previous_zone = current_zone
        if previous_price is None or not snapshot.alert_switch_on:
            return
        now = time.time()
        if now < self._freeze_until_ts or previous_zone != "INSIDE":
            return
        signal = None
        if current_zone == "ABOVE":
            signal = self._build_signal("BREAK_UPPER", tick.price, previous_price, tick.ts_ms, snapshot)
        elif current_zone == "BELOW":
            signal = self._build_signal("BREAK_LOWER", tick.price, previous_price, tick.ts_ms, snapshot)
        if signal is None:
            return
        self._freeze_until_ts = now + self.config.alert_freeze_seconds
        logger.warning(
            "BOLL breakout signal | inst=%s direction=%s prev=%.4f price=%.4f upper=%.4f middle=%.4f lower=%.4f freeze=%ss",
            signal.inst_id, signal.direction, signal.previous_price, signal.price, signal.upper, signal.middle,
            signal.lower, signal.freeze_seconds)
        self._emit(signal)

    async def _queue_tick_event(self, event: MarketTickEvent) -> None:
        if self._tick_event_queue.full():
            logger.error(
                "TICK_EVENT_QUEUE_FULL | price=%.4f tick_ts_ms=%s queue_size=%s",
                event.tick.price,
                event.tick.ts_ms,
                self._tick_event_queue.qsize(),
            )
        await self._tick_event_queue.put(event)
        self._log_tick_queue_backlog()

    async def _tick_event_consumer_loop(self) -> None:
        while self._running:
            event = await self._tick_event_queue.get()
            try:
                await self._emit_tick(event)
            finally:
                self._tick_event_queue.task_done()
                self._log_tick_queue_backlog()

    def _log_tick_queue_backlog(self) -> None:
        queue_size = self._tick_event_queue.qsize()
        level = _monitor_queue_log_level(queue_size)
        if level is None:
            return
        now = time.time()
        if now - self._last_tick_backlog_log_ts < self._tick_backlog_log_seconds:
            return
        self._last_tick_backlog_log_ts = now
        logger.log(
            level,
            "MARKET_TICK_QUEUE_BACKLOG | queue_size=%s maxsize=%s",
            queue_size,
            self._tick_queue_maxsize,
        )

    async def _update_live_candle_from_tick(self, price: float, tick_ts_ms: int) -> None:
        bucket_ts = (tick_ts_ms // self._bar_interval_ms) * self._bar_interval_ms
        async with self._candles_lock:
            if not self._candles:
                self._candles = [Candle(bucket_ts, price, price, price, price, 0.0, False)]
                return
            latest = self._candles[-1]
            if bucket_ts > latest.ts_ms:
                if not latest.confirmed:
                    self._candles[-1] = replace(latest, confirmed=True)
                self._candles.append(Candle(bucket_ts, price, price, price, price, 0.0, False))
                self._candles = self._candles[-self.config.candle_limit:]
                return
            if bucket_ts < latest.ts_ms:
                return
            self._candles[-1] = Candle(latest.ts_ms, latest.open, max(latest.high, price), min(latest.low, price),
                                       price, latest.volume, False)

    def _log_stale_rest_candles_if_needed(self, last_price: float) -> None:
        now_ms = self._now_ms()
        last_success_age_seconds = self._last_success_age_seconds(now_ms)
        if last_success_age_seconds < self._candle_sync_stale_warn_seconds:
            return
        if self._last_boll_stale_log_ts_ms:
            interval_ms = int(self._candle_sync_error_log_interval_seconds * 1000)
            if now_ms - self._last_boll_stale_log_ts_ms < interval_ms:
                return
        latest_candle_ts = self._candles[-1].ts_ms if self._candles else None
        snapshot_candle_ts = self._snapshot.candle_ts_ms if self._snapshot is not None else None
        logger.warning(
            "BOLL_RUNNING_WITH_STALE_REST_CANDLES | last_success_age_seconds=%.1f latest_candle_ts=%s snapshot_candle_ts=%s last_price=%.4f",
            last_success_age_seconds,
            latest_candle_ts,
            snapshot_candle_ts,
            last_price,
        )
        self._last_boll_stale_log_ts_ms = now_ms

    def _build_signal(self, direction: BreakoutDirection, price: float, previous_price: float, tick_ts_ms: int,
                      snapshot: BollSnapshot) -> BreakoutSignal:
        return BreakoutSignal(self.config.inst_id, direction, price, previous_price, tick_ts_ms, snapshot.candle_ts_ms,
                              snapshot.middle, snapshot.upper, snapshot.lower, snapshot.upper_distance_pct,
                              snapshot.lower_distance_pct, self.config.alert_freeze_seconds)

    def _classify_price_zone(self, price: float) -> PriceZone:
        snapshot = self._snapshot
        if snapshot is None or math.isnan(price):
            return "UNKNOWN"
        if price > snapshot.upper:
            return "ABOVE"
        if price < snapshot.lower:
            return "BELOW"
        return "INSIDE"

    def _emit(self, signal: BreakoutSignal) -> None:
        for handler in self.handlers:
            asyncio.create_task(self._run_handler(handler, signal))

    async def _emit_tick(self, event: MarketTickEvent) -> None:
        for handler in self.tick_handlers:
            await self._run_tick_handler(handler, event)

    async def _run_handler(self, handler: SignalHandler, signal: BreakoutSignal) -> None:
        try:
            result = handler(signal)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Signal handler failed")

    async def _run_tick_handler(self, handler: TickHandler, event: MarketTickEvent) -> None:
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Tick handler failed")

    @staticmethod
    def _parse_bar_interval_ms(bar: str) -> int:
        text = bar.strip().lower()
        if text.endswith("m"):
            return int(text[:-1]) * 60 * 1000
        if text.endswith("h"):
            return int(text[:-1]) * 60 * 60 * 1000
        if text.endswith("d"):
            return int(text[:-1]) * 24 * 60 * 60 * 1000
        raise ValueError(f"Unsupported bar interval: {bar}")

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)


def _monitor_queue_log_level(queue_size: int) -> int | None:
    if queue_size < 500:
        return None
    if queue_size < 2000:
        return logging.INFO
    if queue_size < 8000:
        return logging.WARNING
    return logging.ERROR
