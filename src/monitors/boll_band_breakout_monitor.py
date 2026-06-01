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

import aiohttp

logger = logging.getLogger(__name__)
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


class OkxPublicMarketClient:
    def __init__(self, config: BollBandBreakoutMonitorConfig):
        self.config = config

    async def fetch_candles(self, include_live: bool) -> list[Candle]:
        url = f"{self.config.rest_base_url}/api/v5/market/candles"
        params = {"instId": self.config.inst_id, "bar": self.config.bar, "limit": str(self.config.candle_limit)}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, params=params) as resp:
                payload = await resp.json()
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX candle API error: {payload}")
        candles: list[Candle] = []
        for row in payload.get("data", []):
            if len(row) < 6:
                continue
            confirmed = row[8] == "1" if len(row) >= 9 else True
            if not include_live and not confirmed:
                continue
            candles.append(Candle(int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), confirmed))
        candles.sort(key=lambda item: item.ts_ms)
        return candles


class BollBandBreakoutMonitor:
    def __init__(
        self,
        config: BollBandBreakoutMonitorConfig,
        handlers: Optional[list[SignalHandler]] = None,
        tick_handlers: Optional[list[TickHandler]] = None,
    ):
        self.config = config
        self.client = OkxPublicMarketClient(config)
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

    def add_handler(self, handler: SignalHandler) -> None:
        self.handlers.append(handler)

    def add_tick_handler(self, handler: TickHandler) -> None:
        self.tick_handlers.append(handler)

    async def run_forever(self) -> None:
        self._running = True
        await asyncio.gather(self._candle_sync_loop(), self._boll_recalc_loop(), self._tick_loop())

    async def _candle_sync_loop(self) -> None:
        while self._running:
            try:
                await self._sync_candles_from_rest()
            except Exception:
                logger.exception("Failed to sync candles from OKX REST")
            await asyncio.sleep(self.config.candle_poll_seconds)

    async def _sync_candles_from_rest(self) -> None:
        candles = await self.client.fetch_candles(include_live=self.config.use_live_candle)
        if len(candles) < self.config.boll_window:
            logger.warning("Not enough candles for BOLL: %s < %s", len(candles), self.config.boll_window)
            return
        async with self._candles_lock:
            self._candles = candles[-self.config.candle_limit:]

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
        middle, upper, lower = BollCalculator.calculate(closes, self.config.boll_window, self.config.boll_std_multiplier)
        upper_distance_pct = abs(upper - middle) / middle
        lower_distance_pct = abs(middle - lower) / middle
        alert_switch_on = upper_distance_pct >= self.config.band_distance_threshold_pct or lower_distance_pct >= self.config.band_distance_threshold_pct
        is_new_candle = latest.ts_ms != self._latest_candle_ts_ms
        switch_changed = alert_switch_on != self._latest_switch_state
        self._snapshot = BollSnapshot(self.config.inst_id, latest.ts_ms, latest.close, middle, upper, lower, upper_distance_pct, lower_distance_pct, alert_switch_on, self.config.use_live_candle)
        self._latest_candle_ts_ms = latest.ts_ms
        self._latest_switch_state = alert_switch_on
        if self._previous_price is not None:
            self._previous_zone = self._classify_price_zone(self._previous_price)
        if is_new_candle or switch_changed:
            logger.info("BOLL updated | inst=%s candle_ts=%s close=%.4f middle=%.4f upper=%.4f lower=%.4f upper_dist=%.4f%% lower_dist=%.4f%% switch=%s live_mode=%s", self.config.inst_id, latest.ts_ms, latest.close, middle, upper, lower, upper_distance_pct * 100, lower_distance_pct * 100, alert_switch_on, self.config.use_live_candle)

    async def _tick_loop(self) -> None:
        while self._running:
            try:
                await self._connect_tick_ws()
            except Exception:
                logger.exception("Tick websocket disconnected, retrying in 3 seconds")
                await asyncio.sleep(3)

    async def _connect_tick_ws(self) -> None:
        payload = {"op": "subscribe", "args": [{"channel": "trades", "instId": self.config.inst_id}]}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            async with session.ws_connect(self.config.ws_public_url, heartbeat=20, autoping=True) as ws:
                await ws.send_json(payload)
                logger.info("Subscribed OKX trades channel: %s", self.config.inst_id)
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_ws_payload(msg.json())
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    async def _handle_ws_payload(self, payload: dict) -> None:
        if "event" in payload:
            logger.info("OKX websocket event: %s", payload)
            return
        if payload.get("arg", {}).get("channel") != "trades":
            return
        for item in payload.get("data", []):
            try:
                tick = TradeTick(
                    inst_id=self.config.inst_id,
                    price=float(item["px"]),
                    size=float(item.get("sz", 0.0)),
                    side=str(item.get("side", "unknown")),
                    ts_ms=int(item["ts"]),
                )
                await self._process_tick(tick)
            except Exception:
                logger.warning("Invalid trade tick payload: %s", item)

    async def _process_tick(self, tick: TradeTick) -> None:
        if self.config.use_live_candle:
            await self._update_live_candle_from_tick(tick.price, tick.ts_ms)
        snapshot = self._snapshot
        self._emit_tick(MarketTickEvent(tick=tick, boll=snapshot))
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
        logger.warning("BOLL breakout signal | inst=%s direction=%s prev=%.4f price=%.4f upper=%.4f middle=%.4f lower=%.4f freeze=%ss", signal.inst_id, signal.direction, signal.previous_price, signal.price, signal.upper, signal.middle, signal.lower, signal.freeze_seconds)
        self._emit(signal)

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
            self._candles[-1] = Candle(latest.ts_ms, latest.open, max(latest.high, price), min(latest.low, price), price, latest.volume, False)

    def _build_signal(self, direction: BreakoutDirection, price: float, previous_price: float, tick_ts_ms: int, snapshot: BollSnapshot) -> BreakoutSignal:
        return BreakoutSignal(self.config.inst_id, direction, price, previous_price, tick_ts_ms, snapshot.candle_ts_ms, snapshot.middle, snapshot.upper, snapshot.lower, snapshot.upper_distance_pct, snapshot.lower_distance_pct, self.config.alert_freeze_seconds)

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

    def _emit_tick(self, event: MarketTickEvent) -> None:
        for handler in self.tick_handlers:
            asyncio.create_task(self._run_tick_handler(handler, event))

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
