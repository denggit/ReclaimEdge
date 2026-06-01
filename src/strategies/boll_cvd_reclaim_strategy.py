from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer

TradeIntentType = Literal[
    "OPEN_LONG",
    "ADD_LONG",
    "OPEN_SHORT",
    "ADD_SHORT",
    "UPDATE_TP",
]
PositionSide = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class BollCvdReclaimStrategyConfig:
    min_buy_ratio: float = 0.55
    min_sell_ratio: float = 0.55
    add_layer_gap_pct: float = 0.003
    max_layers: int = 3
    order_cooldown_seconds: int = 10
    tp_update_interval_seconds: int = 900

    @classmethod
    def from_env(cls) -> "BollCvdReclaimStrategyConfig":
        return cls(
            min_buy_ratio=float(os.getenv("CVD_MIN_BUY_RATIO", "0.55")),
            min_sell_ratio=float(os.getenv("CVD_MIN_SELL_RATIO", "0.55")),
            add_layer_gap_pct=float(os.getenv("ADD_LAYER_GAP_PCT", "0.003")),
            max_layers=int(os.getenv("MAX_LAYERS", "3")),
            order_cooldown_seconds=int(os.getenv("ORDER_COOLDOWN_SECONDS", "10")),
            tp_update_interval_seconds=int(os.getenv("TP_UPDATE_INTERVAL_SECONDS", "900")),
        )


@dataclass(frozen=True)
class TradeIntent:
    intent_type: TradeIntentType
    side: PositionSide
    price: float
    layer_index: int
    tp_price: float
    reason: str
    size: PositionSize
    fast_cvd: float
    previous_fast_cvd: float
    buy_ratio: float
    sell_ratio: float
    boll_upper: float
    boll_middle: float
    boll_lower: float
    ts_ms: int


@dataclass
class StrategyPositionState:
    side: Optional[PositionSide] = None
    layers: int = 0
    last_entry_price: Optional[float] = None
    tp_price: Optional[float] = None
    last_order_ts_ms: int = 0
    last_tp_update_ts_ms: int = 0


class BollCvdReclaimStrategy:
    """Minimal dry-run strategy for BOLL outside + fast CVD reclaim.

    This module only emits TradeIntent. It does not place orders.
    """

    def __init__(self, config: BollCvdReclaimStrategyConfig, sizer: SimplePositionSizer):
        self.config = config
        self.sizer = sizer
        self.state = StrategyPositionState()

    def on_tick(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> list[TradeIntent]:
        intents: list[TradeIntent] = []
        if not boll.alert_switch_on:
            return intents

        tp_intent = self._maybe_update_tp(price, ts_ms, boll, cvd)
        if tp_intent is not None:
            intents.append(tp_intent)

        if not self._cooldown_ok(ts_ms):
            return intents

        if self._long_setup(price, boll, cvd):
            intent = self._maybe_open_or_add_long(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        if self._short_setup(price, boll, cvd):
            intent = self._maybe_open_or_add_short(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        return intents

    def _long_setup(self, price: float, boll: BollSnapshot, cvd: CvdSnapshot) -> bool:
        if price >= boll.lower:
            return False
        cvd_reclaim = cvd.cross_positive and cvd.buy_ratio >= self.config.min_buy_ratio and cvd.no_new_low
        cvd_absorption = cvd.cvd_increasing and cvd.buy_ratio >= self.config.min_buy_ratio and cvd.no_new_low
        return cvd_reclaim or cvd_absorption

    def _short_setup(self, price: float, boll: BollSnapshot, cvd: CvdSnapshot) -> bool:
        if price <= boll.upper:
            return False
        cvd_reject = cvd.cross_negative and cvd.sell_ratio >= self.config.min_sell_ratio and cvd.no_new_high
        cvd_absorption = cvd.cvd_decreasing and cvd.sell_ratio >= self.config.min_sell_ratio and cvd.no_new_high
        return cvd_reject or cvd_absorption

    def _maybe_open_or_add_long(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None:
            return self._open_position("LONG", "OPEN_LONG", price, ts_ms, boll, cvd, "下轨外侧 + 快速CVD回流/跌不动")
        if self.state.side != "LONG":
            return None
        if self.state.layers >= self.config.max_layers:
            return None
        if self.state.last_entry_price is None:
            return None
        if price > self.state.last_entry_price * (1 - self.config.add_layer_gap_pct):
            return None
        return self._open_position("LONG", "ADD_LONG", price, ts_ms, boll, cvd, "距离上一多仓超过0.3% + 再次跌不动")

    def _maybe_open_or_add_short(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None:
            return self._open_position("SHORT", "OPEN_SHORT", price, ts_ms, boll, cvd, "上轨外侧 + 快速CVD转弱/涨不动")
        if self.state.side != "SHORT":
            return None
        if self.state.layers >= self.config.max_layers:
            return None
        if self.state.last_entry_price is None:
            return None
        if price < self.state.last_entry_price * (1 + self.config.add_layer_gap_pct):
            return None
        return self._open_position("SHORT", "ADD_SHORT", price, ts_ms, boll, cvd, "距离上一空仓超过0.3% + 再次涨不动")

    def _open_position(
        self,
        side: PositionSide,
        intent_type: TradeIntentType,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        reason: str,
    ) -> TradeIntent:
        next_layer = self.state.layers + 1
        tp_price = boll.middle
        size = self.sizer.calculate(price)
        self.state.side = side
        self.state.layers = next_layer
        self.state.last_entry_price = price
        self.state.tp_price = tp_price
        self.state.last_order_ts_ms = ts_ms
        self.state.last_tp_update_ts_ms = ts_ms
        return self._intent(intent_type, side, price, next_layer, tp_price, reason, size, boll, cvd, ts_ms)

    def _maybe_update_tp(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None or self.state.layers <= 0:
            return None
        if ts_ms - self.state.last_tp_update_ts_ms < self.config.tp_update_interval_seconds * 1000:
            return None
        if self.state.tp_price is not None and abs(self.state.tp_price - boll.middle) / boll.middle < 0.0001:
            self.state.last_tp_update_ts_ms = ts_ms
            return None
        self.state.tp_price = boll.middle
        self.state.last_tp_update_ts_ms = ts_ms
        size = self.sizer.calculate(price)
        return self._intent("UPDATE_TP", self.state.side, price, self.state.layers, boll.middle, "15分钟更新一次止盈到最新BOLL中轨", size, boll, cvd, ts_ms)

    def _intent(
        self,
        intent_type: TradeIntentType,
        side: PositionSide,
        price: float,
        layer_index: int,
        tp_price: float,
        reason: str,
        size: PositionSize,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        ts_ms: int,
    ) -> TradeIntent:
        return TradeIntent(
            intent_type=intent_type,
            side=side,
            price=price,
            layer_index=layer_index,
            tp_price=tp_price,
            reason=reason,
            size=size,
            fast_cvd=cvd.fast_cvd,
            previous_fast_cvd=cvd.previous_fast_cvd,
            buy_ratio=cvd.buy_ratio,
            sell_ratio=cvd.sell_ratio,
            boll_upper=boll.upper,
            boll_middle=boll.middle,
            boll_lower=boll.lower,
            ts_ms=ts_ms,
        )

    def _cooldown_ok(self, ts_ms: int) -> bool:
        return ts_ms - self.state.last_order_ts_ms >= self.config.order_cooldown_seconds * 1000
