from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer

logger = logging.getLogger(__name__)

TradeIntentType = Literal[
    "OPEN_LONG",
    "ADD_LONG",
    "OPEN_SHORT",
    "ADD_SHORT",
    "UPDATE_TP",
]
PositionSide = Literal["LONG", "SHORT"]
TpMode = Literal["MIDDLE", "UPPER", "LOWER"]


@dataclass(frozen=True)
class BollCvdReclaimStrategyConfig:
    min_buy_ratio: float = 0.55
    min_sell_ratio: float = 0.55
    add_layer_gap_pct: float = 0.003
    max_layers: int = 3
    order_cooldown_seconds: int = 10
    tp_update_interval_seconds: int = 900
    max_entry_distance_from_extreme_pct: float = 0.002
    max_armed_seconds: int = 900
    breakeven_fee_buffer_pct: float = 0.001

    @classmethod
    def from_env(cls) -> "BollCvdReclaimStrategyConfig":
        return cls(
            min_buy_ratio=float(os.getenv("CVD_MIN_BUY_RATIO", "0.55")),
            min_sell_ratio=float(os.getenv("CVD_MIN_SELL_RATIO", "0.55")),
            add_layer_gap_pct=float(os.getenv("ADD_LAYER_GAP_PCT", "0.003")),
            max_layers=int(os.getenv("MAX_LAYERS", "3")),
            order_cooldown_seconds=int(os.getenv("ORDER_COOLDOWN_SECONDS", "10")),
            tp_update_interval_seconds=int(os.getenv("TP_UPDATE_INTERVAL_SECONDS", "900")),
            max_entry_distance_from_extreme_pct=float(os.getenv("MAX_ENTRY_DISTANCE_FROM_EXTREME_PCT", "0.002")),
            max_armed_seconds=int(os.getenv("MAX_ARMED_SECONDS", "900")),
            breakeven_fee_buffer_pct=float(os.getenv("BREAKEVEN_FEE_BUFFER_PCT", "0.001")),
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
    avg_entry_price: float
    breakeven_price: float
    tp_mode: TpMode


@dataclass
class StrategyPositionState:
    side: Optional[PositionSide] = None
    layers: int = 0
    last_entry_price: Optional[float] = None
    tp_price: Optional[float] = None
    last_order_ts_ms: int = 0
    last_tp_update_ts_ms: int = 0
    last_tp_update_candle_ts_ms: int = 0
    lower_armed: bool = False
    upper_armed: bool = False
    lower_extreme_price: Optional[float] = None
    upper_extreme_price: Optional[float] = None
    lower_armed_ts_ms: int = 0
    upper_armed_ts_ms: int = 0
    total_entry_qty: float = 0.0
    total_entry_notional: float = 0.0
    avg_entry_price: float = 0.0
    breakeven_price: float = 0.0
    tp_mode: TpMode = "MIDDLE"


class BollCvdReclaimStrategy:
    """Minimal strategy for BOLL outside + fast CVD reclaim.

    The strategy is armed after price moves outside a BOLL band. Entry does not
    have to occur while price is still outside the band, but it must occur near
    the recent outside-band extreme. This matches the manual workflow:

    outside band -> watch for stall/reversal -> enter near low/high.
    """

    def __init__(self, config: BollCvdReclaimStrategyConfig, sizer: SimplePositionSizer):
        self.config = config
        self.sizer = sizer
        self.state = StrategyPositionState()

    def on_tick(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> list[TradeIntent]:
        intents: list[TradeIntent] = []

        self._update_armed_state(price, ts_ms, boll)

        # TP maintenance is driven by BOLL candle timestamp. This avoids the old
        # problem where a restart/manual TP update delayed the next 15m update.
        tp_intent = self._maybe_update_tp(price, ts_ms, boll, cvd)
        if tp_intent is not None:
            intents.append(tp_intent)

        if not boll.alert_switch_on:
            return intents

        if not self._cooldown_ok(ts_ms):
            return intents

        if self._long_setup(price, cvd):
            intent = self._maybe_open_or_add_long(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        if self._short_setup(price, cvd):
            intent = self._maybe_open_or_add_short(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        return intents

    def _update_armed_state(self, price: float, ts_ms: int, boll: BollSnapshot) -> None:
        self._expire_armed_state(ts_ms)

        if price < boll.lower:
            if not self.state.lower_armed:
                self.state.lower_armed = True
                self.state.lower_armed_ts_ms = ts_ms
                self.state.lower_extreme_price = price
                logger.info(
                    "LOWER_ARMED | price=%.4f lower=%.4f middle=%.4f max_entry_distance=%.4f%% max_armed=%ss",
                    price,
                    boll.lower,
                    boll.middle,
                    self.config.max_entry_distance_from_extreme_pct * 100,
                    self.config.max_armed_seconds,
                )
            else:
                old_extreme = self.state.lower_extreme_price or price
                self.state.lower_extreme_price = min(old_extreme, price)
                if self.state.lower_extreme_price < old_extreme:
                    logger.debug("LOWER_EXTREME_UPDATED | extreme=%.4f price=%.4f", self.state.lower_extreme_price, price)
            if self.state.upper_armed:
                logger.info("UPPER_ARMED_RESET | reason=opposite_lower_break price=%.4f", price)
            self.state.upper_armed = False
            self.state.upper_extreme_price = None
            self.state.upper_armed_ts_ms = 0
            return

        if price > boll.upper:
            if not self.state.upper_armed:
                self.state.upper_armed = True
                self.state.upper_armed_ts_ms = ts_ms
                self.state.upper_extreme_price = price
                logger.info(
                    "UPPER_ARMED | price=%.4f upper=%.4f middle=%.4f max_entry_distance=%.4f%% max_armed=%ss",
                    price,
                    boll.upper,
                    boll.middle,
                    self.config.max_entry_distance_from_extreme_pct * 100,
                    self.config.max_armed_seconds,
                )
            else:
                old_extreme = self.state.upper_extreme_price or price
                self.state.upper_extreme_price = max(old_extreme, price)
                if self.state.upper_extreme_price > old_extreme:
                    logger.debug("UPPER_EXTREME_UPDATED | extreme=%.4f price=%.4f", self.state.upper_extreme_price, price)
            if self.state.lower_armed:
                logger.info("LOWER_ARMED_RESET | reason=opposite_upper_break price=%.4f", price)
            self.state.lower_armed = False
            self.state.lower_extreme_price = None
            self.state.lower_armed_ts_ms = 0
            return

        # If price mean-reverts all the way to the middle, the original outside-band
        # opportunity is considered stale.
        if self.state.lower_armed and price >= boll.middle:
            logger.info("LOWER_ARMED_RESET | reason=middle_reclaimed price=%.4f middle=%.4f", price, boll.middle)
            self._reset_lower_armed()
        if self.state.upper_armed and price <= boll.middle:
            logger.info("UPPER_ARMED_RESET | reason=middle_reclaimed price=%.4f middle=%.4f", price, boll.middle)
            self._reset_upper_armed()

    def _expire_armed_state(self, ts_ms: int) -> None:
        max_age_ms = self.config.max_armed_seconds * 1000
        if self.state.lower_armed and ts_ms - self.state.lower_armed_ts_ms > max_age_ms:
            logger.info("LOWER_ARMED_RESET | reason=expired age_ms=%s", ts_ms - self.state.lower_armed_ts_ms)
            self._reset_lower_armed()
        if self.state.upper_armed and ts_ms - self.state.upper_armed_ts_ms > max_age_ms:
            logger.info("UPPER_ARMED_RESET | reason=expired age_ms=%s", ts_ms - self.state.upper_armed_ts_ms)
            self._reset_upper_armed()

    def _reset_lower_armed(self) -> None:
        self.state.lower_armed = False
        self.state.lower_extreme_price = None
        self.state.lower_armed_ts_ms = 0

    def _reset_upper_armed(self) -> None:
        self.state.upper_armed = False
        self.state.upper_extreme_price = None
        self.state.upper_armed_ts_ms = 0

    def _long_setup(self, price: float, cvd: CvdSnapshot) -> bool:
        if not self.state.lower_armed or self.state.lower_extreme_price is None:
            return False
        if not self._near_lower_extreme(price):
            return False
        cvd_reclaim = cvd.cross_positive and cvd.buy_ratio >= self.config.min_buy_ratio and cvd.no_new_low
        cvd_absorption = cvd.cvd_increasing and cvd.buy_ratio >= self.config.min_buy_ratio and cvd.no_new_low
        return cvd_reclaim or cvd_absorption

    def _short_setup(self, price: float, cvd: CvdSnapshot) -> bool:
        if not self.state.upper_armed or self.state.upper_extreme_price is None:
            return False
        if not self._near_upper_extreme(price):
            return False
        cvd_reject = cvd.cross_negative and cvd.sell_ratio >= self.config.min_sell_ratio and cvd.no_new_high
        cvd_absorption = cvd.cvd_decreasing and cvd.sell_ratio >= self.config.min_sell_ratio and cvd.no_new_high
        return cvd_reject or cvd_absorption

    def _near_lower_extreme(self, price: float) -> bool:
        extreme = self.state.lower_extreme_price
        if extreme is None:
            return False
        return price <= extreme * (1 + self.config.max_entry_distance_from_extreme_pct)

    def _near_upper_extreme(self, price: float) -> bool:
        extreme = self.state.upper_extreme_price
        if extreme is None:
            return False
        return price >= extreme * (1 - self.config.max_entry_distance_from_extreme_pct)

    def _maybe_open_or_add_long(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None:
            return self._open_position("LONG", "OPEN_LONG", price, ts_ms, boll, cvd, "下轨出轨后armed + 低点附近快速CVD回流/跌不动")
        if self.state.side != "LONG":
            return None
        if self.state.layers >= self.config.max_layers:
            return None
        if self.state.last_entry_price is None:
            return None
        if price > self.state.last_entry_price * (1 - self.config.add_layer_gap_pct):
            return None
        return self._open_position("LONG", "ADD_LONG", price, ts_ms, boll, cvd, "距离上一多仓超过0.3% + 低点附近再次跌不动")

    def _maybe_open_or_add_short(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None:
            return self._open_position("SHORT", "OPEN_SHORT", price, ts_ms, boll, cvd, "上轨出轨后armed + 高点附近快速CVD转弱/涨不动")
        if self.state.side != "SHORT":
            return None
        if self.state.layers >= self.config.max_layers:
            return None
        if self.state.last_entry_price is None:
            return None
        if price < self.state.last_entry_price * (1 + self.config.add_layer_gap_pct):
            return None
        return self._open_position("SHORT", "ADD_SHORT", price, ts_ms, boll, cvd, "距离上一空仓超过0.3% + 高点附近再次涨不动")

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
        size = self.sizer.calculate(price, layer_index=next_layer)
        self._update_position_cost(price, size.eth_qty)
        tp_price, tp_mode = self._select_tp_price(side, boll)
        if tp_mode != "MIDDLE":
            reason = f"{reason} + 中轨不足覆盖含手续费盈亏平衡，TP切换到{tp_mode}"
        self.state.side = side
        self.state.layers = next_layer
        self.state.last_entry_price = price
        self.state.tp_price = tp_price
        self.state.tp_mode = tp_mode
        self.state.last_order_ts_ms = ts_ms
        self.state.last_tp_update_ts_ms = ts_ms
        self.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms
        logger.info(
            "TP_SELECTED | reason=entry side=%s mode=%s avg_entry=%.4f breakeven=%.4f candle_ts=%s middle=%.4f upper=%.4f lower=%.4f tp=%.4f",
            side,
            tp_mode,
            self.state.avg_entry_price,
            self.state.breakeven_price,
            boll.candle_ts_ms,
            boll.middle,
            boll.upper,
            boll.lower,
            tp_price,
        )
        return self._intent(intent_type, side, price, next_layer, tp_price, reason, size, boll, cvd, ts_ms)

    def _maybe_update_tp(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None or self.state.layers <= 0:
            return None
        if self.state.last_tp_update_candle_ts_ms == boll.candle_ts_ms:
            return None

        tp_price, tp_mode = self._select_tp_price(self.state.side, boll)
        self.state.last_tp_update_ts_ms = ts_ms
        self.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms

        if self.state.tp_price is not None and abs(self.state.tp_price - tp_price) / tp_price < 0.0001:
            logger.info(
                "TP_UPDATE_SKIPPED | reason=price_unchanged side=%s mode=%s candle_ts=%s current_tp=%.4f target_tp=%.4f avg_entry=%.4f breakeven=%.4f",
                self.state.side,
                tp_mode,
                boll.candle_ts_ms,
                self.state.tp_price,
                tp_price,
                self.state.avg_entry_price,
                self.state.breakeven_price,
            )
            return None

        self.state.tp_price = tp_price
        self.state.tp_mode = tp_mode
        size = self.sizer.calculate(price, layer_index=self.state.layers)
        logger.info(
            "TP_SELECTED | reason=new_candle side=%s mode=%s avg_entry=%.4f breakeven=%.4f candle_ts=%s middle=%.4f upper=%.4f lower=%.4f tp=%.4f",
            self.state.side,
            tp_mode,
            self.state.avg_entry_price,
            self.state.breakeven_price,
            boll.candle_ts_ms,
            boll.middle,
            boll.upper,
            boll.lower,
            tp_price,
        )
        return self._intent("UPDATE_TP", self.state.side, price, self.state.layers, tp_price, f"新15m K线更新止盈到{tp_mode}轨", size, boll, cvd, ts_ms)

    def _update_position_cost(self, entry_price: float, eth_qty: float) -> None:
        if eth_qty <= 0:
            return
        self.state.total_entry_qty += eth_qty
        self.state.total_entry_notional += entry_price * eth_qty
        self.state.avg_entry_price = self.state.total_entry_notional / self.state.total_entry_qty

    def _select_tp_price(self, side: PositionSide, boll: BollSnapshot) -> tuple[float, TpMode]:
        if self.state.avg_entry_price <= 0:
            return boll.middle, "MIDDLE"
        fee = self.config.breakeven_fee_buffer_pct
        if side == "LONG":
            self.state.breakeven_price = self.state.avg_entry_price * (1 + fee)
            if self.state.breakeven_price > boll.middle:
                return boll.upper, "UPPER"
            return boll.middle, "MIDDLE"
        self.state.breakeven_price = self.state.avg_entry_price * (1 - fee)
        if self.state.breakeven_price < boll.middle:
            return boll.lower, "LOWER"
        return boll.middle, "MIDDLE"

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
            avg_entry_price=self.state.avg_entry_price,
            breakeven_price=self.state.breakeven_price,
            tp_mode=self.state.tp_mode,
        )

    def _cooldown_ok(self, ts_ms: int) -> bool:
        return ts_ms - self.state.last_order_ts_ms >= self.config.order_cooldown_seconds * 1000
