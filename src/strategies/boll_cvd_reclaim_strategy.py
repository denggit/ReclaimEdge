from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer
from src.utils.log import get_logger

logger = get_logger(__name__)

TradeIntentType = Literal[
    "OPEN_LONG",
    "ADD_LONG",
    "OPEN_SHORT",
    "ADD_SHORT",
    "UPDATE_TP",
    "NEAR_TP_REDUCE",
]
PositionSide = Literal["LONG", "SHORT"]
TpMode = Literal["MIDDLE", "UPPER", "LOWER"]
TpPlan = Literal["SINGLE", "SPLIT_PARTIAL_FINAL"]


@dataclass(frozen=True)
class BollCvdReclaimStrategyConfig:
    min_buy_ratio: float = 0.55
    min_sell_ratio: float = 0.55
    add_layer_gap_pct: float = 0.003
    add_layer_gap_pct_layer_7_8: float = 0.004
    add_layer_gap_pct_layer_9_10: float = 0.006
    add_layer_gap_pct_layer_11_plus: float = 0.008
    add_min_avg_improvement_pct: float = 0.0012
    max_layers: int = 3
    order_cooldown_seconds: int = 10
    first_add_block_seconds: int = 1800
    add_min_interval_seconds: int = 600
    add_min_interval_bypass_gap_pct: float = 0.005
    tp_update_interval_seconds: int = 900
    max_entry_distance_from_extreme_pct: float = 0.002
    max_armed_seconds: int = 900
    breakeven_fee_buffer_pct: float = 0.001
    tp_min_net_profit_pct: float = 0.002
    min_outside_pct: float = 0.001
    split_tp_min_layers: int = 4
    split_tp_path_ratio: float = 0.8
    split_tp_partial_ratio: float = 0.5
    split_tp_min_profit_pct: float = 0.004
    near_tp_enabled: bool = False
    near_tp_reduce_enabled: bool = True
    near_tp_shadow_enabled: bool = False
    near_tp_min_progress_ratio: float = 0.88
    near_tp_max_distance_usd: float = 3.0
    near_tp_min_profit_pct: float = 0.004
    near_tp_giveback_usd: float = 3.0
    near_tp_giveback_pct: float = 0.0015
    near_tp_giveback_profit_ratio: float = 0.25
    near_tp_reduce_ratio: float = 0.5
    near_tp_min_reduce_profit_pct: float = 0.004
    near_tp_disable_add_after_reduce: bool = True
    near_tp_protective_sl_enabled: bool = True
    near_tp_protective_sl_profit_pct: float = 0.001
    near_tp_protective_sl_retry_count: int = 3
    near_tp_protective_sl_retry_interval_seconds: float = 1.0
    near_tp_sl_fail_action: str = "MARKET_EXIT"
    near_tp_sl_fail_market_exit_retry_count: int = 3

    @classmethod
    def from_env(cls) -> "BollCvdReclaimStrategyConfig":
        return cls(
            min_buy_ratio=float(os.getenv("CVD_MIN_BUY_RATIO", "0.55")),
            min_sell_ratio=float(os.getenv("CVD_MIN_SELL_RATIO", "0.55")),
            add_layer_gap_pct=float(os.getenv("ADD_LAYER_GAP_PCT", "0.003")),
            add_layer_gap_pct_layer_7_8=float(os.getenv("ADD_LAYER_GAP_PCT_LAYER_7_8", "0.004")),
            add_layer_gap_pct_layer_9_10=float(os.getenv("ADD_LAYER_GAP_PCT_LAYER_9_10", "0.006")),
            add_layer_gap_pct_layer_11_plus=float(os.getenv("ADD_LAYER_GAP_PCT_LAYER_11_PLUS", "0.008")),
            add_min_avg_improvement_pct=float(os.getenv("ADD_MIN_AVG_IMPROVEMENT_PCT", "0.0012")),
            max_layers=int(os.getenv("MAX_LAYERS", "3")),
            order_cooldown_seconds=int(os.getenv("ORDER_COOLDOWN_SECONDS", "10")),
            first_add_block_seconds=int(os.getenv("FIRST_ADD_BLOCK_SECONDS", "1800")),
            add_min_interval_seconds=int(os.getenv("ADD_MIN_INTERVAL_SECONDS", "600")),
            add_min_interval_bypass_gap_pct=float(os.getenv("ADD_MIN_INTERVAL_BYPASS_GAP_PCT", "0.005")),
            tp_update_interval_seconds=int(os.getenv("TP_UPDATE_INTERVAL_SECONDS", "900")),
            max_entry_distance_from_extreme_pct=float(os.getenv("MAX_ENTRY_DISTANCE_FROM_EXTREME_PCT", "0.002")),
            max_armed_seconds=int(os.getenv("MAX_ARMED_SECONDS", "900")),
            breakeven_fee_buffer_pct=float(os.getenv("BREAKEVEN_FEE_BUFFER_PCT", "0.001")),
            tp_min_net_profit_pct=float(os.getenv("TP_MIN_NET_PROFIT_PCT", "0.002")),
            min_outside_pct=float(os.getenv("BOLL_MIN_OUTSIDE_PCT", "0.001")),
            split_tp_min_layers=int(os.getenv("SPLIT_TP_MIN_LAYERS", "4")),
            split_tp_path_ratio=float(os.getenv("SPLIT_TP_PATH_RATIO", "0.8")),
            split_tp_partial_ratio=float(os.getenv("SPLIT_TP_PARTIAL_RATIO", "0.5")),
            split_tp_min_profit_pct=float(os.getenv("SPLIT_TP_MIN_PROFIT_PCT", "0.004")),
            near_tp_enabled=_env_bool("NEAR_TP_ENABLED", False),
            near_tp_reduce_enabled=_env_bool("NEAR_TP_REDUCE_ENABLED", True),
            near_tp_shadow_enabled=_env_bool("NEAR_TP_SHADOW_ENABLED", False),
            near_tp_min_progress_ratio=float(os.getenv("NEAR_TP_MIN_PROGRESS_RATIO", "0.88")),
            near_tp_max_distance_usd=float(os.getenv("NEAR_TP_MAX_DISTANCE_USD", "3")),
            near_tp_min_profit_pct=float(os.getenv("NEAR_TP_MIN_PROFIT_PCT", "0.004")),
            near_tp_giveback_usd=float(os.getenv("NEAR_TP_GIVEBACK_USD", "3")),
            near_tp_giveback_pct=float(os.getenv("NEAR_TP_GIVEBACK_PCT", "0.0015")),
            near_tp_giveback_profit_ratio=float(os.getenv("NEAR_TP_GIVEBACK_PROFIT_RATIO", "0.25")),
            near_tp_reduce_ratio=float(os.getenv("NEAR_TP_REDUCE_RATIO", "0.5")),
            near_tp_min_reduce_profit_pct=float(os.getenv("NEAR_TP_MIN_REDUCE_PROFIT_PCT", "0.004")),
            near_tp_disable_add_after_reduce=_env_bool("NEAR_TP_DISABLE_ADD_AFTER_REDUCE", True),
            near_tp_protective_sl_enabled=_env_bool("NEAR_TP_PROTECTIVE_SL_ENABLED", True),
            near_tp_protective_sl_profit_pct=float(os.getenv("NEAR_TP_PROTECTIVE_SL_PROFIT_PCT", "0.001")),
            near_tp_protective_sl_retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
            near_tp_protective_sl_retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
            near_tp_sl_fail_action=os.getenv("NEAR_TP_SL_FAIL_ACTION", "MARKET_EXIT").strip().upper(),
            near_tp_sl_fail_market_exit_retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
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
    partial_tp_price: float | None = None
    partial_tp_ratio: float = 0.0
    tp_plan: TpPlan = "SINGLE"
    partial_tp_consumed: bool = False
    near_tp_progress_ratio: float = 0.0
    near_tp_best_price: float | None = None
    near_tp_giveback: float = 0.0
    near_tp_giveback_threshold: float = 0.0
    near_tp_reduce_ratio: float = 0.0
    near_tp_protective_sl_price: float | None = None


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
    lower_last_burst_ts_ms: int = 0
    upper_last_burst_ts_ms: int = 0
    lower_deep_enough: bool = False
    upper_deep_enough: bool = False
    total_entry_qty: float = 0.0
    total_entry_notional: float = 0.0
    avg_entry_price: float = 0.0
    breakeven_price: float = 0.0
    tp_mode: TpMode = "MIDDLE"
    partial_tp_price: float | None = None
    partial_tp_ratio: float = 0.0
    tp_plan: TpPlan = "SINGLE"
    partial_tp_consumed: bool = False
    near_tp_armed: bool = False
    near_tp_reduce_pending: bool = False
    near_tp_protected: bool = False
    near_tp_best_price: float | None = None
    near_tp_armed_ts_ms: int = 0
    near_tp_pending_ts_ms: int = 0
    near_tp_trigger_ts_ms: int = 0
    near_tp_protective_sl_price: float | None = None
    near_tp_protective_sl_order_id: str | None = None
    near_tp_add_disabled: bool = False


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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

        near_tp_intent = self._maybe_near_tp_reduce(price, ts_ms, boll, cvd)
        if near_tp_intent is not None:
            intents.append(near_tp_intent)

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
            self._update_lower_deep_enough(boll)
            if self.state.upper_armed:
                logger.info("UPPER_ARMED_RESET | reason=opposite_lower_break price=%.4f", price)
            self._reset_upper_armed()
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
            self._update_upper_deep_enough(boll)
            if self.state.lower_armed:
                logger.info("LOWER_ARMED_RESET | reason=opposite_upper_break price=%.4f", price)
            self._reset_lower_armed()
            return

        # If price mean-reverts all the way to the middle, the original outside-band
        # opportunity is considered stale.
        if self.state.lower_armed and price >= boll.middle:
            logger.info("LOWER_ARMED_RESET | reason=middle_reclaimed price=%.4f middle=%.4f", price, boll.middle)
            self._reset_lower_armed()
        if self.state.upper_armed and price <= boll.middle:
            logger.info("UPPER_ARMED_RESET | reason=middle_reclaimed price=%.4f middle=%.4f", price, boll.middle)
            self._reset_upper_armed()

    def _update_lower_deep_enough(self, boll: BollSnapshot) -> None:
        if self.state.lower_deep_enough or self.state.lower_extreme_price is None:
            return
        threshold = boll.lower * (1 - self.config.min_outside_pct)
        if self.state.lower_extreme_price <= threshold:
            self.state.lower_deep_enough = True
            logger.info(
                "LOWER_DEEP_ENOUGH | extreme=%.4f lower=%.4f required_below=%.4f min_outside=%.4f%%",
                self.state.lower_extreme_price,
                boll.lower,
                threshold,
                self.config.min_outside_pct * 100,
            )

    def _update_upper_deep_enough(self, boll: BollSnapshot) -> None:
        if self.state.upper_deep_enough or self.state.upper_extreme_price is None:
            return
        threshold = boll.upper * (1 + self.config.min_outside_pct)
        if self.state.upper_extreme_price >= threshold:
            self.state.upper_deep_enough = True
            logger.info(
                "UPPER_DEEP_ENOUGH | extreme=%.4f upper=%.4f required_above=%.4f min_outside=%.4f%%",
                self.state.upper_extreme_price,
                boll.upper,
                threshold,
                self.config.min_outside_pct * 100,
            )

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
        self.state.lower_last_burst_ts_ms = 0
        self.state.lower_deep_enough = False

    def _reset_upper_armed(self) -> None:
        self.state.upper_armed = False
        self.state.upper_extreme_price = None
        self.state.upper_armed_ts_ms = 0
        self.state.upper_last_burst_ts_ms = 0
        self.state.upper_deep_enough = False

    def _long_setup(self, price: float, cvd: CvdSnapshot) -> bool:
        if not self.state.lower_armed or self.state.lower_extreme_price is None:
            return False
        if not self.state.lower_deep_enough:
            return False
        if not self._near_lower_extreme(price):
            return False
        cvd_reclaim = cvd.cross_positive and cvd.buy_ratio >= self.config.min_buy_ratio and cvd.no_new_low
        cvd_absorption = cvd.cvd_increasing and cvd.buy_ratio >= self.config.min_buy_ratio and cvd.no_new_low
        return cvd_reclaim or cvd_absorption

    def _short_setup(self, price: float, cvd: CvdSnapshot) -> bool:
        if not self.state.upper_armed or self.state.upper_extreme_price is None:
            return False
        if not self.state.upper_deep_enough:
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
            return self._open_position("LONG", "OPEN_LONG", price, ts_ms, boll, cvd, "下轨出轨深度达标 + 低点附近快速CVD回流/跌不动")
        if self.state.side != "LONG":
            return None
        if self.state.near_tp_add_disabled:
            logger.info("ADD_SKIPPED | reason=near_tp_protected side=LONG price=%.4f layers=%s", price, self.state.layers)
            return None
        if self.state.layers >= self.config.max_layers:
            return None
        if self.state.last_entry_price is None:
            return None
        target_layer = self.state.layers + 1
        timing_ok, timing_reason = self._add_timing_passed("LONG", price, ts_ms)
        if not timing_ok:
            self._log_add_timing_skipped("LONG", timing_reason, price, ts_ms, target_layer)
            return None
        gap_ok, gap_pct, required_price = self._add_gap_passed("LONG", price, target_layer)
        if not gap_ok:
            logger.info(
                "ADD_SKIPPED | reason=add_gap side=LONG price=%.4f layers=%s target_layer=%s last_entry=%.4f required_price=%.4f gap_pct=%.4f%%",
                price,
                self.state.layers,
                target_layer,
                self.state.last_entry_price,
                required_price,
                gap_pct * 100,
            )
            return None
        logger.info(
            "ADD_GAP_PASSED | side=LONG price=%.4f layers=%s target_layer=%s last_entry=%.4f required_price=%.4f gap_pct=%.4f%%",
            price,
            self.state.layers,
            target_layer,
            self.state.last_entry_price,
            required_price,
            gap_pct * 100,
        )
        avg_improvement_ok, improvement_pct, projected_avg = self._add_avg_improvement_passed("LONG", price, target_layer)
        if not avg_improvement_ok:
            logger.info(
                "ADD_SKIPPED | reason=avg_improvement side=LONG price=%.4f layers=%s target_layer=%s avg_entry=%.4f projected_avg_entry=%.4f improvement_pct=%.6f required_improvement_pct=%.6f",
                price,
                self.state.layers,
                target_layer,
                self.state.avg_entry_price,
                projected_avg,
                improvement_pct,
                self.config.add_min_avg_improvement_pct,
            )
            return None
        logger.info(
            "ADD_AVG_IMPROVEMENT_PASSED | side=LONG price=%.4f layers=%s target_layer=%s avg_entry=%.4f projected_avg_entry=%.4f improvement_pct=%.6f required_improvement_pct=%.6f",
            price,
            self.state.layers,
            target_layer,
            self.state.avg_entry_price,
            projected_avg,
            improvement_pct,
            self.config.add_min_avg_improvement_pct,
        )
        return self._open_position(
            "LONG",
            "ADD_LONG",
            price,
            ts_ms,
            boll,
            cvd,
            f"距离上一多仓超过{gap_pct * 100:.2f}% + 补仓后均价改善{improvement_pct * 100:.2f}% + 新出轨深度达标后低点附近再次跌不动",
        )

    def _maybe_open_or_add_short(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None:
            return self._open_position("SHORT", "OPEN_SHORT", price, ts_ms, boll, cvd, "上轨出轨深度达标 + 高点附近快速CVD转弱/涨不动")
        if self.state.side != "SHORT":
            return None
        if self.state.near_tp_add_disabled:
            logger.info("ADD_SKIPPED | reason=near_tp_protected side=SHORT price=%.4f layers=%s", price, self.state.layers)
            return None
        if self.state.layers >= self.config.max_layers:
            return None
        if self.state.last_entry_price is None:
            return None
        target_layer = self.state.layers + 1
        timing_ok, timing_reason = self._add_timing_passed("SHORT", price, ts_ms)
        if not timing_ok:
            self._log_add_timing_skipped("SHORT", timing_reason, price, ts_ms, target_layer)
            return None
        gap_ok, gap_pct, required_price = self._add_gap_passed("SHORT", price, target_layer)
        if not gap_ok:
            logger.info(
                "ADD_SKIPPED | reason=add_gap side=SHORT price=%.4f layers=%s target_layer=%s last_entry=%.4f required_price=%.4f gap_pct=%.4f%%",
                price,
                self.state.layers,
                target_layer,
                self.state.last_entry_price,
                required_price,
                gap_pct * 100,
            )
            return None
        logger.info(
            "ADD_GAP_PASSED | side=SHORT price=%.4f layers=%s target_layer=%s last_entry=%.4f required_price=%.4f gap_pct=%.4f%%",
            price,
            self.state.layers,
            target_layer,
            self.state.last_entry_price,
            required_price,
            gap_pct * 100,
        )
        avg_improvement_ok, improvement_pct, projected_avg = self._add_avg_improvement_passed("SHORT", price, target_layer)
        if not avg_improvement_ok:
            logger.info(
                "ADD_SKIPPED | reason=avg_improvement side=SHORT price=%.4f layers=%s target_layer=%s avg_entry=%.4f projected_avg_entry=%.4f improvement_pct=%.6f required_improvement_pct=%.6f",
                price,
                self.state.layers,
                target_layer,
                self.state.avg_entry_price,
                projected_avg,
                improvement_pct,
                self.config.add_min_avg_improvement_pct,
            )
            return None
        logger.info(
            "ADD_AVG_IMPROVEMENT_PASSED | side=SHORT price=%.4f layers=%s target_layer=%s avg_entry=%.4f projected_avg_entry=%.4f improvement_pct=%.6f required_improvement_pct=%.6f",
            price,
            self.state.layers,
            target_layer,
            self.state.avg_entry_price,
            projected_avg,
            improvement_pct,
            self.config.add_min_avg_improvement_pct,
        )
        return self._open_position(
            "SHORT",
            "ADD_SHORT",
            price,
            ts_ms,
            boll,
            cvd,
            f"距离上一空仓超过{gap_pct * 100:.2f}% + 补仓后均价改善{improvement_pct * 100:.2f}% + 新出轨深度达标后高点附近再次涨不动",
        )

    def _add_layer_gap_pct_for_target_layer(self, target_layer: int) -> float:
        if target_layer >= 11:
            return self.config.add_layer_gap_pct_layer_11_plus
        if target_layer >= 9:
            return self.config.add_layer_gap_pct_layer_9_10
        if target_layer >= 7:
            return self.config.add_layer_gap_pct_layer_7_8
        return self.config.add_layer_gap_pct

    def _add_gap_passed(self, side: PositionSide, price: float, target_layer: int) -> tuple[bool, float, float]:
        gap_pct = self._add_layer_gap_pct_for_target_layer(target_layer)
        last = self.state.last_entry_price
        if last is None or last <= 0:
            return False, gap_pct, 0.0

        if side == "LONG":
            required_price = last * (1 - gap_pct)
            return price <= required_price, gap_pct, required_price

        required_price = last * (1 + gap_pct)
        return price >= required_price, gap_pct, required_price

    def _add_avg_improvement_passed(self, side: PositionSide, price: float, target_layer: int) -> tuple[bool, float, float]:
        required = self.config.add_min_avg_improvement_pct
        if required <= 0:
            return True, 0.0, self.state.avg_entry_price

        old_qty = self.state.total_entry_qty
        old_notional = self.state.total_entry_notional
        old_avg = self.state.avg_entry_price
        size = self.sizer.calculate(price, layer_index=target_layer)
        add_qty = size.eth_qty
        if old_qty <= 0 or old_notional <= 0 or old_avg <= 0 or add_qty <= 0:
            return False, 0.0, old_avg

        projected_qty = old_qty + add_qty
        projected_notional = old_notional + price * add_qty
        projected_avg = projected_notional / projected_qty
        if side == "LONG":
            improvement_pct = (old_avg - projected_avg) / old_avg
        else:
            improvement_pct = (projected_avg - old_avg) / old_avg
        return improvement_pct >= required, improvement_pct, projected_avg

    def _add_timing_passed(self, side: PositionSide, price: float, ts_ms: int) -> tuple[bool, str]:
        last = self.state.last_entry_price
        if last is None or last <= 0:
            return False, "missing_last_entry"

        elapsed_seconds = self._add_elapsed_seconds(ts_ms)
        if self.state.layers == 1:
            if elapsed_seconds < self.config.first_add_block_seconds:
                return False, "first_add_block"
            return True, "ok"

        if self.state.layers >= 2:
            adverse_gap_pct = self._adverse_gap_pct(side, price)
            if (
                elapsed_seconds < self.config.add_min_interval_seconds
                and adverse_gap_pct < self.config.add_min_interval_bypass_gap_pct
            ):
                return False, "add_interval"

        return True, "ok"

    def _add_elapsed_seconds(self, ts_ms: int) -> float:
        return max((ts_ms - self.state.last_order_ts_ms) / 1000, 0.0)

    def _adverse_gap_pct(self, side: PositionSide, price: float) -> float:
        last = self.state.last_entry_price
        if last is None or last <= 0:
            return 0.0
        if side == "LONG":
            return (last - price) / last
        return (price - last) / last

    def _log_add_timing_skipped(self, side: PositionSide, reason: str, price: float, ts_ms: int, target_layer: int) -> None:
        last = self.state.last_entry_price if self.state.last_entry_price is not None else 0.0
        elapsed_seconds = self._add_elapsed_seconds(ts_ms)
        if reason == "first_add_block":
            logger.info(
                "ADD_SKIPPED | reason=first_add_block side=%s price=%.4f layers=%s target_layer=%s last_entry=%.4f elapsed_seconds=%.1f required_seconds=%s",
                side,
                price,
                self.state.layers,
                target_layer,
                last,
                elapsed_seconds,
                self.config.first_add_block_seconds,
            )
            return
        if reason == "add_interval":
            adverse_gap_pct = self._adverse_gap_pct(side, price)
            logger.info(
                "ADD_SKIPPED | reason=add_interval side=%s price=%.4f layers=%s target_layer=%s last_entry=%.4f elapsed_seconds=%.1f required_seconds=%s adverse_gap_pct=%.4f%% bypass_gap_pct=%.4f%%",
                side,
                price,
                self.state.layers,
                target_layer,
                last,
                elapsed_seconds,
                self.config.add_min_interval_seconds,
                adverse_gap_pct * 100,
                self.config.add_min_interval_bypass_gap_pct * 100,
            )
            return
        logger.info(
            "ADD_SKIPPED | reason=%s side=%s price=%.4f layers=%s target_layer=%s last_entry=%.4f elapsed_seconds=%.1f",
            reason,
            side,
            price,
            self.state.layers,
            target_layer,
            last,
            elapsed_seconds,
        )

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
        self.state.partial_tp_consumed = False
        self._reset_near_tp_state()
        tp_price, tp_mode = self._select_tp_price(side, boll)
        partial_tp_price, partial_tp_ratio, tp_plan = self._select_tp_plan(side, tp_price, next_layer)
        if tp_mode != "MIDDLE":
            reason = f"{reason} + 中轨净利润不足阈值，TP切换到{tp_mode}"
        if tp_plan == "SPLIT_PARTIAL_FINAL":
            reason = f"{reason} + 总层数>= {self.config.split_tp_min_layers}，启用分批止盈"
        self.state.side = side
        self.state.layers = next_layer
        self.state.last_entry_price = price
        self.state.tp_price = tp_price
        self.state.tp_mode = tp_mode
        self.state.partial_tp_price = partial_tp_price
        self.state.partial_tp_ratio = partial_tp_ratio
        self.state.tp_plan = tp_plan
        self.state.last_order_ts_ms = ts_ms
        self.state.last_tp_update_ts_ms = ts_ms
        self.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms
        logger.info(
            "TP_SELECTED | reason=entry side=%s mode=%s plan=%s partial_tp=%s partial_ratio=%.2f avg_entry=%.4f breakeven=%.4f candle_ts=%s middle=%.4f upper=%.4f lower=%.4f final_tp=%.4f",
            side,
            tp_mode,
            tp_plan,
            f"{partial_tp_price:.4f}" if partial_tp_price is not None else "-",
            partial_tp_ratio,
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
        if self.state.near_tp_protected or self.state.near_tp_add_disabled:
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
        else:
            partial_tp_price, partial_tp_ratio, tp_plan = self._select_tp_plan(self.state.side, tp_price, self.state.layers)
        self.state.last_tp_update_ts_ms = ts_ms
        self.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms

        if self._tp_plan_unchanged(tp_price, partial_tp_price, partial_tp_ratio, tp_plan):
            logger.info(
                "TP_UPDATE_SKIPPED | reason=plan_unchanged side=%s mode=%s plan=%s candle_ts=%s current_tp=%.4f target_tp=%.4f partial_tp=%s avg_entry=%.4f breakeven=%.4f",
                self.state.side,
                tp_mode,
                tp_plan,
                boll.candle_ts_ms,
                self.state.tp_price,
                tp_price,
                f"{partial_tp_price:.4f}" if partial_tp_price is not None else "-",
                self.state.avg_entry_price,
                self.state.breakeven_price,
            )
            return None

        self.state.tp_price = tp_price
        self.state.tp_mode = tp_mode
        self.state.partial_tp_price = partial_tp_price
        self.state.partial_tp_ratio = partial_tp_ratio
        self.state.tp_plan = tp_plan
        size = self.sizer.calculate(price, layer_index=self.state.layers)
        logger.info(
            "TP_SELECTED | reason=new_candle side=%s mode=%s plan=%s partial_tp=%s partial_ratio=%.2f avg_entry=%.4f breakeven=%.4f candle_ts=%s middle=%.4f upper=%.4f lower=%.4f final_tp=%.4f",
            self.state.side,
            tp_mode,
            tp_plan,
            f"{partial_tp_price:.4f}" if partial_tp_price is not None else "-",
            partial_tp_ratio,
            self.state.avg_entry_price,
            self.state.breakeven_price,
            boll.candle_ts_ms,
            boll.middle,
            boll.upper,
            boll.lower,
            tp_price,
        )
        return self._intent("UPDATE_TP", self.state.side, price, self.state.layers, tp_price, f"新15m K线更新止盈到{tp_mode}轨", size, boll, cvd, ts_ms)

    def _maybe_near_tp_reduce(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if not self.config.near_tp_enabled:
            return None
        if self.state.side is None or self.state.tp_price is None:
            return None
        if self.state.avg_entry_price <= 0 or price <= 0:
            return None
        if self.state.near_tp_protected:
            return None
        if self.state.tp_plan == "SPLIT_PARTIAL_FINAL" and not self.state.partial_tp_consumed:
            return None

        side = self.state.side
        avg = self.state.avg_entry_price
        final_tp = self.state.tp_price
        if side == "LONG":
            if final_tp <= avg:
                return None
            progress = (price - avg) / (final_tp - avg)
            profit_pct = (price - avg) / avg
            near_by_distance = final_tp - price <= self.config.near_tp_max_distance_usd
        else:
            if final_tp >= avg:
                return None
            progress = (avg - price) / (avg - final_tp)
            profit_pct = (avg - price) / avg
            near_by_distance = price - final_tp <= self.config.near_tp_max_distance_usd

        reduce_profit_ok = profit_pct >= self.config.near_tp_min_reduce_profit_pct
        min_profit_seen_ok = profit_pct >= self.config.near_tp_min_profit_pct
        near_by_progress = progress >= self.config.near_tp_min_progress_ratio

        if not self.state.near_tp_armed:
            if (near_by_progress or near_by_distance) and min_profit_seen_ok:
                self.state.near_tp_armed = True
                self.state.near_tp_best_price = price
                self.state.near_tp_armed_ts_ms = ts_ms
                logger.warning(
                    "NEAR_TP_ARMED | side=%s price=%.4f avg_entry=%.4f final_tp=%.4f progress=%.6f profit_pct=%.6f near_by_progress=%s near_by_distance=%s",
                    side,
                    price,
                    avg,
                    final_tp,
                    progress,
                    profit_pct,
                    near_by_progress,
                    near_by_distance,
                )
            else:
                return None

        old_best = self.state.near_tp_best_price if self.state.near_tp_best_price is not None else price
        if side == "LONG":
            best = max(old_best, price)
        else:
            best = min(old_best, price)
        if best != old_best:
            self.state.near_tp_best_price = best
            logger.info("NEAR_TP_BEST_UPDATED | side=%s best_price=%.4f price=%.4f", side, best, price)
        else:
            self.state.near_tp_best_price = best

        if self.state.near_tp_reduce_pending:
            if reduce_profit_ok:
                return self._near_tp_reduce_intent(price, ts_ms, boll, cvd, progress, best, 0.0, 0.0)
            return None

        if side == "LONG":
            giveback = best - price
            floating_profit_path = best - avg
        else:
            giveback = price - best
            floating_profit_path = avg - best
        giveback_threshold = max(
            self.config.near_tp_giveback_usd,
            price * self.config.near_tp_giveback_pct,
            floating_profit_path * self.config.near_tp_giveback_profit_ratio,
        )
        if giveback < giveback_threshold:
            return None

        logger.warning(
            "NEAR_TP_GIVEBACK_TRIGGERED | side=%s price=%.4f best_price=%.4f avg_entry=%.4f final_tp=%.4f giveback=%.6f threshold=%.6f profit_pct=%.6f",
            side,
            price,
            best,
            avg,
            final_tp,
            giveback,
            giveback_threshold,
            profit_pct,
        )
        if not reduce_profit_ok:
            self.state.near_tp_reduce_pending = True
            self.state.near_tp_pending_ts_ms = ts_ms
            logger.warning(
                "NEAR_TP_REDUCE_PENDING | reason=profit_below_min_reduce_profit side=%s price=%.4f profit_pct=%.6f min_reduce_profit_pct=%.6f",
                side,
                price,
                profit_pct,
                self.config.near_tp_min_reduce_profit_pct,
            )
            return None

        return self._near_tp_reduce_intent(price, ts_ms, boll, cvd, progress, best, giveback, giveback_threshold)

    def _near_tp_reduce_intent(
        self,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        progress: float,
        best: float,
        giveback: float,
        giveback_threshold: float,
    ) -> TradeIntent | None:
        side = self.state.side
        if side is None or self.state.tp_price is None:
            return None
        pct = self.config.near_tp_protective_sl_profit_pct
        protective_sl = self.state.avg_entry_price * (1 + pct) if side == "LONG" else self.state.avg_entry_price * (1 - pct)
        size = self.sizer.calculate(price, layer_index=max(self.state.layers, 1))
        if self.config.near_tp_shadow_enabled and not self.config.near_tp_reduce_enabled:
            logger.warning(
                "NEAR_TP_REDUCE_SHADOW | side=%s price=%.4f avg_entry=%.4f final_tp=%.4f progress=%.6f best_price=%.4f giveback=%.6f threshold=%.6f reduce_ratio=%.4f protective_sl=%.4f",
                side,
                price,
                self.state.avg_entry_price,
                self.state.tp_price,
                progress,
                best,
                giveback,
                giveback_threshold,
                self.config.near_tp_reduce_ratio,
                protective_sl,
            )
            return None
        if not self.config.near_tp_reduce_enabled:
            return None

        self.state.near_tp_trigger_ts_ms = ts_ms
        logger.warning(
            "NEAR_TP_REDUCE_SIGNAL | side=%s price=%.4f avg_entry=%.4f final_tp=%.4f progress=%.6f best_price=%.4f giveback=%.6f threshold=%.6f reduce_ratio=%.4f protective_sl=%.4f",
            side,
            price,
            self.state.avg_entry_price,
            self.state.tp_price,
            progress,
            best,
            giveback,
            giveback_threshold,
            self.config.near_tp_reduce_ratio,
            protective_sl,
        )
        return TradeIntent(
            intent_type="NEAR_TP_REDUCE",
            side=side,
            price=price,
            layer_index=self.state.layers,
            tp_price=self.state.tp_price,
            reason="near_tp_giveback_protection",
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
            partial_tp_price=None,
            partial_tp_ratio=0.0,
            tp_plan="SINGLE",
            partial_tp_consumed=True,
            near_tp_progress_ratio=progress,
            near_tp_best_price=best,
            near_tp_giveback=giveback,
            near_tp_giveback_threshold=giveback_threshold,
            near_tp_reduce_ratio=self.config.near_tp_reduce_ratio,
            near_tp_protective_sl_price=protective_sl,
        )

    def _update_position_cost(self, entry_price: float, eth_qty: float) -> None:
        if eth_qty <= 0:
            return
        self.state.total_entry_qty += eth_qty
        self.state.total_entry_notional += entry_price * eth_qty
        self.state.avg_entry_price = self.state.total_entry_notional / self.state.total_entry_qty

    def _reset_near_tp_state(self) -> None:
        self.state.near_tp_armed = False
        self.state.near_tp_reduce_pending = False
        self.state.near_tp_protected = False
        self.state.near_tp_best_price = None
        self.state.near_tp_armed_ts_ms = 0
        self.state.near_tp_pending_ts_ms = 0
        self.state.near_tp_trigger_ts_ms = 0
        self.state.near_tp_protective_sl_price = None
        self.state.near_tp_protective_sl_order_id = None
        self.state.near_tp_add_disabled = False

    def _select_tp_price(self, side: PositionSide, boll: BollSnapshot) -> tuple[float, TpMode]:
        if self.state.avg_entry_price <= 0:
            return boll.middle, "MIDDLE"
        fee = self.config.breakeven_fee_buffer_pct
        min_net_profit = self.config.tp_min_net_profit_pct
        required_profit_pct = fee + min_net_profit
        if side == "LONG":
            self.state.breakeven_price = self.state.avg_entry_price * (1 + fee)
            middle_required_price = self.state.avg_entry_price * (1 + required_profit_pct)
            if boll.middle < middle_required_price:
                return boll.upper, "UPPER"
            return boll.middle, "MIDDLE"
        self.state.breakeven_price = self.state.avg_entry_price * (1 - fee)
        middle_required_price = self.state.avg_entry_price * (1 - required_profit_pct)
        if boll.middle > middle_required_price:
            return boll.lower, "LOWER"
        return boll.middle, "MIDDLE"

    def _select_tp_plan(self, side: PositionSide, final_tp: float, layers: int) -> tuple[float | None, float, TpPlan]:
        if layers < self.config.split_tp_min_layers:
            return None, 0.0, "SINGLE"
        if self.state.partial_tp_consumed:
            return None, 0.0, "SINGLE"
        avg_entry = self.state.avg_entry_price
        if avg_entry <= 0 or final_tp <= 0:
            return None, 0.0, "SINGLE"
        partial_ratio = min(max(self.config.split_tp_partial_ratio, 0.0), 1.0)
        path_ratio = min(max(self.config.split_tp_path_ratio, 0.0), 1.0)
        if partial_ratio <= 0 or partial_ratio >= 1 or path_ratio <= 0 or path_ratio >= 1:
            return None, 0.0, "SINGLE"
        min_profit_pct = abs(self.config.split_tp_min_profit_pct)

        if side == "LONG":
            min_tp = avg_entry * (1 + min_profit_pct)
            if final_tp <= min_tp:
                return None, 0.0, "SINGLE"
            path_tp = avg_entry + (final_tp - avg_entry) * path_ratio
            partial_tp = max(path_tp, min_tp)
            if partial_tp >= final_tp:
                return None, 0.0, "SINGLE"
            return partial_tp, partial_ratio, "SPLIT_PARTIAL_FINAL"

        min_tp = avg_entry * (1 - min_profit_pct)
        if final_tp >= min_tp:
            return None, 0.0, "SINGLE"
        path_tp = avg_entry - (avg_entry - final_tp) * path_ratio
        partial_tp = min(path_tp, min_tp)
        if partial_tp <= final_tp:
            return None, 0.0, "SINGLE"
        return partial_tp, partial_ratio, "SPLIT_PARTIAL_FINAL"

    def _tp_plan_unchanged(self, tp_price: float, partial_tp_price: float | None, partial_tp_ratio: float, tp_plan: TpPlan) -> bool:
        if self.state.tp_price is None:
            return False
        if abs(self.state.tp_price - tp_price) / tp_price >= 0.0001:
            return False
        if self.state.tp_plan != tp_plan:
            return False
        if abs(self.state.partial_tp_ratio - partial_tp_ratio) >= 0.0001:
            return False
        if self.state.partial_tp_price is None or partial_tp_price is None:
            return self.state.partial_tp_price is None and partial_tp_price is None
        return abs(self.state.partial_tp_price - partial_tp_price) / partial_tp_price < 0.0001

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
            partial_tp_price=self.state.partial_tp_price,
            partial_tp_ratio=self.state.partial_tp_ratio,
            tp_plan=self.state.tp_plan,
            partial_tp_consumed=self.state.partial_tp_consumed,
        )

    def _cooldown_ok(self, ts_ms: int) -> bool:
        return ts_ms - self.state.last_order_ts_ms >= self.config.order_cooldown_seconds * 1000
