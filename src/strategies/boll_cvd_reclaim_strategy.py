from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.position_management.sidecar.model import trim_sidecar_legs_for_state
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
    "MARKET_EXIT_RUNNER",
]
PositionSide = Literal["LONG", "SHORT"]
TpMode = Literal["MIDDLE", "UPPER", "LOWER"]
TpPlan = Literal["SINGLE", "SPLIT_PARTIAL_FINAL", "MIDDLE_RUNNER", "THREE_STAGE_RUNNER"]


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
    split_tp_enabled: bool = True
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
    middle_runner_enabled: bool = False
    middle_runner_first_close_ratio: float = 0.8
    middle_runner_extension_trigger_ratio: float = 0.6
    middle_runner_disable_add_after_partial: bool = True
    middle_runner_protective_sl_enabled: bool = True
    three_stage_runner_enabled: bool = False
    three_stage_tp1_ratio: float = 0.60
    three_stage_tp2_ratio: float = 0.20
    three_stage_runner_ratio: float = 0.20
    three_stage_post_tp1_protective_sl_enabled: bool = True
    three_stage_post_tp1_sl_extension_trigger_ratio: float = 0.6
    runner_dynamic_enabled: bool = True
    # Reserved for future timer-based updates; current runner TP/SL refreshes are driven by 15m BOLL candle_ts_ms.
    runner_dynamic_update_seconds: int = 900
    runner_tp_initial_outer_extra_pct: float = 0.010
    runner_tp_step_pct: float = 0.001
    runner_tp_min_outer_extra_pct: float = 0.004
    runner_sl_initial_outer_distance_ratio: float = 1.00
    runner_sl_step_ratio: float = 0.10
    runner_sl_min_outer_distance_ratio: float = 0.50
    runner_reverse_burst_exit_enabled: bool = True
    runner_reverse_burst_arm_delay_seconds: int = 60
    runner_reverse_burst_confirm_seconds: int = 5
    runner_reverse_sell_ratio: float = 0.58
    runner_reverse_buy_ratio: float = 0.58
    runner_reverse_strong_ratio: float = 0.62
    runner_reverse_min_price_damage_pct: float = 0.0015
    runner_reverse_recovery_cancel_pct: float = 0.001
    runner_max_trend_seconds_after_second_tp: int = 18000

    @classmethod
    def from_env(cls) -> "BollCvdReclaimStrategyConfig":
        middle_runner_first_close_ratio = min(max(float(os.getenv("MIDDLE_RUNNER_FIRST_CLOSE_RATIO", "0.8")), 0.1), 0.95)
        middle_runner_enabled = _env_bool("MIDDLE_RUNNER_ENABLED", False)
        near_tp_enabled = _env_bool("NEAR_TP_ENABLED", False)
        three_stage_runner_enabled = _env_bool("THREE_STAGE_RUNNER_ENABLED", False)
        if middle_runner_enabled and near_tp_enabled:
            raise RuntimeError("MIDDLE_RUNNER_ENABLED=true requires NEAR_TP_ENABLED=false; Middle Runner and Near-TP Reduce are mutually exclusive.")
        if three_stage_runner_enabled and near_tp_enabled:
            raise RuntimeError("THREE_STAGE_RUNNER_ENABLED=true requires NEAR_TP_ENABLED=false; Three-Stage Runner and Near-TP Reduce are mutually exclusive.")
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
            split_tp_enabled=_env_bool("SPLIT_TP_ENABLED", True),
            split_tp_min_layers=int(os.getenv("SPLIT_TP_MIN_LAYERS", "4")),
            split_tp_path_ratio=float(os.getenv("SPLIT_TP_PATH_RATIO", "0.8")),
            split_tp_partial_ratio=float(os.getenv("SPLIT_TP_PARTIAL_RATIO", "0.5")),
            split_tp_min_profit_pct=float(os.getenv("SPLIT_TP_MIN_PROFIT_PCT", "0.004")),
            near_tp_enabled=near_tp_enabled,
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
            middle_runner_enabled=middle_runner_enabled,
            middle_runner_first_close_ratio=middle_runner_first_close_ratio,
            middle_runner_extension_trigger_ratio=float(os.getenv("MIDDLE_RUNNER_EXTENSION_TRIGGER_RATIO", "0.6")),
            middle_runner_disable_add_after_partial=_env_bool("MIDDLE_RUNNER_DISABLE_ADD_AFTER_PARTIAL", True),
            middle_runner_protective_sl_enabled=_env_bool("MIDDLE_RUNNER_PROTECTIVE_SL_ENABLED", True),
            three_stage_runner_enabled=three_stage_runner_enabled,
            three_stage_tp1_ratio=float(os.getenv("THREE_STAGE_TP1_RATIO", "0.60")),
            three_stage_tp2_ratio=float(os.getenv("THREE_STAGE_TP2_RATIO", "0.20")),
            three_stage_runner_ratio=float(os.getenv("THREE_STAGE_RUNNER_RATIO", "0.20")),
            three_stage_post_tp1_protective_sl_enabled=_env_bool("THREE_STAGE_POST_TP1_PROTECTIVE_SL_ENABLED", True),
            three_stage_post_tp1_sl_extension_trigger_ratio=float(os.getenv("THREE_STAGE_POST_TP1_SL_EXTENSION_TRIGGER_RATIO", "0.6")),
            runner_dynamic_enabled=_env_bool("RUNNER_DYNAMIC_ENABLED", True),
            runner_dynamic_update_seconds=int(os.getenv("RUNNER_DYNAMIC_UPDATE_SECONDS", "900")),
            runner_tp_initial_outer_extra_pct=float(os.getenv("RUNNER_TP_INITIAL_OUTER_EXTRA_PCT", "0.010")),
            runner_tp_step_pct=float(os.getenv("RUNNER_TP_STEP_PCT", "0.001")),
            runner_tp_min_outer_extra_pct=float(os.getenv("RUNNER_TP_MIN_OUTER_EXTRA_PCT", "0.004")),
            runner_sl_initial_outer_distance_ratio=float(os.getenv("RUNNER_SL_INITIAL_OUTER_DISTANCE_RATIO", "1.00")),
            runner_sl_step_ratio=float(os.getenv("RUNNER_SL_STEP_RATIO", "0.10")),
            runner_sl_min_outer_distance_ratio=float(os.getenv("RUNNER_SL_MIN_OUTER_DISTANCE_RATIO", "0.50")),
            runner_reverse_burst_exit_enabled=_env_bool("RUNNER_REVERSE_BURST_EXIT_ENABLED", True),
            runner_reverse_burst_arm_delay_seconds=int(os.getenv("RUNNER_REVERSE_BURST_ARM_DELAY_SECONDS", "60")),
            runner_reverse_burst_confirm_seconds=int(os.getenv("RUNNER_REVERSE_BURST_CONFIRM_SECONDS", "5")),
            runner_reverse_sell_ratio=float(os.getenv("RUNNER_REVERSE_SELL_RATIO", "0.58")),
            runner_reverse_buy_ratio=float(os.getenv("RUNNER_REVERSE_BUY_RATIO", "0.58")),
            runner_reverse_strong_ratio=float(os.getenv("RUNNER_REVERSE_STRONG_RATIO", "0.62")),
            runner_reverse_min_price_damage_pct=float(os.getenv("RUNNER_REVERSE_MIN_PRICE_DAMAGE_PCT", "0.0015")),
            runner_reverse_recovery_cancel_pct=float(os.getenv("RUNNER_REVERSE_RECOVERY_CANCEL_PCT", "0.001")),
            runner_max_trend_seconds_after_second_tp=int(os.getenv("RUNNER_MAX_TREND_SECONDS_AFTER_SECOND_TP", "18000")),
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
    middle_runner_enabled_for_position: bool = False
    middle_runner_pending: bool = False
    middle_runner_active: bool = False
    middle_runner_first_close_ratio: float = 0.0
    middle_runner_keep_ratio: float = 0.0
    middle_runner_first_tp_price: float | None = None
    middle_runner_final_tp_price: float | None = None
    middle_runner_protective_sl_price: float | None = None
    middle_runner_protective_sl_order_id: str | None = None
    middle_runner_extension_triggered: bool = False
    middle_runner_add_disabled: bool = False
    three_stage_tp1_price: float | None = None
    three_stage_tp1_ratio: float = 0.0
    three_stage_tp2_price: float | None = None
    three_stage_tp2_ratio: float = 0.0
    three_stage_runner_tp_price: float | None = None
    three_stage_runner_ratio: float = 0.0
    three_stage_runner_sl_price: float | None = None
    three_stage_tp1_consumed: bool = False
    three_stage_tp2_consumed: bool = False
    three_stage_post_tp1_protective_sl_price: float | None = None
    three_stage_post_tp1_protective_sl_order_id: str | None = None
    three_stage_post_tp1_sl_extension_triggered: bool = False
    three_stage_post_tp1_protected: bool = False
    trend_runner_active: bool = False
    trend_runner_tp_price: float | None = None
    trend_runner_sl_price: float | None = None
    trend_runner_tp_order_id: str | None = None
    trend_runner_sl_order_id: str | None = None
    trend_runner_exit_reason: str | None = None
    trend_runner_adjust_count: int = 0
    protected_order_ids: tuple[str, ...] = ()
    managed_core_contracts: str | None = None
    managed_core_eth_qty: float = 0.0


@dataclass
class StrategyPositionState:
    side: Optional[PositionSide] = None
    layers: int = 0
    last_entry_price: Optional[float] = None
    tp_price: Optional[float] = None
    last_order_ts_ms: int = 0
    first_entry_ts_ms: int = 0
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
    middle_runner_enabled_for_position: bool = False
    middle_runner_pending: bool = False
    middle_runner_active: bool = False
    middle_runner_first_close_ratio: float = 0.0
    middle_runner_keep_ratio: float = 0.0
    middle_runner_first_tp_price: float | None = None
    middle_runner_final_tp_price: float | None = None
    middle_runner_protective_sl_price: float | None = None
    middle_runner_protective_sl_order_id: str | None = None
    middle_runner_extension_triggered: bool = False
    middle_runner_add_disabled: bool = False
    middle_runner_size_mismatch_protected: bool = False
    middle_runner_size_mismatch_warning_ts_ms: int = 0
    three_stage_runner_enabled_for_position: bool = False
    three_stage_tp1_price: float | None = None
    three_stage_tp2_price: float | None = None
    three_stage_runner_initial_tp_price: float | None = None
    three_stage_tp1_ratio: float = 0.0
    three_stage_tp2_ratio: float = 0.0
    three_stage_runner_ratio: float = 0.0
    three_stage_tp1_consumed: bool = False
    three_stage_tp2_consumed: bool = False
    three_stage_post_tp1_protective_sl_price: float | None = None
    three_stage_post_tp1_protective_sl_order_id: str | None = None
    three_stage_post_tp1_sl_extension_triggered: bool = False
    three_stage_post_tp1_protected: bool = False
    trend_runner_active: bool = False
    trend_runner_trend_start_ts_ms: int = 0
    trend_runner_adjust_count: int = 0
    trend_runner_last_update_candle_ts_ms: int = 0
    trend_runner_tp_price: float | None = None
    trend_runner_sl_price: float | None = None
    trend_runner_tp_order_id: str | None = None
    trend_runner_sl_order_id: str | None = None
    trend_runner_exit_reason: str | None = None
    trend_runner_reverse_candidate: bool = False
    trend_runner_reverse_start_ts_ms: int = 0
    trend_runner_reverse_start_price: float | None = None
    trend_runner_reverse_extreme_price: float | None = None
    trend_runner_reverse_fast_cvd_start: float = 0.0
    trend_runner_reverse_samples: list = None
    sidecar_enabled_for_position: bool = False
    sidecar_margin_pct: float = 0.0
    sidecar_tp_pct: float = 0.0
    sidecar_total_qty: float = 0.0
    sidecar_open_qty: float = 0.0
    sidecar_total_notional: float = 0.0
    sidecar_realized_qty: float = 0.0
    sidecar_legs: list[dict] = field(default_factory=list)
    sidecar_dirty: bool = False
    sidecar_halt_reason: str | None = None
    core_contracts: str | None = None
    core_eth_qty: float = 0.0
    tp_order_id: str | None = None
    tp_order_ids: list[str] = field(default_factory=list)


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

        runner_exit_intent = self._maybe_trend_runner_market_exit(price, ts_ms, boll, cvd)
        if runner_exit_intent is not None:
            return [runner_exit_intent]

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
        if self.state.three_stage_runner_enabled_for_position or self.state.trend_runner_active:
            logger.info("ADD_SKIPPED | reason=three_stage_runner side=LONG price=%.4f layers=%s", price, self.state.layers)
            return None
        if self.state.middle_runner_add_disabled or self.state.middle_runner_active:
            logger.info("ADD_SKIPPED | reason=middle_runner_active side=LONG price=%.4f layers=%s", price, self.state.layers)
            return None
        if self.state.layers >= self.config.max_layers:
            return None
        if self.state.last_entry_price is None:
            return None
        target_layer = self.state.layers + 1
        timing_ok, timing_reason = self._add_timing_passed("LONG", price, ts_ms, target_layer)
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
        if self.state.three_stage_runner_enabled_for_position or self.state.trend_runner_active:
            logger.info("ADD_SKIPPED | reason=three_stage_runner side=SHORT price=%.4f layers=%s", price, self.state.layers)
            return None
        if self.state.middle_runner_add_disabled or self.state.middle_runner_active:
            logger.info("ADD_SKIPPED | reason=middle_runner_active side=SHORT price=%.4f layers=%s", price, self.state.layers)
            return None
        if self.state.layers >= self.config.max_layers:
            return None
        if self.state.last_entry_price is None:
            return None
        target_layer = self.state.layers + 1
        timing_ok, timing_reason = self._add_timing_passed("SHORT", price, ts_ms, target_layer)
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

    def _add_min_interval_bypass_gap_pct_for_target_layer(self, target_layer: int) -> float:
        return self._add_layer_gap_pct_for_target_layer(target_layer) * 2

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

    def _add_timing_passed(self, side: PositionSide, price: float, ts_ms: int, target_layer: int) -> tuple[bool, str]:
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
            bypass_gap_pct = self._add_min_interval_bypass_gap_pct_for_target_layer(target_layer)
            if (
                elapsed_seconds < self.config.add_min_interval_seconds
                and adverse_gap_pct < bypass_gap_pct
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
            bypass_gap_pct = self._add_min_interval_bypass_gap_pct_for_target_layer(target_layer)
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
                bypass_gap_pct * 100,
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
        if next_layer == 1:
            self.state.sidecar_enabled_for_position = bool(getattr(self.sizer.config, "sidecar_enabled", False))
            self.state.sidecar_margin_pct = (
                float(getattr(self.sizer.config, "sidecar_margin_pct", 0.0) or 0.0)
                if self.state.sidecar_enabled_for_position
                else 0.0
            )
            self.state.sidecar_tp_pct = (
                float(getattr(self.sizer.config, "sidecar_tp_pct", 0.0) or 0.0)
                if self.state.sidecar_enabled_for_position
                else 0.0
            )
            self.state.sidecar_total_qty = 0.0
            self.state.sidecar_open_qty = 0.0
            self.state.sidecar_total_notional = 0.0
            self.state.sidecar_realized_qty = 0.0
            self.state.sidecar_legs = []
            self.state.sidecar_dirty = False
            self.state.sidecar_halt_reason = None
        self._update_position_cost(price, size.eth_qty)
        self.state.partial_tp_consumed = False
        self._reset_near_tp_state()
        self._reset_middle_runner_state()
        self._reset_three_stage_runner_state()
        tp_price, tp_mode = self._select_tp_price(side, boll)
        partial_tp_price, partial_tp_ratio, tp_plan = self._select_tp_plan(side, tp_price, next_layer, tp_mode=tp_mode, boll=boll)
        if tp_plan == "MIDDLE_RUNNER":
            tp_price = boll.upper if side == "LONG" else boll.lower
        if tp_plan == "THREE_STAGE_RUNNER":
            tp_price = boll.upper if side == "LONG" else boll.lower
        if tp_mode != "MIDDLE":
            reason = f"{reason} + 中轨净利润不足阈值，TP切换到{tp_mode}"
        if tp_plan == "SPLIT_PARTIAL_FINAL":
            reason = f"{reason} + 总层数>= {self.config.split_tp_min_layers}，启用分批止盈"
        if tp_plan == "MIDDLE_RUNNER":
            reason = f"{reason} + 中轨先平{partial_tp_ratio * 100:.0f}%，剩余runner到外轨"
        if tp_plan == "THREE_STAGE_RUNNER":
            reason = f"{reason} + 三段式趋势Runner：中轨{self.config.three_stage_tp1_ratio * 100:.0f}%/外轨{self.config.three_stage_tp2_ratio * 100:.0f}%/Runner{self.config.three_stage_runner_ratio * 100:.0f}%"
        self.state.side = side
        self.state.layers = next_layer
        self.state.last_entry_price = price
        self.state.tp_price = tp_price
        self.state.tp_mode = tp_mode
        self.state.partial_tp_price = partial_tp_price
        self.state.partial_tp_ratio = partial_tp_ratio
        self.state.tp_plan = tp_plan
        if tp_plan == "MIDDLE_RUNNER":
            self._set_middle_runner_planned(partial_tp_price, tp_price)
        if tp_plan == "THREE_STAGE_RUNNER":
            self._set_three_stage_runner_planned(side, boll)
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
        if tp_plan == "THREE_STAGE_RUNNER":
            logger.warning(
                "THREE_STAGE_RUNNER_PLANNED | side=%s tp1=%.4f tp1_ratio=%.4f tp2=%.4f tp2_ratio=%.4f runner_tp=%s runner_sl=%s runner_ratio=%.4f",
                side,
                self.state.three_stage_tp1_price or 0.0,
                self.state.three_stage_tp1_ratio,
                self.state.three_stage_tp2_price or 0.0,
                self.state.three_stage_tp2_ratio,
                f"{self.state.trend_runner_tp_price:.4f}" if self.state.trend_runner_tp_price is not None else "-",
                f"{self.state.trend_runner_sl_price:.4f}" if self.state.trend_runner_sl_price is not None else "-",
                self.state.three_stage_runner_ratio,
            )
        return self._intent(intent_type, side, price, next_layer, tp_price, reason, size, boll, cvd, ts_ms)

    def _maybe_update_tp(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None or self.state.layers <= 0:
            return None
        trend_runner_needs_initial_orders = (
            self.state.trend_runner_active
            and (self.state.trend_runner_tp_price is None or self.state.trend_runner_sl_price is None)
        )
        if self.state.last_tp_update_candle_ts_ms == boll.candle_ts_ms and not trend_runner_needs_initial_orders:
            return None
        if self._three_stage_waiting_tp2():
            old_post_tp1_sl = self.state.three_stage_post_tp1_protective_sl_price
            if self.config.three_stage_post_tp1_protective_sl_enabled:
                calculated_sl = self._calculate_three_stage_post_tp1_protective_sl(self.state.side, price, boll)
                protective_sl = self._tighten_optional_three_stage_post_tp1_sl(self.state.side, old_post_tp1_sl, calculated_sl)
                extension_sl = self._apply_three_stage_post_tp1_extension_trigger(self.state.side, price, boll, protective_sl)
                protective_sl = self._tighten_optional_three_stage_post_tp1_sl(self.state.side, old_post_tp1_sl, extension_sl)
                self.state.three_stage_post_tp1_protective_sl_price = protective_sl
            else:
                protective_sl = old_post_tp1_sl
            self.state.last_tp_update_ts_ms = ts_ms
            self.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms
            post_tp1_sl_changed = (
                protective_sl is not None
                and (
                    old_post_tp1_sl is None
                    or abs(protective_sl - old_post_tp1_sl) / protective_sl >= 0.0001
                )
            )
            if post_tp1_sl_changed:
                size = self.sizer.calculate(price, layer_index=self.state.layers)
                logger.warning(
                    "THREE_STAGE_TP1_PROTECTIVE_SL_UPDATE_SIGNAL | side=%s old_sl=%s new_sl=%.4f candle_ts=%s tp2_price=%s",
                    self.state.side,
                    f"{old_post_tp1_sl:.4f}" if old_post_tp1_sl is not None else "-",
                    protective_sl,
                    boll.candle_ts_ms,
                    self.state.three_stage_tp2_price,
                )
                return self._intent("UPDATE_TP", self.state.side, price, self.state.layers, self.state.tp_price or self.state.three_stage_tp2_price or price, "three_stage_post_tp1_protective_sl_update", size, boll, cvd, ts_ms)
            logger.info(
                "TP_UPDATE_SKIPPED | reason=three_stage_waiting_tp2 side=%s candle_ts=%s tp2_price=%s protective_sl=%s",
                self.state.side,
                boll.candle_ts_ms,
                self.state.three_stage_tp2_price,
                protective_sl,
            )
            return None

        old_runner_sl = self.state.middle_runner_protective_sl_price
        old_trend_runner_tp = self.state.trend_runner_tp_price
        old_trend_runner_sl = self.state.trend_runner_sl_price
        tp_price, tp_mode = self._select_tp_price(self.state.side, boll)
        if self.state.trend_runner_active:
            tp_mode = "UPPER" if self.state.side == "LONG" else "LOWER"
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
            if self.config.runner_dynamic_enabled:
                tp_price, runner_sl, tp_extra_pct, sl_distance_ratio = self._calculate_trend_runner_dynamic_orders(
                    self.state.side,
                    boll,
                    self.state.trend_runner_adjust_count,
                    self.state.trend_runner_sl_price,
                )
                self.state.trend_runner_tp_price = tp_price
                self.state.trend_runner_sl_price = runner_sl
                logger.warning(
                    "TREND_RUNNER_UPDATE | side=%s old_tp=%s new_tp=%.4f old_sl=%s new_sl=%.4f adjust_count=%s tp_extra_pct=%.6f sl_distance_ratio=%.6f candle_ts=%s",
                    self.state.side,
                    f"{old_trend_runner_tp:.4f}" if old_trend_runner_tp is not None else "-",
                    tp_price,
                    f"{old_trend_runner_sl:.4f}" if old_trend_runner_sl is not None else "-",
                    runner_sl,
                    self.state.trend_runner_adjust_count,
                    tp_extra_pct,
                    sl_distance_ratio,
                    boll.candle_ts_ms,
                )
            else:
                tp_price = self.state.trend_runner_tp_price or tp_price
        elif self.state.middle_runner_active:
            tp_price = boll.upper if self.state.side == "LONG" else boll.lower
            tp_mode = "UPPER" if self.state.side == "LONG" else "LOWER"
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
            calculated_sl = self._calculate_middle_runner_protective_sl(self.state.side, price, boll)
            protective_sl = self._tighten_optional_middle_runner_sl(self.state.side, old_runner_sl, calculated_sl)
            extension_sl = self._apply_middle_runner_extension_trigger(self.state.side, price, boll, protective_sl)
            protective_sl = self._tighten_optional_middle_runner_sl(self.state.side, old_runner_sl, extension_sl)
            self.state.middle_runner_final_tp_price = tp_price
            self.state.middle_runner_protective_sl_price = protective_sl
        elif self.state.middle_runner_pending:
            logger.info(
                "TP_UPDATE_SKIPPED | reason=middle_runner_plan_locked side=%s candle_ts=%s first_tp=%s final_tp=%s",
                self.state.side,
                boll.candle_ts_ms,
                self.state.middle_runner_first_tp_price,
                self.state.middle_runner_final_tp_price,
            )
            self.state.last_tp_update_ts_ms = ts_ms
            self.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms
            return None
        elif self.state.three_stage_runner_enabled_for_position and not self.state.trend_runner_active:
            logger.info(
                "TP_UPDATE_SKIPPED | reason=three_stage_plan_locked side=%s candle_ts=%s tp1=%s tp2=%s",
                self.state.side,
                boll.candle_ts_ms,
                self.state.three_stage_tp1_price,
                self.state.three_stage_tp2_price,
            )
            self.state.last_tp_update_ts_ms = ts_ms
            self.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms
            return None
        elif self.state.near_tp_protected or self.state.near_tp_add_disabled:
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
        else:
            partial_tp_price, partial_tp_ratio, tp_plan = self._select_tp_plan(self.state.side, tp_price, self.state.layers, tp_mode=tp_mode, boll=boll)
            if tp_plan == "MIDDLE_RUNNER":
                tp_price = boll.upper if self.state.side == "LONG" else boll.lower
            if tp_plan == "THREE_STAGE_RUNNER":
                tp_price = boll.upper if self.state.side == "LONG" else boll.lower
            if tp_plan == "MIDDLE_RUNNER":
                self._set_middle_runner_planned(partial_tp_price, tp_price)
            elif tp_plan == "THREE_STAGE_RUNNER":
                self._set_three_stage_runner_planned(self.state.side, boll)
            elif self.state.middle_runner_pending and not self.state.middle_runner_active:
                self._reset_middle_runner_state()
            elif (
                self.state.three_stage_runner_enabled_for_position
                and not self.state.trend_runner_active
                and not self._three_stage_waiting_tp2()
            ):
                self._reset_three_stage_runner_state()
        self.state.last_tp_update_ts_ms = ts_ms
        self.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms

        runner_sl_changed = (
            self.state.middle_runner_active
            and self.state.middle_runner_protective_sl_price is not None
            and (
                old_runner_sl is None
                or abs(self.state.middle_runner_protective_sl_price - old_runner_sl) / self.state.middle_runner_protective_sl_price >= 0.0001
            )
        )
        trend_runner_orders_changed = (
            self.state.trend_runner_active
            and (
                old_trend_runner_tp is None
                or old_trend_runner_sl is None
                or self.state.trend_runner_tp_price is None
                or self.state.trend_runner_sl_price is None
                or abs(self.state.trend_runner_tp_price - old_trend_runner_tp) / self.state.trend_runner_tp_price >= 0.0001
                or abs(self.state.trend_runner_sl_price - old_trend_runner_sl) / self.state.trend_runner_sl_price >= 0.0001
            )
        )
        if self._tp_plan_unchanged(tp_price, partial_tp_price, partial_tp_ratio, tp_plan) and not runner_sl_changed and not trend_runner_orders_changed:
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
        if tp_plan == "MIDDLE_RUNNER":
            self._set_middle_runner_planned(partial_tp_price, tp_price)
        if tp_plan == "THREE_STAGE_RUNNER":
            self._set_three_stage_runner_planned(self.state.side, boll)
        if self.state.trend_runner_active and self.config.runner_dynamic_enabled:
            self.state.trend_runner_adjust_count += 1
            self.state.trend_runner_last_update_candle_ts_ms = boll.candle_ts_ms
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

    def _maybe_trend_runner_market_exit(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if not self.state.trend_runner_active or self.state.side is None:
            return None
        side = self.state.side
        sl_price = self.state.trend_runner_sl_price
        tp_price = self.state.trend_runner_tp_price
        if tp_price is not None:
            if side == "LONG" and price >= tp_price:
                return self._runner_market_exit_intent(price, ts_ms, boll, cvd, "trend_runner_tp_crossed")
            if side == "SHORT" and price <= tp_price:
                return self._runner_market_exit_intent(price, ts_ms, boll, cvd, "trend_runner_tp_crossed")

        if sl_price is not None:
            if side == "LONG" and price <= sl_price:
                return self._runner_market_exit_intent(price, ts_ms, boll, cvd, "trend_runner_sl_failsafe")
            if side == "SHORT" and price >= sl_price:
                return self._runner_market_exit_intent(price, ts_ms, boll, cvd, "trend_runner_sl_failsafe")

        if side == "LONG" and price < boll.middle:
            return self._runner_market_exit_intent(price, ts_ms, boll, cvd, "trend_runner_middle_lost")
        if side == "SHORT" and price > boll.middle:
            return self._runner_market_exit_intent(price, ts_ms, boll, cvd, "trend_runner_middle_lost")

        start_ts = int(self.state.trend_runner_trend_start_ts_ms or 0)
        max_trend_ms = int(self.config.runner_max_trend_seconds_after_second_tp * 1000)
        if start_ts > 0 and max_trend_ms > 0 and ts_ms - start_ts >= max_trend_ms:
            logger.warning(
                "TREND_RUNNER_MAX_TIME_EXIT | side=%s start_ts_ms=%s ts_ms=%s max_seconds=%s",
                side,
                start_ts,
                ts_ms,
                self.config.runner_max_trend_seconds_after_second_tp,
            )
            return self._runner_market_exit_intent(price, ts_ms, boll, cvd, "trend_runner_max_time_after_second_tp")

        reverse_reason = self._maybe_confirm_trend_runner_reverse_burst(side, price, ts_ms, cvd)
        if reverse_reason is not None:
            return self._runner_market_exit_intent(price, ts_ms, boll, cvd, reverse_reason)
        return None

    def _maybe_confirm_trend_runner_reverse_burst(self, side: PositionSide, price: float, ts_ms: int, cvd: CvdSnapshot) -> str | None:
        if not self.config.runner_reverse_burst_exit_enabled:
            return None
        start_ts = int(self.state.trend_runner_trend_start_ts_ms or 0)
        if start_ts <= 0 or ts_ms - start_ts < self.config.runner_reverse_burst_arm_delay_seconds * 1000:
            return None

        samples = self.state.trend_runner_reverse_samples
        if samples is None:
            samples = []
            self.state.trend_runner_reverse_samples = samples

        candidate = self._trend_runner_reverse_candidate(side, cvd)
        if not self.state.trend_runner_reverse_candidate:
            if not candidate:
                return None
            self.state.trend_runner_reverse_candidate = True
            self.state.trend_runner_reverse_start_ts_ms = ts_ms
            self.state.trend_runner_reverse_start_price = price
            self.state.trend_runner_reverse_extreme_price = price
            self.state.trend_runner_reverse_fast_cvd_start = cvd.fast_cvd
            samples.clear()
            samples.append((ts_ms, cvd.buy_ratio, cvd.sell_ratio, cvd.fast_cvd, price))
            logger.warning(
                "TREND_RUNNER_REVERSE_CANDIDATE | side=%s price=%.4f fast_cvd=%.8f buy_ratio=%.4f sell_ratio=%.4f",
                side,
                price,
                cvd.fast_cvd,
                cvd.buy_ratio,
                cvd.sell_ratio,
            )
            return None

        if side == "LONG":
            self.state.trend_runner_reverse_extreme_price = min(self.state.trend_runner_reverse_extreme_price or price, price)
        else:
            self.state.trend_runner_reverse_extreme_price = max(self.state.trend_runner_reverse_extreme_price or price, price)
        samples.append((ts_ms, cvd.buy_ratio, cvd.sell_ratio, cvd.fast_cvd, price))
        cutoff_ts = ts_ms - max(self.config.runner_reverse_burst_confirm_seconds * 1000, 1)
        self.state.trend_runner_reverse_samples = [sample for sample in samples if sample[0] >= cutoff_ts]

        elapsed_ms = ts_ms - int(self.state.trend_runner_reverse_start_ts_ms or ts_ms)
        if elapsed_ms < self.config.runner_reverse_burst_confirm_seconds * 1000:
            return None

        confirmed = self._trend_runner_reverse_confirmed(side, price, cvd)
        if confirmed:
            logger.warning("TREND_RUNNER_REVERSE_CONFIRMED | side=%s price=%.4f elapsed_ms=%s", side, price, elapsed_ms)
            self._reset_trend_runner_reverse_state()
            return "trend_runner_reverse_burst_confirmed"

        logger.warning("TREND_RUNNER_REVERSE_CANCELLED | side=%s price=%.4f elapsed_ms=%s", side, price, elapsed_ms)
        self._reset_trend_runner_reverse_state()
        return None

    def _trend_runner_reverse_candidate(self, side: PositionSide, cvd: CvdSnapshot) -> bool:
        if side == "LONG":
            return bool(
                cvd.down_burst
                or (
                    cvd.sell_ratio >= self.config.runner_reverse_strong_ratio
                    and cvd.fast_cvd < 0
                    and cvd.cvd_decreasing
                )
            )
        return bool(
            cvd.up_burst
            or (
                cvd.buy_ratio >= self.config.runner_reverse_strong_ratio
                and cvd.fast_cvd > 0
                and cvd.cvd_increasing
            )
        )

    def _trend_runner_reverse_confirmed(self, side: PositionSide, current_price: float, cvd: CvdSnapshot) -> bool:
        samples = self.state.trend_runner_reverse_samples or []
        if not samples:
            return False
        start_price = self.state.trend_runner_reverse_start_price
        extreme = self.state.trend_runner_reverse_extreme_price
        if start_price is None or start_price <= 0 or extreme is None or extreme <= 0:
            return False
        if side == "LONG":
            avg_sell_ratio = sum(float(sample[2]) for sample in samples) / len(samples)
            price_damage_pct = (start_price - current_price) / start_price
            recovery_pct = (current_price - extreme) / extreme
            return (
                avg_sell_ratio >= self.config.runner_reverse_sell_ratio
                and cvd.fast_cvd < self.state.trend_runner_reverse_fast_cvd_start
                and price_damage_pct >= self.config.runner_reverse_min_price_damage_pct
                and recovery_pct < self.config.runner_reverse_recovery_cancel_pct
            )

        avg_buy_ratio = sum(float(sample[1]) for sample in samples) / len(samples)
        price_damage_pct = (current_price - start_price) / start_price
        recovery_pct = (extreme - current_price) / extreme
        return (
            avg_buy_ratio >= self.config.runner_reverse_buy_ratio
            and cvd.fast_cvd > self.state.trend_runner_reverse_fast_cvd_start
            and price_damage_pct >= self.config.runner_reverse_min_price_damage_pct
            and recovery_pct < self.config.runner_reverse_recovery_cancel_pct
        )

    def _runner_market_exit_intent(
        self,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        reason: str,
    ) -> TradeIntent | None:
        side = self.state.side
        if side is None:
            return None
        self.state.trend_runner_exit_reason = reason
        size = self.sizer.calculate(price, layer_index=max(self.state.layers, 1))
        logger.warning(
            "TREND_RUNNER_MARKET_EXIT_SIGNAL | side=%s price=%.4f reason=%s runner_tp=%s runner_sl=%s active=%s adjust_count=%s",
            side,
            price,
            reason,
            self.state.trend_runner_tp_price,
            self.state.trend_runner_sl_price,
            self.state.trend_runner_active,
            self.state.trend_runner_adjust_count,
        )
        return TradeIntent(
            intent_type="MARKET_EXIT_RUNNER",
            side=side,
            price=price,
            layer_index=self.state.layers,
            tp_price=self.state.trend_runner_tp_price or self.state.tp_price or price,
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
            partial_tp_price=None,
            partial_tp_ratio=0.0,
            tp_plan="SINGLE",
            partial_tp_consumed=True,
            trend_runner_active=True,
            trend_runner_tp_price=self.state.trend_runner_tp_price,
            trend_runner_sl_price=self.state.trend_runner_sl_price,
            trend_runner_tp_order_id=self.state.trend_runner_tp_order_id,
            trend_runner_sl_order_id=self.state.trend_runner_sl_order_id,
            trend_runner_exit_reason=reason,
            trend_runner_adjust_count=self.state.trend_runner_adjust_count,
        )

    def _maybe_near_tp_reduce(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if not self.config.near_tp_enabled:
            return None
        if self.state.side is None or self.state.tp_price is None:
            return None
        if self.state.avg_entry_price <= 0 or price <= 0:
            return None
        if self.state.near_tp_protected:
            return None
        if self.state.tp_plan == "MIDDLE_RUNNER" or self.state.middle_runner_pending or self.state.middle_runner_active:
            return None
        if self.state.tp_plan == "THREE_STAGE_RUNNER" or self.state.three_stage_runner_enabled_for_position or self.state.trend_runner_active:
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

    def _reset_middle_runner_state(self) -> None:
        self.state.middle_runner_enabled_for_position = False
        self.state.middle_runner_pending = False
        self.state.middle_runner_active = False
        self.state.middle_runner_first_close_ratio = 0.0
        self.state.middle_runner_keep_ratio = 0.0
        self.state.middle_runner_first_tp_price = None
        self.state.middle_runner_final_tp_price = None
        self.state.middle_runner_protective_sl_price = None
        self.state.middle_runner_protective_sl_order_id = None
        self.state.middle_runner_extension_triggered = False
        self.state.middle_runner_add_disabled = False
        self.state.middle_runner_size_mismatch_protected = False
        self.state.middle_runner_size_mismatch_warning_ts_ms = 0

    def _reset_three_stage_runner_state(self) -> None:
        self.state.three_stage_runner_enabled_for_position = False
        self.state.three_stage_tp1_price = None
        self.state.three_stage_tp2_price = None
        self.state.three_stage_runner_initial_tp_price = None
        self.state.three_stage_tp1_ratio = 0.0
        self.state.three_stage_tp2_ratio = 0.0
        self.state.three_stage_runner_ratio = 0.0
        self.state.three_stage_tp1_consumed = False
        self.state.three_stage_tp2_consumed = False
        self.state.three_stage_post_tp1_protective_sl_price = None
        self.state.three_stage_post_tp1_protective_sl_order_id = None
        self.state.three_stage_post_tp1_sl_extension_triggered = False
        self.state.three_stage_post_tp1_protected = False
        self.state.trend_runner_active = False
        self.state.trend_runner_trend_start_ts_ms = 0
        self.state.trend_runner_adjust_count = 0
        self.state.trend_runner_last_update_candle_ts_ms = 0
        self.state.trend_runner_tp_price = None
        self.state.trend_runner_sl_price = None
        self.state.trend_runner_tp_order_id = None
        self.state.trend_runner_sl_order_id = None
        self.state.trend_runner_exit_reason = None
        self._reset_trend_runner_reverse_state()

    def _reset_trend_runner_reverse_state(self) -> None:
        self.state.trend_runner_reverse_candidate = False
        self.state.trend_runner_reverse_start_ts_ms = 0
        self.state.trend_runner_reverse_start_price = None
        self.state.trend_runner_reverse_extreme_price = None
        self.state.trend_runner_reverse_fast_cvd_start = 0.0
        self.state.trend_runner_reverse_samples = []

    def _set_middle_runner_planned(self, first_tp_price: float | None, final_tp_price: float) -> None:
        first_close_ratio = min(max(self.config.middle_runner_first_close_ratio, 0.1), 0.95)
        self.state.middle_runner_enabled_for_position = True
        self.state.middle_runner_pending = True
        self.state.middle_runner_active = False
        self.state.middle_runner_first_close_ratio = first_close_ratio
        self.state.middle_runner_keep_ratio = 1 - first_close_ratio
        self.state.middle_runner_first_tp_price = first_tp_price
        self.state.middle_runner_final_tp_price = final_tp_price
        self.state.middle_runner_protective_sl_price = None
        self.state.middle_runner_protective_sl_order_id = None
        self.state.middle_runner_extension_triggered = False
        self.state.middle_runner_add_disabled = False
        self.state.middle_runner_size_mismatch_protected = False
        self.state.middle_runner_size_mismatch_warning_ts_ms = 0

    def _set_three_stage_runner_planned(self, side: PositionSide, boll: BollSnapshot) -> None:
        tp1_ratio, tp2_ratio, runner_ratio = self._normalized_three_stage_ratios()
        self.state.three_stage_runner_enabled_for_position = True
        self.state.three_stage_tp1_price = boll.middle
        self.state.three_stage_tp2_price = boll.upper if side == "LONG" else boll.lower
        self.state.three_stage_runner_initial_tp_price = None
        self.state.three_stage_tp1_ratio = tp1_ratio
        self.state.three_stage_tp2_ratio = tp2_ratio
        self.state.three_stage_runner_ratio = runner_ratio
        self.state.three_stage_tp1_consumed = False
        self.state.three_stage_tp2_consumed = False
        self.state.three_stage_post_tp1_protective_sl_price = None
        self.state.three_stage_post_tp1_protective_sl_order_id = None
        self.state.three_stage_post_tp1_sl_extension_triggered = False
        self.state.three_stage_post_tp1_protected = False
        self.state.trend_runner_active = False
        self.state.trend_runner_trend_start_ts_ms = 0
        self.state.trend_runner_adjust_count = 0
        self.state.trend_runner_last_update_candle_ts_ms = 0
        self.state.trend_runner_tp_price = None
        self.state.trend_runner_sl_price = None
        self.state.trend_runner_tp_order_id = None
        self.state.trend_runner_sl_order_id = None
        self.state.trend_runner_exit_reason = None
        self._reset_trend_runner_reverse_state()

    def _normalized_three_stage_ratios(self) -> tuple[float, float, float]:
        tp1 = max(float(self.config.three_stage_tp1_ratio), 0.0)
        tp2 = max(float(self.config.three_stage_tp2_ratio), 0.0)
        runner = max(float(self.config.three_stage_runner_ratio), 0.0)
        total = tp1 + tp2 + runner
        if total <= 0:
            return 0.60, 0.20, 0.20
        return tp1 / total, tp2 / total, runner / total

    def _calculate_runner_initial_tp(self, side: PositionSide, boll: BollSnapshot) -> float:
        extra = max(float(self.config.runner_tp_initial_outer_extra_pct), 0.0)
        if side == "LONG":
            return boll.upper * (1 + extra)
        return boll.lower * (1 - extra)

    def _calculate_trend_runner_dynamic_orders(
        self,
        side: PositionSide,
        boll: BollSnapshot,
        adjust_count: int,
        old_sl: float | None,
    ) -> tuple[float, float, float, float]:
        tp_extra_pct = max(
            self.config.runner_tp_min_outer_extra_pct,
            self.config.runner_tp_initial_outer_extra_pct - self.config.runner_tp_step_pct * adjust_count,
        )
        sl_distance_ratio = max(
            self.config.runner_sl_min_outer_distance_ratio,
            self.config.runner_sl_initial_outer_distance_ratio - self.config.runner_sl_step_ratio * adjust_count,
        )
        if side == "LONG":
            new_tp = boll.upper * (1 + tp_extra_pct)
            calculated_sl = boll.upper - (boll.upper - boll.middle) * sl_distance_ratio
            new_sl = calculated_sl if old_sl is None else max(old_sl, calculated_sl)
        else:
            new_tp = boll.lower * (1 - tp_extra_pct)
            calculated_sl = boll.lower + (boll.middle - boll.lower) * sl_distance_ratio
            new_sl = calculated_sl if old_sl is None else min(old_sl, calculated_sl)
        return new_tp, new_sl, tp_extra_pct, sl_distance_ratio

    def _calculate_middle_runner_protective_sl(self, side: PositionSide, current_price: float, boll: BollSnapshot) -> float | None:
        avg_entry = self.state.avg_entry_price
        if avg_entry <= 0 or current_price <= 0:
            return None
        fee = self.config.breakeven_fee_buffer_pct
        if side == "LONG":
            after_partial_breakeven = avg_entry * (1 + fee)
            candidate_1 = (after_partial_breakeven + boll.middle) / 2
            candidate_2 = (boll.lower + boll.middle) / 2
            protective_sl = max(candidate_1, candidate_2)
            if protective_sl >= current_price:
                logger.warning(
                    "MIDDLE_RUNNER_ORDER_WARNING | reason=long_sl_not_below_current side=LONG current_price=%.4f protective_sl_price=%.4f boll_middle=%.4f boll_lower=%.4f",
                    current_price,
                    protective_sl,
                    boll.middle,
                    boll.lower,
                )
                return None
            return protective_sl

        after_partial_breakeven = avg_entry * (1 - fee)
        candidate_1 = (after_partial_breakeven + boll.middle) / 2
        candidate_2 = (boll.upper + boll.middle) / 2
        protective_sl = min(candidate_1, candidate_2)
        if protective_sl <= current_price:
            logger.warning(
                "MIDDLE_RUNNER_ORDER_WARNING | reason=short_sl_not_above_current side=SHORT current_price=%.4f protective_sl_price=%.4f boll_middle=%.4f boll_upper=%.4f",
                current_price,
                protective_sl,
                boll.middle,
                boll.upper,
            )
            return None
        return protective_sl

    def _tighten_middle_runner_sl(self, side: PositionSide, old_sl: float, new_sl: float) -> float:
        if side == "LONG":
            return max(old_sl, new_sl)
        return min(old_sl, new_sl)

    def _tighten_optional_middle_runner_sl(self, side: PositionSide, old_sl: float | None, new_sl: float | None) -> float | None:
        if new_sl is None:
            return old_sl
        if old_sl is None:
            return new_sl
        return self._tighten_middle_runner_sl(side, old_sl, new_sl)

    def _apply_middle_runner_extension_trigger(
        self,
        side: PositionSide,
        current_price: float,
        boll: BollSnapshot,
        protective_sl: float | None,
    ) -> float | None:
        ratio = min(max(self.config.middle_runner_extension_trigger_ratio, 0.0), 1.0)
        if side == "LONG":
            trigger_price = boll.middle + (boll.upper - boll.middle) * ratio
            if current_price < trigger_price:
                return protective_sl
            new_sl = boll.middle if protective_sl is None else max(protective_sl, boll.middle)
        else:
            trigger_price = boll.middle - (boll.middle - boll.lower) * ratio
            if current_price > trigger_price:
                return protective_sl
            new_sl = boll.middle if protective_sl is None else min(protective_sl, boll.middle)
        if not self.state.middle_runner_extension_triggered:
            logger.warning(
                "MIDDLE_RUNNER_EXTENSION_TRIGGERED | side=%s current_price=%.4f extension_trigger_price=%.4f protective_sl_price=%.4f boll_middle=%.4f boll_upper=%.4f boll_lower=%.4f",
                side,
                current_price,
                trigger_price,
                new_sl,
                boll.middle,
                boll.upper,
                boll.lower,
            )
        self.state.middle_runner_extension_triggered = True
        return new_sl

    def _calculate_three_stage_post_tp1_protective_sl(self, side: PositionSide, current_price: float, boll: BollSnapshot) -> float | None:
        avg_entry = self.state.avg_entry_price
        tp1_price = self.state.three_stage_tp1_price
        tp1_ratio = float(self.state.three_stage_tp1_ratio or 0.0)
        if avg_entry <= 0 or current_price <= 0 or tp1_price is None or tp1_ratio <= 0 or tp1_ratio >= 1:
            return None
        fee = self.config.breakeven_fee_buffer_pct
        if side == "LONG":
            post_tp1_breakeven = avg_entry - tp1_ratio * (float(tp1_price) - avg_entry) / (1 - tp1_ratio)
            post_tp1_breakeven_buffered = post_tp1_breakeven * (1 + fee)
            candidate_1 = (post_tp1_breakeven_buffered + boll.middle) / 2
            candidate_2 = (boll.lower + boll.middle) / 2
            protective_sl = max(candidate_1, candidate_2)
            if protective_sl >= current_price:
                logger.warning(
                    "THREE_STAGE_POST_TP1_SL_WARNING | reason=long_sl_not_below_current side=LONG current_price=%.4f protective_sl_price=%.4f avg_entry=%.4f tp1_price=%.4f tp1_ratio=%.4f boll_middle=%.4f boll_lower=%.4f",
                    current_price,
                    protective_sl,
                    avg_entry,
                    float(tp1_price),
                    tp1_ratio,
                    boll.middle,
                    boll.lower,
                )
                return None
            return protective_sl

        post_tp1_breakeven = avg_entry + tp1_ratio * (avg_entry - float(tp1_price)) / (1 - tp1_ratio)
        post_tp1_breakeven_buffered = post_tp1_breakeven * (1 - fee)
        candidate_1 = (post_tp1_breakeven_buffered + boll.middle) / 2
        candidate_2 = (boll.upper + boll.middle) / 2
        protective_sl = min(candidate_1, candidate_2)
        if protective_sl <= current_price:
            logger.warning(
                "THREE_STAGE_POST_TP1_SL_WARNING | reason=short_sl_not_above_current side=SHORT current_price=%.4f protective_sl_price=%.4f avg_entry=%.4f tp1_price=%.4f tp1_ratio=%.4f boll_middle=%.4f boll_upper=%.4f",
                current_price,
                protective_sl,
                avg_entry,
                float(tp1_price),
                tp1_ratio,
                boll.middle,
                boll.upper,
            )
            return None
        return protective_sl

    def _tighten_three_stage_post_tp1_sl(self, side: PositionSide, old_sl: float, new_sl: float) -> float:
        if side == "LONG":
            return max(old_sl, new_sl)
        return min(old_sl, new_sl)

    def _tighten_optional_three_stage_post_tp1_sl(self, side: PositionSide, old_sl: float | None, new_sl: float | None) -> float | None:
        if new_sl is None:
            return old_sl
        if old_sl is None:
            return new_sl
        return self._tighten_three_stage_post_tp1_sl(side, old_sl, new_sl)

    def _apply_three_stage_post_tp1_extension_trigger(
        self,
        side: PositionSide,
        current_price: float,
        boll: BollSnapshot,
        protective_sl: float | None,
    ) -> float | None:
        ratio = min(max(self.config.three_stage_post_tp1_sl_extension_trigger_ratio, 0.0), 1.0)
        if side == "LONG":
            trigger_price = boll.middle + (boll.upper - boll.middle) * ratio
            if current_price < trigger_price:
                return protective_sl
            new_sl = boll.middle if protective_sl is None else max(protective_sl, boll.middle)
        else:
            trigger_price = boll.middle - (boll.middle - boll.lower) * ratio
            if current_price > trigger_price:
                return protective_sl
            new_sl = boll.middle if protective_sl is None else min(protective_sl, boll.middle)
        if not self.state.three_stage_post_tp1_sl_extension_triggered:
            logger.warning(
                "THREE_STAGE_POST_TP1_EXTENSION_TRIGGERED | side=%s current_price=%.4f extension_trigger_price=%.4f protective_sl_price=%.4f boll_middle=%.4f boll_upper=%.4f boll_lower=%.4f",
                side,
                current_price,
                trigger_price,
                new_sl,
                boll.middle,
                boll.upper,
                boll.lower,
            )
        self.state.three_stage_post_tp1_sl_extension_triggered = True
        return new_sl

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

    def _select_tp_plan(
        self,
        side: PositionSide,
        final_tp: float,
        layers: int,
        *,
        tp_mode: TpMode | None = None,
        boll: BollSnapshot | None = None,
    ) -> tuple[float | None, float, TpPlan]:
        if self._three_stage_runner_plan_allowed(tp_mode, boll):
            tp1_ratio, _tp2_ratio, _runner_ratio = self._normalized_three_stage_ratios()
            return boll.middle, tp1_ratio, "THREE_STAGE_RUNNER"
        if self.config.three_stage_runner_enabled:
            return None, 0.0, "SINGLE"
        if self._middle_runner_plan_allowed(tp_mode, boll):
            first_close_ratio = min(max(self.config.middle_runner_first_close_ratio, 0.1), 0.95)
            return boll.middle, first_close_ratio, "MIDDLE_RUNNER"
        if not self.config.split_tp_enabled:
            return None, 0.0, "SINGLE"
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

    def _three_stage_runner_plan_allowed(self, tp_mode: TpMode | None, boll: BollSnapshot | None) -> bool:
        if not self.config.three_stage_runner_enabled:
            return False
        if tp_mode != "MIDDLE" or boll is None:
            return False
        if self.state.near_tp_protected or self.state.near_tp_add_disabled:
            return False
        if self.state.partial_tp_consumed:
            return False
        if (
            self.state.middle_runner_enabled_for_position
            or self.state.middle_runner_pending
            or self.state.middle_runner_active
            or self.state.tp_plan == "MIDDLE_RUNNER"
            or self.state.trend_runner_active
        ):
            return False
        return True

    def _three_stage_waiting_tp2(self) -> bool:
        return bool(
            self.state.three_stage_runner_enabled_for_position
            and self.state.three_stage_tp1_consumed
            and not self.state.three_stage_tp2_consumed
            and not self.state.trend_runner_active
        )

    def _middle_runner_plan_allowed(self, tp_mode: TpMode | None, boll: BollSnapshot | None) -> bool:
        if not self.config.middle_runner_enabled:
            return False
        if tp_mode != "MIDDLE" or boll is None:
            return False
        if self.state.near_tp_protected or self.state.near_tp_add_disabled:
            return False
        if self.state.partial_tp_consumed:
            return False
        if self.state.middle_runner_active:
            return False
        if (
            self.state.three_stage_runner_enabled_for_position
            or self.state.tp_plan == "THREE_STAGE_RUNNER"
            or self.state.three_stage_tp1_consumed
            or self.state.three_stage_tp2_consumed
        ):
            return False
        return True

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
            middle_runner_enabled_for_position=self.state.middle_runner_enabled_for_position,
            middle_runner_pending=self.state.middle_runner_pending,
            middle_runner_active=self.state.middle_runner_active,
            middle_runner_first_close_ratio=self.state.middle_runner_first_close_ratio,
            middle_runner_keep_ratio=self.state.middle_runner_keep_ratio,
            middle_runner_first_tp_price=self.state.middle_runner_first_tp_price,
            middle_runner_final_tp_price=self.state.middle_runner_final_tp_price,
            middle_runner_protective_sl_price=self.state.middle_runner_protective_sl_price,
            middle_runner_protective_sl_order_id=self.state.middle_runner_protective_sl_order_id,
            middle_runner_extension_triggered=self.state.middle_runner_extension_triggered,
            middle_runner_add_disabled=self.state.middle_runner_add_disabled,
            three_stage_tp1_price=self.state.three_stage_tp1_price,
            three_stage_tp1_ratio=self.state.three_stage_tp1_ratio,
            three_stage_tp2_price=self.state.three_stage_tp2_price,
            three_stage_tp2_ratio=self.state.three_stage_tp2_ratio,
            three_stage_runner_tp_price=self.state.trend_runner_tp_price,
            three_stage_runner_ratio=self.state.three_stage_runner_ratio,
            three_stage_runner_sl_price=self.state.trend_runner_sl_price,
            three_stage_tp1_consumed=self.state.three_stage_tp1_consumed,
            three_stage_tp2_consumed=self.state.three_stage_tp2_consumed,
            three_stage_post_tp1_protective_sl_price=self.state.three_stage_post_tp1_protective_sl_price,
            three_stage_post_tp1_protective_sl_order_id=self.state.three_stage_post_tp1_protective_sl_order_id,
            three_stage_post_tp1_sl_extension_triggered=self.state.three_stage_post_tp1_sl_extension_triggered,
            three_stage_post_tp1_protected=self.state.three_stage_post_tp1_protected,
            trend_runner_active=self.state.trend_runner_active,
            trend_runner_tp_price=self.state.trend_runner_tp_price,
            trend_runner_sl_price=self.state.trend_runner_sl_price,
            trend_runner_tp_order_id=self.state.trend_runner_tp_order_id,
            trend_runner_sl_order_id=self.state.trend_runner_sl_order_id,
            trend_runner_exit_reason=self.state.trend_runner_exit_reason,
            trend_runner_adjust_count=self.state.trend_runner_adjust_count,
            protected_order_ids=self._protected_order_ids(),
            managed_core_contracts=self._managed_core_contracts_for_intent(intent_type),
            managed_core_eth_qty=self._managed_core_eth_qty_for_intent(intent_type),
        )

    def _managed_core_contracts_for_intent(self, intent_type: TradeIntentType) -> str | None:
        if not self.state.sidecar_enabled_for_position:
            return None
        if intent_type in {"UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"}:
            return self.state.core_contracts
        return None

    def _managed_core_eth_qty_for_intent(self, intent_type: TradeIntentType) -> float:
        if not self.state.sidecar_enabled_for_position:
            return 0.0
        if intent_type in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT", "UPDATE_TP"}:
            return float(self.state.total_entry_qty or 0.0)
        return float(self.state.core_eth_qty or 0.0)

    def _protected_order_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        max_legs = int(getattr(self.sizer.config, "sidecar_max_legs", 10) or 10)
        for leg in trim_sidecar_legs_for_state(self.state.sidecar_legs, max_legs):
            if leg.get("status") == "OPEN" and leg.get("tp_order_id"):
                ids.append(str(leg["tp_order_id"]))
        for order_id in (
            self.state.near_tp_protective_sl_order_id,
            self.state.middle_runner_protective_sl_order_id,
            self.state.three_stage_post_tp1_protective_sl_order_id,
            self.state.trend_runner_sl_order_id,
        ):
            if order_id:
                ids.append(str(order_id))
        return tuple(dict.fromkeys(ids))

    def _cooldown_ok(self, ts_ms: int) -> bool:
        return ts_ms - self.state.last_order_ts_ms >= self.config.order_cooldown_seconds * 1000
