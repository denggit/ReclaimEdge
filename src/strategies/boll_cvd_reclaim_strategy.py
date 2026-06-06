from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.position_management.cost_basis import calculate_remaining_breakeven_price
from src.position_management.sidecar.model import trim_sidecar_legs_for_state
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer
from src.strategies import add_layer_gates
from src.strategies import middle_runner as middle_runner_helpers
from src.strategies import near_tp_reduce as near_tp_helpers
from src.strategies import three_stage_runner as three_stage_helpers
from src.strategies import tp_plan_selector
from src.strategies import trend_runner as trend_runner_helpers
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


def _price_changed(old: float | None, new: float | None, threshold: float = 0.0001) -> bool:
    if old is None or new is None:
        return old is not None or new is not None
    if new == 0:
        return old != 0
    return abs(float(old) - float(new)) / abs(float(new)) >= threshold


def _interpolate_to_middle(anchor: float, middle: float, ratio: float) -> float:
    return anchor + (middle - anchor) * ratio


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
    add_freeze_chain_enabled: bool = True
    add_min_interval_bypass_multiplier: float = 2.0
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
    runner_protective_sl_time_tighten_enabled: bool = True
    runner_protective_sl_time_tighten_step_ratio: float = 0.05
    runner_protective_sl_time_tighten_max_ratio: float = 1.0
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
    three_stage_pre_tp1_degrade_enabled: bool = True
    three_stage_pre_tp1_middle_runner_after_seconds: int = 10800
    three_stage_pre_tp1_single_after_seconds: int = 21600
    # TP-only BOLL window (15) used exclusively for take-profit prices.
    # BOLL_WINDOW=20 remains the structure window for entry/add/risk/SL/runner.
    tp_boll_enabled: bool = True
    tp_boll_window: int = 15

    def __post_init__(self) -> None:
        if (
                self.three_stage_pre_tp1_degrade_enabled
                and self.three_stage_pre_tp1_single_after_seconds <= self.three_stage_pre_tp1_middle_runner_after_seconds
        ):
            raise RuntimeError(
                "THREE_STAGE_PRE_TP1_SINGLE_AFTER_SECONDS must be greater than "
                "THREE_STAGE_PRE_TP1_MIDDLE_RUNNER_AFTER_SECONDS"
            )

    @classmethod
    def from_env(cls) -> "BollCvdReclaimStrategyConfig":
        middle_runner_first_close_ratio = min(max(float(os.getenv("MIDDLE_RUNNER_FIRST_CLOSE_RATIO", "0.8")), 0.1),
                                              0.95)
        middle_runner_enabled = _env_bool("MIDDLE_RUNNER_ENABLED", False)
        near_tp_enabled = _env_bool("NEAR_TP_ENABLED", False)
        three_stage_runner_enabled = _env_bool("THREE_STAGE_RUNNER_ENABLED", False)
        if middle_runner_enabled and near_tp_enabled:
            raise RuntimeError(
                "MIDDLE_RUNNER_ENABLED=true requires NEAR_TP_ENABLED=false; Middle Runner and Near-TP Reduce are mutually exclusive.")
        if three_stage_runner_enabled and near_tp_enabled:
            raise RuntimeError(
                "THREE_STAGE_RUNNER_ENABLED=true requires NEAR_TP_ENABLED=false; Three-Stage Runner and Near-TP Reduce are mutually exclusive.")
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
            add_freeze_chain_enabled=_env_bool("ADD_FREEZE_CHAIN_ENABLED", True),
            add_min_interval_bypass_multiplier=float(os.getenv("ADD_MIN_INTERVAL_BYPASS_MULTIPLIER", "2")),
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
            near_tp_protective_sl_retry_interval_seconds=float(
                os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
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
            three_stage_post_tp1_sl_extension_trigger_ratio=float(
                os.getenv("THREE_STAGE_POST_TP1_SL_EXTENSION_TRIGGER_RATIO", "0.6")),
            runner_protective_sl_time_tighten_enabled=_env_bool("RUNNER_PROTECTIVE_SL_TIME_TIGHTEN_ENABLED", True),
            runner_protective_sl_time_tighten_step_ratio=float(
                os.getenv("RUNNER_PROTECTIVE_SL_TIME_TIGHTEN_STEP_RATIO", "0.05")),
            runner_protective_sl_time_tighten_max_ratio=float(
                os.getenv("RUNNER_PROTECTIVE_SL_TIME_TIGHTEN_MAX_RATIO", "1.0")),
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
            runner_max_trend_seconds_after_second_tp=int(
                os.getenv("RUNNER_MAX_TREND_SECONDS_AFTER_SECOND_TP", "18000")),
            three_stage_pre_tp1_degrade_enabled=_env_bool("THREE_STAGE_PRE_TP1_DEGRADE_ENABLED", True),
            three_stage_pre_tp1_middle_runner_after_seconds=int(
                os.getenv("THREE_STAGE_PRE_TP1_MIDDLE_RUNNER_AFTER_SECONDS", "10800")),
            three_stage_pre_tp1_single_after_seconds=int(
                os.getenv("THREE_STAGE_PRE_TP1_SINGLE_AFTER_SECONDS", "21600")),
            tp_boll_enabled=_env_bool("TP_BOLL_ENABLED", True),
            tp_boll_window=int(os.getenv("TP_BOLL_WINDOW", "15")),
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


@dataclass(frozen=True)
class MainTpUpdatePlan:
    should_update: bool
    reason: str
    tp_price: float
    tp_mode: TpMode
    tp_plan: TpPlan
    partial_tp_price: float | None
    partial_tp_ratio: float
    three_stage_tp1_price: float | None = None
    three_stage_tp2_price: float | None = None
    middle_runner_first_tp_price: float | None = None
    middle_runner_final_tp_price: float | None = None
    protective_sl_price: float | None = None
    log_reason: str = ""


@dataclass
class StrategyPositionState:
    side: Optional[PositionSide] = None
    layers: int = 0
    last_entry_price: Optional[float] = None
    tp_price: Optional[float] = None
    last_order_ts_ms: int = 0
    first_entry_ts_ms: int = 0
    add_freeze_until_ts_ms: int = 0
    add_freeze_penalty_count: int = 0
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
    position_cost_entry_notional: float = 0.0
    position_cost_exit_notional: float = 0.0
    position_cost_remaining_qty: float = 0.0
    net_remaining_breakeven_price: float = 0.0
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
    middle_runner_sl_diag_last_signature: str | None = None
    middle_runner_sl_time_tighten_candle_count: int = 0
    middle_runner_sl_time_tighten_last_candle_ts_ms: int = 0
    middle_runner_sl_time_tighten_log_candle_ts_ms: int = 0
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
    three_stage_post_tp1_sl_diag_last_signature: str | None = None
    three_stage_post_tp1_sl_time_tighten_candle_count: int = 0
    three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms: int = 0
    three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms: int = 0
    three_stage_pre_tp1_degrade_stage: str | None = None
    three_stage_pre_tp1_degraded_ts_ms: int = 0
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
    near_tp_sidecar_skip_logged: bool = False
    last_add_skip_log_reason: str | None = None
    last_add_skip_log_ts_ms: int = 0
    core_contracts: str | None = None
    core_eth_qty: float = 0.0
    tp_order_id: str | None = None
    tp_order_ids: list[str] = field(default_factory=list)
    startup_force_tp_reconcile: bool = False


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
                    logger.debug("LOWER_EXTREME_UPDATED | extreme=%.4f price=%.4f", self.state.lower_extreme_price,
                                 price)
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
                    logger.debug("UPPER_EXTREME_UPDATED | extreme=%.4f price=%.4f", self.state.upper_extreme_price,
                                 price)
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

    def _maybe_open_or_add_long(self, price: float, ts_ms: int, boll: BollSnapshot,
                                cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None:
            return self._open_position("LONG", "OPEN_LONG", price, ts_ms, boll, cvd,
                                       "下轨出轨深度达标 + 低点附近快速CVD回流/跌不动")
        if self.state.side != "LONG":
            return None
        if self.state.near_tp_add_disabled:
            self._log_add_skip_once_per_window(reason="near_tp_protected", side="LONG", price=price, ts_ms=ts_ms)
            return None
        if self.state.trend_runner_active:
            self._log_add_skip_once_per_window(reason="trend_runner_active", side="LONG", price=price, ts_ms=ts_ms)
            return None
        if (
                self.state.three_stage_runner_enabled_for_position
                and (self.state.three_stage_tp1_consumed or self.state.three_stage_tp2_consumed)
        ):
            self._log_add_skip_once_per_window(reason="three_stage_after_tp1", side="LONG", price=price, ts_ms=ts_ms)
            return None
        if self.state.middle_runner_add_disabled or self.state.middle_runner_active:
            self._log_add_skip_once_per_window(reason="middle_runner_active", side="LONG", price=price, ts_ms=ts_ms)
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
        avg_improvement_ok, improvement_pct, projected_avg = self._add_avg_improvement_passed("LONG", price,
                                                                                              target_layer)
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

    def _maybe_open_or_add_short(self, price: float, ts_ms: int, boll: BollSnapshot,
                                 cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None:
            return self._open_position("SHORT", "OPEN_SHORT", price, ts_ms, boll, cvd,
                                       "上轨出轨深度达标 + 高点附近快速CVD转弱/涨不动")
        if self.state.side != "SHORT":
            return None
        if self.state.near_tp_add_disabled:
            self._log_add_skip_once_per_window(reason="near_tp_protected", side="SHORT", price=price, ts_ms=ts_ms)
            return None
        if self.state.trend_runner_active:
            self._log_add_skip_once_per_window(reason="trend_runner_active", side="SHORT", price=price, ts_ms=ts_ms)
            return None
        if (
                self.state.three_stage_runner_enabled_for_position
                and (self.state.three_stage_tp1_consumed or self.state.three_stage_tp2_consumed)
        ):
            self._log_add_skip_once_per_window(reason="three_stage_after_tp1", side="SHORT", price=price, ts_ms=ts_ms)
            return None
        if self.state.middle_runner_add_disabled or self.state.middle_runner_active:
            self._log_add_skip_once_per_window(reason="middle_runner_active", side="SHORT", price=price, ts_ms=ts_ms)
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
        avg_improvement_ok, improvement_pct, projected_avg = self._add_avg_improvement_passed("SHORT", price,
                                                                                              target_layer)
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
        return add_layer_gates.add_layer_gap_pct_for_target_layer(
            target_layer=target_layer,
            add_layer_gap_pct=self.config.add_layer_gap_pct,
            add_layer_gap_pct_layer_7_8=self.config.add_layer_gap_pct_layer_7_8,
            add_layer_gap_pct_layer_9_10=self.config.add_layer_gap_pct_layer_9_10,
            add_layer_gap_pct_layer_11_plus=self.config.add_layer_gap_pct_layer_11_plus,
        )

    def _add_min_interval_bypass_gap_pct_for_target_layer(self, target_layer: int) -> float:
        return add_layer_gates.add_min_interval_bypass_gap_pct_for_target_layer(
            target_layer=target_layer,
            add_layer_gap_pct=self.config.add_layer_gap_pct,
            add_layer_gap_pct_layer_7_8=self.config.add_layer_gap_pct_layer_7_8,
            add_layer_gap_pct_layer_9_10=self.config.add_layer_gap_pct_layer_9_10,
            add_layer_gap_pct_layer_11_plus=self.config.add_layer_gap_pct_layer_11_plus,
        )

    def _add_gap_passed(self, side: PositionSide, price: float, target_layer: int) -> tuple[bool, float, float]:
        decision = add_layer_gates.check_add_gap(
            side=side,
            price=price,
            last_entry_price=self.state.last_entry_price,
            target_layer=target_layer,
            add_layer_gap_pct=self.config.add_layer_gap_pct,
            add_layer_gap_pct_layer_7_8=self.config.add_layer_gap_pct_layer_7_8,
            add_layer_gap_pct_layer_9_10=self.config.add_layer_gap_pct_layer_9_10,
            add_layer_gap_pct_layer_11_plus=self.config.add_layer_gap_pct_layer_11_plus,
        )
        return decision.ok, decision.gap_pct, decision.required_price

    def _add_avg_improvement_passed(self, side: PositionSide, price: float, target_layer: int) -> tuple[
        bool, float, float]:
        size = self.sizer.calculate(price, layer_index=target_layer)
        add_qty = size.eth_qty
        decision = add_layer_gates.check_add_avg_improvement(
            side=side,
            price=price,
            required_improvement_pct=self.config.add_min_avg_improvement_pct,
            old_qty=self.state.total_entry_qty,
            old_notional=self.state.total_entry_notional,
            old_avg=self.state.avg_entry_price,
            add_qty=add_qty,
        )
        return decision.ok, decision.improvement_pct, decision.projected_avg

    def _add_timing_passed(self, side: PositionSide, price: float, ts_ms: int, target_layer: int) -> tuple[bool, str]:
        decision = add_layer_gates.check_base_add_timing(
            side=side,
            price=price,
            ts_ms=ts_ms,
            target_layer=target_layer,
            layers=self.state.layers,
            last_entry_price=self.state.last_entry_price,
            last_order_ts_ms=self.state.last_order_ts_ms,
            first_add_block_seconds=self.config.first_add_block_seconds,
            add_min_interval_seconds=self.config.add_min_interval_seconds,
            add_layer_gap_pct=self.config.add_layer_gap_pct,
            add_layer_gap_pct_layer_7_8=self.config.add_layer_gap_pct_layer_7_8,
            add_layer_gap_pct_layer_9_10=self.config.add_layer_gap_pct_layer_9_10,
            add_layer_gap_pct_layer_11_plus=self.config.add_layer_gap_pct_layer_11_plus,
        )
        return decision.ok, decision.reason

    def _add_elapsed_seconds(self, ts_ms: int) -> float:
        return add_layer_gates.add_elapsed_seconds(ts_ms=ts_ms, last_order_ts_ms=self.state.last_order_ts_ms)

    def _adverse_gap_pct(self, side: PositionSide, price: float) -> float:
        return add_layer_gates.adverse_gap_pct(side=side, price=price, last_entry_price=self.state.last_entry_price)

    def _log_add_skip_once_per_window(
            self,
            *,
            reason: str,
            side: PositionSide,
            price: float,
            ts_ms: int,
            min_interval_ms: int = 60_000,
    ) -> None:
        if (
                self.state.last_add_skip_log_reason == reason
                and ts_ms - int(self.state.last_add_skip_log_ts_ms or 0) < min_interval_ms
        ):
            return
        self.state.last_add_skip_log_reason = reason
        self.state.last_add_skip_log_ts_ms = ts_ms
        logger.info(
            "ADD_SKIPPED | reason=%s side=%s price=%.4f layers=%s",
            reason,
            side,
            price,
            self.state.layers,
        )

    def _log_add_timing_skipped(self, side: PositionSide, reason: str, price: float, ts_ms: int,
                                target_layer: int) -> None:
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
            self.state.first_entry_ts_ms = ts_ms
            self.state.add_freeze_until_ts_ms = 0
            self.state.add_freeze_penalty_count = 0
            self.state.three_stage_pre_tp1_degrade_stage = None
            self.state.three_stage_pre_tp1_degraded_ts_ms = 0
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
            self.state.position_cost_entry_notional = 0.0
            self.state.position_cost_exit_notional = 0.0
            self.state.position_cost_remaining_qty = 0.0
            self.state.net_remaining_breakeven_price = 0.0
            self.state.last_add_skip_log_reason = None
            self.state.last_add_skip_log_ts_ms = 0
        self.state.side = side
        self._update_position_cost(price, size.eth_qty)
        self.state.partial_tp_consumed = False
        self._reset_near_tp_state()
        self._reset_middle_runner_state()
        self._reset_three_stage_runner_state()
        tp_price, tp_mode = self._select_tp_price(side, boll)
        partial_tp_price, partial_tp_ratio, tp_plan = self._select_tp_plan(side, tp_price, next_layer, tp_mode=tp_mode,
                                                                           boll=boll)
        if tp_plan == "MIDDLE_RUNNER":
            tp_price, _tp_src = self._select_tp_outer(side, boll)
        if tp_plan == "THREE_STAGE_RUNNER":
            tp_price, _tp_src = self._select_tp_outer(side, boll)
        if tp_mode != "MIDDLE":
            reason = f"{reason} + 中轨净利润不足阈值，TP切换到{tp_mode}"
        if tp_plan == "SPLIT_PARTIAL_FINAL":
            reason = f"{reason} + 总层数>= {self.config.split_tp_min_layers}，启用分批止盈"
        if tp_plan == "MIDDLE_RUNNER":
            reason = f"{reason} + 中轨先平{partial_tp_ratio * 100:.0f}%，剩余runner到外轨"
        if tp_plan == "THREE_STAGE_RUNNER":
            reason = f"{reason} + 三段式趋势Runner：中轨{self.config.three_stage_tp1_ratio * 100:.0f}%/外轨{self.config.three_stage_tp2_ratio * 100:.0f}%/Runner{self.config.three_stage_runner_ratio * 100:.0f}%"
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
        self._log_tp_boll_price_selected(
            phase="initial",
            boll=boll,
            tp_price=tp_price,
            tp_mode=tp_mode,
            tp_plan=tp_plan,
            partial_tp_price=partial_tp_price,
            tp1_price=self.state.three_stage_tp1_price if tp_plan == "THREE_STAGE_RUNNER" else None,
            tp2_price=self.state.three_stage_tp2_price if tp_plan == "THREE_STAGE_RUNNER" else None,
            first_tp_price=self.state.middle_runner_first_tp_price if tp_plan == "MIDDLE_RUNNER" else None,
            final_tp_price=self.state.middle_runner_final_tp_price if tp_plan == "MIDDLE_RUNNER" else None,
        )
        return self._intent(intent_type, side, price, next_layer, tp_price, reason, size, boll, cvd, ts_ms)

    def _maybe_update_tp(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        if self.state.side is None or self.state.layers <= 0:
            return None
        trend_runner_needs_initial_orders = (
                self.state.trend_runner_active
                and (self.state.trend_runner_tp_price is None or self.state.trend_runner_sl_price is None)
        )
        force_reconcile = bool(getattr(self.state, "startup_force_tp_reconcile", False))
        if (
                self.state.last_tp_update_candle_ts_ms == boll.candle_ts_ms
                and not trend_runner_needs_initial_orders
                and not force_reconcile
        ):
            return None
        if force_reconcile:
            logger.warning(
                "STARTUP_FORCE_TP_RECONCILE_ARMED | side=%s layers=%s tp_plan=%s candle_ts=%s last_tp_update_candle_ts_ms=%s",
                self.state.side,
                self.state.layers,
                self.state.tp_plan,
                boll.candle_ts_ms,
                self.state.last_tp_update_candle_ts_ms,
            )
        if self._three_stage_waiting_tp2():
            old_post_tp1_sl = self.state.three_stage_post_tp1_protective_sl_price
            old_tp2_price = self.state.three_stage_tp2_price
            new_tp2_price, _tp2_src = self._select_tp_outer(self.state.side, boll)
            if self.config.three_stage_post_tp1_protective_sl_enabled:
                self._advance_runner_sl_time_tighten_candle_count(
                    target="three_stage_post_tp1",
                    candle_ts_ms=int(getattr(boll, "candle_ts_ms", 0) or 0),
                )
                calculated_sl = self._calculate_three_stage_post_tp1_protective_sl(self.state.side, price, boll)
                extension_sl = self._apply_three_stage_post_tp1_extension_trigger(self.state.side, price, boll,
                                                                                  calculated_sl)
                protective_sl = self._tighten_optional_three_stage_post_tp1_sl(self.state.side, old_post_tp1_sl,
                                                                               extension_sl)
                self.state.three_stage_post_tp1_protective_sl_price = protective_sl
            else:
                protective_sl = old_post_tp1_sl
            self.state.three_stage_tp2_price = new_tp2_price
            self.state.tp_price = new_tp2_price
            self.state.tp_mode = "UPPER" if self.state.side == "LONG" else "LOWER"
            self.state.tp_plan = "THREE_STAGE_RUNNER"
            self.state.partial_tp_price = None
            self.state.partial_tp_ratio = 0.0
            self.state.last_tp_update_ts_ms = ts_ms
            self.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms
            post_tp1_sl_changed = protective_sl is not None and _price_changed(old_post_tp1_sl, protective_sl)
            tp2_changed = _price_changed(old_tp2_price, new_tp2_price)
            if post_tp1_sl_changed or tp2_changed or force_reconcile:
                self.state.startup_force_tp_reconcile = False
                size = self.sizer.calculate(price, layer_index=self.state.layers)
                reason_text = "startup_force_tp_reconcile" if force_reconcile else "three_stage_post_tp1_dynamic_tp_sl_update"
                logger.warning(
                    "THREE_STAGE_TP1_PROTECTIVE_SL_UPDATE_SIGNAL | side=%s old_sl=%s new_sl=%s old_tp2=%s new_tp2=%.4f candle_ts=%s force_reconcile=%s",
                    self.state.side,
                    f"{old_post_tp1_sl:.4f}" if old_post_tp1_sl is not None else "-",
                    f"{protective_sl:.4f}" if protective_sl is not None else "-",
                    f"{old_tp2_price:.4f}" if old_tp2_price is not None else "-",
                    new_tp2_price,
                    boll.candle_ts_ms,
                    force_reconcile,
                )
                self._log_tp_boll_price_selected(
                    phase="waiting_tp2_dynamic",
                    boll=boll,
                    tp_price=new_tp2_price,
                    tp_mode="UPPER" if self.state.side == "LONG" else "LOWER",
                    tp_plan="THREE_STAGE_RUNNER",
                    tp2_price=new_tp2_price,
                )
                return self._intent("UPDATE_TP", self.state.side, price, self.state.layers, new_tp2_price, reason_text,
                                    size, boll, cvd, ts_ms)
            self.state.startup_force_tp_reconcile = False
            logger.info(
                "TP_UPDATE_SKIPPED | reason=three_stage_waiting_tp2_plan_unchanged side=%s candle_ts=%s tp2_price=%s protective_sl=%s",
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
        middle_profit_fallback_locked = False
        reason_override: str | None = None

        # ── Unified middle-profit eligibility enforcement ──
        # Before any complex TP mode is allowed, the middle band must offer
        # sufficient net profit relative to the effective breakeven.
        # Exceptions: already-executed stages (TP1 consumed, runner active).
        if tp_mode != "MIDDLE":
            outer, _outer_src = self._select_tp_outer(self.state.side, boll)
            outer_mode: TpMode = "UPPER" if self.state.side == "LONG" else "LOWER"
            effective_be = self._effective_breakeven_for_tp_selection(self.state.side)
            min_profit = self.config.tp_min_net_profit_pct
            required_middle = effective_be * (1 + min_profit) if self.state.side == "LONG" else effective_be * (
                    1 - min_profit)

            # Three-Stage: only reset when TP1 has NOT been consumed and trend runner is NOT active
            if (
                    self.state.three_stage_runner_enabled_for_position
                    and not self.state.three_stage_tp1_consumed
                    and not self.state.trend_runner_active
            ):
                old_tp1 = self.state.three_stage_tp1_price
                old_tp2 = self.state.three_stage_tp2_price
                self._reset_three_stage_runner_state()
                logger.warning(
                    "THREE_STAGE_DISABLED_MIDDLE_PROFIT_INSUFFICIENT | "
                    "side=%s effective_breakeven=%.4f middle=%.4f required_middle=%.4f "
                    "outer=%.4f old_tp1=%s old_tp2=%s candle_ts=%s",
                    self.state.side,
                    effective_be,
                    boll.middle,
                    required_middle,
                    outer,
                    f"{old_tp1:.4f}" if old_tp1 is not None else "-",
                    f"{old_tp2:.4f}" if old_tp2 is not None else "-",
                    boll.candle_ts_ms,
                )
                tp_price = outer
                tp_mode = outer_mode
                partial_tp_price = None
                partial_tp_ratio = 0.0
                tp_plan = "SINGLE"
                reason_override = "three_stage_middle_profit_insufficient_single_outer"
                middle_profit_fallback_locked = True

            # Middle Runner pending (first close NOT done): reset
            elif self.state.middle_runner_pending and not self.state.middle_runner_active:
                self._reset_middle_runner_state()
                logger.warning(
                    "MIDDLE_RUNNER_DISABLED_MIDDLE_PROFIT_INSUFFICIENT | "
                    "side=%s effective_breakeven=%.4f middle=%.4f required_middle=%.4f "
                    "outer=%.4f candle_ts=%s",
                    self.state.side,
                    effective_be,
                    boll.middle,
                    required_middle,
                    outer,
                    boll.candle_ts_ms,
                )
                tp_price = outer
                tp_mode = outer_mode
                partial_tp_price = None
                partial_tp_ratio = 0.0
                tp_plan = "SINGLE"
                reason_override = "middle_runner_middle_profit_insufficient_single_outer"
                middle_profit_fallback_locked = True

            # SPLIT partial NOT consumed: fall back to SINGLE outer
            elif self.state.tp_plan == "SPLIT_PARTIAL_FINAL" and not self.state.partial_tp_consumed:
                logger.warning(
                    "SPLIT_TP_DISABLED_MIDDLE_PROFIT_INSUFFICIENT | "
                    "side=%s effective_breakeven=%.4f middle=%.4f required_middle=%.4f "
                    "outer=%.4f candle_ts=%s",
                    self.state.side,
                    effective_be,
                    boll.middle,
                    required_middle,
                    outer,
                    boll.candle_ts_ms,
                )
                tp_price = outer
                tp_mode = outer_mode
                partial_tp_price = None
                partial_tp_ratio = 0.0
                tp_plan = "SINGLE"
                middle_profit_fallback_locked = True

            # Any other unfulfilled complex plan: fall back to SINGLE outer
            elif (
                    self.state.tp_plan != "SINGLE"
                    and not self.state.trend_runner_active
                    and not self.state.middle_runner_active
                    and not self.state.three_stage_tp1_consumed
                    and not self.state.three_stage_tp2_consumed
            ):
                logger.warning(
                    "COMPLEX_TP_DISABLED_MIDDLE_PROFIT_INSUFFICIENT | "
                    "side=%s effective_breakeven=%.4f middle=%.4f required_middle=%.4f "
                    "outer=%.4f old_plan=%s candle_ts=%s",
                    self.state.side,
                    effective_be,
                    boll.middle,
                    required_middle,
                    outer,
                    self.state.tp_plan,
                    boll.candle_ts_ms,
                )
                tp_price = outer
                tp_mode = outer_mode
                partial_tp_price = None
                partial_tp_ratio = 0.0
                tp_plan = "SINGLE"
                middle_profit_fallback_locked = True

        if middle_profit_fallback_locked:
            pass
        elif (degrade_target := self._three_stage_pre_tp1_degrade_target(ts_ms)) == "SINGLE":
            tp_price, tp_mode = self._degrade_three_stage_pre_tp1_to_single(ts_ms, boll)
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
            reason_override = "three_stage_pre_tp1_degraded_to_single"
        elif degrade_target == "MIDDLE_RUNNER":
            self._degrade_three_stage_pre_tp1_to_middle_runner(ts_ms, boll)
            tp_price = self.state.tp_price or self._select_tp_outer(self.state.side, boll)[0]
            tp_mode = self.state.tp_mode
            partial_tp_price = self.state.partial_tp_price
            partial_tp_ratio = self.state.partial_tp_ratio
            tp_plan = "MIDDLE_RUNNER"
            reason_override = "three_stage_pre_tp1_degraded_to_middle_runner"
        elif self.state.trend_runner_active:
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
            tp_price, _tp_src = self._select_tp_outer(self.state.side, boll)
            tp_mode = "UPPER" if self.state.side == "LONG" else "LOWER"
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
            self._advance_runner_sl_time_tighten_candle_count(
                target="middle_runner",
                candle_ts_ms=int(getattr(boll, "candle_ts_ms", 0) or 0),
            )
            calculated_sl = self._calculate_middle_runner_protective_sl(self.state.side, price, boll)
            extension_sl = self._apply_middle_runner_extension_trigger(self.state.side, price, boll, calculated_sl)
            protective_sl = self._tighten_optional_middle_runner_sl(self.state.side, old_runner_sl, extension_sl)
            self.state.middle_runner_final_tp_price = tp_price
            self.state.middle_runner_protective_sl_price = protective_sl
        elif self.state.middle_runner_pending:
            tp_price, _tp_src = self._select_tp_outer(self.state.side, boll)
            tp_mode = "UPPER" if self.state.side == "LONG" else "LOWER"
            partial_tp_price, _ptp_src = self._select_tp_middle_with_profit_fallback(self.state.side, boll)
            partial_tp_ratio = self.state.middle_runner_first_close_ratio or min(
                max(self.config.middle_runner_first_close_ratio, 0.1), 0.95)
            tp_plan = "MIDDLE_RUNNER"
            self.state.middle_runner_first_tp_price = partial_tp_price
            self.state.middle_runner_final_tp_price = tp_price
        elif self.state.three_stage_runner_enabled_for_position and not self.state.trend_runner_active:
            tp_price, _tp_src = self._select_tp_outer(self.state.side, boll)
            tp_mode = "UPPER" if self.state.side == "LONG" else "LOWER"
            partial_tp_price, _ptp_src = self._select_tp_middle_with_profit_fallback(self.state.side, boll)
            self._update_three_stage_dynamic_targets_without_reset(self.state.side, boll)
            tp1_ratio = self.state.three_stage_tp1_ratio
            partial_tp_ratio = tp1_ratio
            tp_plan = "THREE_STAGE_RUNNER"
        elif self.state.near_tp_protected or self.state.near_tp_add_disabled:
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
        else:
            partial_tp_price, partial_tp_ratio, tp_plan = self._select_tp_plan(self.state.side, tp_price,
                                                                               self.state.layers, tp_mode=tp_mode,
                                                                               boll=boll)
            if tp_plan == "MIDDLE_RUNNER":
                tp_price, _tp_src = self._select_tp_outer(self.state.side, boll)
            if tp_plan == "THREE_STAGE_RUNNER":
                tp_price, _tp_src = self._select_tp_outer(self.state.side, boll)
            if tp_plan == "MIDDLE_RUNNER":
                self._set_middle_runner_planned(partial_tp_price, tp_price)
            elif tp_plan == "THREE_STAGE_RUNNER":
                self._update_three_stage_dynamic_targets_without_reset(self.state.side, boll)
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
                        or abs(
                    self.state.middle_runner_protective_sl_price - old_runner_sl) / self.state.middle_runner_protective_sl_price >= 0.0001
                )
        )
        trend_runner_orders_changed = (
                self.state.trend_runner_active
                and (
                        old_trend_runner_tp is None
                        or old_trend_runner_sl is None
                        or self.state.trend_runner_tp_price is None
                        or self.state.trend_runner_sl_price is None
                        or abs(
                    self.state.trend_runner_tp_price - old_trend_runner_tp) / self.state.trend_runner_tp_price >= 0.0001
                        or abs(
                    self.state.trend_runner_sl_price - old_trend_runner_sl) / self.state.trend_runner_sl_price >= 0.0001
                )
        )
        if (
                reason_override is None
                and self._tp_plan_unchanged(tp_price, partial_tp_price, partial_tp_ratio, tp_plan)
                and not runner_sl_changed
                and not trend_runner_orders_changed
                and not force_reconcile
        ):
            self.state.startup_force_tp_reconcile = False
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

        self.state.startup_force_tp_reconcile = False
        self.state.tp_price = tp_price
        self.state.tp_mode = tp_mode
        self.state.partial_tp_price = partial_tp_price
        self.state.partial_tp_ratio = partial_tp_ratio
        self.state.tp_plan = tp_plan
        if tp_plan == "MIDDLE_RUNNER":
            self._set_middle_runner_planned(partial_tp_price, tp_price)
        if tp_plan == "THREE_STAGE_RUNNER":
            self._update_three_stage_dynamic_targets_without_reset(self.state.side, boll)
        if self.state.trend_runner_active and self.config.runner_dynamic_enabled:
            self.state.trend_runner_adjust_count += 1
            self.state.trend_runner_last_update_candle_ts_ms = boll.candle_ts_ms
        size = self.sizer.calculate(price, layer_index=self.state.layers)
        if reason_override is not None:
            reason_text = reason_override
        else:
            reason_text = f"startup_force_tp_reconcile" if force_reconcile else f"新15m K线更新止盈到{tp_mode}轨"
        logger.warning(
            "TP_SELECTED | reason=%s side=%s mode=%s plan=%s partial_tp=%s partial_ratio=%.2f avg_entry=%.4f breakeven=%.4f candle_ts=%s middle=%.4f upper=%.4f lower=%.4f final_tp=%.4f force_reconcile=%s",
            reason_text,
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
            force_reconcile,
        )
        self._log_tp_boll_price_selected(
            phase="waiting_tp2" if self._three_stage_waiting_tp2() else "update",
            boll=boll,
            tp_price=tp_price,
            tp_mode=tp_mode,
            tp_plan=tp_plan,
            partial_tp_price=partial_tp_price,
            tp1_price=self.state.three_stage_tp1_price if tp_plan == "THREE_STAGE_RUNNER" else None,
            tp2_price=self.state.three_stage_tp2_price if tp_plan == "THREE_STAGE_RUNNER" else None,
            first_tp_price=self.state.middle_runner_first_tp_price if tp_plan == "MIDDLE_RUNNER" else None,
            final_tp_price=self.state.middle_runner_final_tp_price if tp_plan == "MIDDLE_RUNNER" else None,
        )
        return self._intent("UPDATE_TP", self.state.side, price, self.state.layers, tp_price, reason_text, size, boll,
                            cvd, ts_ms)

    def _three_stage_pre_tp1_age_seconds(self, ts_ms: int) -> float:
        first_entry_ts_ms = int(getattr(self.state, "first_entry_ts_ms", 0) or 0)
        if first_entry_ts_ms <= 0:
            first_entry_ts_ms = int(getattr(self.state, "last_order_ts_ms", 0) or 0)
        if first_entry_ts_ms <= 0:
            return 0.0
        return max((ts_ms - first_entry_ts_ms) / 1000, 0.0)

    def _three_stage_pre_tp1_degrade_target(self, ts_ms: int) -> str | None:
        if not self.config.three_stage_pre_tp1_degrade_enabled:
            return None
        if self.state.three_stage_tp1_consumed:
            return None
        if self.state.three_stage_tp2_consumed:
            return None
        if self.state.trend_runner_active:
            return None
        if self.state.middle_runner_active:
            return None
        if self.state.partial_tp_consumed:
            return None

        age = self._three_stage_pre_tp1_age_seconds(ts_ms)
        if self.state.three_stage_pre_tp1_degrade_stage == "SINGLE":
            return None
        if self.state.three_stage_pre_tp1_degrade_stage == "MIDDLE_RUNNER":
            if age >= self.config.three_stage_pre_tp1_single_after_seconds:
                return "SINGLE"
            return None

        if not self.state.three_stage_runner_enabled_for_position and not self.config.three_stage_runner_enabled:
            return None
        if age >= self.config.three_stage_pre_tp1_single_after_seconds:
            return "SINGLE"
        if age >= self.config.three_stage_pre_tp1_middle_runner_after_seconds:
            return "MIDDLE_RUNNER"
        return None

    def _degrade_three_stage_pre_tp1_to_middle_runner(self, ts_ms: int, boll: BollSnapshot) -> None:
        if self.state.side is None:
            return
        old_tp1 = self.state.three_stage_tp1_price
        old_tp2 = self.state.three_stage_tp2_price
        old_tp_plan = self.state.tp_plan
        age_seconds = self._three_stage_pre_tp1_age_seconds(ts_ms)
        final_tp, _tp_src = self._select_tp_outer(self.state.side, boll)
        first_tp, _first_src = self._select_tp_middle_with_profit_fallback(self.state.side, boll)

        self._reset_three_stage_runner_state()
        self._set_middle_runner_planned(first_tp, final_tp)
        self.state.tp_plan = "MIDDLE_RUNNER"
        self.state.tp_price = final_tp
        self.state.tp_mode = "UPPER" if self.state.side == "LONG" else "LOWER"
        self.state.partial_tp_price = first_tp
        self.state.partial_tp_ratio = self.state.middle_runner_first_close_ratio
        self.state.three_stage_pre_tp1_degrade_stage = "MIDDLE_RUNNER"
        self.state.three_stage_pre_tp1_degraded_ts_ms = ts_ms
        logger.warning(
            "THREE_STAGE_PRE_TP1_DEGRADED | from=%s to=MIDDLE_RUNNER side=%s age_seconds=%.1f old_tp1=%s old_tp2=%s new_first_tp=%.4f new_final_tp=%.4f first_entry_ts_ms=%s",
            old_tp_plan,
            self.state.side,
            age_seconds,
            f"{old_tp1:.4f}" if old_tp1 is not None else "-",
            f"{old_tp2:.4f}" if old_tp2 is not None else "-",
            first_tp,
            final_tp,
            self.state.first_entry_ts_ms,
        )

    def _degrade_three_stage_pre_tp1_to_single(self, ts_ms: int, boll: BollSnapshot) -> tuple[float, TpMode]:
        if self.state.side is None:
            return boll.middle, "MIDDLE"
        old_plan = self.state.tp_plan
        tp_price, tp_mode = self._select_tp_price(self.state.side, boll)
        age_seconds = self._three_stage_pre_tp1_age_seconds(ts_ms)
        self._reset_three_stage_runner_state()
        self._reset_middle_runner_state()
        self.state.tp_plan = "SINGLE"
        self.state.tp_price = tp_price
        self.state.tp_mode = tp_mode
        self.state.partial_tp_price = None
        self.state.partial_tp_ratio = 0.0
        self.state.three_stage_pre_tp1_degrade_stage = "SINGLE"
        self.state.three_stage_pre_tp1_degraded_ts_ms = ts_ms
        logger.warning(
            "THREE_STAGE_PRE_TP1_DEGRADED | from=%s to=SINGLE side=%s age_seconds=%.1f tp_price=%.4f tp_mode=%s middle=%.4f upper=%.4f lower=%.4f first_entry_ts_ms=%s",
            old_plan,
            self.state.side,
            age_seconds,
            tp_price,
            tp_mode,
            boll.middle,
            boll.upper,
            boll.lower,
            self.state.first_entry_ts_ms,
        )
        return tp_price, tp_mode

    def _maybe_trend_runner_market_exit(self, price: float, ts_ms: int, boll: BollSnapshot,
                                        cvd: CvdSnapshot) -> TradeIntent | None:
        if not self.state.trend_runner_active or self.state.side is None:
            return None
        side = self.state.side

        decision = trend_runner_helpers.trend_runner_market_exit_reason(
            side=side,
            price=price,
            boll_middle=boll.middle,
            tp_price=self.state.trend_runner_tp_price,
            sl_price=self.state.trend_runner_sl_price,
            trend_start_ts_ms=self.state.trend_runner_trend_start_ts_ms,
            ts_ms=ts_ms,
            runner_max_trend_seconds_after_second_tp=self.config.runner_max_trend_seconds_after_second_tp,
        )

        if decision.reason is not None:
            if decision.reason == "trend_runner_max_time_after_second_tp":
                logger.warning(
                    "TREND_RUNNER_MAX_TIME_EXIT | side=%s start_ts_ms=%s ts_ms=%s max_seconds=%s",
                    side,
                    int(self.state.trend_runner_trend_start_ts_ms or 0),
                    ts_ms,
                    self.config.runner_max_trend_seconds_after_second_tp,
                )
            return self._runner_market_exit_intent(price, ts_ms, boll, cvd, decision.reason)

        reverse_reason = self._maybe_confirm_trend_runner_reverse_burst(side, price, ts_ms, cvd)
        if reverse_reason is not None:
            return self._runner_market_exit_intent(price, ts_ms, boll, cvd, reverse_reason)
        return None

    def _maybe_confirm_trend_runner_reverse_burst(self, side: PositionSide, price: float, ts_ms: int,
                                                  cvd: CvdSnapshot) -> str | None:
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

        self.state.trend_runner_reverse_extreme_price = trend_runner_helpers.update_trend_runner_reverse_extreme_price(
            side=side,
            current_extreme_price=self.state.trend_runner_reverse_extreme_price,
            price=price,
        )
        samples.append((ts_ms, cvd.buy_ratio, cvd.sell_ratio, cvd.fast_cvd, price))
        cutoff_ts = ts_ms - max(self.config.runner_reverse_burst_confirm_seconds * 1000, 1)
        self.state.trend_runner_reverse_samples = trend_runner_helpers.prune_trend_runner_reverse_samples(
            samples=samples,
            cutoff_ts_ms=cutoff_ts,
        )

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
        decision = trend_runner_helpers.trend_runner_reverse_candidate(
            side=side,
            up_burst=cvd.up_burst,
            down_burst=cvd.down_burst,
            buy_ratio=cvd.buy_ratio,
            sell_ratio=cvd.sell_ratio,
            fast_cvd=cvd.fast_cvd,
            cvd_increasing=cvd.cvd_increasing,
            cvd_decreasing=cvd.cvd_decreasing,
            runner_reverse_strong_ratio=self.config.runner_reverse_strong_ratio,
        )
        return decision.is_candidate

    def _trend_runner_reverse_confirmed(self, side: PositionSide, current_price: float, cvd: CvdSnapshot) -> bool:
        samples = self.state.trend_runner_reverse_samples or []
        decision = trend_runner_helpers.trend_runner_reverse_confirmed(
            side=side,
            current_price=current_price,
            samples=samples,
            start_price=self.state.trend_runner_reverse_start_price,
            extreme_price=self.state.trend_runner_reverse_extreme_price,
            fast_cvd_start=self.state.trend_runner_reverse_fast_cvd_start,
            current_fast_cvd=cvd.fast_cvd,
            runner_reverse_sell_ratio=self.config.runner_reverse_sell_ratio,
            runner_reverse_buy_ratio=self.config.runner_reverse_buy_ratio,
            runner_reverse_min_price_damage_pct=self.config.runner_reverse_min_price_damage_pct,
            runner_reverse_recovery_cancel_pct=self.config.runner_reverse_recovery_cancel_pct,
        )
        return decision.confirmed

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

    def _maybe_near_tp_reduce(self, price: float, ts_ms: int, boll: BollSnapshot,
                              cvd: CvdSnapshot) -> TradeIntent | None:
        if not self.config.near_tp_enabled:
            return None
        sidecar_gate = near_tp_helpers.near_tp_sidecar_skip_allowed(
            sidecar_enabled_for_position=self.state.sidecar_enabled_for_position,
        )
        if not sidecar_gate.allowed:
            if sidecar_gate.reason == "sidecar_enabled" and not self.state.near_tp_sidecar_skip_logged:
                logger.info(
                    "NEAR_TP_REDUCE_SKIPPED | reason=sidecar_enabled side=%s price=%.4f sidecar_open_qty=%.8f",
                    self.state.side,
                    price,
                    self.state.sidecar_open_qty,
                )
                self.state.near_tp_sidecar_skip_logged = True
            return None
        if self.state.side is None or self.state.tp_price is None:
            return None
        if self.state.avg_entry_price <= 0 or price <= 0:
            return None
        if self.state.near_tp_protected:
            return None

        plan_gate = near_tp_helpers.near_tp_plan_allowed(
            tp_plan=self.state.tp_plan,
            middle_runner_pending=self.state.middle_runner_pending,
            middle_runner_active=self.state.middle_runner_active,
            three_stage_runner_enabled_for_position=self.state.three_stage_runner_enabled_for_position,
            trend_runner_active=self.state.trend_runner_active,
            partial_tp_consumed=self.state.partial_tp_consumed,
        )
        if not plan_gate.allowed:
            return None

        side = self.state.side
        avg = self.state.avg_entry_price
        final_tp = self.state.tp_price

        progress_result = near_tp_helpers.calculate_near_tp_progress(
            side=side,
            price=price,
            avg_entry_price=avg,
            final_tp_price=final_tp,
            near_tp_max_distance_usd=self.config.near_tp_max_distance_usd,
            near_tp_min_reduce_profit_pct=self.config.near_tp_min_reduce_profit_pct,
            near_tp_min_profit_pct=self.config.near_tp_min_profit_pct,
            near_tp_min_progress_ratio=self.config.near_tp_min_progress_ratio,
        )
        if progress_result is None:
            return None

        progress = progress_result.progress
        profit_pct = progress_result.profit_pct
        near_by_distance = progress_result.near_by_distance
        near_by_progress = progress_result.near_by_progress
        reduce_profit_ok = progress_result.reduce_profit_ok
        min_profit_seen_ok = progress_result.min_profit_seen_ok

        if not self.state.near_tp_armed:
            arming = near_tp_helpers.should_arm_near_tp(progress=progress_result)
            if arming:
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

        best_decision = near_tp_helpers.update_near_tp_best_price(
            side=side,
            old_best_price=self.state.near_tp_best_price,
            price=price,
        )
        best = best_decision.best_price
        if best_decision.changed:
            self.state.near_tp_best_price = best
            logger.info("NEAR_TP_BEST_UPDATED | side=%s best_price=%.4f price=%.4f", side, best, price)
        else:
            self.state.near_tp_best_price = best

        if self.state.near_tp_reduce_pending:
            if near_tp_helpers.near_tp_pending_can_reduce(reduce_profit_ok=reduce_profit_ok):
                return self._near_tp_reduce_intent(price, ts_ms, boll, cvd, progress, best, 0.0, 0.0)
            return None

        giveback_result = near_tp_helpers.calculate_near_tp_giveback(
            side=side,
            price=price,
            avg_entry_price=avg,
            best_price=best,
            near_tp_giveback_usd=self.config.near_tp_giveback_usd,
            near_tp_giveback_pct=self.config.near_tp_giveback_pct,
            near_tp_giveback_profit_ratio=self.config.near_tp_giveback_profit_ratio,
        )
        giveback = giveback_result.giveback
        giveback_threshold = giveback_result.threshold
        if not giveback_result.triggered:
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
        protective_sl = near_tp_helpers.calculate_near_tp_protective_sl(
            side=side,
            avg_entry_price=self.state.avg_entry_price,
            near_tp_protective_sl_profit_pct=self.config.near_tp_protective_sl_profit_pct,
        )
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
        self.state.position_cost_entry_notional += entry_price * eth_qty
        self.state.position_cost_remaining_qty += eth_qty
        self._refresh_net_remaining_breakeven_price()

    def _refresh_net_remaining_breakeven_price(self) -> None:
        if self.state.side not in {"LONG", "SHORT"}:
            self.state.net_remaining_breakeven_price = 0.0
            return
        basis = calculate_remaining_breakeven_price(
            side=self.state.side,
            entry_notional=self.state.position_cost_entry_notional,
            exit_notional=self.state.position_cost_exit_notional,
            remaining_qty=self.state.position_cost_remaining_qty,
            fee_buffer_pct=self.config.breakeven_fee_buffer_pct,
        )
        self.state.net_remaining_breakeven_price = float(basis.buffered_breakeven_price or 0.0)

    def _apply_near_tp_state_values(self, values: near_tp_helpers.NearTpStateValues) -> None:
        self.state.near_tp_armed = values.near_tp_armed
        self.state.near_tp_reduce_pending = values.near_tp_reduce_pending
        self.state.near_tp_protected = values.near_tp_protected
        self.state.near_tp_best_price = values.near_tp_best_price
        self.state.near_tp_armed_ts_ms = values.near_tp_armed_ts_ms
        self.state.near_tp_pending_ts_ms = values.near_tp_pending_ts_ms
        self.state.near_tp_trigger_ts_ms = values.near_tp_trigger_ts_ms
        self.state.near_tp_protective_sl_price = values.near_tp_protective_sl_price
        self.state.near_tp_protective_sl_order_id = values.near_tp_protective_sl_order_id
        self.state.near_tp_add_disabled = values.near_tp_add_disabled
        self.state.near_tp_sidecar_skip_logged = values.near_tp_sidecar_skip_logged

    def _reset_near_tp_state(self) -> None:
        values = near_tp_helpers.reset_near_tp_state_values()
        self._apply_near_tp_state_values(values)

    def _apply_middle_runner_state_values(self, values: middle_runner_helpers.MiddleRunnerStateValues) -> None:
        self.state.middle_runner_enabled_for_position = values.middle_runner_enabled_for_position
        self.state.middle_runner_pending = values.middle_runner_pending
        self.state.middle_runner_active = values.middle_runner_active
        self.state.middle_runner_first_close_ratio = values.middle_runner_first_close_ratio
        self.state.middle_runner_keep_ratio = values.middle_runner_keep_ratio
        self.state.middle_runner_first_tp_price = values.middle_runner_first_tp_price
        self.state.middle_runner_final_tp_price = values.middle_runner_final_tp_price
        self.state.middle_runner_protective_sl_price = values.middle_runner_protective_sl_price
        self.state.middle_runner_protective_sl_order_id = values.middle_runner_protective_sl_order_id
        self.state.middle_runner_extension_triggered = values.middle_runner_extension_triggered
        self.state.middle_runner_add_disabled = values.middle_runner_add_disabled
        self.state.middle_runner_size_mismatch_protected = values.middle_runner_size_mismatch_protected
        self.state.middle_runner_size_mismatch_warning_ts_ms = values.middle_runner_size_mismatch_warning_ts_ms
        self.state.middle_runner_sl_time_tighten_candle_count = values.middle_runner_sl_time_tighten_candle_count
        self.state.middle_runner_sl_time_tighten_last_candle_ts_ms = values.middle_runner_sl_time_tighten_last_candle_ts_ms
        self.state.middle_runner_sl_time_tighten_log_candle_ts_ms = values.middle_runner_sl_time_tighten_log_candle_ts_ms

    def _apply_trend_runner_state_values(self, values: trend_runner_helpers.TrendRunnerStateValues) -> None:
        self.state.trend_runner_active = values.trend_runner_active
        self.state.trend_runner_trend_start_ts_ms = values.trend_runner_trend_start_ts_ms
        self.state.trend_runner_adjust_count = values.trend_runner_adjust_count
        self.state.trend_runner_last_update_candle_ts_ms = values.trend_runner_last_update_candle_ts_ms
        self.state.trend_runner_tp_price = values.trend_runner_tp_price
        self.state.trend_runner_sl_price = values.trend_runner_sl_price
        self.state.trend_runner_tp_order_id = values.trend_runner_tp_order_id
        self.state.trend_runner_sl_order_id = values.trend_runner_sl_order_id
        self.state.trend_runner_exit_reason = values.trend_runner_exit_reason

    def _apply_trend_runner_reverse_state_values(self,
                                                 values: trend_runner_helpers.TrendRunnerReverseStateValues) -> None:
        self.state.trend_runner_reverse_candidate = values.trend_runner_reverse_candidate
        self.state.trend_runner_reverse_start_ts_ms = values.trend_runner_reverse_start_ts_ms
        self.state.trend_runner_reverse_start_price = values.trend_runner_reverse_start_price
        self.state.trend_runner_reverse_extreme_price = values.trend_runner_reverse_extreme_price
        self.state.trend_runner_reverse_fast_cvd_start = values.trend_runner_reverse_fast_cvd_start
        self.state.trend_runner_reverse_samples = values.trend_runner_reverse_samples

    def _reset_middle_runner_state(self) -> None:
        values = middle_runner_helpers.reset_middle_runner_state_values()
        self._apply_middle_runner_state_values(values)

    # ── Three-Stage state value appliers ──────────────────────────────────

    def _apply_three_stage_state_values(self, values: three_stage_helpers.ThreeStageStateValues) -> None:
        """Write back all Three-Stage state fields from a value object.

        Does NOT write Trend Runner fields.
        """
        self.state.three_stage_runner_enabled_for_position = values.three_stage_runner_enabled_for_position
        self.state.three_stage_tp1_price = values.three_stage_tp1_price
        self.state.three_stage_tp2_price = values.three_stage_tp2_price
        self.state.three_stage_runner_initial_tp_price = values.three_stage_runner_initial_tp_price
        self.state.three_stage_tp1_ratio = values.three_stage_tp1_ratio
        self.state.three_stage_tp2_ratio = values.three_stage_tp2_ratio
        self.state.three_stage_runner_ratio = values.three_stage_runner_ratio
        self.state.three_stage_tp1_consumed = values.three_stage_tp1_consumed
        self.state.three_stage_tp2_consumed = values.three_stage_tp2_consumed
        self.state.three_stage_post_tp1_protective_sl_price = values.three_stage_post_tp1_protective_sl_price
        self.state.three_stage_post_tp1_protective_sl_order_id = values.three_stage_post_tp1_protective_sl_order_id
        self.state.three_stage_post_tp1_sl_extension_triggered = values.three_stage_post_tp1_sl_extension_triggered
        self.state.three_stage_post_tp1_protected = values.three_stage_post_tp1_protected
        self.state.three_stage_post_tp1_sl_time_tighten_candle_count = values.three_stage_post_tp1_sl_time_tighten_candle_count
        self.state.three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms = values.three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms
        self.state.three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms = values.three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms

    def _apply_three_stage_dynamic_target_values(self,
                                                 values: three_stage_helpers.ThreeStageDynamicTargetValues) -> None:
        """Write back dynamic target update fields from a value object.

        Does NOT write consumed flags, protective SL, or extension flags.
        """
        self.state.three_stage_runner_enabled_for_position = values.three_stage_runner_enabled_for_position
        self.state.three_stage_tp1_price = values.three_stage_tp1_price
        self.state.three_stage_tp2_price = values.three_stage_tp2_price
        self.state.three_stage_tp1_ratio = values.three_stage_tp1_ratio
        self.state.three_stage_tp2_ratio = values.three_stage_tp2_ratio
        self.state.three_stage_runner_ratio = values.three_stage_runner_ratio

    def _reset_three_stage_runner_state(self) -> None:
        values = three_stage_helpers.reset_three_stage_state_values()
        self._apply_three_stage_state_values(values)
        trend_values = trend_runner_helpers.reset_trend_runner_state_values()
        self._apply_trend_runner_state_values(trend_values)
        self._reset_trend_runner_reverse_state()

    def _reset_trend_runner_reverse_state(self) -> None:
        values = trend_runner_helpers.reset_trend_runner_reverse_state_values()
        self._apply_trend_runner_reverse_state_values(values)

    def _reset_middle_runner_sl_time_tighten_state(self) -> None:
        self.state.middle_runner_sl_time_tighten_candle_count = 0
        self.state.middle_runner_sl_time_tighten_last_candle_ts_ms = 0
        self.state.middle_runner_sl_time_tighten_log_candle_ts_ms = 0

    def _reset_three_stage_post_tp1_sl_time_tighten_state(self) -> None:
        count, last_ts, log_ts = three_stage_helpers.reset_three_stage_post_tp1_sl_time_tighten_values()
        self.state.three_stage_post_tp1_sl_time_tighten_candle_count = count
        self.state.three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms = last_ts
        self.state.three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms = log_ts

    def _set_middle_runner_planned(self, first_tp_price: float | None, final_tp_price: float) -> None:
        values = middle_runner_helpers.planned_middle_runner_state_values(
            first_tp_price=first_tp_price,
            final_tp_price=final_tp_price,
            configured_first_close_ratio=self.config.middle_runner_first_close_ratio,
        )
        self._apply_middle_runner_state_values(values)

    def _set_three_stage_runner_planned(self, side: PositionSide, boll: BollSnapshot) -> None:
        tp1_ratio, tp2_ratio, runner_ratio = self._normalized_three_stage_ratios()
        tp_mid, _tp_mid_src = self._select_tp_middle_with_profit_fallback(side, boll)
        tp_outer, _tp_outer_src = self._select_tp_outer(side, boll)
        ratios = three_stage_helpers.ThreeStageRatios(
            tp1_ratio=tp1_ratio,
            tp2_ratio=tp2_ratio,
            runner_ratio=runner_ratio,
        )
        values = three_stage_helpers.planned_three_stage_state_values(
            tp1_price=tp_mid,
            tp2_price=tp_outer,
            ratios=ratios,
        )
        self._apply_three_stage_state_values(values)
        trend_values = trend_runner_helpers.reset_trend_runner_state_values()
        self._apply_trend_runner_state_values(trend_values)
        self._reset_trend_runner_reverse_state()

    def _update_three_stage_dynamic_targets_without_reset(self, side: PositionSide, boll: BollSnapshot) -> None:
        tp1_ratio, tp2_ratio, runner_ratio = self._normalized_three_stage_ratios()
        tp_mid, _tp_mid_src = self._select_tp_middle_with_profit_fallback(side, boll)
        tp_outer, _tp_outer_src = self._select_tp_outer(side, boll)
        ratios = three_stage_helpers.ThreeStageRatios(
            tp1_ratio=tp1_ratio,
            tp2_ratio=tp2_ratio,
            runner_ratio=runner_ratio,
        )
        values = three_stage_helpers.update_three_stage_dynamic_target_values(
            tp1_price=tp_mid,
            tp2_price=tp_outer,
            ratios=ratios,
        )
        self._apply_three_stage_dynamic_target_values(values)

    def _normalized_three_stage_ratios(self) -> tuple[float, float, float]:
        ratios = three_stage_helpers.normalize_three_stage_ratios(
            tp1_ratio=self.config.three_stage_tp1_ratio,
            tp2_ratio=self.config.three_stage_tp2_ratio,
            runner_ratio=self.config.three_stage_runner_ratio,
        )
        return ratios.tp1_ratio, ratios.tp2_ratio, ratios.runner_ratio

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
        decision = trend_runner_helpers.calculate_trend_runner_dynamic_orders(
            side=side,
            boll_middle=boll.middle,
            boll_upper=boll.upper,
            boll_lower=boll.lower,
            adjust_count=adjust_count,
            current_sl_price=old_sl,
            runner_tp_initial_outer_extra_pct=self.config.runner_tp_initial_outer_extra_pct,
            runner_tp_step_pct=self.config.runner_tp_step_pct,
            runner_tp_min_outer_extra_pct=self.config.runner_tp_min_outer_extra_pct,
            runner_sl_initial_outer_distance_ratio=self.config.runner_sl_initial_outer_distance_ratio,
            runner_sl_step_ratio=self.config.runner_sl_step_ratio,
            runner_sl_min_outer_distance_ratio=self.config.runner_sl_min_outer_distance_ratio,
        )
        return decision.tp_price, decision.sl_price, decision.tp_extra_pct, decision.sl_distance_ratio

    def _advance_runner_sl_time_tighten_candle_count(
            self,
            *,
            target: Literal["middle_runner", "three_stage_post_tp1"],
            candle_ts_ms: int,
    ) -> int:
        if target == "middle_runner":
            count_attr = "middle_runner_sl_time_tighten_candle_count"
            last_attr = "middle_runner_sl_time_tighten_last_candle_ts_ms"
        else:
            count_attr = "three_stage_post_tp1_sl_time_tighten_candle_count"
            last_attr = "three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms"

        count = int(getattr(self.state, count_attr, 0) or 0)
        if candle_ts_ms <= 0:
            return count
        last_candle_ts_ms = int(getattr(self.state, last_attr, 0) or 0)
        if last_candle_ts_ms <= 0:
            setattr(self.state, last_attr, candle_ts_ms)
            return 0
        if candle_ts_ms != last_candle_ts_ms:
            count += 1
            setattr(self.state, count_attr, count)
            setattr(self.state, last_attr, candle_ts_ms)
        return count

    def _seed_runner_sl_time_tighten_activation_candle(
            self,
            *,
            target: Literal["middle_runner", "three_stage_post_tp1"],
            candle_ts_ms: int,
    ) -> None:
        if candle_ts_ms <= 0:
            candle_ts_ms = int(getattr(self.state, "last_tp_update_candle_ts_ms", 0) or 0)
        if candle_ts_ms <= 0:
            return

        if target == "middle_runner":
            self.state.middle_runner_sl_time_tighten_candle_count = 0
            self.state.middle_runner_sl_time_tighten_last_candle_ts_ms = candle_ts_ms
            self.state.middle_runner_sl_time_tighten_log_candle_ts_ms = 0
        else:
            self.state.three_stage_post_tp1_sl_time_tighten_candle_count = 0
            self.state.three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms = candle_ts_ms
            self.state.three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms = 0

    def _runner_sl_time_tighten_ratio(self, candle_count: int) -> float:
        base_ratio = 0.50
        step = max(float(self.config.runner_protective_sl_time_tighten_step_ratio), 0.0)
        max_ratio = min(max(float(self.config.runner_protective_sl_time_tighten_max_ratio), base_ratio), 1.0)
        return min(base_ratio + max(int(candle_count), 0) * step, max_ratio, 1.0)

    def _calculate_middle_runner_protective_sl(self, side: PositionSide, current_price: float,
                                               boll: BollSnapshot) -> float | None:
        avg_entry = float(self.state.avg_entry_price or 0.0)
        base_breakeven = float(getattr(self.state, "net_remaining_breakeven_price", 0.0) or 0.0)
        fee = self.config.breakeven_fee_buffer_pct
        ratio = (
            self._runner_sl_time_tighten_ratio(self.state.middle_runner_sl_time_tighten_candle_count)
            if self.config.runner_protective_sl_time_tighten_enabled
            else 0.50
        )
        decision = middle_runner_helpers.calculate_middle_runner_protective_sl(
            side=side,
            current_price=current_price,
            avg_entry_price=avg_entry,
            net_remaining_breakeven_price=base_breakeven,
            breakeven_fee_buffer_pct=fee,
            boll_middle=boll.middle,
            boll_upper=boll.upper,
            boll_lower=boll.lower,
            sl_tighten_ratio=ratio,
        )
        if decision.reason == "missing_cost_basis":
            return None
        if decision.reason != "calculated":
            # Reconstruct the raw protective SL that was found invalid for the log signature.
            _raw_sl = (
                min(max(decision.candidate_cost, decision.candidate_structure), boll.middle)
                if side == "LONG"
                else max(min(decision.candidate_cost, decision.candidate_structure), boll.middle)
            )
            self._log_middle_runner_sl_diagnostic_once(
                side,
                decision.reason,
                current_price,
                base_breakeven,
                decision.candidate_cost,
                decision.candidate_structure,
                _raw_sl,
                boll,
            )
            return None
        self._log_middle_runner_sl_time_tightened_once(
            side,
            ratio,
            decision.candidate_cost,
            decision.candidate_structure,
            decision.protective_sl,
            boll,
        )
        self._log_middle_runner_sl_diagnostic_once(
            side,
            "calculated",
            current_price,
            base_breakeven,
            decision.candidate_cost,
            decision.candidate_structure,
            decision.protective_sl,
            boll,
        )
        return decision.protective_sl

    def _log_middle_runner_sl_diagnostic_once(
            self,
            side: PositionSide,
            reason: str,
            current_price: float,
            net_remaining_breakeven: float,
            candidate_cost: float,
            candidate_structure: float,
            protective_sl: float | None,
            boll: BollSnapshot,
    ) -> None:
        protective_sl_for_signature = float(protective_sl or 0.0)
        signature = (
            f"{side}|{getattr(boll, 'candle_ts_ms', 0)}|{round(net_remaining_breakeven, 4)}|"
            f"{round(candidate_cost, 4)}|{round(candidate_structure, 4)}|"
            f"{round(protective_sl_for_signature, 4)}|{reason}"
        )
        if self.state.middle_runner_sl_diag_last_signature == signature:
            return
        self.state.middle_runner_sl_diag_last_signature = signature
        breakeven_source = "net_remaining_breakeven" if net_remaining_breakeven > 0 else "avg_entry_fallback"
        protective_sl_text = f"{protective_sl:.4f}" if protective_sl is not None else "-"
        logger.warning(
            "MIDDLE_RUNNER_PROTECTIVE_SL_DIAG | side=%s reason=%s current_price=%.4f avg_entry=%.4f net_remaining_breakeven=%.4f breakeven_source=%s candidate_cost=%.4f candidate_structure=%.4f protective_sl=%s candle_ts=%s middle=%.4f upper=%.4f lower=%.4f position_cost_entry_notional=%.4f position_cost_exit_notional=%.4f position_cost_remaining_qty=%.8f",
            side,
            reason,
            current_price,
            float(self.state.avg_entry_price or 0.0),
            net_remaining_breakeven,
            breakeven_source,
            candidate_cost,
            candidate_structure,
            protective_sl_text,
            getattr(boll, "candle_ts_ms", 0),
            boll.middle,
            boll.upper,
            boll.lower,
            float(getattr(self.state, "position_cost_entry_notional", 0.0) or 0.0),
            float(getattr(self.state, "position_cost_exit_notional", 0.0) or 0.0),
            float(getattr(self.state, "position_cost_remaining_qty", 0.0) or 0.0),
        )

    def _log_middle_runner_sl_time_tightened_once(
            self,
            side: PositionSide,
            ratio: float,
            candidate_cost: float,
            candidate_structure: float,
            protective_sl: float,
            boll: BollSnapshot,
    ) -> None:
        candle_ts_ms = int(getattr(boll, "candle_ts_ms", 0) or 0)
        if ratio <= 0.50 or candle_ts_ms <= 0:
            return
        if self.state.middle_runner_sl_time_tighten_log_candle_ts_ms == candle_ts_ms:
            return
        self.state.middle_runner_sl_time_tighten_log_candle_ts_ms = candle_ts_ms
        logger.warning(
            "MIDDLE_RUNNER_SL_TIME_TIGHTENED | side=%s candle_count=%s ratio=%.4f candidate_cost=%.4f candidate_structure=%.4f protective_sl=%.4f middle=%.4f upper=%.4f lower=%.4f candle_ts=%s extension_triggered=%s",
            side,
            int(getattr(self.state, "middle_runner_sl_time_tighten_candle_count", 0) or 0),
            ratio,
            candidate_cost,
            candidate_structure,
            protective_sl,
            boll.middle,
            boll.upper,
            boll.lower,
            candle_ts_ms,
            bool(getattr(self.state, "middle_runner_extension_triggered", False)),
        )

    def _tighten_middle_runner_sl(self, side: PositionSide, old_sl: float, new_sl: float) -> float:
        return middle_runner_helpers.tighten_middle_runner_sl(side=side, old_sl=old_sl, new_sl=new_sl)

    def _tighten_optional_middle_runner_sl(self, side: PositionSide, old_sl: float | None,
                                           new_sl: float | None) -> float | None:
        return middle_runner_helpers.tighten_optional_middle_runner_sl(side=side, old_sl=old_sl, new_sl=new_sl)

    def _apply_middle_runner_extension_trigger(
            self,
            side: PositionSide,
            current_price: float,
            boll: BollSnapshot,
            protective_sl: float | None,
    ) -> float | None:
        decision = middle_runner_helpers.apply_middle_runner_extension_trigger(
            side=side,
            current_price=current_price,
            protective_sl=protective_sl,
            boll_middle=boll.middle,
            boll_upper=boll.upper,
            boll_lower=boll.lower,
            extension_trigger_ratio=self.config.middle_runner_extension_trigger_ratio,
            already_triggered=self.state.middle_runner_extension_triggered,
        )
        if decision.extension_triggered:
            if not self.state.middle_runner_extension_triggered:
                logger.warning(
                    "MIDDLE_RUNNER_EXTENSION_TRIGGERED | side=%s current_price=%.4f extension_trigger_price=%.4f protective_sl_price=%.4f boll_middle=%.4f boll_upper=%.4f boll_lower=%.4f",
                    side,
                    current_price,
                    decision.trigger_price,
                    decision.protective_sl,
                    boll.middle,
                    boll.upper,
                    boll.lower,
                )
            self.state.middle_runner_extension_triggered = True
        return decision.protective_sl

    def _calculate_three_stage_post_tp1_protective_sl(self, side: PositionSide, current_price: float,
                                                      boll: BollSnapshot) -> float | None:
        avg_entry = float(self.state.avg_entry_price or 0.0)
        tp1_price = self.state.three_stage_tp1_price
        tp1_ratio = float(self.state.three_stage_tp1_ratio or 0.0)
        base_breakeven = float(getattr(self.state, "net_remaining_breakeven_price", 0.0) or 0.0)
        fee = self.config.breakeven_fee_buffer_pct
        ratio = (
            self._runner_sl_time_tighten_ratio(self.state.three_stage_post_tp1_sl_time_tighten_candle_count)
            if self.config.runner_protective_sl_time_tighten_enabled
            else 0.50
        )
        decision = three_stage_helpers.calculate_three_stage_post_tp1_protective_sl(
            side=side,
            current_price=current_price,
            avg_entry_price=avg_entry,
            net_remaining_breakeven_price=base_breakeven,
            breakeven_fee_buffer_pct=fee,
            tp1_price=tp1_price,
            tp1_ratio=tp1_ratio,
            boll_middle=boll.middle,
            boll_upper=boll.upper,
            boll_lower=boll.lower,
            sl_tighten_ratio=ratio,
        )
        if decision.reason in ("missing_tp1_price", "missing_cost_basis", "invalid_tp1_ratio"):
            if current_price > 0 and base_breakeven <= 0:
                self._log_three_stage_post_tp1_sl_diagnostic_once(
                    side,
                    "missing_cost_basis",
                    current_price,
                    base_breakeven,
                    0.0,
                    0.0,
                    None,
                    boll,
                )
            return None
        if decision.reason != "calculated":
            # Reconstruct the raw protective SL that was found invalid for the log signature.
            _raw_sl = (
                min(max(decision.candidate_cost, decision.candidate_structure), boll.middle)
                if side == "LONG"
                else max(min(decision.candidate_cost, decision.candidate_structure), boll.middle)
            )
            self._log_three_stage_post_tp1_sl_diagnostic_once(
                side,
                decision.reason,
                current_price,
                base_breakeven,
                decision.candidate_cost,
                decision.candidate_structure,
                _raw_sl,
                boll,
            )
            return None
        self._log_three_stage_post_tp1_sl_time_tightened_once(
            side,
            ratio,
            decision.candidate_cost,
            decision.candidate_structure,
            decision.protective_sl,
            boll,
        )
        self._log_three_stage_post_tp1_sl_diagnostic_once(
            side,
            "calculated",
            current_price,
            base_breakeven,
            decision.candidate_cost,
            decision.candidate_structure,
            decision.protective_sl,
            boll,
        )
        return decision.protective_sl

    def _log_three_stage_post_tp1_sl_diagnostic_once(
            self,
            side: PositionSide,
            reason: str,
            current_price: float,
            net_remaining_breakeven: float,
            candidate_cost: float,
            candidate_structure: float,
            protective_sl: float | None,
            boll: BollSnapshot,
    ) -> None:
        protective_sl_for_signature = float(protective_sl or 0.0)
        signature = (
            f"{side}|{getattr(boll, 'candle_ts_ms', 0)}|{round(net_remaining_breakeven, 4)}|"
            f"{round(candidate_cost, 4)}|{round(candidate_structure, 4)}|"
            f"{round(protective_sl_for_signature, 4)}|{reason}"
        )
        if self.state.three_stage_post_tp1_sl_diag_last_signature == signature:
            return
        self.state.three_stage_post_tp1_sl_diag_last_signature = signature
        breakeven_source = "net_remaining_breakeven" if net_remaining_breakeven > 0 else "avg_entry_fallback"
        protective_sl_text = f"{protective_sl:.4f}" if protective_sl is not None else "-"
        tp1_price = getattr(self.state, "three_stage_tp1_price", None)
        tp1_price_text = f"{float(tp1_price):.4f}" if tp1_price is not None else "-"
        logger.warning(
            "THREE_STAGE_POST_TP1_PROTECTIVE_SL_DIAG | side=%s reason=%s current_price=%.4f avg_entry=%.4f net_remaining_breakeven=%.4f breakeven_source=%s tp1_price=%s tp1_ratio=%.4f candidate_cost=%.4f candidate_structure=%.4f protective_sl=%s candle_ts=%s middle=%.4f upper=%.4f lower=%.4f position_cost_entry_notional=%.4f position_cost_exit_notional=%.4f position_cost_remaining_qty=%.8f",
            side,
            reason,
            current_price,
            float(self.state.avg_entry_price or 0.0),
            net_remaining_breakeven,
            breakeven_source,
            tp1_price_text,
            float(getattr(self.state, "three_stage_tp1_ratio", 0.0) or 0.0),
            candidate_cost,
            candidate_structure,
            protective_sl_text,
            getattr(boll, "candle_ts_ms", 0),
            boll.middle,
            boll.upper,
            boll.lower,
            float(getattr(self.state, "position_cost_entry_notional", 0.0) or 0.0),
            float(getattr(self.state, "position_cost_exit_notional", 0.0) or 0.0),
            float(getattr(self.state, "position_cost_remaining_qty", 0.0) or 0.0),
        )

    def _log_three_stage_post_tp1_sl_time_tightened_once(
            self,
            side: PositionSide,
            ratio: float,
            candidate_cost: float,
            candidate_structure: float,
            protective_sl: float,
            boll: BollSnapshot,
    ) -> None:
        candle_ts_ms = int(getattr(boll, "candle_ts_ms", 0) or 0)
        if ratio <= 0.50 or candle_ts_ms <= 0:
            return
        if self.state.three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms == candle_ts_ms:
            return
        self.state.three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms = candle_ts_ms
        logger.warning(
            "THREE_STAGE_POST_TP1_SL_TIME_TIGHTENED | side=%s candle_count=%s ratio=%.4f candidate_cost=%.4f candidate_structure=%.4f protective_sl=%.4f middle=%.4f upper=%.4f lower=%.4f candle_ts=%s extension_triggered=%s",
            side,
            int(getattr(self.state, "three_stage_post_tp1_sl_time_tighten_candle_count", 0) or 0),
            ratio,
            candidate_cost,
            candidate_structure,
            protective_sl,
            boll.middle,
            boll.upper,
            boll.lower,
            candle_ts_ms,
            bool(getattr(self.state, "three_stage_post_tp1_sl_extension_triggered", False)),
        )

    def _tighten_three_stage_post_tp1_sl(self, side: PositionSide, old_sl: float, new_sl: float) -> float:
        return three_stage_helpers.tighten_three_stage_post_tp1_sl(side=side, old_sl=old_sl, new_sl=new_sl)

    def _tighten_optional_three_stage_post_tp1_sl(self, side: PositionSide, old_sl: float | None,
                                                  new_sl: float | None) -> float | None:
        return three_stage_helpers.tighten_optional_three_stage_post_tp1_sl(side=side, old_sl=old_sl, new_sl=new_sl)

    def _apply_three_stage_post_tp1_extension_trigger(
            self,
            side: PositionSide,
            current_price: float,
            boll: BollSnapshot,
            protective_sl: float | None,
    ) -> float | None:
        decision = three_stage_helpers.apply_three_stage_post_tp1_extension_trigger(
            side=side,
            current_price=current_price,
            protective_sl=protective_sl,
            boll_middle=boll.middle,
            boll_upper=boll.upper,
            boll_lower=boll.lower,
            extension_trigger_ratio=self.config.three_stage_post_tp1_sl_extension_trigger_ratio,
        )
        if decision.extension_triggered:
            if not self.state.three_stage_post_tp1_sl_extension_triggered:
                logger.warning(
                    "THREE_STAGE_POST_TP1_EXTENSION_TRIGGERED | side=%s current_price=%.4f extension_trigger_price=%.4f protective_sl_price=%.4f boll_middle=%.4f boll_upper=%.4f boll_lower=%.4f",
                    side,
                    current_price,
                    decision.trigger_price,
                    decision.protective_sl,
                    boll.middle,
                    boll.upper,
                    boll.lower,
                )
            self.state.three_stage_post_tp1_sl_extension_triggered = True
        return decision.protective_sl

    # ── TP BOLL resolver ──────────────────────────────────────────────
    # Centralised helpers that select the correct price source for every
    # TP price calculation.  All TP prices MUST flow through these methods
    # so that TP_BOLL_WINDOW=15 is applied consistently and the fallback
    # to structure BOLL20 is automatic.

    def _tp_band_snapshot(self, boll: BollSnapshot) -> tp_plan_selector.TpBandSnapshot:
        """Build a TpBandSnapshot from a BollSnapshot (thin adapter)."""
        return tp_plan_selector.TpBandSnapshot(
            middle=float(boll.middle),
            upper=float(boll.upper),
            lower=float(boll.lower),
            tp_middle=getattr(boll, "tp_middle", None),
            tp_upper=getattr(boll, "tp_upper", None),
            tp_lower=getattr(boll, "tp_lower", None),
            tp_window=getattr(boll, "tp_window", None),
        )

    def _tp_boll_available(self, boll: BollSnapshot) -> bool:
        """True when a valid TP-only BOLL snapshot is present."""
        return tp_plan_selector.tp_boll_available(
            tp_boll_enabled=self.config.tp_boll_enabled,
            tp_middle=getattr(boll, "tp_middle", None),
            tp_upper=getattr(boll, "tp_upper", None),
            tp_lower=getattr(boll, "tp_lower", None),
        )

    def _select_tp_middle(self, boll: BollSnapshot) -> tuple[float, str]:
        """Return (middle_price, source) preferring TP_BOLL15 middle."""
        tp_band = self._tp_band_snapshot(boll)
        sel = tp_plan_selector.select_tp_middle(
            tp_band=tp_band,
            tp_boll_enabled=self.config.tp_boll_enabled,
        )
        return sel.price, sel.source

    def _select_tp_middle_with_profit_fallback(
            self,
            side: PositionSide,
            boll: BollSnapshot,
    ) -> tuple[float, str]:
        """Return (middle_price, source) for TP1 / first TP with profit-distance fallback.

        Unlike _select_tp_middle() which is the raw low-level resolver, this
        helper enforces the min-net-profit check so that a TP1 price is never
        worse than what _select_tp_price() would have accepted for SINGLE mode.

        LONG:  TP_BOLL15 middle first → structure BOLL20 middle if TP_BOLL15
               profit is insufficient → TP_BOLL15 middle as last resort.
        SHORT: TP_BOLL15 middle first → structure BOLL20 middle if TP_BOLL15
               profit is insufficient → TP_BOLL15 middle as last resort.
        """
        effective_be = self._effective_breakeven_for_tp_selection(side)
        tp_band = self._tp_band_snapshot(boll)
        sel = tp_plan_selector.select_tp_middle_with_profit_fallback(
            side=side,
            effective_be=effective_be,
            min_net_profit=self.config.tp_min_net_profit_pct,
            tp_band=tp_band,
            tp_boll_enabled=self.config.tp_boll_enabled,
        )
        return sel.price, sel.source

    def _select_tp_outer(self, side: PositionSide, boll: BollSnapshot) -> tuple[float, str]:
        """Return (outer_price, source) for the given side."""
        tp_band = self._tp_band_snapshot(boll)
        sel = tp_plan_selector.select_tp_outer(
            side=side,
            tp_band=tp_band,
            tp_boll_enabled=self.config.tp_boll_enabled,
        )
        return sel.price, sel.source

    def _log_tp_boll_price_selected(
            self,
            *,
            phase: str,
            boll: BollSnapshot,
            tp_price: float,
            tp_mode: TpMode,
            tp_plan: TpPlan,
            partial_tp_price: float | None = None,
            tp1_price: float | None = None,
            tp2_price: float | None = None,
            first_tp_price: float | None = None,
            final_tp_price: float | None = None,
            tp1_source: str | None = None,
            tp2_source: str | None = None,
    ) -> None:
        """Log TP_BOLL_PRICE_SELECTED at initial TP gen or 15m UPDATE_TP only."""
        tp_boll_avail = self._tp_boll_available(boll)
        tp_mid, tp_mid_src = self._select_tp_middle(boll)
        tp_outer, tp_outer_src = self._select_tp_outer(self.state.side or "LONG", boll)

        # Determine the effective TP price sources.
        # Auto-detect profit fallback: if the actual TP1/first-tp price equals the
        # structure BOLL middle while TP_BOLL middle was available but unused,
        # we know the profit check forced a fallback.
        actual_mid_price: float | None = None
        if tp1_price is not None:
            actual_mid_price = tp1_price
        elif first_tp_price is not None:
            actual_mid_price = first_tp_price
        elif tp_mode == "MIDDLE" and tp_plan == "SINGLE":
            actual_mid_price = tp_price

        if (tp_boll_avail and actual_mid_price is not None
                and boll.tp_middle is not None
                and abs(actual_mid_price - float(boll.tp_middle)) > 0.0001
                and abs(actual_mid_price - float(boll.middle)) < 0.0001):
            resolved_middle_source = "STRUCTURE_BOLL_PROFIT_FALLBACK"
        else:
            resolved_middle_source = tp_mid_src

        # Callers may override with explicit source hints.
        if tp_plan in ("THREE_STAGE_RUNNER", "MIDDLE_RUNNER"):
            middle_source = tp1_source or resolved_middle_source
            outer_source = tp2_source or tp_outer_src
        elif tp_mode == "MIDDLE":
            middle_source = tp1_source or resolved_middle_source
            outer_source = "N/A"
        elif tp_mode in ("UPPER", "LOWER"):
            middle_source = "N/A"
            outer_source = tp_outer_src
        else:
            middle_source = "N/A"
            outer_source = "N/A"

        tp_mid_str = f"{boll.tp_middle:.4f}" if tp_boll_avail and boll.tp_middle is not None else "-"
        tp_up_str = f"{boll.tp_upper:.4f}" if tp_boll_avail and boll.tp_upper is not None else "-"
        tp_lo_str = f"{boll.tp_lower:.4f}" if tp_boll_avail and boll.tp_lower is not None else "-"

        tp1_str = (
            f"{tp1_price:.4f}" if tp1_price is not None
            else f"{first_tp_price:.4f}" if first_tp_price is not None
            else f"{partial_tp_price:.4f}" if partial_tp_price is not None
            else "-"
        )
        tp2_str = (
            f"{tp2_price:.4f}" if tp2_price is not None
            else f"{final_tp_price:.4f}" if final_tp_price is not None
            else "-"
        )

        reason = "available" if tp_boll_avail else "fallback_structure_boll"

        logger.info(
            "TP_BOLL_PRICE_SELECTED | mode=%s phase=%s side=%s tp_boll_enabled=%s "
            "tp_window=%s tp1_source=%s tp2_source=%s tp_price_source=%s "
            "structure_middle=%.4f structure_upper=%.4f structure_lower=%.4f "
            "tp_middle=%s tp_upper=%s tp_lower=%s "
            "final_tp1_price=%s final_tp2_price=%s final_tp_price=%.4f "
            "reason=tp_boll_%s",
            tp_plan, phase, self.state.side or "-", self.config.tp_boll_enabled,
            boll.tp_window if tp_boll_avail else "-",
            middle_source, outer_source,
            outer_source if tp_plan == "SINGLE" else middle_source,
            boll.middle, boll.upper, boll.lower,
            tp_mid_str, tp_up_str, tp_lo_str,
            tp1_str, tp2_str, tp_price,
            reason,
        )

    def _effective_breakeven_for_tp_selection(self, side: PositionSide) -> float:
        return tp_plan_selector.effective_breakeven_for_tp_selection(
            side=side,
            net_remaining_breakeven_price=float(getattr(self.state, "net_remaining_breakeven_price", 0.0) or 0.0),
            avg_entry_price=self.state.avg_entry_price,
            breakeven_fee_buffer_pct=self.config.breakeven_fee_buffer_pct,
        )

    def _select_tp_price(self, side: PositionSide, boll: BollSnapshot) -> tuple[float, TpMode]:
        """Select TP price preferring TP_BOLL15, with fallback to structure BOLL20.

        The profit-distance check is preserved exactly as before; only the price
        *candidate* source changes.
        """
        effective_be = self._effective_breakeven_for_tp_selection(side)
        if effective_be <= 0:
            return float(boll.middle), "MIDDLE"

        # Write breakeven_price when effective_be > 0 (preserves original side-effect)
        self.state.breakeven_price = effective_be

        tp_band = self._tp_band_snapshot(boll)
        sel = tp_plan_selector.select_tp_price(
            side=side,
            effective_be=effective_be,
            min_net_profit=self.config.tp_min_net_profit_pct,
            tp_band=tp_band,
            tp_boll_enabled=self.config.tp_boll_enabled,
        )
        return sel.price, sel.mode

    def _select_tp_plan(
            self,
            side: PositionSide,
            final_tp: float,
            layers: int,
            *,
            tp_mode: TpMode | None = None,
            boll: BollSnapshot | None = None,
    ) -> tuple[float | None, float, TpPlan]:
        # Pre-compute profit fallback price and plan-allowed decisions
        # (these depend on state/config which the pure function does not access)
        tp_mid_fb_price: float
        if boll is not None:
            tp_mid_fb_price, _fb_src = self._select_tp_middle_with_profit_fallback(side, boll)
        else:
            tp_mid_fb_price = 0.0

        three_stage_allowed = self._three_stage_runner_plan_allowed(tp_mode, boll)
        middle_runner_allowed = self._middle_runner_plan_allowed(tp_mode, boll)

        tp1_ratio: float = 0.0
        if three_stage_allowed:
            tp1_ratio, _tp2r, _rr = self._normalized_three_stage_ratios()

        sel = tp_plan_selector.select_tp_plan(
            side=side,
            final_tp=final_tp,
            layers=layers,
            tp_mode=tp_mode,
            boll_exists=boll is not None,
            three_stage_pre_tp1_degrade_stage=self.state.three_stage_pre_tp1_degrade_stage,
            middle_runner_first_close_ratio=self.config.middle_runner_first_close_ratio,
            tp_middle_profit_fallback_price=tp_mid_fb_price,
            three_stage_runner_plan_allowed=three_stage_allowed,
            three_stage_tp1_ratio=tp1_ratio,
            three_stage_runner_enabled=self.config.three_stage_runner_enabled,
            middle_runner_plan_allowed=middle_runner_allowed,
            split_tp_enabled=self.config.split_tp_enabled,
            split_tp_min_layers=self.config.split_tp_min_layers,
            partial_tp_consumed=self.state.partial_tp_consumed,
            avg_entry=self.state.avg_entry_price,
            split_tp_partial_ratio=self.config.split_tp_partial_ratio,
            split_tp_path_ratio=self.config.split_tp_path_ratio,
            split_tp_min_profit_pct=self.config.split_tp_min_profit_pct,
        )
        return sel.partial_tp_price, sel.partial_tp_ratio, sel.tp_plan

    def _three_stage_runner_plan_allowed(self, tp_mode: TpMode | None, boll: BollSnapshot | None) -> bool:
        return tp_plan_selector.three_stage_runner_plan_allowed(
            three_stage_runner_enabled=self.config.three_stage_runner_enabled,
            three_stage_pre_tp1_degrade_stage=self.state.three_stage_pre_tp1_degrade_stage,
            tp_mode=tp_mode,
            boll_exists=boll is not None,
            near_tp_protected=self.state.near_tp_protected,
            near_tp_add_disabled=self.state.near_tp_add_disabled,
            partial_tp_consumed=self.state.partial_tp_consumed,
            middle_runner_enabled_for_position=self.state.middle_runner_enabled_for_position,
            middle_runner_pending=self.state.middle_runner_pending,
            middle_runner_active=self.state.middle_runner_active,
            tp_plan=self.state.tp_plan,
            trend_runner_active=self.state.trend_runner_active,
        )

    def _three_stage_waiting_tp2(self) -> bool:
        return bool(
            self.state.three_stage_runner_enabled_for_position
            and self.state.three_stage_tp1_consumed
            and not self.state.three_stage_tp2_consumed
            and not self.state.trend_runner_active
        )

    def _middle_runner_plan_allowed(self, tp_mode: TpMode | None, boll: BollSnapshot | None) -> bool:
        return tp_plan_selector.middle_runner_plan_allowed(
            middle_runner_enabled=self.config.middle_runner_enabled,
            tp_mode=tp_mode,
            boll_exists=boll is not None,
            near_tp_protected=self.state.near_tp_protected,
            near_tp_add_disabled=self.state.near_tp_add_disabled,
            partial_tp_consumed=self.state.partial_tp_consumed,
            middle_runner_active=self.state.middle_runner_active,
            three_stage_runner_enabled_for_position=self.state.three_stage_runner_enabled_for_position,
            tp_plan=self.state.tp_plan,
            three_stage_tp1_consumed=self.state.three_stage_tp1_consumed,
            three_stage_tp2_consumed=self.state.three_stage_tp2_consumed,
        )

    def _tp_plan_unchanged(self, tp_price: float, partial_tp_price: float | None, partial_tp_ratio: float,
                           tp_plan: TpPlan) -> bool:
        return tp_plan_selector.tp_plan_unchanged(
            current_tp_price=self.state.tp_price,
            current_tp_plan=self.state.tp_plan,
            current_partial_tp_price=self.state.partial_tp_price,
            current_partial_tp_ratio=self.state.partial_tp_ratio,
            new_tp_price=tp_price,
            new_partial_tp_price=partial_tp_price,
            new_partial_tp_ratio=partial_tp_ratio,
            new_tp_plan=tp_plan,
        ).unchanged

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
            if leg.get("status") in {"OPEN", "OPEN_UNPROTECTED"} and leg.get("tp_order_id"):
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
