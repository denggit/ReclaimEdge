from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.position_management.cost_basis import calculate_remaining_breakeven_price
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer
from src.strategies import add_layer_gates
from src.strategies import middle_runner as middle_runner_helpers
from src.strategies import three_stage_runner as three_stage_helpers
from src.strategies import tp_plan_selector
from src.strategies import trend_runner as trend_runner_helpers
from src.strategies.tp_lifecycle import is_pre_tp1_lifecycle
from src.utils.log import get_logger

logger = get_logger(__name__)

TradeIntentType = Literal[
    "OPEN_LONG",
    "ADD_LONG",
    "OPEN_SHORT",
    "ADD_SHORT",
    "UPDATE_TP",
    "MARKET_EXIT_RUNNER",
]
PositionSide = Literal["LONG", "SHORT"]
TpMode = Literal["MIDDLE", "UPPER", "LOWER"]
TpPlan = Literal["SINGLE", "MIDDLE_RUNNER", "THREE_STAGE_RUNNER"]


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
    add_gap_mode: str = "linear"
    add_gap_base_pct: float = 0.003
    add_gap_step_pct: float = 0.001
    add_min_avg_improvement_pct: float = 0.0012
    order_cooldown_seconds: int = 10
    first_add_block_seconds: int = 1800
    add_min_interval_seconds: int = 600
    add_freeze_chain_enabled: bool = True
    add_min_interval_bypass_multiplier: float = 2.0
    tp_update_interval_seconds: int = 900
    max_entry_distance_from_extreme_pct: float = 0.002
    max_armed_seconds: int = 900
    breakeven_fee_buffer_pct: float = 0.001
    tp_min_net_profit_pct: float = 0.004
    min_outside_pct: float = 0.001
    entry_reclaim_inside_band: bool = True
    entry_reclaim_buffer_pct: float = 0.0
    entry_sl_buffer_pct: float = 0.0005
    entry_min_reward_risk: float = 1.0
    entry_fee_slippage_buffer_pct: float = 0.001
    entry_max_stop_distance_pct: float = 0.012
    entry_protective_sl_retry_count: int = 3
    entry_protective_sl_retry_interval_seconds: float = 1.0
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

    # ── Middle Bucket Split ────────────────────────────────────────────
    middle_bucket_split_enabled: bool = False
    middle_bucket_split_fast_ratio: float = 0.70
    middle_bucket_split_fast_sl_fee_buffer_pct: float = 0.001
    middle_bucket_split_fast_sl_enabled: bool = True
    middle_bucket_split_fast_sl_invalid_action: str = "MARKET_EXIT"
    middle_bucket_split_fast_sl_fail_action: str = "MARKET_EXIT"

    # ── Three-Stage TP2 structure BOLL ─────────────────────────────────
    three_stage_tp2_use_structure_boll: bool = True

    # ── Entry RR Target ───────────────────────────────────────────────
    # Which price to use for entry reward/risk filtering.
    # STRUCTURE_MIDDLE — BOLL20 middle (default, conservative).
    # FINAL_TP         — the actual selected take-profit price.
    entry_rr_target: str = "STRUCTURE_MIDDLE"

    # ── CVD Structure Entry ──────────────────────────────────────────
    entry_cvd_structure_mode: str = "DIVERGENCE_OR_ABSORPTION"
    entry_cvd_divergence_enabled: bool = True
    entry_cvd_absorption_enabled: bool = True
    entry_cvd_structure_min_outside_pct: float = 0.001

    # ── Reclaim Soft Confirm ─────────────────────────────────────────
    entry_reclaim_confirm_seconds: float = 1.0
    entry_reclaim_outside_tolerance_pct: float = 0.0002
    entry_reclaim_new_extreme_buffer_pct: float = 0.0001

    # ── Setup Lifetime ───────────────────────────────────────────────
    entry_max_extreme_to_reclaim_seconds: int = 900
    entry_max_total_setup_seconds: int = 1800
    entry_max_reclaim_cycles: int = 3

    # ── Post-Entry SL Cooldown ────────────────────────────────────────
    post_entry_sl_cooldown_enabled: bool = True
    post_entry_sl_cooldown_seconds: int = 1800
    post_entry_sl_cooldown_scope: str = "GLOBAL"

    # ── Extreme Retest Add ────────────────────────────────────────────
    extreme_retest_add_enabled: bool = False
    extreme_retest_pivot_left_bars: int = 2
    extreme_retest_pivot_right_bars: int = 2
    extreme_retest_anchor_max_age_candles: int = 12
    extreme_retest_sweep_max_age_seconds: float = 900.0
    extreme_retest_near_extreme_pct: float = 0.0015
    extreme_retest_reclaim_pct: float = 0.0005
    extreme_retest_min_reverse_ratio: float = 0.55
    extreme_retest_one_add_per_anchor: bool = True

    def __post_init__(self) -> None:
        mode = str(self.add_gap_mode or "linear").strip().lower()
        if mode != "linear":
            raise RuntimeError(f"ADD_GAP_MODE={self.add_gap_mode!r} is not supported; currently supports only 'linear'")
        if self.add_gap_base_pct <= 0:
            raise RuntimeError(f"ADD_GAP_BASE_PCT={self.add_gap_base_pct} must be > 0")
        if self.add_gap_step_pct < 0:
            raise RuntimeError(f"ADD_GAP_STEP_PCT={self.add_gap_step_pct} must be >= 0")
        if (
                self.three_stage_pre_tp1_degrade_enabled
                and self.three_stage_pre_tp1_single_after_seconds <= self.three_stage_pre_tp1_middle_runner_after_seconds
        ):
            raise RuntimeError(
                "THREE_STAGE_PRE_TP1_SINGLE_AFTER_SECONDS must be greater than "
                "THREE_STAGE_PRE_TP1_MIDDLE_RUNNER_AFTER_SECONDS"
            )
        if self.entry_rr_target not in {"STRUCTURE_MIDDLE", "FINAL_TP"}:
            raise RuntimeError(
                f"ENTRY_RR_TARGET={self.entry_rr_target!r} is not supported; "
                f"must be STRUCTURE_MIDDLE or FINAL_TP"
            )
        if self.entry_reclaim_confirm_seconds < 0:
            raise RuntimeError(
                f"ENTRY_RECLAIM_CONFIRM_SECONDS={self.entry_reclaim_confirm_seconds} must be >= 0"
            )
        if self.entry_cvd_structure_min_outside_pct < 0:
            raise RuntimeError(
                f"ENTRY_CVD_STRUCTURE_MIN_OUTSIDE_PCT={self.entry_cvd_structure_min_outside_pct} must be >= 0"
            )
        if self.entry_max_extreme_to_reclaim_seconds <= 0:
            raise RuntimeError(
                f"ENTRY_MAX_EXTREME_TO_RECLAIM_SECONDS={self.entry_max_extreme_to_reclaim_seconds} must be > 0"
            )
        if self.entry_max_total_setup_seconds <= 0:
            raise RuntimeError(
                f"ENTRY_MAX_TOTAL_SETUP_SECONDS={self.entry_max_total_setup_seconds} must be > 0"
            )
        if self.entry_max_reclaim_cycles < 0:
            raise RuntimeError(
                f"ENTRY_MAX_RECLAIM_CYCLES={self.entry_max_reclaim_cycles} must be >= 0"
            )
        if self.entry_reclaim_outside_tolerance_pct < 0:
            raise RuntimeError(
                f"ENTRY_RECLAIM_OUTSIDE_TOLERANCE_PCT={self.entry_reclaim_outside_tolerance_pct} must be >= 0"
            )
        if self.entry_reclaim_new_extreme_buffer_pct < 0:
            raise RuntimeError(
                f"ENTRY_RECLAIM_NEW_EXTREME_BUFFER_PCT={self.entry_reclaim_new_extreme_buffer_pct} must be >= 0"
            )
        if self.entry_cvd_structure_mode not in {"DIVERGENCE_ONLY", "ABSORPTION_ONLY", "DIVERGENCE_OR_ABSORPTION"}:
            raise RuntimeError(
                f"ENTRY_CVD_STRUCTURE_MODE={self.entry_cvd_structure_mode!r} "
                f"must be DIVERGENCE_ONLY, ABSORPTION_ONLY, or DIVERGENCE_OR_ABSORPTION"
            )
        if self.post_entry_sl_cooldown_seconds < 0:
            raise RuntimeError(
                f"POST_ENTRY_SL_COOLDOWN_SECONDS={self.post_entry_sl_cooldown_seconds} must be >= 0"
            )
        if self.post_entry_sl_cooldown_scope not in {"GLOBAL", "SIDE"}:
            raise RuntimeError(
                f"POST_ENTRY_SL_COOLDOWN_SCOPE={self.post_entry_sl_cooldown_scope!r} must be GLOBAL or SIDE"
            )
        if self.middle_bucket_split_enabled:
            _fr = self.middle_bucket_split_fast_ratio
            if not (0.05 <= _fr <= 0.95):
                raise RuntimeError(
                    f"MIDDLE_BUCKET_SPLIT_FAST_RATIO={_fr} is out of range [0.05, 0.95]; "
                    f"this is a live position ratio — refusing to proceed with dangerous value"
                )

    @classmethod
    def from_env(cls) -> "BollCvdReclaimStrategyConfig":
        middle_runner_first_close_ratio = min(max(float(os.getenv("MIDDLE_RUNNER_FIRST_CLOSE_RATIO", "0.8")), 0.1),
                                              0.95)
        middle_runner_enabled = _env_bool("MIDDLE_RUNNER_ENABLED", False)
        three_stage_runner_enabled = _env_bool("THREE_STAGE_RUNNER_ENABLED", False)
        return cls(
            min_buy_ratio=float(os.getenv("CVD_MIN_BUY_RATIO", "0.55")),
            min_sell_ratio=float(os.getenv("CVD_MIN_SELL_RATIO", "0.55")),
            add_gap_mode=os.getenv("ADD_GAP_MODE", "linear"),
            add_gap_base_pct=float(os.getenv("ADD_GAP_BASE_PCT", "0.003")),
            add_gap_step_pct=float(os.getenv("ADD_GAP_STEP_PCT", "0.001")),
            add_min_avg_improvement_pct=float(os.getenv("ADD_MIN_AVG_IMPROVEMENT_PCT", "0.0012")),
            order_cooldown_seconds=int(os.getenv("ORDER_COOLDOWN_SECONDS", "10")),
            first_add_block_seconds=int(os.getenv("FIRST_ADD_BLOCK_SECONDS", "1800")),
            add_min_interval_seconds=int(os.getenv("ADD_MIN_INTERVAL_SECONDS", "600")),
            add_freeze_chain_enabled=_env_bool("ADD_FREEZE_CHAIN_ENABLED", True),
            add_min_interval_bypass_multiplier=float(os.getenv("ADD_MIN_INTERVAL_BYPASS_MULTIPLIER", "2")),
            tp_update_interval_seconds=int(os.getenv("TP_UPDATE_INTERVAL_SECONDS", "900")),
            max_entry_distance_from_extreme_pct=float(os.getenv("MAX_ENTRY_DISTANCE_FROM_EXTREME_PCT", "0.002")),
            max_armed_seconds=int(os.getenv("MAX_ARMED_SECONDS", "900")),
            breakeven_fee_buffer_pct=float(os.getenv("BREAKEVEN_FEE_BUFFER_PCT", "0.001")),
            tp_min_net_profit_pct=float(os.getenv("TP_MIN_NET_PROFIT_PCT", "0.004")),
            min_outside_pct=float(os.getenv("BOLL_MIN_OUTSIDE_PCT", "0.001")),
            entry_reclaim_inside_band=_env_bool("ENTRY_RECLAIM_INSIDE_BAND", True),
            entry_reclaim_buffer_pct=float(os.getenv("ENTRY_RECLAIM_BUFFER_PCT", "0")),
            entry_sl_buffer_pct=float(os.getenv("ENTRY_SL_BUFFER_PCT", "0.0005")),
            entry_min_reward_risk=float(os.getenv("ENTRY_MIN_REWARD_RISK", "1.0")),
            entry_fee_slippage_buffer_pct=float(os.getenv("ENTRY_FEE_SLIPPAGE_BUFFER_PCT", "0.001")),
            entry_max_stop_distance_pct=float(os.getenv("ENTRY_MAX_STOP_DISTANCE_PCT", "0.012")),
            entry_protective_sl_retry_count=int(os.getenv("ENTRY_PROTECTIVE_SL_RETRY_COUNT", "3")),
            entry_protective_sl_retry_interval_seconds=float(os.getenv("ENTRY_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
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
            middle_bucket_split_enabled=_env_bool("MIDDLE_BUCKET_SPLIT_ENABLED", False),
            middle_bucket_split_fast_ratio=float(os.getenv("MIDDLE_BUCKET_SPLIT_FAST_RATIO", "0.70")),
            middle_bucket_split_fast_sl_fee_buffer_pct=float(
                os.getenv("MIDDLE_BUCKET_SPLIT_FAST_SL_FEE_BUFFER_PCT", "0.001")),
            middle_bucket_split_fast_sl_enabled=_env_bool("MIDDLE_BUCKET_SPLIT_FAST_SL_ENABLED", True),
            middle_bucket_split_fast_sl_invalid_action=os.getenv(
                "MIDDLE_BUCKET_SPLIT_FAST_SL_INVALID_ACTION", "MARKET_EXIT").strip().upper(),
            middle_bucket_split_fast_sl_fail_action=os.getenv(
                "MIDDLE_BUCKET_FAST_SL_FAIL_ACTION", "MARKET_EXIT").strip().upper(),
            three_stage_tp2_use_structure_boll=_env_bool("THREE_STAGE_TP2_USE_STRUCTURE_BOLL", True),
            # ── Entry RR Target ──────────────────────────────────────
            entry_rr_target=os.getenv("ENTRY_RR_TARGET", "STRUCTURE_MIDDLE").strip().upper(),
            # ── Extreme Retest Add ────────────────────────────────────
            extreme_retest_add_enabled=_env_bool("EXTREME_RETEST_ADD_ENABLED", False),
            extreme_retest_pivot_left_bars=int(os.getenv("EXTREME_RETEST_PIVOT_LEFT_BARS", "2")),
            extreme_retest_pivot_right_bars=int(os.getenv("EXTREME_RETEST_PIVOT_RIGHT_BARS", "2")),
            extreme_retest_anchor_max_age_candles=int(os.getenv("EXTREME_RETEST_ANCHOR_MAX_AGE_CANDLES", "12")),
            extreme_retest_sweep_max_age_seconds=float(os.getenv("EXTREME_RETEST_SWEEP_MAX_AGE_SECONDS", "900")),
            extreme_retest_near_extreme_pct=float(os.getenv("EXTREME_RETEST_NEAR_EXTREME_PCT", "0.0015")),
            extreme_retest_reclaim_pct=float(os.getenv("EXTREME_RETEST_RECLAIM_PCT", "0.0005")),
            extreme_retest_min_reverse_ratio=float(os.getenv("EXTREME_RETEST_MIN_REVERSE_RATIO", "0.55")),
            extreme_retest_one_add_per_anchor=_env_bool("EXTREME_RETEST_ONE_ADD_PER_ANCHOR", True),
            # ── CVD Structure Entry ──────────────────────────────────
            entry_cvd_structure_mode=os.getenv("ENTRY_CVD_STRUCTURE_MODE", "DIVERGENCE_OR_ABSORPTION").strip().upper(),
            entry_cvd_divergence_enabled=_env_bool("ENTRY_CVD_DIVERGENCE_ENABLED", True),
            entry_cvd_absorption_enabled=_env_bool("ENTRY_CVD_ABSORPTION_ENABLED", True),
            entry_cvd_structure_min_outside_pct=float(os.getenv("ENTRY_CVD_STRUCTURE_MIN_OUTSIDE_PCT", "0.001")),
            # ── Reclaim Soft Confirm ─────────────────────────────────
            entry_reclaim_confirm_seconds=float(os.getenv("ENTRY_RECLAIM_CONFIRM_SECONDS", "1.0")),
            entry_reclaim_outside_tolerance_pct=float(os.getenv("ENTRY_RECLAIM_OUTSIDE_TOLERANCE_PCT", "0.0002")),
            entry_reclaim_new_extreme_buffer_pct=float(os.getenv("ENTRY_RECLAIM_NEW_EXTREME_BUFFER_PCT", "0.0001")),
            # ── Setup Lifetime ───────────────────────────────────────
            entry_max_extreme_to_reclaim_seconds=int(os.getenv("ENTRY_MAX_EXTREME_TO_RECLAIM_SECONDS", "900")),
            entry_max_total_setup_seconds=int(os.getenv("ENTRY_MAX_TOTAL_SETUP_SECONDS", "1800")),
            entry_max_reclaim_cycles=int(os.getenv("ENTRY_MAX_RECLAIM_CYCLES", "3")),
            # ── Post-Entry SL Cooldown ────────────────────────────────
            post_entry_sl_cooldown_enabled=_env_bool("POST_ENTRY_SL_COOLDOWN_ENABLED", True),
            post_entry_sl_cooldown_seconds=int(os.getenv("POST_ENTRY_SL_COOLDOWN_SECONDS", "1800")),
            post_entry_sl_cooldown_scope=os.getenv("POST_ENTRY_SL_COOLDOWN_SCOPE", "GLOBAL").strip().upper(),
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
    entry_protective_sl_price: float | None = None
    entry_protective_sl_order_id: str | None = None
    entry_protective_sl_protected: bool = False
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

    # ── Middle Bucket Split fields ────────────────────────────────────
    middle_bucket_split_active: bool = False
    middle_bucket_split_fast_consumed: bool = False
    middle_bucket_split_slow_consumed: bool = False

    middle_bucket_split_fast_price: float | None = None
    middle_bucket_split_slow_price: float | None = None
    middle_bucket_split_effective_price: float | None = None

    middle_bucket_split_middle_bucket_ratio: float = 0.0
    middle_bucket_split_fast_ratio_of_bucket: float = 0.0
    middle_bucket_split_slow_ratio_of_bucket: float = 0.0
    middle_bucket_split_fast_total_ratio: float = 0.0
    middle_bucket_split_slow_total_ratio: float = 0.0

    middle_bucket_split_reason: str | None = None
    middle_bucket_split_fast_sl_price: float | None = None
    middle_bucket_split_fast_sl_order_id: str | None = None
    middle_bucket_split_fast_sl_protected: bool = False


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

    # ── CVD Structure Entry state ───────────────────────────────────
    lower_first_armed_ts_ms: int = 0
    upper_first_armed_ts_ms: int = 0

    lower_extreme_ts_ms: int = 0
    upper_extreme_ts_ms: int = 0

    lower_reference_fast_cvd: float | None = None
    upper_reference_fast_cvd: float | None = None

    lower_extreme_fast_cvd: float | None = None
    upper_extreme_fast_cvd: float | None = None

    lower_cvd_divergence_confirmed: bool = False
    upper_cvd_divergence_confirmed: bool = False

    lower_cvd_absorption_confirmed: bool = False
    upper_cvd_absorption_confirmed: bool = False

    lower_reclaim_seen: bool = False
    upper_reclaim_seen: bool = False

    lower_reclaim_ts_ms: int = 0
    upper_reclaim_ts_ms: int = 0

    lower_reclaim_cycle_count: int = 0
    upper_reclaim_cycle_count: int = 0

    # ── Post-Entry SL Cooldown state ────────────────────────────────
    post_entry_sl_cooldown_until_ts_ms: int = 0
    post_entry_sl_cooldown_side: str | None = None
    post_entry_sl_cooldown_reason: str | None = None

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
    entry_protective_sl_price: float | None = None
    entry_protective_sl_order_id: str | None = None
    entry_protective_sl_protected: bool = False
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
    last_add_skip_log_reason: str | None = None
    last_add_skip_log_ts_ms: int = 0
    core_contracts: str | None = None
    core_eth_qty: float = 0.0
    tp_order_id: str | None = None
    tp_order_ids: list[str] = field(default_factory=list)
    startup_force_tp_reconcile: bool = False

    # ── Middle Bucket Split state ─────────────────────────────────────
    middle_bucket_split_active: bool = False
    middle_bucket_split_fast_consumed: bool = False
    middle_bucket_split_slow_consumed: bool = False

    middle_bucket_split_fast_price: float | None = None
    middle_bucket_split_slow_price: float | None = None
    middle_bucket_split_effective_price: float | None = None

    middle_bucket_split_middle_bucket_ratio: float = 0.0
    middle_bucket_split_fast_ratio_of_bucket: float = 0.0
    middle_bucket_split_slow_ratio_of_bucket: float = 0.0
    middle_bucket_split_fast_total_ratio: float = 0.0
    middle_bucket_split_slow_total_ratio: float = 0.0

    middle_bucket_split_reason: str | None = None

    middle_bucket_split_fast_sl_price: float | None = None
    middle_bucket_split_fast_sl_order_id: str | None = None
    middle_bucket_split_fast_sl_protected: bool = False
    middle_bucket_split_fast_sl_invalid_action_taken: str | None = None

    middle_bucket_split_add_disabled: bool = False

    # ── Delayed market exit state (persisted, survives restart) ────────
    delayed_market_exit_armed: bool = False
    delayed_market_exit_reason: str | None = None
    delayed_market_exit_context: str | None = None
    delayed_market_exit_side: str | None = None
    delayed_market_exit_position_id: str | None = None
    delayed_market_exit_source_event: str | None = None
    delayed_market_exit_armed_ts_ms: int | None = None
    delayed_market_exit_deadline_ts_ms: int | None = None
    delayed_market_exit_manual_intervention_required: bool = False
    delayed_market_exit_last_error: str | None = None
    # ── Idempotency fields ────────────────────────────────────────────
    delayed_market_exit_status: str | None = None
    # None | "ARMED" | "WAITING_FLAT" | "FAILED" | "CLEARED"
    delayed_market_exit_executed_ts_ms: int | None = None
    delayed_market_exit_exit_attempt_count: int = 0
    delayed_market_exit_last_exit_message: str | None = None

    # ── Extreme Retest Add state ──────────────────────────────────────
    extreme_retest_anchor_side: Optional[str] = None
    extreme_retest_anchor_kind: Optional[str] = None
    extreme_retest_anchor_price: Optional[float] = None
    extreme_retest_anchor_candle_ts_ms: Optional[int] = None
    extreme_retest_anchor_boll_upper: Optional[float] = None
    extreme_retest_anchor_boll_lower: Optional[float] = None

    extreme_retest_sweep_seen: bool = False
    extreme_retest_sweep_extreme_price: Optional[float] = None
    extreme_retest_sweep_first_seen_ts_ms: Optional[int] = None
    extreme_retest_sweep_last_seen_ts_ms: Optional[int] = None

    extreme_retest_consumed_watermark_price: Optional[float] = None
    extreme_retest_consumed_anchor_ts_ms: Optional[int] = None


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

        self._update_armed_state(price, ts_ms, boll, cvd)

        runner_exit_intent = self._maybe_trend_runner_market_exit(price, ts_ms, boll, cvd)
        if runner_exit_intent is not None:
            return [runner_exit_intent]

        # TP maintenance is driven by BOLL candle timestamp. This avoids the old
        # problem where a restart/manual TP update delayed the next 15m update.
        tp_intent = self._maybe_update_tp(price, ts_ms, boll, cvd)
        if tp_intent is not None:
            intents.append(tp_intent)

        if not boll.alert_switch_on:
            return intents

        if not self._cooldown_ok(ts_ms):
            return intents

        if not self._post_entry_sl_cooldown_ok("LONG", ts_ms):
            return intents
        if not self._post_entry_sl_cooldown_ok("SHORT", ts_ms):
            return intents

        if self._long_setup(price, cvd, boll):
            intent = self._maybe_open_or_add_long(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        if self._short_setup(price, cvd, boll):
            intent = self._maybe_open_or_add_short(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        return intents

    def _update_armed_state(self, price: float, ts_ms: int, boll: BollSnapshot,
                            cvd: CvdSnapshot | None = None) -> None:
        self._expire_armed_state(ts_ms)

        # ── Total setup timeout check ──────────────────────────────────
        _total_ms = self.config.entry_max_total_setup_seconds * 1000
        if self.state.lower_first_armed_ts_ms > 0 and ts_ms - self.state.lower_first_armed_ts_ms > _total_ms:
            logger.info("LOWER_SETUP_EXPIRED | reason=total_setup_timeout age_ms=%s max_ms=%s",
                        ts_ms - self.state.lower_first_armed_ts_ms, _total_ms)
            self._reset_lower_armed()
        if self.state.upper_first_armed_ts_ms > 0 and ts_ms - self.state.upper_first_armed_ts_ms > _total_ms:
            logger.info("UPPER_SETUP_EXPIRED | reason=total_setup_timeout age_ms=%s max_ms=%s",
                        ts_ms - self.state.upper_first_armed_ts_ms, _total_ms)
            self._reset_upper_armed()

        if price < boll.lower:
            self._update_lower_outside(price, ts_ms, boll, cvd)
            if self.state.upper_armed:
                logger.info("UPPER_ARMED_RESET | reason=opposite_lower_break price=%.4f", price)
            self._reset_upper_armed()
            return

        if price > boll.upper:
            self._update_upper_outside(price, ts_ms, boll, cvd)
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

    def _update_lower_outside(self, price: float, ts_ms: int, boll: BollSnapshot,
                               cvd: CvdSnapshot | None) -> None:
        """Handle a tick where price is below the lower BOLL band."""
        if not self.state.lower_armed:
            # ── First arm ──────────────────────────────────────────────
            self.state.lower_armed = True
            self.state.lower_armed_ts_ms = ts_ms
            self.state.lower_first_armed_ts_ms = ts_ms
            self.state.lower_extreme_price = price
            if cvd is not None:
                self.state.lower_reference_fast_cvd = cvd.fast_cvd
            _fast_cvd_str = f" fast_cvd={cvd.fast_cvd:.8f}" if cvd is not None else ""
            logger.info(
                "LOWER_ARMED | price=%.4f lower=%.4f middle=%.4f max_entry_distance=%.4f%% max_armed=%ss%s",
                price, boll.lower, boll.middle,
                self.config.max_entry_distance_from_extreme_pct * 100,
                self.config.max_armed_seconds, _fast_cvd_str,
            )
        elif self.state.lower_reclaim_seen:
            # ── Previously reclaimed, now outside again ────────────────
            self._handle_lower_rebreak_after_reclaim(price, ts_ms, boll, cvd)
        else:
            # ── Normal extreme update during outside excursion ─────────
            self._update_lower_extreme(price, ts_ms, boll, cvd)

        self._update_lower_deep_enough(boll)
        if cvd is not None:
            self._check_lower_cvd_structure(cvd, boll, ts_ms)

    def _update_upper_outside(self, price: float, ts_ms: int, boll: BollSnapshot,
                               cvd: CvdSnapshot | None) -> None:
        """Handle a tick where price is above the upper BOLL band."""
        if not self.state.upper_armed:
            # ── First arm ──────────────────────────────────────────────
            self.state.upper_armed = True
            self.state.upper_armed_ts_ms = ts_ms
            self.state.upper_first_armed_ts_ms = ts_ms
            self.state.upper_extreme_price = price
            if cvd is not None:
                self.state.upper_reference_fast_cvd = cvd.fast_cvd
            _fast_cvd_str = f" fast_cvd={cvd.fast_cvd:.8f}" if cvd is not None else ""
            logger.info(
                "UPPER_ARMED | price=%.4f upper=%.4f middle=%.4f max_entry_distance=%.4f%% max_armed=%ss%s",
                price, boll.upper, boll.middle,
                self.config.max_entry_distance_from_extreme_pct * 100,
                self.config.max_armed_seconds, _fast_cvd_str,
            )
        elif self.state.upper_reclaim_seen:
            # ── Previously reclaimed, now outside again ────────────────
            self._handle_upper_rebreak_after_reclaim(price, ts_ms, boll, cvd)
        else:
            # ── Normal extreme update during outside excursion ─────────
            self._update_upper_extreme(price, ts_ms, boll, cvd)

        self._update_upper_deep_enough(boll)
        if cvd is not None:
            self._check_upper_cvd_structure(cvd, boll, ts_ms)

    def _update_lower_extreme(self, price: float, ts_ms: int, boll: BollSnapshot,
                               cvd: CvdSnapshot | None) -> None:
        """Update lower extreme during ongoing outside excursion."""
        old_extreme = self.state.lower_extreme_price
        if old_extreme is None or price >= old_extreme:
            return
        buffer_pct = self.config.entry_reclaim_new_extreme_buffer_pct
        if price >= old_extreme * (1 - buffer_pct):
            return  # within noise buffer, not a real new extreme
        # Real new extreme — update price and timestamp
        # (extreme_fast_cvd is managed by _check_lower_cvd_structure)
        self.state.lower_extreme_price = price
        self.state.lower_extreme_ts_ms = ts_ms
        logger.debug("LOWER_EXTREME_UPDATED | extreme=%.4f price=%.4f", price, price)

    def _update_upper_extreme(self, price: float, ts_ms: int, boll: BollSnapshot,
                               cvd: CvdSnapshot | None) -> None:
        """Update upper extreme during ongoing outside excursion."""
        old_extreme = self.state.upper_extreme_price
        if old_extreme is None or price <= old_extreme:
            return
        buffer_pct = self.config.entry_reclaim_new_extreme_buffer_pct
        if price <= old_extreme * (1 + buffer_pct):
            return  # within noise buffer, not a real new extreme
        # Real new extreme — update price and timestamp
        # (extreme_fast_cvd is managed by _check_upper_cvd_structure)
        self.state.upper_extreme_price = price
        self.state.upper_extreme_ts_ms = ts_ms
        logger.debug("UPPER_EXTREME_UPDATED | extreme=%.4f price=%.4f", price, price)

    def _handle_lower_rebreak_after_reclaim(self, price: float, ts_ms: int, boll: BollSnapshot,
                                             cvd: CvdSnapshot | None) -> None:
        """Handle price going back below lower band after reclaim was seen."""
        old_extreme = self.state.lower_extreme_price or price
        buffer_pct = self.config.entry_reclaim_new_extreme_buffer_pct
        new_extreme_threshold = old_extreme * (1 - buffer_pct)
        if price < new_extreme_threshold:
            # Breaks old extreme → cancel pending, new extreme, increment cycle
            self.state.lower_reclaim_cycle_count += 1
            if self.state.lower_reclaim_cycle_count > self.config.entry_max_reclaim_cycles:
                logger.info("LOWER_SETUP_EXPIRED | reason=max_reclaim_cycles cycles=%s max=%s",
                            self.state.lower_reclaim_cycle_count, self.config.entry_max_reclaim_cycles)
                self._reset_lower_armed()
                return
            self.state.lower_reclaim_seen = False
            self.state.lower_reclaim_ts_ms = 0
            self.state.lower_extreme_price = price
            self.state.lower_extreme_ts_ms = ts_ms
            logger.info(
                "LOWER_RECLAIM_PENDING_CANCELLED_BY_NEW_EXTREME | old_extreme=%.4f new_extreme=%.4f "
                "cycle_count=%s max_cycles=%s ts_ms=%s",
                old_extreme, price, self.state.lower_reclaim_cycle_count,
                self.config.entry_max_reclaim_cycles, ts_ms,
            )
        else:
            # Minor breach → soft reset timer only
            self.state.lower_reclaim_ts_ms = 0
            logger.info(
                "LOWER_RECLAIM_CONFIRM_RESET | reason=minor_outside_noise price=%.4f extreme=%.4f "
                "lower=%.4f ts_ms=%s",
                price, old_extreme, boll.lower, ts_ms,
            )

    def _handle_upper_rebreak_after_reclaim(self, price: float, ts_ms: int, boll: BollSnapshot,
                                             cvd: CvdSnapshot | None) -> None:
        """Handle price going back above upper band after reclaim was seen."""
        old_extreme = self.state.upper_extreme_price or price
        buffer_pct = self.config.entry_reclaim_new_extreme_buffer_pct
        new_extreme_threshold = old_extreme * (1 + buffer_pct)
        if price > new_extreme_threshold:
            # Breaks old extreme → cancel pending, new extreme, increment cycle
            self.state.upper_reclaim_cycle_count += 1
            if self.state.upper_reclaim_cycle_count > self.config.entry_max_reclaim_cycles:
                logger.info("UPPER_SETUP_EXPIRED | reason=max_reclaim_cycles cycles=%s max=%s",
                            self.state.upper_reclaim_cycle_count, self.config.entry_max_reclaim_cycles)
                self._reset_upper_armed()
                return
            self.state.upper_reclaim_seen = False
            self.state.upper_reclaim_ts_ms = 0
            self.state.upper_extreme_price = price
            self.state.upper_extreme_ts_ms = ts_ms
            logger.info(
                "UPPER_RECLAIM_PENDING_CANCELLED_BY_NEW_EXTREME | old_extreme=%.4f new_extreme=%.4f "
                "cycle_count=%s max_cycles=%s ts_ms=%s",
                old_extreme, price, self.state.upper_reclaim_cycle_count,
                self.config.entry_max_reclaim_cycles, ts_ms,
            )
        else:
            # Minor breach → soft reset timer only
            self.state.upper_reclaim_ts_ms = 0
            logger.info(
                "UPPER_RECLAIM_CONFIRM_RESET | reason=minor_outside_noise price=%.4f extreme=%.4f "
                "upper=%.4f ts_ms=%s",
                price, old_extreme, boll.upper, ts_ms,
            )

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

    def _lower_cvd_structure_ok(self) -> bool:
        """Check whether lower-side CVD structure is confirmed per mode."""
        # If neither divergence nor absorption is enabled, skip CVD structure gate
        if not self.config.entry_cvd_divergence_enabled and not self.config.entry_cvd_absorption_enabled:
            return True
        mode = self.config.entry_cvd_structure_mode
        if mode == "DIVERGENCE_ONLY":
            return self.state.lower_cvd_divergence_confirmed
        if mode == "ABSORPTION_ONLY":
            return self.state.lower_cvd_absorption_confirmed
        # DIVERGENCE_OR_ABSORPTION
        return self.state.lower_cvd_divergence_confirmed or self.state.lower_cvd_absorption_confirmed

    def _upper_cvd_structure_ok(self) -> bool:
        """Check whether upper-side CVD structure is confirmed per mode."""
        # If neither divergence nor absorption is enabled, skip CVD structure gate
        if not self.config.entry_cvd_divergence_enabled and not self.config.entry_cvd_absorption_enabled:
            return True
        mode = self.config.entry_cvd_structure_mode
        if mode == "DIVERGENCE_ONLY":
            return self.state.upper_cvd_divergence_confirmed
        if mode == "ABSORPTION_ONLY":
            return self.state.upper_cvd_absorption_confirmed
        # DIVERGENCE_OR_ABSORPTION
        return self.state.upper_cvd_divergence_confirmed or self.state.upper_cvd_absorption_confirmed

    def _check_lower_cvd_structure(self, cvd: CvdSnapshot, boll: BollSnapshot, ts_ms: int) -> None:
        """Evaluate both divergence and absorption during lower outside excursion."""
        if not self.state.lower_deep_enough:
            return
        if self._lower_cvd_structure_ok():
            return
        extreme = self.state.lower_extreme_price
        if extreme is None or extreme <= 0:
            return

        # ── First time reaching valid extreme depth — record baseline ──
        if self.state.lower_extreme_fast_cvd is None:
            self.state.lower_extreme_fast_cvd = cvd.fast_cvd
            self.state.lower_extreme_ts_ms = ts_ms
            outside_pct = (boll.lower - extreme) / boll.lower * 100
            logger.info(
                "LOWER_VALID_EXTREME | extreme_price=%.4f boll_lower=%.4f outside_pct=%.4f%% fast_cvd=%.8f ts_ms=%s",
                extreme, boll.lower, outside_pct, cvd.fast_cvd, ts_ms,
            )
            # Also check absorption on first extreme
            self._check_lower_absorption(extreme, ts_ms)
            return

        # ── Divergence: compare current fast_cvd vs stored extreme_fast_cvd ──
        if self.config.entry_cvd_divergence_enabled and not self.state.lower_cvd_divergence_confirmed:
            if cvd.fast_cvd >= self.state.lower_extreme_fast_cvd:
                self.state.lower_cvd_divergence_confirmed = True
                logger.info(
                    "LOWER_CVD_DIVERGENCE_CONFIRMED | old_extreme_fast_cvd=%.8f new_fast_cvd=%.8f "
                    "extreme_price=%.4f price=%.4f ts_ms=%s",
                    self.state.lower_extreme_fast_cvd, cvd.fast_cvd, extreme, cvd.price, ts_ms,
                )
            else:
                # CVD making new low — update reference for next comparison
                self.state.lower_extreme_fast_cvd = cvd.fast_cvd

        # ── Absorption: compare extreme_fast_cvd vs reference_fast_cvd ──
        self._check_lower_absorption(extreme, ts_ms)

    def _check_lower_absorption(self, extreme: float, ts_ms: int) -> None:
        """Check lower-side single-sweep absorption."""
        if not self.config.entry_cvd_absorption_enabled or self.state.lower_cvd_absorption_confirmed:
            return
        if self.state.lower_reference_fast_cvd is not None and self.state.lower_extreme_fast_cvd is not None:
            if self.state.lower_extreme_fast_cvd >= self.state.lower_reference_fast_cvd:
                self.state.lower_cvd_absorption_confirmed = True
                logger.info(
                    "LOWER_CVD_ABSORPTION_CONFIRMED | reference_fast_cvd=%.8f extreme_fast_cvd=%.8f "
                    "extreme_price=%.4f ts_ms=%s",
                    self.state.lower_reference_fast_cvd, self.state.lower_extreme_fast_cvd, extreme, ts_ms,
                )

    def _check_upper_absorption(self, extreme: float, ts_ms: int) -> None:
        """Check upper-side single-sweep absorption."""
        if not self.config.entry_cvd_absorption_enabled or self.state.upper_cvd_absorption_confirmed:
            return
        if self.state.upper_reference_fast_cvd is not None and self.state.upper_extreme_fast_cvd is not None:
            if self.state.upper_extreme_fast_cvd <= self.state.upper_reference_fast_cvd:
                self.state.upper_cvd_absorption_confirmed = True
                logger.info(
                    "UPPER_CVD_ABSORPTION_CONFIRMED | reference_fast_cvd=%.8f extreme_fast_cvd=%.8f "
                    "extreme_price=%.4f ts_ms=%s",
                    self.state.upper_reference_fast_cvd, self.state.upper_extreme_fast_cvd, extreme, ts_ms,
                )

    def _check_upper_cvd_structure(self, cvd: CvdSnapshot, boll: BollSnapshot, ts_ms: int) -> None:
        """Evaluate both divergence and absorption during upper outside excursion."""
        if not self.state.upper_deep_enough:
            return
        if self._upper_cvd_structure_ok():
            return
        extreme = self.state.upper_extreme_price
        if extreme is None or extreme <= 0:
            return

        # ── First time reaching valid extreme depth — record baseline ──
        if self.state.upper_extreme_fast_cvd is None:
            self.state.upper_extreme_fast_cvd = cvd.fast_cvd
            self.state.upper_extreme_ts_ms = ts_ms
            outside_pct = (extreme - boll.upper) / boll.upper * 100
            logger.info(
                "UPPER_VALID_EXTREME | extreme_price=%.4f boll_upper=%.4f outside_pct=%.4f%% fast_cvd=%.8f ts_ms=%s",
                extreme, boll.upper, outside_pct, cvd.fast_cvd, ts_ms,
            )
            # Also check absorption on first extreme
            self._check_upper_absorption(extreme, ts_ms)
            return

        # ── Divergence: compare current fast_cvd vs stored extreme_fast_cvd ──
        if self.config.entry_cvd_divergence_enabled and not self.state.upper_cvd_divergence_confirmed:
            if cvd.fast_cvd <= self.state.upper_extreme_fast_cvd:
                self.state.upper_cvd_divergence_confirmed = True
                logger.info(
                    "UPPER_CVD_DIVERGENCE_CONFIRMED | old_extreme_fast_cvd=%.8f new_fast_cvd=%.8f "
                    "extreme_price=%.4f price=%.4f ts_ms=%s",
                    self.state.upper_extreme_fast_cvd, cvd.fast_cvd, extreme, cvd.price, ts_ms,
                )
            else:
                # CVD making new high — update reference for next comparison
                self.state.upper_extreme_fast_cvd = cvd.fast_cvd

        # ── Absorption: compare extreme_fast_cvd vs reference_fast_cvd ──
        self._check_upper_absorption(extreme, ts_ms)

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
        # ── Clear CVD structure / reclaim state (NOT cooldown) ──────────
        self.state.lower_first_armed_ts_ms = 0
        self.state.lower_extreme_ts_ms = 0
        self.state.lower_reference_fast_cvd = None
        self.state.lower_extreme_fast_cvd = None
        self.state.lower_cvd_divergence_confirmed = False
        self.state.lower_cvd_absorption_confirmed = False
        self.state.lower_reclaim_seen = False
        self.state.lower_reclaim_ts_ms = 0
        self.state.lower_reclaim_cycle_count = 0

    def _reset_upper_armed(self) -> None:
        self.state.upper_armed = False
        self.state.upper_extreme_price = None
        self.state.upper_armed_ts_ms = 0
        self.state.upper_last_burst_ts_ms = 0
        self.state.upper_deep_enough = False
        # ── Clear CVD structure / reclaim state (NOT cooldown) ──────────
        self.state.upper_first_armed_ts_ms = 0
        self.state.upper_extreme_ts_ms = 0
        self.state.upper_reference_fast_cvd = None
        self.state.upper_extreme_fast_cvd = None
        self.state.upper_cvd_divergence_confirmed = False
        self.state.upper_cvd_absorption_confirmed = False
        self.state.upper_reclaim_seen = False
        self.state.upper_reclaim_ts_ms = 0
        self.state.upper_reclaim_cycle_count = 0

    def _long_setup(self, price: float, cvd: CvdSnapshot, boll: BollSnapshot) -> bool:
        if not self.state.lower_armed or self.state.lower_extreme_price is None:
            return False
        if not self.state.lower_deep_enough:
            return False

        # ── CVD structure gate ──────────────────────────────────────────
        if not self._lower_cvd_structure_ok():
            return False

        # ── Reclaim soft confirm state machine ──────────────────────────
        if self.config.entry_reclaim_confirm_seconds > 0:
            tolerance = self.config.entry_reclaim_outside_tolerance_pct

            # Check if price went back outside during confirmation
            if self.state.lower_reclaim_seen and self.state.lower_reclaim_ts_ms > 0:
                if price < boll.lower * (1 - tolerance):
                    # Outside band beyond tolerance → soft reset timer
                    # (the _update_armed_state handles new extreme vs minor breach)
                    # Here we just reset reclaim_ts_ms so the timer restarts
                    self.state.lower_reclaim_ts_ms = 0
                    return False

            # Timer was reset → wait for price to come back inside, then restart
            if self.state.lower_reclaim_seen and self.state.lower_reclaim_ts_ms == 0:
                if price >= boll.lower:
                    self.state.lower_reclaim_ts_ms = cvd.ts_ms
                    logger.info(
                        "LOWER_RECLAIM_CONFIRM_RESET | reason=timer_restarted reclaim_ts_ms=%s price=%.4f lower=%.4f",
                        cvd.ts_ms, price, boll.lower,
                    )
                return False

            if not self.state.lower_reclaim_seen:
                # First tick back inside band
                inside_band = price >= boll.lower * (1 + self.config.entry_reclaim_buffer_pct)
                if not inside_band:
                    return False

                # Check extreme-to-reclaim time window
                if self.state.lower_extreme_ts_ms > 0:
                    elapsed_ms = cvd.ts_ms - self.state.lower_extreme_ts_ms
                    if elapsed_ms > self.config.entry_max_extreme_to_reclaim_seconds * 1000:
                        logger.info(
                            "LOWER_SETUP_EXPIRED | reason=extreme_to_reclaim_timeout "
                            "elapsed_ms=%s max_ms=%s",
                            elapsed_ms,
                            self.config.entry_max_extreme_to_reclaim_seconds * 1000,
                        )
                        self._reset_lower_armed()
                        return False

                self.state.lower_reclaim_seen = True
                self.state.lower_reclaim_ts_ms = cvd.ts_ms
                logger.info(
                    "LOWER_RECLAIM_PENDING | reclaim_ts_ms=%s confirm_seconds=%.1f price=%.4f lower=%.4f",
                    cvd.ts_ms, self.config.entry_reclaim_confirm_seconds, price, boll.lower,
                )
                return False

            # Check if enough continuous time has elapsed
            confirm_ms = int(self.config.entry_reclaim_confirm_seconds * 1000)
            if cvd.ts_ms - self.state.lower_reclaim_ts_ms < confirm_ms:
                return False

            logger.info(
                "LOWER_RECLAIM_CONFIRMED | reclaim_ts_ms=%s ts_ms=%s elapsed_ms=%s",
                self.state.lower_reclaim_ts_ms, cvd.ts_ms, cvd.ts_ms - self.state.lower_reclaim_ts_ms,
            )

        # ── Inside-band reclaim check ───────────────────────────────────
        if self.config.entry_reclaim_inside_band and price < boll.lower * (1 + self.config.entry_reclaim_buffer_pct):
            return False

        # ── CVD direction check at entry ────────────────────────────────
        cvd_direction_ok = (
            (cvd.cross_positive or cvd.cvd_increasing)
            and cvd.buy_ratio >= self.config.min_buy_ratio
            and cvd.no_new_low
        )
        return cvd_direction_ok

    def _short_setup(self, price: float, cvd: CvdSnapshot, boll: BollSnapshot) -> bool:
        if not self.state.upper_armed or self.state.upper_extreme_price is None:
            return False
        if not self.state.upper_deep_enough:
            return False

        # ── CVD structure gate ──────────────────────────────────────────
        if not self._upper_cvd_structure_ok():
            return False

        # ── Reclaim soft confirm state machine ──────────────────────────
        if self.config.entry_reclaim_confirm_seconds > 0:
            tolerance = self.config.entry_reclaim_outside_tolerance_pct

            # Check if price went back outside during confirmation
            if self.state.upper_reclaim_seen and self.state.upper_reclaim_ts_ms > 0:
                if price > boll.upper * (1 + tolerance):
                    # Outside band beyond tolerance → soft reset timer
                    self.state.upper_reclaim_ts_ms = 0
                    return False

            # Timer was reset → wait for price to come back inside, then restart
            if self.state.upper_reclaim_seen and self.state.upper_reclaim_ts_ms == 0:
                if price <= boll.upper:
                    self.state.upper_reclaim_ts_ms = cvd.ts_ms
                    logger.info(
                        "UPPER_RECLAIM_CONFIRM_RESET | reason=timer_restarted reclaim_ts_ms=%s price=%.4f upper=%.4f",
                        cvd.ts_ms, price, boll.upper,
                    )
                return False

            if not self.state.upper_reclaim_seen:
                # First tick back inside band
                inside_band = price <= boll.upper * (1 - self.config.entry_reclaim_buffer_pct)
                if not inside_band:
                    return False

                # Check extreme-to-reclaim time window
                if self.state.upper_extreme_ts_ms > 0:
                    elapsed_ms = cvd.ts_ms - self.state.upper_extreme_ts_ms
                    if elapsed_ms > self.config.entry_max_extreme_to_reclaim_seconds * 1000:
                        logger.info(
                            "UPPER_SETUP_EXPIRED | reason=extreme_to_reclaim_timeout "
                            "elapsed_ms=%s max_ms=%s",
                            elapsed_ms,
                            self.config.entry_max_extreme_to_reclaim_seconds * 1000,
                        )
                        self._reset_upper_armed()
                        return False

                self.state.upper_reclaim_seen = True
                self.state.upper_reclaim_ts_ms = cvd.ts_ms
                logger.info(
                    "UPPER_RECLAIM_PENDING | reclaim_ts_ms=%s confirm_seconds=%.1f price=%.4f upper=%.4f",
                    cvd.ts_ms, self.config.entry_reclaim_confirm_seconds, price, boll.upper,
                )
                return False

            # Check if enough continuous time has elapsed
            confirm_ms = int(self.config.entry_reclaim_confirm_seconds * 1000)
            if cvd.ts_ms - self.state.upper_reclaim_ts_ms < confirm_ms:
                return False

            logger.info(
                "UPPER_RECLAIM_CONFIRMED | reclaim_ts_ms=%s ts_ms=%s elapsed_ms=%s",
                self.state.upper_reclaim_ts_ms, cvd.ts_ms, cvd.ts_ms - self.state.upper_reclaim_ts_ms,
            )

        # ── Inside-band reclaim check ───────────────────────────────────
        if self.config.entry_reclaim_inside_band and price > boll.upper * (1 - self.config.entry_reclaim_buffer_pct):
            return False

        # ── CVD direction check at entry ────────────────────────────────
        cvd_direction_ok = (
            (cvd.cross_negative or cvd.cvd_decreasing)
            and cvd.sell_ratio >= self.config.min_sell_ratio
            and cvd.no_new_high
        )
        return cvd_direction_ok

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


    def _entry_protective_sl_price(self, side: PositionSide) -> float | None:
        if side == "LONG":
            extreme = self.state.lower_extreme_price
            if extreme is None or extreme <= 0:
                return None
            return extreme * (1 - self.config.entry_sl_buffer_pct)
        if side == "SHORT":
            extreme = self.state.upper_extreme_price
            if extreme is None or extreme <= 0:
                return None
            return extreme * (1 + self.config.entry_sl_buffer_pct)
        return None

    def _entry_reward_risk_check(
        self,
        *,
        side: PositionSide,
        entry_price: float,
        tp_price: float,
        stop_price: float,
    ) -> tuple[bool, str, float, float, float]:
        if entry_price <= 0 or tp_price <= 0 or stop_price <= 0:
            return False, "invalid_price", 0.0, 0.0, 0.0
        if side == "LONG":
            stop_distance_pct = (entry_price - stop_price) / entry_price
            reward_pct = (tp_price - entry_price) / entry_price
        else:
            stop_distance_pct = (stop_price - entry_price) / entry_price
            reward_pct = (entry_price - tp_price) / entry_price
        if stop_distance_pct <= 0:
            return False, "invalid_stop_distance", stop_distance_pct, reward_pct, 0.0
        if reward_pct <= 0:
            return False, "invalid_reward_distance", stop_distance_pct, reward_pct, 0.0
        if self.config.entry_max_stop_distance_pct > 0 and stop_distance_pct > self.config.entry_max_stop_distance_pct:
            return False, "stop_distance_too_wide", stop_distance_pct, reward_pct, 0.0
        effective_risk_pct = stop_distance_pct + self.config.entry_fee_slippage_buffer_pct
        reward_risk = reward_pct / effective_risk_pct if effective_risk_pct > 0 else 0.0
        if reward_risk < self.config.entry_min_reward_risk:
            return False, "reward_risk_below_min", stop_distance_pct, reward_pct, reward_risk
        return True, "ok", stop_distance_pct, reward_pct, reward_risk

    def _entry_reward_risk_target_price(
        self,
        *,
        side: PositionSide,
        boll: BollSnapshot,
        final_tp_price: float,
    ) -> tuple[float, str]:
        """Return the price to use for entry reward/risk filtering.

        STRUCTURE_MIDDLE — BOLL20 middle band (default, conservative entry quality).
        FINAL_TP         — the actual selected take-profit price (legacy behaviour).
        """
        if self.config.entry_rr_target == "STRUCTURE_MIDDLE":
            return float(boll.middle), "STRUCTURE_MIDDLE"
        return final_tp_price, "FINAL_TP"

    def _entry_add_flow(self):
        if not hasattr(self, "_entry_add_flow_coordinator"):
            from src.strategies.entry_add_flow_coordinator import EntryAddFlowCoordinator
            self._entry_add_flow_coordinator = EntryAddFlowCoordinator(self)
        return self._entry_add_flow_coordinator

    def _maybe_open_or_add_long(self, price: float, ts_ms: int, boll: BollSnapshot,
                                cvd: CvdSnapshot) -> TradeIntent | None:
        return self._entry_add_flow().maybe_open_or_add_long(price, ts_ms, boll, cvd)

    def _maybe_open_or_add_short(self, price: float, ts_ms: int, boll: BollSnapshot,
                                 cvd: CvdSnapshot) -> TradeIntent | None:
        return self._entry_add_flow().maybe_open_or_add_short(price, ts_ms, boll, cvd)

    def _add_layer_gap_pct_for_target_layer(self, target_layer: int) -> float:
        return add_layer_gates.add_layer_gap_pct_for_target_layer(
            target_layer=target_layer,
            add_gap_mode=self.config.add_gap_mode,
            add_gap_base_pct=self.config.add_gap_base_pct,
            add_gap_step_pct=self.config.add_gap_step_pct,
        )

    def _add_min_interval_bypass_gap_pct_for_target_layer(self, target_layer: int) -> float:
        return add_layer_gates.add_min_interval_bypass_gap_pct_for_target_layer(
            target_layer=target_layer,
            add_gap_mode=self.config.add_gap_mode,
            add_gap_base_pct=self.config.add_gap_base_pct,
            add_gap_step_pct=self.config.add_gap_step_pct,
        )

    def _add_gap_passed(self, side: PositionSide, price: float, target_layer: int) -> tuple[bool, float, float]:
        decision = add_layer_gates.check_add_gap(
            side=side,
            price=price,
            last_entry_price=self.state.last_entry_price,
            target_layer=target_layer,
            add_gap_mode=self.config.add_gap_mode,
            add_gap_base_pct=self.config.add_gap_base_pct,
            add_gap_step_pct=self.config.add_gap_step_pct,
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
            add_gap_mode=self.config.add_gap_mode,
            add_gap_base_pct=self.config.add_gap_base_pct,
            add_gap_step_pct=self.config.add_gap_step_pct,
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
    ) -> TradeIntent | None:
        return self._entry_add_flow().open_position(
            side, intent_type, price, ts_ms, boll, cvd, reason)

    def _tp_update(self):
        if not hasattr(self, "_tp_update_coordinator"):
            from src.strategies.tp_update_coordinator import TpUpdateCoordinator
            self._tp_update_coordinator = TpUpdateCoordinator(self)
        return self._tp_update_coordinator

    def _intent_factory(self):
        if not hasattr(self, "_strategy_intent_factory"):
            from src.strategies.strategy_intent_factory import StrategyIntentFactory
            self._strategy_intent_factory = StrategyIntentFactory(self)
        return self._strategy_intent_factory

    def _maybe_update_tp(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> TradeIntent | None:
        return self._tp_update().maybe_update_tp(price, ts_ms, boll, cvd)

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
        if not is_pre_tp1_lifecycle(self.state):
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
        can_refresh_degrade_stage = is_pre_tp1_lifecycle(self.state)
        final_tp, final_src = self._select_valid_tp_outer_with_profit_fallback(self.state.side, boll)
        first_tp, _first_src = self._select_valid_tp_middle_with_profit_fallback(self.state.side, boll)

        if first_tp is None:
            effective_be = self._effective_breakeven_for_tp_selection(self.state.side)
            required_middle = self._required_middle_for_profit(self.state.side, effective_be)
            logger.warning(
                "THREE_STAGE_PRE_TP1_MIDDLE_DEGRADE_SKIPPED_MIDDLE_PROFIT_INSUFFICIENT | "
                "side=%s effective_breakeven=%.4f required_middle=%.4f "
                "tp_boll_middle=%s structure_middle=%.4f selected_outer=%.4f outer_source=%s "
                "age_seconds=%.1f first_entry_ts_ms=%s",
                self.state.side,
                effective_be,
                required_middle,
                self._format_optional_price(getattr(boll, "tp_middle", None)),
                boll.middle,
                final_tp,
                final_src,
                age_seconds,
                self.state.first_entry_ts_ms,
            )
            self._fallback_to_single_outer_due_middle_profit_insufficient(
                side=self.state.side,
                boll=boll,
                ts_ms=ts_ms,
                reason="three_stage_pre_tp1_middle_degrade_middle_profit_insufficient",
            )
            return

        self._reset_three_stage_runner_state()
        self._set_middle_runner_planned(first_tp, final_tp)
        self.state.tp_plan = "MIDDLE_RUNNER"
        self.state.tp_price = final_tp
        self.state.tp_mode = "UPPER" if self.state.side == "LONG" else "LOWER"
        self.state.partial_tp_price = first_tp
        self.state.partial_tp_ratio = self.state.middle_runner_first_close_ratio
        if can_refresh_degrade_stage:
            self.state.three_stage_pre_tp1_degrade_stage = "MIDDLE_RUNNER"
            self.state.three_stage_pre_tp1_degraded_ts_ms = ts_ms
        else:
            logger.info(
                "PRE_TP1_DEGRADE_REFRESH_SKIPPED_POST_TP1 | target=MIDDLE_RUNNER side=%s age_seconds=%.1f",
                self.state.side,
                age_seconds,
            )
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
        can_refresh_degrade_stage = is_pre_tp1_lifecycle(self.state)
        self._reset_three_stage_runner_state()
        self._reset_middle_runner_state()
        self.state.tp_plan = "SINGLE"
        self.state.tp_price = tp_price
        self.state.tp_mode = tp_mode
        self.state.partial_tp_price = None
        self.state.partial_tp_ratio = 0.0
        if can_refresh_degrade_stage:
            self.state.three_stage_pre_tp1_degrade_stage = "SINGLE"
            self.state.three_stage_pre_tp1_degraded_ts_ms = ts_ms
        else:
            logger.info(
                "PRE_TP1_DEGRADE_REFRESH_SKIPPED_POST_TP1 | target=SINGLE side=%s age_seconds=%.1f",
                self.state.side,
                age_seconds,
            )
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
        return self._intent_factory().build_runner_market_exit_intent(
            side=side,
            price=price,
            layer_index=self.state.layers,
            tp_price=self.state.trend_runner_tp_price or self.state.tp_price or price,
            reason=reason,
            size=size,
            boll=boll,
            cvd=cvd,
            ts_ms=ts_ms,
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
        tp_mid, _tp_mid_src = self._select_valid_tp_middle_with_profit_fallback(side, boll)
        tp_outer, tp_outer_src = self._select_three_stage_tp2_outer(side, boll)
        if tp_mid is None:
            effective_be = self._effective_breakeven_for_tp_selection(side)
            required_middle = self._required_middle_for_profit(side, effective_be)
            logger.warning(
                "THREE_STAGE_PLAN_SKIPPED_MIDDLE_PROFIT_INSUFFICIENT | "
                "side=%s effective_breakeven=%.4f required_middle=%.4f "
                "tp_boll_middle=%s structure_middle=%.4f selected_outer=%.4f outer_source=%s candle_ts=%s",
                side,
                effective_be,
                required_middle,
                self._format_optional_price(getattr(boll, "tp_middle", None)),
                boll.middle,
                tp_outer,
                tp_outer_src,
                boll.candle_ts_ms,
            )
            self._fallback_to_single_outer_due_middle_profit_insufficient(
                side=side,
                boll=boll,
                ts_ms=int(getattr(self.state, "last_order_ts_ms", 0) or 0),
                reason="three_stage_plan_middle_profit_insufficient",
            )
            return
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

    def _update_three_stage_dynamic_targets_without_reset(self, side: PositionSide, boll: BollSnapshot) -> bool:
        tp1_ratio, tp2_ratio, runner_ratio = self._normalized_three_stage_ratios()
        tp_mid, _tp_mid_src = self._select_valid_tp_middle_with_profit_fallback(side, boll)
        if tp_mid is None:
            return False
        tp_outer, _tp_outer_src = self._select_three_stage_tp2_outer(side, boll)
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
        return True

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
        # sl_tighten_ratio is no longer used; pass 0.0 as a placeholder.
        # The pure function ignores it.
        decision = middle_runner_helpers.calculate_middle_runner_protective_sl(
            side=side,
            current_price=current_price,
            avg_entry_price=avg_entry,
            net_remaining_breakeven_price=base_breakeven,
            breakeven_fee_buffer_pct=fee,
            boll_middle=boll.middle,
            boll_upper=boll.upper,
            boll_lower=boll.lower,
            sl_tighten_ratio=0.0,
        )
        if decision.reason == "missing_cost_basis":
            return None
        if decision.reason != "calculated":
            # Reconstruct the raw protective SL that was found invalid for the log signature.
            _raw_sl = (
                max(decision.candidate_cost, decision.candidate_structure)
                if side == "LONG"
                else min(decision.candidate_cost, decision.candidate_structure)
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
        # sl_tighten_ratio is no longer used; pass 0.0 as a placeholder.
        # The pure function ignores it.
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
            sl_tighten_ratio=0.0,
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
                max(decision.candidate_cost, decision.candidate_structure)
                if side == "LONG"
                else min(decision.candidate_cost, decision.candidate_structure)
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

    def _select_valid_tp_middle_with_profit_fallback(
            self,
            side: PositionSide,
            boll: BollSnapshot,
    ) -> tuple[float | None, str]:
        effective_be = self._effective_breakeven_for_tp_selection(side)
        if effective_be <= 0:
            return None, "MISSING_EFFECTIVE_BREAKEVEN"

        self.state.breakeven_price = effective_be
        min_net_profit = abs(float(self.config.tp_min_net_profit_pct))
        tp_mid = (
            float(getattr(boll, "tp_middle", 0.0) or 0.0)
            if self._tp_boll_available(boll)
            else None
        )
        structure_mid = float(boll.middle)

        if side == "LONG":
            required = effective_be * (1 + min_net_profit)
            if tp_mid is not None and tp_mid >= required:
                return tp_mid, "TP_BOLL"
            if structure_mid >= required:
                return structure_mid, "STRUCTURE_BOLL_PROFIT_FALLBACK"
            return None, "MIDDLE_PROFIT_INSUFFICIENT"

        required = effective_be * (1 - min_net_profit)
        if tp_mid is not None and tp_mid <= required:
            return tp_mid, "TP_BOLL"
        if structure_mid <= required:
            return structure_mid, "STRUCTURE_BOLL_PROFIT_FALLBACK"
        return None, "MIDDLE_PROFIT_INSUFFICIENT"

    def _required_middle_for_profit(self, side: PositionSide, effective_be: float) -> float:
        min_net_profit = abs(float(self.config.tp_min_net_profit_pct))
        if effective_be <= 0:
            return 0.0
        if side == "LONG":
            return effective_be * (1 + min_net_profit)
        return effective_be * (1 - min_net_profit)

    def _format_optional_price(self, value: float | None) -> str:
        if value is None:
            return "-"
        return f"{float(value):.4f}"

    def _fallback_to_single_outer_due_middle_profit_insufficient(
            self,
            *,
            side: PositionSide,
            boll: BollSnapshot,
            ts_ms: int,
            reason: str,
    ) -> tuple[float, TpMode]:
        can_apply_pre_tp1_fallback = is_pre_tp1_lifecycle(self.state)
        if not can_apply_pre_tp1_fallback:
            current_tp_price = self.state.tp_price or self._select_valid_tp_outer_with_profit_fallback(side, boll)[0]
            current_tp_mode = self.state.tp_mode or ("UPPER" if side == "LONG" else "LOWER")
            logger.info(
                "PRE_TP1_FALLBACK_SKIPPED_POST_TP1 | side=%s reason=%s tp_plan=%s tp_price=%s tp_mode=%s "
                "tp1_consumed=%s tp2_consumed=%s trend_runner_active=%s middle_runner_active=%s partial_tp_consumed=%s",
                side,
                reason,
                self.state.tp_plan,
                current_tp_price,
                current_tp_mode,
                self.state.three_stage_tp1_consumed,
                self.state.three_stage_tp2_consumed,
                self.state.trend_runner_active,
                self.state.middle_runner_active,
                self.state.partial_tp_consumed,
            )
            return current_tp_price, current_tp_mode

        outer, _outer_src = self._select_valid_tp_outer_with_profit_fallback(side, boll)
        tp_mode: TpMode = "UPPER" if side == "LONG" else "LOWER"
        self._reset_three_stage_runner_state()
        self._reset_middle_runner_state()
        self.state.tp_plan = "SINGLE"
        self.state.tp_price = outer
        self.state.tp_mode = tp_mode
        self.state.partial_tp_price = None
        self.state.partial_tp_ratio = 0.0
        self.state.three_stage_pre_tp1_degrade_stage = "SINGLE"
        self.state.three_stage_pre_tp1_degraded_ts_ms = ts_ms
        return outer, tp_mode

    def _select_tp_outer(self, side: PositionSide, boll: BollSnapshot) -> tuple[float, str]:
        """Return (outer_price, source) for the given side."""
        tp_band = self._tp_band_snapshot(boll)
        sel = tp_plan_selector.select_tp_outer(
            side=side,
            tp_band=tp_band,
            tp_boll_enabled=self.config.tp_boll_enabled,
        )
        return sel.price, sel.source

    def _select_valid_tp_outer_with_profit_fallback(
            self,
            side: PositionSide,
            boll: BollSnapshot,
            *,
            log_warning: bool = True,
    ) -> tuple[float, str]:
        """Return (outer_price, source) with profit-distance fallback.

        1) TP_BOLL15 outer first.
        2) Structure BOLL20 outer if TP_BOLL15 outer profit is insufficient.
        3) Farther outer as last resort (logs TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK
           only when log_warning=True).
        """
        effective_be = self._effective_breakeven_for_tp_selection(side)
        tp_band = self._tp_band_snapshot(boll)
        sel = tp_plan_selector.select_tp_outer_with_profit_fallback(
            side=side,
            effective_be=effective_be,
            min_net_profit=self.config.tp_min_net_profit_pct,
            tp_band=tp_band,
            tp_boll_enabled=self.config.tp_boll_enabled,
        )
        if log_warning and sel.source == "TP_OUTER_HALF_MIN_PROFIT_FALLBACK":
            effective_be_val = self._effective_breakeven_for_tp_selection(side)
            half_min_profit_pct = abs(float(self.config.tp_min_net_profit_pct)) * 0.5
            tp_boll_outer_raw = (
                getattr(boll, "tp_upper", None)
                if side == "LONG"
                else getattr(boll, "tp_lower", None)
            )
            structure_outer_raw = float(boll.upper) if side == "LONG" else float(boll.lower)
            raw_outer_value = (
                float(tp_boll_outer_raw)
                if self._tp_boll_available(boll) and tp_boll_outer_raw is not None
                else structure_outer_raw
            )
            logger.warning(
                "CORE_TP_OUTER_UNPROFITABLE_HALF_MIN_FALLBACK | "
                "side=%s effective_breakeven=%.4f half_min_profit_pct=%.6f "
                "selected_tp=%.4f raw_outer=%s tp_boll_outer=%s structure_outer=%.4f candle_ts=%s",
                side,
                effective_be_val,
                half_min_profit_pct,
                sel.price,
                self._format_optional_price(raw_outer_value),
                self._format_optional_price(tp_boll_outer_raw),
                structure_outer_raw,
                boll.candle_ts_ms,
            )
        if log_warning and sel.source == "TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK":
            effective_be_val = self._effective_breakeven_for_tp_selection(side)
            min_net_profit_abs = abs(float(self.config.tp_min_net_profit_pct))
            required = (
                effective_be_val * (1 + min_net_profit_abs)
                if side == "LONG"
                else effective_be_val * (1 - min_net_profit_abs)
            )
            logger.warning(
                "TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK | "
                "side=%s effective_breakeven=%.4f required_outer=%.4f "
                "tp_boll_outer=%s structure_outer=%.4f selected_outer=%.4f candle_ts=%s",
                side,
                effective_be_val,
                required,
                self._format_optional_price(
                    tp_band.tp_upper if side == "LONG" else tp_band.tp_lower
                ),
                tp_band.upper if side == "LONG" else tp_band.lower,
                sel.price,
                boll.candle_ts_ms,
            )
        return sel.price, sel.source

    def _select_three_stage_tp2_outer(
        self,
        side: PositionSide,
        boll: BollSnapshot,
        *,
        log_warning: bool = True,
    ) -> tuple[float, str]:
        """Select TP2 outer price for Three-Stage Runner.

        Default semantics (THREE_STAGE_TP2_USE_STRUCTURE_BOLL=true):
          LONG  => structure BOLL20 upper
          SHORT => structure BOLL20 lower

        TP2 is the structural confirmation gate before Trend Runner.
        Trend Runner itself uses structure BOLL20 upper/lower/middle,
        so TP2 must be aligned with the same structure.

        If structure BOLL20 outer does not satisfy min net profit,
        choose the farther direction-correct outer between structure BOLL20
        and TP_BOLL outer if available, and log a warning.

        When THREE_STAGE_TP2_USE_STRUCTURE_BOLL=false, delegate to
        _select_valid_tp_outer_with_profit_fallback() to preserve old behavior.
        """
        if not self.config.three_stage_tp2_use_structure_boll:
            return self._select_valid_tp_outer_with_profit_fallback(
                side,
                boll,
                log_warning=log_warning,
            )

        effective_be = self._effective_breakeven_for_tp_selection(side)
        min_net_profit = abs(float(self.config.tp_min_net_profit_pct))

        if side == "LONG":
            structure_outer = float(boll.upper)
        else:
            structure_outer = float(boll.lower)

        # Determine if TP_BOLL15 outer is available
        tp_boll_avail = self._tp_boll_available(boll)
        tp_boll_outer: float | None = None
        if tp_boll_avail:
            if side == "LONG":
                tp_boll_outer = float(boll.tp_upper) if boll.tp_upper is not None else None
            else:
                tp_boll_outer = float(boll.tp_lower) if boll.tp_lower is not None else None

        # Without a valid effective breakeven, default to structure outer
        if effective_be <= 0:
            return structure_outer, "STRUCTURE_BOLL_THREE_STAGE_TP2"

        if side == "LONG":
            required = effective_be * (1 + min_net_profit)
            if structure_outer >= required:
                return structure_outer, "STRUCTURE_BOLL_THREE_STAGE_TP2"

            fallback_candidates = [structure_outer]
            if tp_boll_outer is not None:
                fallback_candidates.append(tp_boll_outer)
            fallback = max(fallback_candidates)

            if log_warning:
                tp_boll_str = f"{tp_boll_outer:.4f}" if tp_boll_outer is not None else "-"
                logger.warning(
                    "THREE_STAGE_TP2_STRUCTURE_OUTER_PROFIT_INSUFFICIENT_FALLBACK | "
                    "side=%s effective_breakeven=%.4f required=%.4f "
                    "structure_outer=%.4f tp_boll_outer=%s fallback=%.4f candle_ts=%s",
                    side,
                    effective_be,
                    required,
                    structure_outer,
                    tp_boll_str,
                    fallback,
                    boll.candle_ts_ms,
                )
            return fallback, "THREE_STAGE_TP2_PROFIT_FALLBACK"

        # SHORT
        required = effective_be * (1 - min_net_profit)
        if structure_outer <= required:
            return structure_outer, "STRUCTURE_BOLL_THREE_STAGE_TP2"

        fallback_candidates = [structure_outer]
        if tp_boll_outer is not None:
            fallback_candidates.append(tp_boll_outer)
        fallback = min(fallback_candidates)

        if log_warning:
            tp_boll_str = f"{tp_boll_outer:.4f}" if tp_boll_outer is not None else "-"
            logger.warning(
                "THREE_STAGE_TP2_STRUCTURE_OUTER_PROFIT_INSUFFICIENT_FALLBACK | "
                "side=%s effective_breakeven=%.4f required=%.4f "
                "structure_outer=%.4f tp_boll_outer=%s fallback=%.4f candle_ts=%s",
                side,
                effective_be,
                required,
                structure_outer,
                tp_boll_str,
                fallback,
                boll.candle_ts_ms,
            )
        return fallback, "THREE_STAGE_TP2_PROFIT_FALLBACK"

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
        # Use Three-Stage TP2 selector when plan is THREE_STAGE_RUNNER
        # so the fallback outer_source aligns with the actual TP2 selection.
        if tp_plan == "THREE_STAGE_RUNNER":
            tp_outer, tp_outer_src = self._select_three_stage_tp2_outer(
                self.state.side or "LONG", boll, log_warning=False)
        else:
            tp_outer, tp_outer_src = self._select_valid_tp_outer_with_profit_fallback(
                self.state.side or "LONG", boll, log_warning=False)

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
        tp_mid_fb_price: float | None
        if boll is not None:
            tp_mid_fb_price, _fb_src = self._select_valid_tp_middle_with_profit_fallback(side, boll)
        else:
            tp_mid_fb_price = None

        three_stage_allowed = self._three_stage_runner_plan_allowed(tp_mode, boll)
        middle_runner_allowed = self._middle_runner_plan_allowed(tp_mode, boll)

        tp1_ratio: float = 0.0
        if three_stage_allowed:
            tp1_ratio, _tp2r, _rr = self._normalized_three_stage_ratios()

        if (
                tp_mid_fb_price is None
                and (
                    three_stage_allowed
                    or middle_runner_allowed
                    or self.state.three_stage_pre_tp1_degrade_stage == "MIDDLE_RUNNER"
                )
        ):
            return None, 0.0, "SINGLE"

        sel = tp_plan_selector.select_tp_plan(
            side=side,
            final_tp=final_tp,
            layers=layers,
            tp_mode=tp_mode,
            boll_exists=boll is not None,
            three_stage_pre_tp1_degrade_stage=self.state.three_stage_pre_tp1_degrade_stage,
            middle_runner_first_close_ratio=self.config.middle_runner_first_close_ratio,
            tp_middle_profit_fallback_price=tp_mid_fb_price or 0.0,
            three_stage_runner_plan_allowed=three_stage_allowed,
            three_stage_tp1_ratio=tp1_ratio,
            three_stage_runner_enabled=self.config.three_stage_runner_enabled,
            middle_runner_plan_allowed=middle_runner_allowed,
        )
        return sel.partial_tp_price, sel.partial_tp_ratio, sel.tp_plan

    def _three_stage_runner_plan_allowed(self, tp_mode: TpMode | None, boll: BollSnapshot | None) -> bool:
        return tp_plan_selector.three_stage_runner_plan_allowed(
            three_stage_runner_enabled=self.config.three_stage_runner_enabled,
            three_stage_pre_tp1_degrade_stage=self.state.three_stage_pre_tp1_degrade_stage,
            tp_mode=tp_mode,
            boll_exists=boll is not None,
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
        return self._intent_factory().build_intent(
            intent_type=intent_type,
            side=side,
            price=price,
            layer_index=layer_index,
            tp_price=tp_price,
            reason=reason,
            size=size,
            boll=boll,
            cvd=cvd,
            ts_ms=ts_ms,
        )

    def _managed_core_contracts_for_intent(self, intent_type: TradeIntentType) -> str | None:
        return self._intent_factory().managed_core_contracts_for_intent(intent_type)

    def _managed_core_eth_qty_for_intent(self, intent_type: TradeIntentType) -> float:
        return self._intent_factory().managed_core_eth_qty_for_intent(intent_type)

    def _protected_order_ids(self) -> tuple[str, ...]:
        return self._intent_factory().protected_order_ids()

    def _cooldown_ok(self, ts_ms: int) -> bool:
        return ts_ms - self.state.last_order_ts_ms >= self.config.order_cooldown_seconds * 1000

    def _post_entry_sl_cooldown_ok(self, side: PositionSide, ts_ms: int) -> bool:
        """Check whether post-entry-SL cooldown allows a new entry.

        Returns True if entry is allowed, False if blocked by cooldown.
        """
        if not self.config.post_entry_sl_cooldown_enabled:
            return True
        if self.state.post_entry_sl_cooldown_until_ts_ms <= 0:
            return True
        if ts_ms >= self.state.post_entry_sl_cooldown_until_ts_ms:
            remaining_ms = max(self.state.post_entry_sl_cooldown_until_ts_ms - ts_ms, 0)
            logger.info(
                "POST_ENTRY_SL_COOLDOWN_EXPIRED | until_ts_ms=%s ts_ms=%s side=%s reason=%s",
                self.state.post_entry_sl_cooldown_until_ts_ms,
                ts_ms,
                self.state.post_entry_sl_cooldown_side,
                self.state.post_entry_sl_cooldown_reason,
            )
            self.state.post_entry_sl_cooldown_until_ts_ms = 0
            self.state.post_entry_sl_cooldown_side = None
            self.state.post_entry_sl_cooldown_reason = None
            return True

        # Cooldown is active
        scope = self.config.post_entry_sl_cooldown_scope
        if scope == "GLOBAL":
            logger.info(
                "POST_ENTRY_SL_COOLDOWN_ACTIVE | side=%s scope=GLOBAL "
                "until_ts_ms=%s remaining_ms=%s cooldown_side=%s reason=%s",
                side,
                self.state.post_entry_sl_cooldown_until_ts_ms,
                self.state.post_entry_sl_cooldown_until_ts_ms - ts_ms,
                self.state.post_entry_sl_cooldown_side,
                self.state.post_entry_sl_cooldown_reason,
            )
            return False
        if scope == "SIDE" and side == self.state.post_entry_sl_cooldown_side:
            logger.info(
                "POST_ENTRY_SL_COOLDOWN_ACTIVE | side=%s scope=SIDE "
                "until_ts_ms=%s remaining_ms=%s reason=%s",
                side,
                self.state.post_entry_sl_cooldown_until_ts_ms,
                self.state.post_entry_sl_cooldown_until_ts_ms - ts_ms,
                self.state.post_entry_sl_cooldown_reason,
            )
            return False
        return True

    def arm_post_entry_sl_cooldown(self, ts_ms: int, side: str, reason: str) -> None:
        """Arm post-entry-SL cooldown after an initial entry protective SL exit."""
        if not self.config.post_entry_sl_cooldown_enabled:
            return
        cooldown_ms = self.config.post_entry_sl_cooldown_seconds * 1000
        self.state.post_entry_sl_cooldown_until_ts_ms = ts_ms + cooldown_ms
        self.state.post_entry_sl_cooldown_side = side
        self.state.post_entry_sl_cooldown_reason = reason
        logger.warning(
            "POST_ENTRY_SL_COOLDOWN_ARMED | side=%s until_ts_ms=%s cooldown_seconds=%s "
            "scope=%s reason=%s ts_ms=%s",
            side,
            self.state.post_entry_sl_cooldown_until_ts_ms,
            self.config.post_entry_sl_cooldown_seconds,
            self.config.post_entry_sl_cooldown_scope,
            reason,
            ts_ms,
        )
