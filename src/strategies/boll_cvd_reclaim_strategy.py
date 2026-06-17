from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.position_management.cost_basis import calculate_remaining_breakeven_price
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer
from src.strategies import middle_runner as middle_runner_helpers
from src.strategies import three_stage_runner as three_stage_helpers
from src.strategies import tp_plan_selector
from src.strategies import trend_runner as trend_runner_helpers
from src.strategies.regime.router import RegimeRouter, RouterInput
from src.strategies.regime.types import (
    AnchoredCvdState,
    BandSnapshot,
    RegimeDecision,
    RegimeDecisionType,
    TrendState,
)
from src.strategies.anchored_orderflow import (
    AnchoredOrderflowSnapshot,
    AnchoredOrderflowTracker,
)
from src.strategies.reclaim_anchored_divergence import (
    AnchoredDivergenceConfig,
    AnchoredDivergenceDecision,
    evaluate_anchored_divergence,
)
from src.strategies.sweep_volume_profile import SweepVolumeProfile
from src.strategies.tp_lifecycle import is_pre_tp1_lifecycle
from src.strategies.trend_breakout import TrendBreakoutAssessor, TrendBreakoutDecision
from src.strategies.trend_breakout_metrics import (
    TrendBreakoutMetricsTracker,
)
from src.strategies.trend_middle_trailing_sl import (
    calculate_trend_middle_sl,
)
from src.utils.log import get_logger

logger = get_logger(__name__)

TradeIntentType = Literal[
    "OPEN_LONG",
    "OPEN_SHORT",
    "UPDATE_TP",
    "UPDATE_TREND_SL",
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
    order_cooldown_seconds: int = 10
    tp_update_interval_seconds: int = 900
    max_armed_seconds: int = 900
    breakeven_fee_buffer_pct: float = 0.001
    tp_min_net_profit_pct: float = 0.004
    min_outside_pct: float = 0.001
    entry_reclaim_inside_band: bool = True
    entry_reclaim_buffer_pct: float = 0.0
    entry_sl_buffer_pct: float = 0.0005
    entry_min_reward_risk: float = 1.0
    entry_fee_slippage_buffer_pct: float = 0.001
    entry_max_stop_distance_pct: float = 0.0
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
    post_entry_sl_cooldown_scope: str = "SIDE"

    # ── Reclaim V2 ────────────────────────────────────────────────────
    entry_reclaim_v2_enabled: bool = True
    entry_reclaim_require_anchored_divergence: bool = True
    # ── Sweep Volume Profile / POC ────────────────────────────────────
    entry_sweep_profile_enabled: bool = True
    entry_sweep_profile_bucket_pct: float = 0.0002
    entry_poc_stop_enabled: bool = True
    entry_poc_stop_min_tail_pct: float = 0.008
    entry_poc_stop_buffer_pct: float = 0.001
    entry_extreme_stop_buffer_pct: float = 0.001
    entry_reclaim_min_cvd_recovery: float = 0.0
    entry_reclaim_min_cvd_follow_through: float = 0.0
    entry_reclaim_max_inside_depth_ratio: float = 0.15
    reclaim_extreme_log_interval_seconds: int = 10
    reclaim_no_entry_log_interval_seconds: int = 60

    # ── Trend Breakout Entry ────────────────────────────────────────────
    trend_breakout_enabled: bool = False
    trend_middle_trailing_sl_enabled: bool = True
    trend_middle_sl_buffer_pct: float = 0.001
    trend_max_stop_distance_pct: float = 0.02
    trend_sl_update_interval_seconds: int = 900
    trend_compression_valid_after_seconds: int = 7200
    trend_confirm_min_seconds: int = 900
    trend_confirm_max_seconds: int = 1200
    trend_range_expansion_ratio_min: float = 3.0
    trend_volume_expansion_ratio_min: float = 3.0
    trend_outside_occupancy_min_ratio: float = 0.70
    trend_min_new_extreme_count: int = 2
    trend_max_inside_reclaim_seconds: int = 3
    trend_cvd_min_buy_ratio: float = 0.58
    trend_cvd_min_sell_ratio: float = 0.58
    trend_cvd_max_pullback_ratio: float = 0.45
    # ── Trend Candle Close Confirmation ───────────────────────────────
    trend_confirm_require_candle_close: bool = True
    # ── Pre-Breakout Directional Pressure ─────────────────────────────
    trend_pre_breakout_pressure_enabled: bool = True
    trend_pre_breakout_min_cvd_ratio: float = 0.55
    trend_pre_breakout_max_pullback_ratio: float = 0.45
    trend_pre_breakout_min_observe_seconds: int = 300
    trend_pre_breakout_pressure_min_score: float = 0.60

    # ── Trend Upgrade Add-on ──────────────────────────────────────────
    trend_upgrade_addon_enabled: bool = False
    trend_upgrade_profit_reinvest_ratio: float = 0.30
    trend_upgrade_max_addon_risk_pct: float = 0.002
    trend_upgrade_max_total_notional_multiplier: float = 1.0
    trend_upgrade_require_tp1_consumed: bool = True
    trend_upgrade_require_tp2_consumed: bool = True
    trend_upgrade_min_runner_remaining_ratio: float = 0.05
    trend_upgrade_min_trend_confidence: float = 0.80

    def __post_init__(self) -> None:
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
        # ── Reclaim V2 validation ──────────────────────────────────────
        if self.entry_sweep_profile_bucket_pct <= 0:
            raise RuntimeError(
                f"ENTRY_SWEEP_PROFILE_BUCKET_PCT={self.entry_sweep_profile_bucket_pct} must be > 0"
            )
        if self.entry_poc_stop_min_tail_pct < 0:
            raise RuntimeError(
                f"ENTRY_POC_STOP_MIN_TAIL_PCT={self.entry_poc_stop_min_tail_pct} must be >= 0"
            )
        if self.entry_poc_stop_buffer_pct < 0:
            raise RuntimeError(
                f"ENTRY_POC_STOP_BUFFER_PCT={self.entry_poc_stop_buffer_pct} must be >= 0"
            )
        if self.entry_extreme_stop_buffer_pct < 0:
            raise RuntimeError(
                f"ENTRY_EXTREME_STOP_BUFFER_PCT={self.entry_extreme_stop_buffer_pct} must be >= 0"
            )
        # ── Trend Breakout validation ────────────────────────────────────
        if self.trend_middle_sl_buffer_pct < 0:
            raise RuntimeError(
                f"TREND_MIDDLE_SL_BUFFER_PCT={self.trend_middle_sl_buffer_pct} must be >= 0"
            )
        if self.trend_max_stop_distance_pct <= 0:
            raise RuntimeError(
                f"TREND_MAX_STOP_DISTANCE_PCT={self.trend_max_stop_distance_pct} must be > 0"
            )
        if self.trend_sl_update_interval_seconds <= 0:
            raise RuntimeError(
                f"TREND_SL_UPDATE_INTERVAL_SECONDS={self.trend_sl_update_interval_seconds} must be > 0"
            )
        if self.trend_confirm_min_seconds <= 0:
            raise RuntimeError(
                f"TREND_CONFIRM_MIN_SECONDS={self.trend_confirm_min_seconds} must be > 0"
            )
        if self.trend_confirm_max_seconds < self.trend_confirm_min_seconds:
            raise RuntimeError(
                f"TREND_CONFIRM_MAX_SECONDS={self.trend_confirm_max_seconds} must be >= "
                f"TREND_CONFIRM_MIN_SECONDS={self.trend_confirm_min_seconds}"
            )
        if self.trend_range_expansion_ratio_min <= 0:
            raise RuntimeError(
                f"TREND_RANGE_EXPANSION_RATIO_MIN={self.trend_range_expansion_ratio_min} must be > 0"
            )
        if self.trend_volume_expansion_ratio_min <= 0:
            raise RuntimeError(
                f"TREND_VOLUME_EXPANSION_RATIO_MIN={self.trend_volume_expansion_ratio_min} must be > 0"
            )
        if not (0 < self.trend_outside_occupancy_min_ratio <= 1):
            raise RuntimeError(
                f"TREND_OUTSIDE_OCCUPANCY_MIN_RATIO={self.trend_outside_occupancy_min_ratio} must be in (0, 1]"
            )
        if self.trend_min_new_extreme_count < 0:
            raise RuntimeError(
                f"TREND_MIN_NEW_EXTREME_COUNT={self.trend_min_new_extreme_count} must be >= 0"
            )
        if self.trend_max_inside_reclaim_seconds < 0:
            raise RuntimeError(
                f"TREND_MAX_INSIDE_RECLAIM_SECONDS={self.trend_max_inside_reclaim_seconds} must be >= 0"
            )
        if not (0 < self.trend_cvd_min_buy_ratio <= 1):
            raise RuntimeError(
                f"TREND_CVD_MIN_BUY_RATIO={self.trend_cvd_min_buy_ratio} must be in (0, 1]"
            )
        if not (0 < self.trend_cvd_min_sell_ratio <= 1):
            raise RuntimeError(
                f"TREND_CVD_MIN_SELL_RATIO={self.trend_cvd_min_sell_ratio} must be in (0, 1]"
            )
        if not (0 <= self.trend_cvd_max_pullback_ratio <= 1):
            raise RuntimeError(
                f"TREND_CVD_MAX_PULLBACK_RATIO={self.trend_cvd_max_pullback_ratio} must be in [0, 1]"
            )
        # ── Trend Pre-Breakout Pressure validation ─────────────────────
        if self.trend_pre_breakout_pressure_enabled:
            if not (0.5 <= self.trend_pre_breakout_min_cvd_ratio <= 1):
                raise RuntimeError(
                    f"TREND_PRE_BREAKOUT_MIN_CVD_RATIO={self.trend_pre_breakout_min_cvd_ratio} "
                    f"must be in [0.5, 1]"
                )
            if not (0 <= self.trend_pre_breakout_max_pullback_ratio <= 1):
                raise RuntimeError(
                    f"TREND_PRE_BREAKOUT_MAX_PULLBACK_RATIO={self.trend_pre_breakout_max_pullback_ratio} "
                    f"must be in [0, 1]"
                )
            if self.trend_pre_breakout_min_observe_seconds <= 0:
                raise RuntimeError(
                    f"TREND_PRE_BREAKOUT_MIN_OBSERVE_SECONDS={self.trend_pre_breakout_min_observe_seconds} "
                    f"must be > 0"
                )
            if not (0 <= self.trend_pre_breakout_pressure_min_score <= 1):
                raise RuntimeError(
                    f"TREND_PRE_BREAKOUT_PRESSURE_MIN_SCORE={self.trend_pre_breakout_pressure_min_score} "
                    f"must be in [0, 1]"
                )
        if self.middle_bucket_split_enabled:
            _fr = self.middle_bucket_split_fast_ratio
            if not (0.05 <= _fr <= 0.95):
                raise RuntimeError(
                    f"MIDDLE_BUCKET_SPLIT_FAST_RATIO={_fr} is out of range [0.05, 0.95]; "
                    f"this is a live position ratio — refusing to proceed with dangerous value"
                )
        # ── Trend Upgrade Add-on validation ────────────────────────────
        if self.trend_upgrade_addon_enabled:
            if not (0 < self.trend_upgrade_profit_reinvest_ratio <= 1):
                raise RuntimeError(
                    f"TREND_UPGRADE_PROFIT_REINVEST_RATIO={self.trend_upgrade_profit_reinvest_ratio} must be in (0, 1]"
                )
            if not (0 < self.trend_upgrade_max_addon_risk_pct <= 0.05):
                raise RuntimeError(
                    f"TREND_UPGRADE_MAX_ADDON_RISK_PCT={self.trend_upgrade_max_addon_risk_pct} must be in (0, 0.05]"
                )
            if not (0 < self.trend_upgrade_max_total_notional_multiplier <= 5):
                raise RuntimeError(
                    f"TREND_UPGRADE_MAX_TOTAL_NOTIONAL_MULTIPLIER={self.trend_upgrade_max_total_notional_multiplier} must be in (0, 5]"
                )
            if not (0.01 <= self.trend_upgrade_min_runner_remaining_ratio <= 1):
                raise RuntimeError(
                    f"TREND_UPGRADE_MIN_RUNNER_REMAINING_RATIO={self.trend_upgrade_min_runner_remaining_ratio} must be in [0.01, 1]"
                )
            if not (0.5 <= self.trend_upgrade_min_trend_confidence <= 1):
                raise RuntimeError(
                    f"TREND_UPGRADE_MIN_TREND_CONFIDENCE={self.trend_upgrade_min_trend_confidence} must be in [0.5, 1]"
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
            order_cooldown_seconds=int(os.getenv("ORDER_COOLDOWN_SECONDS", "10")),
            tp_update_interval_seconds=int(os.getenv("TP_UPDATE_INTERVAL_SECONDS", "900")),
            max_armed_seconds=int(os.getenv("MAX_ARMED_SECONDS", "900")),
            breakeven_fee_buffer_pct=float(os.getenv("BREAKEVEN_FEE_BUFFER_PCT", "0.001")),
            tp_min_net_profit_pct=float(os.getenv("TP_MIN_NET_PROFIT_PCT", "0.004")),
            min_outside_pct=float(os.getenv("BOLL_MIN_OUTSIDE_PCT", "0.001")),
            entry_reclaim_inside_band=_env_bool("ENTRY_RECLAIM_INSIDE_BAND", True),
            entry_reclaim_buffer_pct=float(os.getenv("ENTRY_RECLAIM_BUFFER_PCT", "0")),
            entry_sl_buffer_pct=float(os.getenv("ENTRY_SL_BUFFER_PCT", "0.0005")),
            entry_min_reward_risk=float(os.getenv("ENTRY_MIN_REWARD_RISK", "1.0")),
            entry_fee_slippage_buffer_pct=float(os.getenv("ENTRY_FEE_SLIPPAGE_BUFFER_PCT", "0.001")),
            entry_max_stop_distance_pct=float(os.getenv("ENTRY_MAX_STOP_DISTANCE_PCT", "0")),
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
            post_entry_sl_cooldown_scope=os.getenv("POST_ENTRY_SL_COOLDOWN_SCOPE", "SIDE").strip().upper(),
            # ── Reclaim V2 ──────────────────────────────────────────────
            entry_reclaim_v2_enabled=_env_bool("ENTRY_RECLAIM_V2_ENABLED", True),
            entry_reclaim_require_anchored_divergence=_env_bool("ENTRY_RECLAIM_REQUIRE_ANCHORED_DIVERGENCE", True),
            entry_sweep_profile_enabled=_env_bool("ENTRY_SWEEP_PROFILE_ENABLED", True),
            entry_sweep_profile_bucket_pct=float(os.getenv("ENTRY_SWEEP_PROFILE_BUCKET_PCT", "0.0002")),
            entry_poc_stop_enabled=_env_bool("ENTRY_POC_STOP_ENABLED", True),
            entry_poc_stop_min_tail_pct=float(os.getenv("ENTRY_POC_STOP_MIN_TAIL_PCT", "0.008")),
            entry_poc_stop_buffer_pct=float(os.getenv("ENTRY_POC_STOP_BUFFER_PCT", "0.001")),
            entry_extreme_stop_buffer_pct=float(os.getenv("ENTRY_EXTREME_STOP_BUFFER_PCT", "0.001")),
            entry_reclaim_min_cvd_recovery=float(os.getenv("ENTRY_RECLAIM_MIN_CVD_RECOVERY", "0")),
            entry_reclaim_min_cvd_follow_through=float(os.getenv("ENTRY_RECLAIM_MIN_CVD_FOLLOW_THROUGH", "0")),
            entry_reclaim_max_inside_depth_ratio=float(os.getenv("ENTRY_RECLAIM_MAX_INSIDE_DEPTH_RATIO", "0.15")),
            reclaim_extreme_log_interval_seconds=int(os.getenv("RECLAIM_EXTREME_LOG_INTERVAL_SECONDS", "10")),
            reclaim_no_entry_log_interval_seconds=int(os.getenv("RECLAIM_NO_ENTRY_LOG_INTERVAL_SECONDS", "60")),
            # ── Trend Breakout Entry ────────────────────────────────────
            trend_breakout_enabled=_env_bool("TREND_BREAKOUT_ENABLED", False),
            trend_middle_trailing_sl_enabled=_env_bool("TREND_MIDDLE_TRAILING_SL_ENABLED", True),
            trend_middle_sl_buffer_pct=float(os.getenv("TREND_MIDDLE_SL_BUFFER_PCT", "0.001")),
            trend_max_stop_distance_pct=float(os.getenv("TREND_MAX_STOP_DISTANCE_PCT", "0.02")),
            trend_sl_update_interval_seconds=int(os.getenv("TREND_SL_UPDATE_INTERVAL_SECONDS", "900")),
            trend_compression_valid_after_seconds=int(os.getenv("TREND_COMPRESSION_VALID_AFTER_SECONDS", "7200")),
            trend_confirm_min_seconds=int(os.getenv("TREND_CONFIRM_MIN_SECONDS", "900")),
            trend_confirm_max_seconds=int(os.getenv("TREND_CONFIRM_MAX_SECONDS", "1200")),
            trend_range_expansion_ratio_min=float(os.getenv("TREND_RANGE_EXPANSION_RATIO_MIN", "3.0")),
            trend_volume_expansion_ratio_min=float(os.getenv("TREND_VOLUME_EXPANSION_RATIO_MIN", "3.0")),
            trend_outside_occupancy_min_ratio=float(os.getenv("TREND_OUTSIDE_OCCUPANCY_MIN_RATIO", "0.70")),
            trend_min_new_extreme_count=int(os.getenv("TREND_MIN_NEW_EXTREME_COUNT", "2")),
            trend_max_inside_reclaim_seconds=int(os.getenv("TREND_MAX_INSIDE_RECLAIM_SECONDS", "3")),
            trend_cvd_min_buy_ratio=float(os.getenv("TREND_CVD_MIN_BUY_RATIO", "0.58")),
            trend_cvd_min_sell_ratio=float(os.getenv("TREND_CVD_MIN_SELL_RATIO", "0.58")),
            trend_cvd_max_pullback_ratio=float(os.getenv("TREND_CVD_MAX_PULLBACK_RATIO", "0.45")),
            # ── Trend Candle Close Confirmation ───────────────────────────
            trend_confirm_require_candle_close=_env_bool("TREND_CONFIRM_REQUIRE_CANDLE_CLOSE", True),
            # ── Trend Pre-Breakout Directional Pressure ───────────────────
            trend_pre_breakout_pressure_enabled=_env_bool("TREND_PRE_BREAKOUT_PRESSURE_ENABLED", True),
            trend_pre_breakout_min_cvd_ratio=float(os.getenv("TREND_PRE_BREAKOUT_MIN_CVD_RATIO", "0.55")),
            trend_pre_breakout_max_pullback_ratio=float(os.getenv("TREND_PRE_BREAKOUT_MAX_PULLBACK_RATIO", "0.45")),
            trend_pre_breakout_min_observe_seconds=int(os.getenv("TREND_PRE_BREAKOUT_MIN_OBSERVE_SECONDS", "300")),
            trend_pre_breakout_pressure_min_score=float(os.getenv("TREND_PRE_BREAKOUT_PRESSURE_MIN_SCORE", "0.60")),
            # ── Trend Upgrade Add-on ──────────────────────────────────
            trend_upgrade_addon_enabled=_env_bool("TREND_UPGRADE_ADDON_ENABLED", False),
            trend_upgrade_profit_reinvest_ratio=float(os.getenv("TREND_UPGRADE_PROFIT_REINVEST_RATIO", "0.30")),
            trend_upgrade_max_addon_risk_pct=float(os.getenv("TREND_UPGRADE_MAX_ADDON_RISK_PCT", "0.002")),
            trend_upgrade_max_total_notional_multiplier=float(
                os.getenv("TREND_UPGRADE_MAX_TOTAL_NOTIONAL_MULTIPLIER", "1.0")),
            trend_upgrade_require_tp1_consumed=_env_bool("TREND_UPGRADE_REQUIRE_TP1_CONSUMED", True),
            trend_upgrade_require_tp2_consumed=_env_bool("TREND_UPGRADE_REQUIRE_TP2_CONSUMED", True),
            trend_upgrade_min_runner_remaining_ratio=float(
                os.getenv("TREND_UPGRADE_MIN_RUNNER_REMAINING_RATIO", "0.05")),
            trend_upgrade_min_trend_confidence=float(os.getenv("TREND_UPGRADE_MIN_TREND_CONFIDENCE", "0.80")),
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

    # ── Entry Regime ────────────────────────────────────────────────────
    # "MEAN_REVERSION" | "TREND_BREAKOUT" | None
    entry_regime: str | None = None


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
    add_freeze_until_ts_ms: int = 0  # DEPRECATED: no-add mode, kept for live_state_store compat
    add_freeze_penalty_count: int = 0  # DEPRECATED: no-add mode, kept for live_state_store compat
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
    lower_reclaim_confirmed_logged: bool = False
    upper_reclaim_confirmed_logged: bool = False

    lower_reclaim_cvd_follow_through_logged: bool = False
    upper_reclaim_cvd_follow_through_logged: bool = False

    # ── Reclaim V2 attempt reject lock ────────────────────────────────
    lower_reclaim_rejected_until_next_outside: bool = False
    upper_reclaim_rejected_until_next_outside: bool = False

    # ── Reclaim V2: event-anchored cumulative CVD divergence ──────────
    # LOWER / LONG side
    lower_outside_observed: bool = False
    lower_anchor_price: float | None = None
    lower_anchor_ts_ms: int = 0
    lower_anchor_cumulative_cvd: float | None = None

    lower_first_extreme_price: float | None = None
    lower_first_extreme_ts_ms: int = 0
    lower_first_extreme_anchored_cvd: float | None = None

    lower_previous_extreme_price: float | None = None
    lower_previous_extreme_ts_ms: int = 0
    lower_previous_extreme_anchored_cvd: float | None = None

    lower_anchored_divergence_confirmed: bool = False
    lower_anchored_divergence_ts_ms: int = 0

    lower_sweep_profile: object | None = None  # SweepVolumeProfile

    # UPPER / SHORT side
    upper_outside_observed: bool = False
    upper_anchor_price: float | None = None
    upper_anchor_ts_ms: int = 0
    upper_anchor_cumulative_cvd: float | None = None

    upper_first_extreme_price: float | None = None
    upper_first_extreme_ts_ms: int = 0
    upper_first_extreme_anchored_cvd: float | None = None

    upper_previous_extreme_price: float | None = None
    upper_previous_extreme_ts_ms: int = 0
    upper_previous_extreme_anchored_cvd: float | None = None

    upper_anchored_divergence_confirmed: bool = False
    upper_anchored_divergence_ts_ms: int = 0

    upper_sweep_profile: object | None = None  # SweepVolumeProfile

    # ── Divergence extreme + reference band (saved at divergence confirm) ──
    lower_divergence_extreme_price: float | None = None
    lower_divergence_extreme_ts_ms: int = 0
    lower_divergence_extreme_anchored_cvd: float | None = None
    lower_divergence_ref_lower: float | None = None
    lower_divergence_ref_middle: float | None = None

    upper_divergence_extreme_price: float | None = None
    upper_divergence_extreme_ts_ms: int = 0
    upper_divergence_extreme_anchored_cvd: float | None = None
    upper_divergence_ref_upper: float | None = None
    upper_divergence_ref_middle: float | None = None

    # ── Post-Entry SL Cooldown state ────────────────────────────────
    post_entry_sl_cooldown_until_ts_ms: int = 0
    post_entry_sl_cooldown_side: str | None = None
    post_entry_sl_cooldown_reason: str | None = None

    # ── Reclaim V2 observability state ──────────────────────────────
    lower_last_extreme_snapshot_log_ts_ms: int = 0
    upper_last_extreme_snapshot_log_ts_ms: int = 0

    lower_extreme_snapshot_pending: bool = False
    upper_extreme_snapshot_pending: bool = False

    lower_last_logged_extreme_price: float | None = None
    upper_last_logged_extreme_price: float | None = None

    lower_last_extreme_divergence_reason: str | None = None
    upper_last_extreme_divergence_reason: str | None = None

    lower_last_extreme_divergence_confirmed: bool = False
    upper_last_extreme_divergence_confirmed: bool = False

    # ── Reclaim V2 coherent snapshot cache (from same divergence evaluation) ──
    lower_last_snapshot_prev_extreme_price: float | None = None
    lower_last_snapshot_prev_extreme_cvd: float | None = None
    lower_last_snapshot_curr_extreme_price: float | None = None
    lower_last_snapshot_curr_extreme_cvd: float | None = None
    lower_last_snapshot_price_extension_pct: float = 0.0
    lower_last_snapshot_cvd_recovery: float = 0.0
    lower_last_snapshot_divergence_confirmed: bool = False
    lower_last_snapshot_divergence_reason: str | None = None

    upper_last_snapshot_prev_extreme_price: float | None = None
    upper_last_snapshot_prev_extreme_cvd: float | None = None
    upper_last_snapshot_curr_extreme_price: float | None = None
    upper_last_snapshot_curr_extreme_cvd: float | None = None
    upper_last_snapshot_price_extension_pct: float = 0.0
    upper_last_snapshot_cvd_recovery: float = 0.0
    upper_last_snapshot_divergence_confirmed: bool = False
    upper_last_snapshot_divergence_reason: str | None = None

    lower_last_no_entry_log_ts_ms: int = 0
    upper_last_no_entry_log_ts_ms: int = 0

    lower_last_no_entry_reason: str | None = None
    upper_last_no_entry_reason: str | None = None

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

    # ── Trend Breakout Entry state ────────────────────────────────────────
    entry_regime: str | None = None  # None | "MEAN_REVERSION" | "TREND_BREAKOUT" | "TREND_UPGRADE" | "TREND_UPGRADE_ADDON"
    trend_breakout_active: bool = False
    trend_trailing_sl_price: float | None = None
    trend_last_sl_update_ts_ms: int = 0

    # ── Trend Upgrade Add-on state ─────────────────────────────────────────
    trend_upgrade_active: bool = False
    trend_upgrade_addon_active: bool = False
    trend_upgrade_addon_count: int = 0
    trend_upgrade_addon_entry_price: float | None = None
    trend_upgrade_addon_qty: float = 0.0
    trend_upgrade_addon_risk_budget_usdt: float = 0.0
    trend_upgrade_addon_sl_price: float | None = None
    trend_upgrade_last_ts_ms: int = 0
    # ── Position management mode: tracks the active management strategy ──
    # "MEAN_REVERSION" | "TREND_BREAKOUT" | "TREND_UPGRADE" | "TREND_UPGRADE_ADDON" | None
    position_management_mode: str | None = None

    # ── Extreme Retest Add state (DEPRECATED: kept for live_state_store backward compat) ──
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


@dataclass(frozen=True)
class _ExtremeUpdateResult:
    """Structured result from _update_lower_extreme / _update_upper_extreme.

    Only when new_extreme_detected=True should divergence be evaluated.
    Absorption may still be checked on the first valid extreme tick.
    """

    new_extreme_detected: bool
    old_extreme_price: float | None = None
    new_extreme_price: float | None = None
    old_extreme_fast_cvd: float | None = None
    new_fast_cvd: float | None = None


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
        self.regime_router = RegimeRouter()
        self.trend_assessor: TrendBreakoutAssessor | None = None
        self.trend_metrics_tracker: TrendBreakoutMetricsTracker | None = None
        self._last_throttled_log_ts_ms: dict[str, int] = {}
        # ── Reclaim V2 anchored orderflow trackers ─────────────────────
        self._lower_orderflow = AnchoredOrderflowTracker()
        self._upper_orderflow = AnchoredOrderflowTracker()

    def _log_info_throttled(self, key: str, interval_ms: int, ts_ms: int, message: str, *args) -> None:
        """Log at INFO level at most once per interval_ms for each unique key."""
        last_ts = self._last_throttled_log_ts_ms.get(key)
        if last_ts is None or ts_ms - last_ts >= interval_ms:
            self._last_throttled_log_ts_ms[key] = ts_ms
            logger.info(message, *args)

    def on_tick(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> list[TradeIntent]:
        intents: list[TradeIntent] = []

        # ── Post-entry SL cooldown blocking (pure predicates, pre-setup) ──
        long_blocked_by_post_sl = self._post_entry_sl_cooldown_blocks_side("LONG", ts_ms)
        short_blocked_by_post_sl = self._post_entry_sl_cooldown_blocks_side("SHORT", ts_ms)

        # Defensive: clean up any pre-existing armed state for blocked sides
        # (armed before cooldown was triggered).  _update_armed_state() will
        # prevent new setups from being created for blocked sides.
        self._discard_cooldown_blocked_setups(
            long_blocked=long_blocked_by_post_sl,
            short_blocked=short_blocked_by_post_sl,
            ts_ms=ts_ms,
        )

        self._update_armed_state(
            price,
            ts_ms,
            boll,
            cvd,
            long_blocked_by_post_sl=long_blocked_by_post_sl,
            short_blocked_by_post_sl=short_blocked_by_post_sl,
        )

        runner_exit_intent = self._maybe_trend_runner_market_exit(price, ts_ms, boll, cvd)
        if runner_exit_intent is not None:
            return [runner_exit_intent]

        # TP maintenance is driven by BOLL candle timestamp. This avoids the old
        # problem where a restart/manual TP update delayed the next 15m update.
        tp_intent = self._maybe_update_tp(price, ts_ms, boll, cvd)
        if tp_intent is not None:
            intents.append(tp_intent)

        # Already in a position — update trend detectors and check Trend Upgrade
        if self.state.side is not None:
            # Always refresh trend detectors even when holding position
            self._maybe_update_trend_detectors(price, ts_ms, boll, cvd)
            trend_upgrade_intent = self._maybe_trend_upgrade_addon(price, ts_ms, boll, cvd)
            if trend_upgrade_intent is not None:
                intents.append(trend_upgrade_intent)
            return intents

        if not boll.alert_switch_on:
            return intents

        if not self._cooldown_ok(ts_ms):
            return intents

        # MR setup gates (side effects for reclaim state machine)
        long_setup_ok = (
            not long_blocked_by_post_sl
            and self._long_setup(price, cvd, boll)
        )
        short_setup_ok = (
            not short_blocked_by_post_sl
            and self._short_setup(price, cvd, boll)
        )

        # Regime routing: trend vs mean-reversion arbitration
        regime_decision = self._route_regime(
            price=price, ts_ms=ts_ms, boll=boll, cvd=cvd,
            mr_long_allowed=long_setup_ok,
            mr_short_allowed=short_setup_ok,
        )

        if regime_decision is None:
            # Trend breakout disabled — fall through to existing MR-only logic
            if long_setup_ok:
                intent = self._maybe_open_or_add_long(price, ts_ms, boll, cvd)
                if intent is not None:
                    intents.append(intent)
            if short_setup_ok:
                intent = self._maybe_open_or_add_short(price, ts_ms, boll, cvd)
                if intent is not None:
                    intents.append(intent)
            return intents

        # Execute based on regime decision
        dt = regime_decision.decision_type
        if dt == RegimeDecisionType.TREND_LONG:
            intent = self._maybe_trend_entry("LONG", price, ts_ms, boll, cvd, regime_decision)
            if intent is not None:
                intents.append(intent)
        elif dt == RegimeDecisionType.TREND_SHORT:
            intent = self._maybe_trend_entry("SHORT", price, ts_ms, boll, cvd, regime_decision)
            if intent is not None:
                intents.append(intent)
        elif dt == RegimeDecisionType.MEAN_REVERSION_LONG and long_setup_ok:
            intent = self._maybe_open_or_add_long(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)
        elif dt == RegimeDecisionType.MEAN_REVERSION_SHORT and short_setup_ok:
            intent = self._maybe_open_or_add_short(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)
        elif dt == RegimeDecisionType.CONFLICT_NO_TRADE:
            logger.warning(
                "REGIME_CONFLICT_NO_TRADE | reason=%s confidence=%.2f trend_state=%s",
                regime_decision.reason, regime_decision.confidence,
                regime_decision.trend_state,
            )
        # NO_TRADE, TREND_UPGRADE_* → skip

        return intents

    def _update_armed_state(
        self,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot | None = None,
        *,
        long_blocked_by_post_sl: bool = False,
        short_blocked_by_post_sl: bool = False,
    ) -> None:
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
            # price below lower maps to LONG mean-reversion setup
            if long_blocked_by_post_sl:
                self._log_post_entry_sl_cooldown_discard(side="LONG", ts_ms=ts_ms)
                if self.state.lower_armed:
                    self._reset_lower_armed()
                else:
                    self._log_reclaim_no_entry_reason(
                        side="LOWER",
                        reason="post_entry_sl_cooldown",
                        price=price,
                        boll=boll,
                        cvd=cvd,
                    )

                # Still reset opposite upper setup — market broke opposite side
                if self.state.upper_armed:
                    logger.info("UPPER_ARMED_RESET | reason=opposite_lower_break price=%.4f", price)
                    self._reset_upper_armed()
                return

            self._update_lower_outside(price, ts_ms, boll, cvd)
            if self.state.upper_armed:
                logger.info("UPPER_ARMED_RESET | reason=opposite_lower_break price=%.4f", price)
            self._reset_upper_armed()
            return

        if price > boll.upper:
            # price above upper maps to SHORT mean-reversion setup
            if short_blocked_by_post_sl:
                self._log_post_entry_sl_cooldown_discard(side="SHORT", ts_ms=ts_ms)
                if self.state.upper_armed:
                    self._reset_upper_armed()
                else:
                    self._log_reclaim_no_entry_reason(
                        side="UPPER",
                        reason="post_entry_sl_cooldown",
                        price=price,
                        boll=boll,
                        cvd=cvd,
                    )

                # Still reset opposite lower setup — market broke opposite side
                if self.state.lower_armed:
                    logger.info("LOWER_ARMED_RESET | reason=opposite_upper_break price=%.4f", price)
                    self._reset_lower_armed()
                return

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

        # ── Reclaim V2: abort setup without divergence on inside return ─
        # When price returns inside the band without anchored divergence,
        # the setup can never become a valid entry — abort once and reset.
        # This replaces the old per-minute no_anchored_divergence heartbeat log
        # which is now a one-shot LOWER_RECLAIM_ABORTED / UPPER_RECLAIM_ABORTED.
        if (
            self.config.entry_reclaim_v2_enabled
            and price >= boll.lower
            and price <= boll.upper
        ):
            self._abort_lower_v2_if_inside_without_divergence(price=price, ts_ms=ts_ms, boll=boll)
            self._abort_upper_v2_if_inside_without_divergence(price=price, ts_ms=ts_ms, boll=boll)

    # ── Reclaim V2 abort helpers ──────────────────────────────────────────

    def _abort_lower_v2_if_inside_without_divergence(
        self,
        *,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
    ) -> bool:
        """Abort a lower V2 setup that returned inside without anchored divergence.

        Once price returns inside the lower band without a confirmed
        anchored divergence, the setup can never become a valid entry.
        Log once and reset — no repeated heartbeat.
        """
        if not self.config.entry_reclaim_v2_enabled:
            return False
        if not self.state.lower_outside_observed:
            return False
        if self.state.lower_anchored_divergence_confirmed:
            return False
        # Price has returned inside / above lower band
        if price < boll.lower:
            return False

        logger.info(
            "LOWER_RECLAIM_ABORTED | reason=inside_return_without_anchored_divergence "
            "price=%.4f lower=%.4f middle=%.4f first_extreme=%.4f previous_extreme=%.4f "
            "previous_extreme_cvd=%.4f ts_ms=%s",
            price,
            boll.lower,
            boll.middle,
            self.state.lower_first_extreme_price or 0.0,
            self.state.lower_previous_extreme_price or 0.0,
            self.state.lower_previous_extreme_anchored_cvd or 0.0,
            ts_ms,
        )

        self._reset_lower_armed()
        return True

    def _abort_upper_v2_if_inside_without_divergence(
        self,
        *,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
    ) -> bool:
        """Abort an upper V2 setup that returned inside without anchored divergence.

        Mirror of :meth:`_abort_lower_v2_if_inside_without_divergence` for SHORT.
        """
        if not self.config.entry_reclaim_v2_enabled:
            return False
        if not self.state.upper_outside_observed:
            return False
        if self.state.upper_anchored_divergence_confirmed:
            return False
        # Price has returned inside / below upper band
        if price > boll.upper:
            return False

        logger.info(
            "UPPER_RECLAIM_ABORTED | reason=inside_return_without_anchored_divergence "
            "price=%.4f upper=%.4f middle=%.4f first_extreme=%.4f previous_extreme=%.4f "
            "previous_extreme_cvd=%.4f ts_ms=%s",
            price,
            boll.upper,
            boll.middle,
            self.state.upper_first_extreme_price or 0.0,
            self.state.upper_previous_extreme_price or 0.0,
            self.state.upper_previous_extreme_anchored_cvd or 0.0,
            ts_ms,
        )

        self._reset_upper_armed()
        return True

    def _update_lower_outside(self, price: float, ts_ms: int, boll: BollSnapshot,
                               cvd: CvdSnapshot | None) -> None:
        """Handle a tick where price is below the lower BOLL band."""
        # ── Reclaim V2 path ─────────────────────────────────────────────
        if self.config.entry_reclaim_v2_enabled:
            self._update_lower_outside_v2(price, ts_ms, boll, cvd)
            self._update_lower_deep_enough(boll)
            # Still run old absorption/divergence check for logging only
            if cvd is not None and not self.state.lower_deep_enough:
                pass  # deep_enough evaluated above; skip redundant
            return

        # ── Legacy path ─────────────────────────────────────────────────
        new_extreme_detected = False
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
                "LOWER_ARMED | price=%.4f lower=%.4f middle=%.4f max_armed=%ss%s",
                price, boll.lower, boll.middle,
                self.config.max_armed_seconds, _fast_cvd_str,
            )
            # First arm is the first extreme — allow absorption evaluation
            new_extreme_detected = True
        elif self.state.lower_reclaim_seen:
            # ── Previously reclaimed, now outside again ────────────────
            self._handle_lower_rebreak_after_reclaim(price, ts_ms, boll, cvd)
        else:
            # ── Normal extreme update during outside excursion ─────────
            result = self._update_lower_extreme(price, ts_ms, boll, cvd)
            new_extreme_detected = result.new_extreme_detected

        self._update_lower_deep_enough(boll)
        if cvd is not None:
            self._check_lower_cvd_structure(cvd, boll, ts_ms, new_extreme_detected=new_extreme_detected)

    def _update_upper_outside(self, price: float, ts_ms: int, boll: BollSnapshot,
                               cvd: CvdSnapshot | None) -> None:
        """Handle a tick where price is above the upper BOLL band."""
        # ── Reclaim V2 path ─────────────────────────────────────────────
        if self.config.entry_reclaim_v2_enabled:
            self._update_upper_outside_v2(price, ts_ms, boll, cvd)
            self._update_upper_deep_enough(boll)
            if cvd is not None and not self.state.upper_deep_enough:
                pass
            return

        # ── Legacy path ─────────────────────────────────────────────────
        new_extreme_detected = False
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
                "UPPER_ARMED | price=%.4f upper=%.4f middle=%.4f max_armed=%ss%s",
                price, boll.upper, boll.middle,
                self.config.max_armed_seconds, _fast_cvd_str,
            )
            # First arm is the first extreme — allow absorption evaluation
            new_extreme_detected = True
        elif self.state.upper_reclaim_seen:
            # ── Previously reclaimed, now outside again ────────────────
            self._handle_upper_rebreak_after_reclaim(price, ts_ms, boll, cvd)
        else:
            # ── Normal extreme update during outside excursion ─────────
            result = self._update_upper_extreme(price, ts_ms, boll, cvd)
            new_extreme_detected = result.new_extreme_detected

        self._update_upper_deep_enough(boll)
        if cvd is not None:
            self._check_upper_cvd_structure(cvd, boll, ts_ms, new_extreme_detected=new_extreme_detected)

    def _update_lower_extreme(self, price: float, ts_ms: int, boll: BollSnapshot,
                               cvd: CvdSnapshot | None) -> _ExtremeUpdateResult:
        """Update lower extreme during ongoing outside excursion.

        Returns a structured result so the caller can decide whether to
        evaluate divergence (only when new_extreme_detected=True).
        """
        old_extreme = self.state.lower_extreme_price
        old_fast_cvd = self.state.lower_extreme_fast_cvd
        new_fast_cvd = cvd.fast_cvd if cvd is not None else None

        if old_extreme is None or price >= old_extreme:
            return _ExtremeUpdateResult(
                new_extreme_detected=False,
                old_extreme_price=old_extreme,
                old_extreme_fast_cvd=old_fast_cvd,
                new_fast_cvd=new_fast_cvd,
            )
        buffer_pct = self.config.entry_reclaim_new_extreme_buffer_pct
        if price >= old_extreme * (1 - buffer_pct):
            return _ExtremeUpdateResult(
                new_extreme_detected=False,
                old_extreme_price=old_extreme,
                old_extreme_fast_cvd=old_fast_cvd,
                new_fast_cvd=new_fast_cvd,
            )
        # Real new extreme — update price and timestamp
        # (extreme_fast_cvd is managed by _check_lower_cvd_structure)
        self.state.lower_extreme_price = price
        self.state.lower_extreme_ts_ms = ts_ms
        logger.debug("LOWER_EXTREME_UPDATED | extreme=%.4f price=%.4f", price, price)
        return _ExtremeUpdateResult(
            new_extreme_detected=True,
            old_extreme_price=old_extreme,
            new_extreme_price=price,
            old_extreme_fast_cvd=old_fast_cvd,
            new_fast_cvd=new_fast_cvd,
        )

    def _update_upper_extreme(self, price: float, ts_ms: int, boll: BollSnapshot,
                               cvd: CvdSnapshot | None) -> _ExtremeUpdateResult:
        """Update upper extreme during ongoing outside excursion.

        Returns a structured result so the caller can decide whether to
        evaluate divergence (only when new_extreme_detected=True).
        """
        old_extreme = self.state.upper_extreme_price
        old_fast_cvd = self.state.upper_extreme_fast_cvd
        new_fast_cvd = cvd.fast_cvd if cvd is not None else None

        if old_extreme is None or price <= old_extreme:
            return _ExtremeUpdateResult(
                new_extreme_detected=False,
                old_extreme_price=old_extreme,
                old_extreme_fast_cvd=old_fast_cvd,
                new_fast_cvd=new_fast_cvd,
            )
        buffer_pct = self.config.entry_reclaim_new_extreme_buffer_pct
        if price <= old_extreme * (1 + buffer_pct):
            return _ExtremeUpdateResult(
                new_extreme_detected=False,
                old_extreme_price=old_extreme,
                old_extreme_fast_cvd=old_fast_cvd,
                new_fast_cvd=new_fast_cvd,
            )
        # Real new extreme — update price and timestamp
        # (extreme_fast_cvd is managed by _check_upper_cvd_structure)
        self.state.upper_extreme_price = price
        self.state.upper_extreme_ts_ms = ts_ms
        logger.debug("UPPER_EXTREME_UPDATED | extreme=%.4f price=%.4f", price, price)
        return _ExtremeUpdateResult(
            new_extreme_detected=True,
            old_extreme_price=old_extreme,
            new_extreme_price=price,
            old_extreme_fast_cvd=old_fast_cvd,
            new_fast_cvd=new_fast_cvd,
        )

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
            self.state.lower_reclaim_confirmed_logged = False
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
            self.state.lower_reclaim_confirmed_logged = False
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
            self.state.upper_reclaim_confirmed_logged = False
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
            self.state.upper_reclaim_confirmed_logged = False
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
        """Check whether lower-side CVD structure is confirmed per mode.

        When Reclaim V2 is enabled, only event-anchored cumulative CVD
        divergence can pass this gate.  Old absorption / fast-CVD paths
        are logged but do NOT produce entry intents.
        """
        # ── Reclaim V2: anchored divergence only ────────────────────────
        if self.config.entry_reclaim_v2_enabled:
            return self.state.lower_anchored_divergence_confirmed
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
        """Check whether upper-side CVD structure is confirmed per mode.

        When Reclaim V2 is enabled, only event-anchored cumulative CVD
        divergence can pass this gate.  Old absorption / fast-CVD paths
        are logged but do NOT produce entry intents.
        """
        # ── Reclaim V2: anchored divergence only ────────────────────────
        if self.config.entry_reclaim_v2_enabled:
            return self.state.upper_anchored_divergence_confirmed
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

    def _check_lower_cvd_structure(self, cvd: CvdSnapshot, boll: BollSnapshot, ts_ms: int,
                                    new_extreme_detected: bool = False) -> None:
        """Evaluate both divergence and absorption during lower outside excursion.

        Divergence is only evaluated when new_extreme_detected=True — i.e. only
        on ticks that actually break a new price extreme.  Absorption is still
        evaluated on the first valid extreme tick (when extreme_fast_cvd is
        first recorded) regardless of the flag.
        """
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

        # ── Divergence: only evaluate when price broke a new extreme ──
        if new_extreme_detected and self.config.entry_cvd_divergence_enabled and not self.state.lower_cvd_divergence_confirmed:
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
        elif not new_extreme_detected and self.config.entry_cvd_divergence_enabled and not self.state.lower_cvd_divergence_confirmed:
            # CVD trend update on non-extreme ticks: update extreme_fast_cvd
            # only if CVD confirms (makes new low), to keep reference for
            # future divergence comparison.
            if cvd.fast_cvd < self.state.lower_extreme_fast_cvd:
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

    # ── Reclaim V2 helpers ──────────────────────────────────────────────

    @staticmethod
    def _cumulative_cvd(cvd: CvdSnapshot) -> float:
        """Return the cumulative CVD delta (buy - sell volume)."""
        return float(cvd.cumulative_buy_volume - cvd.cumulative_sell_volume)

    def _select_entry_stop_price(
        self,
        *,
        side: PositionSide,
        entry_price: float,
    ) -> tuple[float | None, str]:
        """Select the entry protective stop-loss price.

        Uses POC-based stop when the extreme is a distant sweep tail;
        otherwise falls back to the classic extreme-based stop.

        Returns (stop_price | None, mode_str) where mode is one of:
        POC_OUTWARD, EXTREME_OUTWARD, EXTREME_CLASSIC.
        """
        if side == "LONG":
            extreme = self.state.lower_extreme_price
            poc = self._lower_poc_price()
        else:
            extreme = self.state.upper_extreme_price
            poc = self._upper_poc_price()

        # ── Classic fallback when extreme is missing ─────────────────────
        if extreme is None or extreme <= 0 or entry_price <= 0:
            return None, "MISSING_EXTREME"

        extreme_stop = (
            extreme * (1.0 - self.config.entry_extreme_stop_buffer_pct)
            if side == "LONG"
            else extreme * (1.0 + self.config.entry_extreme_stop_buffer_pct)
        )

        # ── POC stop candidate ───────────────────────────────────────────
        if not self.config.entry_poc_stop_enabled or poc is None or poc <= 0:
            if side == "LONG":
                sl = extreme * (1.0 - self.config.entry_sl_buffer_pct)
            else:
                sl = extreme * (1.0 + self.config.entry_sl_buffer_pct)
            return sl, "EXTREME_CLASSIC"

        poc_stop = (
            poc * (1.0 - self.config.entry_poc_stop_buffer_pct)
            if side == "LONG"
            else poc * (1.0 + self.config.entry_poc_stop_buffer_pct)
        )

        # ── Entry-extreme distance check ───────────────────────────────────
        if side == "LONG":
            entry_extreme_distance_pct = (entry_price - extreme) / entry_price
            use_poc = (
                entry_extreme_distance_pct >= self.config.entry_poc_stop_min_tail_pct
                and poc_stop < entry_price
                and poc_stop > extreme_stop
            )
        else:
            entry_extreme_distance_pct = (extreme - entry_price) / entry_price
            use_poc = (
                entry_extreme_distance_pct >= self.config.entry_poc_stop_min_tail_pct
                and poc_stop > entry_price
                and poc_stop < extreme_stop
            )

        if use_poc:
            logger.info(
                "ENTRY_SL_SELECTED | side=%s mode=POC_OUTWARD entry=%.4f poc=%.4f extreme=%.4f "
                "entry_extreme_distance_pct=%.6f sl=%.4f",
                side, entry_price, poc, extreme, entry_extreme_distance_pct, poc_stop,
            )
            return poc_stop, "POC_OUTWARD"

        logger.info(
            "ENTRY_SL_SELECTED | side=%s mode=EXTREME_OUTWARD entry=%.4f poc=%.4f extreme=%.4f "
            "entry_extreme_distance_pct=%.6f sl=%.4f",
            side, entry_price, poc, extreme, entry_extreme_distance_pct, extreme_stop,
        )
        return extreme_stop, "EXTREME_OUTWARD"

    def _lower_poc_price(self) -> float | None:
        """Return the lower sweep profile POC price, or None."""
        sp = self.state.lower_sweep_profile
        if sp is None:
            return None
        return sp.poc_price()  # type: ignore[union-attr]

    def _upper_poc_price(self) -> float | None:
        """Return the upper sweep profile POC price, or None."""
        sp = self.state.upper_sweep_profile
        if sp is None:
            return None
        return sp.poc_price()  # type: ignore[union-attr]

    # ── Reclaim V2 lower-side state machine ──────────────────────────────

    def _update_lower_outside_v2(
        self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot | None
    ) -> None:
        """Reclaim V2 lower-side (LONG) state machine.

        Uses ``AnchoredOrderflowTracker`` for episode volume/CVD/extreme
        tracking.  Evaluates anchored CVD divergence when a new lower
        extreme is detected.

        Phases:
          anchor DOWN event → record extremes → divergence confirmed → ARMED.
        """
        if cvd is None:
            return

        cum_cvd = self._cumulative_cvd(cvd)

        # ── Phase 1: first outside tick → anchor the tracker ────────────
        if not self._lower_orderflow.initialised:
            self._lower_orderflow.anchor(
                direction="DOWN",
                ts_ms=ts_ms,
                price=price,
                cumulative_cvd=cum_cvd,
                cumulative_buy_volume=cvd.cumulative_buy_volume,
                cumulative_sell_volume=cvd.cumulative_sell_volume,
            )
            self.state.lower_outside_observed = True
            self.state.lower_anchor_price = price
            self.state.lower_anchor_ts_ms = ts_ms
            self.state.lower_anchor_cumulative_cvd = cum_cvd
            self._ensure_lower_sweep_profile()
            self._previous_lower_extreme_count = 0
            logger.info(
                "LOWER_OUTSIDE_OBSERVED | price=%.4f lower=%.4f anchor_cum_cvd=%.4f ts_ms=%s",
                price, boll.lower, cum_cvd, ts_ms,
            )
            self._record_sweep_volume("LOWER", price, cvd)
            return

        # ── Record sweep volume every tick ─────────────────────────────
        self._record_sweep_volume("LOWER", price, cvd)

        # ── Unlock reclaim retry when price re-enters outside ──────────
        if self.state.lower_reclaim_rejected_until_next_outside:
            logger.info(
                "LOWER_RECLAIM_RETRY_ENABLED | reason=reentered_outside_after_late_cvd_reject "
                "price=%.4f lower=%.4f ts_ms=%s",
                price, boll.lower, ts_ms,
            )
        self.state.lower_reclaim_rejected_until_next_outside = False
        self.state.lower_reclaim_cvd_follow_through_logged = False

        # ── Update orderflow tracker ───────────────────────────────────
        prev_extreme_count: int = getattr(self, "_previous_lower_extreme_count", 0)
        snap = self._lower_orderflow.update(
            ts_ms=ts_ms,
            price=price,
            cumulative_cvd=cum_cvd,
            cumulative_buy_volume=cvd.cumulative_buy_volume,
            cumulative_sell_volume=cvd.cumulative_sell_volume,
        )
        new_extreme_this_tick = snap.new_extreme_count > prev_extreme_count
        self._previous_lower_extreme_count = snap.new_extreme_count

        # Only update extreme_ts_ms on truly new extremes (not every outside tick)
        if new_extreme_this_tick and snap.last_extreme_price > 0:
            self.state.lower_extreme_price = snap.last_extreme_price
            self.state.lower_extreme_ts_ms = ts_ms

        # ── Divergence already confirmed — handle re-break ─────────────
        if self.state.lower_anchored_divergence_confirmed:
            # Cancel in-progress reclaim on re-break
            if self.state.lower_reclaim_seen:
                self.state.lower_reclaim_seen = False
                self.state.lower_reclaim_ts_ms = 0
                self.state.lower_reclaim_confirmed_logged = False
                logger.info(
                    "LOWER_RECLAIM_CANCELLED | reason=re_broke_outside "
                    "price=%.4f lower=%.4f ts_ms=%s",
                    price, boll.lower, ts_ms,
                )
            # Check for new extreme beyond divergence extreme → invalidate old divergence
            div_extreme = self.state.lower_divergence_extreme_price
            if div_extreme is not None and price < div_extreme:
                self.state.lower_anchored_divergence_confirmed = False
                self.state.lower_anchored_divergence_ts_ms = 0
                self.state.lower_armed = False
                self.state.lower_armed_ts_ms = 0
                self.state.lower_cvd_divergence_confirmed = False
                # Promote divergence extreme as previous for re-evaluation
                self.state.lower_previous_extreme_price = div_extreme
                self.state.lower_previous_extreme_ts_ms = self.state.lower_divergence_extreme_ts_ms
                self.state.lower_previous_extreme_anchored_cvd = self.state.lower_divergence_extreme_anchored_cvd
                logger.info(
                    "LOWER_DIVERGENCE_INVALIDATED | reason=new_extreme_beyond_divergence "
                    "price=%.4f div_extreme=%.4f ts_ms=%s",
                    price, div_extreme, ts_ms,
                )
                # fall through to re-evaluate divergence below
            else:
                return

        # ── Phase 2: first extreme — must be deep enough, do NOT arm ────
        if self.state.lower_first_extreme_price is None:
            if snap.new_extreme_count >= 1 and snap.last_extreme_price > 0:
                is_deep_enough = price <= boll.lower * (1 - self.config.min_outside_pct)
                if not is_deep_enough:
                    # Not deep enough yet — continue observing, don't record first extreme
                    return
                self.state.lower_first_extreme_price = snap.last_extreme_price
                self.state.lower_first_extreme_ts_ms = ts_ms
                self.state.lower_first_extreme_anchored_cvd = snap.last_extreme_anchored_cvd
                self.state.lower_previous_extreme_price = snap.last_extreme_price
                self.state.lower_previous_extreme_ts_ms = ts_ms
                self.state.lower_previous_extreme_anchored_cvd = snap.last_extreme_anchored_cvd
                logger.info(
                    "LOWER_FIRST_VALID_EXTREME | price=%.4f anchored_cvd=%.4f deep_enough=True ts_ms=%s",
                    snap.last_extreme_price, snap.last_extreme_anchored_cvd, ts_ms,
                )
            return

        # ── Phase 3: new extreme this tick → evaluate divergence ───────
        if not new_extreme_this_tick:
            return

        prev_price = self.state.lower_previous_extreme_price
        prev_cvd = self.state.lower_previous_extreme_anchored_cvd

        decision = evaluate_anchored_divergence(
            side="LONG",
            previous_extreme_price=prev_price,
            previous_anchored_cvd=prev_cvd,
            current_extreme_price=snap.last_extreme_price,
            current_anchored_cvd=snap.last_extreme_anchored_cvd,
            config=AnchoredDivergenceConfig(
                min_price_extension_pct=self.config.entry_reclaim_new_extreme_buffer_pct,
                min_cvd_recovery=self.config.entry_reclaim_min_cvd_recovery,
            ),
        )

        # Cache divergence check result for snapshot / no-entry logs
        self.state.lower_last_extreme_divergence_confirmed = decision.confirmed
        self.state.lower_last_extreme_divergence_reason = decision.reason

        # ── Cache coherent snapshot pair from this divergence evaluation ──
        self.state.lower_last_snapshot_prev_extreme_price = decision.previous_extreme_price
        self.state.lower_last_snapshot_prev_extreme_cvd = decision.previous_anchored_cvd
        self.state.lower_last_snapshot_curr_extreme_price = decision.current_extreme_price
        self.state.lower_last_snapshot_curr_extreme_cvd = decision.current_anchored_cvd
        self.state.lower_last_snapshot_price_extension_pct = decision.price_extension_pct
        self.state.lower_last_snapshot_cvd_recovery = decision.cvd_recovery
        self.state.lower_last_snapshot_divergence_confirmed = decision.confirmed
        self.state.lower_last_snapshot_divergence_reason = decision.reason
        self.state.lower_extreme_snapshot_pending = True

        if decision.confirmed:
            self.state.lower_anchored_divergence_confirmed = True
            self.state.lower_anchored_divergence_ts_ms = ts_ms
            self.state.lower_armed = True
            self.state.lower_armed_ts_ms = ts_ms
            self.state.lower_first_armed_ts_ms = ts_ms
            self.state.lower_cvd_divergence_confirmed = True
            # Save divergence extreme + reference band
            self.state.lower_divergence_extreme_price = snap.last_extreme_price
            self.state.lower_divergence_extreme_ts_ms = ts_ms
            self.state.lower_divergence_extreme_anchored_cvd = snap.last_extreme_anchored_cvd
            self.state.lower_divergence_ref_lower = boll.lower
            self.state.lower_divergence_ref_middle = boll.middle
            logger.info(
                "LOWER_ANCHORED_CVD_DIVERGENCE_CONFIRMED | "
                "prev_price=%.4f prev_cvd=%.4f curr_price=%.4f curr_cvd=%.4f "
                "cvd_recovery=%.4f price_ext_pct=%.6f ref_lower=%.4f ref_middle=%.4f ts_ms=%s",
                prev_price, prev_cvd,
                snap.last_extreme_price, snap.last_extreme_anchored_cvd,
                decision.cvd_recovery, decision.price_extension_pct,
                boll.lower, boll.middle, ts_ms,
            )
            logger.info("LOWER_ARMED | reason=anchored_divergence price=%.4f ts_ms=%s", price, ts_ms)
        else:
            # Update previous extreme reference for next comparison
            self.state.lower_previous_extreme_price = snap.last_extreme_price
            self.state.lower_previous_extreme_ts_ms = ts_ms
            self.state.lower_previous_extreme_anchored_cvd = snap.last_extreme_anchored_cvd
            logger.debug(
                "LOWER_NEW_EXTREME_NO_DIVERGENCE | price=%.4f cum_cvd=%.4f "
                "reason=%s ts_ms=%s",
                snap.last_extreme_price, cum_cvd, decision.reason, ts_ms,
            )

        # ── Throttled extreme snapshot log (after divergence evaluation) ──
        self._maybe_log_reclaim_extreme_snapshot(
            side="LOWER", price=price, boll=boll, cvd=cvd, snap=snap,
        )

    # ── Reclaim V2 upper-side state machine ──────────────────────────────

    def _update_upper_outside_v2(
        self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot | None
    ) -> None:
        """Reclaim V2 upper-side (SHORT) state machine.

        Uses ``AnchoredOrderflowTracker`` for episode volume/CVD/extreme
        tracking.  Evaluates anchored CVD divergence when a new upper
        extreme is detected.

        Phases:
          anchor UP event → record extremes → divergence confirmed → ARMED.
        """
        if cvd is None:
            return

        cum_cvd = self._cumulative_cvd(cvd)

        # ── Phase 1: first outside tick → anchor the tracker ────────────
        if not self._upper_orderflow.initialised:
            self._upper_orderflow.anchor(
                direction="UP",
                ts_ms=ts_ms,
                price=price,
                cumulative_cvd=cum_cvd,
                cumulative_buy_volume=cvd.cumulative_buy_volume,
                cumulative_sell_volume=cvd.cumulative_sell_volume,
            )
            self.state.upper_outside_observed = True
            self.state.upper_anchor_price = price
            self.state.upper_anchor_ts_ms = ts_ms
            self.state.upper_anchor_cumulative_cvd = cum_cvd
            self._ensure_upper_sweep_profile()
            self._previous_upper_extreme_count = 0
            logger.info(
                "UPPER_OUTSIDE_OBSERVED | price=%.4f upper=%.4f anchor_cum_cvd=%.4f ts_ms=%s",
                price, boll.upper, cum_cvd, ts_ms,
            )
            self._record_sweep_volume("UPPER", price, cvd)
            return

        # ── Record sweep volume every tick ─────────────────────────────
        self._record_sweep_volume("UPPER", price, cvd)

        # ── Unlock reclaim retry when price re-enters outside ──────────
        if self.state.upper_reclaim_rejected_until_next_outside:
            logger.info(
                "UPPER_RECLAIM_RETRY_ENABLED | reason=reentered_outside_after_late_cvd_reject "
                "price=%.4f upper=%.4f ts_ms=%s",
                price, boll.upper, ts_ms,
            )
        self.state.upper_reclaim_rejected_until_next_outside = False
        self.state.upper_reclaim_cvd_follow_through_logged = False

        # ── Update orderflow tracker ───────────────────────────────────
        prev_extreme_count: int = getattr(self, "_previous_upper_extreme_count", 0)
        snap = self._upper_orderflow.update(
            ts_ms=ts_ms,
            price=price,
            cumulative_cvd=cum_cvd,
            cumulative_buy_volume=cvd.cumulative_buy_volume,
            cumulative_sell_volume=cvd.cumulative_sell_volume,
        )
        new_extreme_this_tick = snap.new_extreme_count > prev_extreme_count
        self._previous_upper_extreme_count = snap.new_extreme_count

        # Only update extreme_ts_ms on truly new extremes (not every outside tick)
        if new_extreme_this_tick and snap.last_extreme_price > 0:
            self.state.upper_extreme_price = snap.last_extreme_price
            self.state.upper_extreme_ts_ms = ts_ms

        # ── Divergence already confirmed — handle re-break ─────────────
        if self.state.upper_anchored_divergence_confirmed:
            # Cancel in-progress reclaim on re-break
            if self.state.upper_reclaim_seen:
                self.state.upper_reclaim_seen = False
                self.state.upper_reclaim_ts_ms = 0
                self.state.upper_reclaim_confirmed_logged = False
                logger.info(
                    "UPPER_RECLAIM_CANCELLED | reason=re_broke_outside "
                    "price=%.4f upper=%.4f ts_ms=%s",
                    price, boll.upper, ts_ms,
                )
            # Check for new extreme beyond divergence extreme → invalidate old divergence
            div_extreme = self.state.upper_divergence_extreme_price
            if div_extreme is not None and price > div_extreme:
                self.state.upper_anchored_divergence_confirmed = False
                self.state.upper_anchored_divergence_ts_ms = 0
                self.state.upper_armed = False
                self.state.upper_armed_ts_ms = 0
                self.state.upper_cvd_divergence_confirmed = False
                self.state.upper_previous_extreme_price = div_extreme
                self.state.upper_previous_extreme_ts_ms = self.state.upper_divergence_extreme_ts_ms
                self.state.upper_previous_extreme_anchored_cvd = self.state.upper_divergence_extreme_anchored_cvd
                logger.info(
                    "UPPER_DIVERGENCE_INVALIDATED | reason=new_extreme_beyond_divergence "
                    "price=%.4f div_extreme=%.4f ts_ms=%s",
                    price, div_extreme, ts_ms,
                )
                # fall through to re-evaluate divergence below
            else:
                return

        # ── Phase 2: first extreme — must be deep enough, do NOT arm ────
        if self.state.upper_first_extreme_price is None:
            if snap.new_extreme_count >= 1 and snap.last_extreme_price > 0:
                is_deep_enough = price >= boll.upper * (1 + self.config.min_outside_pct)
                if not is_deep_enough:
                    # Not deep enough yet — continue observing, don't record first extreme
                    return
                self.state.upper_first_extreme_price = snap.last_extreme_price
                self.state.upper_first_extreme_ts_ms = ts_ms
                self.state.upper_first_extreme_anchored_cvd = snap.last_extreme_anchored_cvd
                self.state.upper_previous_extreme_price = snap.last_extreme_price
                self.state.upper_previous_extreme_ts_ms = ts_ms
                self.state.upper_previous_extreme_anchored_cvd = snap.last_extreme_anchored_cvd
                logger.info(
                    "UPPER_FIRST_VALID_EXTREME | price=%.4f anchored_cvd=%.4f deep_enough=True ts_ms=%s",
                    snap.last_extreme_price, snap.last_extreme_anchored_cvd, ts_ms,
                )
            return

        # ── Phase 3: new extreme this tick → evaluate divergence ───────
        if not new_extreme_this_tick:
            return

        prev_price = self.state.upper_previous_extreme_price
        prev_cvd = self.state.upper_previous_extreme_anchored_cvd

        decision = evaluate_anchored_divergence(
            side="SHORT",
            previous_extreme_price=prev_price,
            previous_anchored_cvd=prev_cvd,
            current_extreme_price=snap.last_extreme_price,
            current_anchored_cvd=snap.last_extreme_anchored_cvd,
            config=AnchoredDivergenceConfig(
                min_price_extension_pct=self.config.entry_reclaim_new_extreme_buffer_pct,
                min_cvd_recovery=self.config.entry_reclaim_min_cvd_recovery,
            ),
        )

        # Cache divergence check result for snapshot / no-entry logs
        self.state.upper_last_extreme_divergence_confirmed = decision.confirmed
        self.state.upper_last_extreme_divergence_reason = decision.reason

        # ── Cache coherent snapshot pair from this divergence evaluation ──
        self.state.upper_last_snapshot_prev_extreme_price = decision.previous_extreme_price
        self.state.upper_last_snapshot_prev_extreme_cvd = decision.previous_anchored_cvd
        self.state.upper_last_snapshot_curr_extreme_price = decision.current_extreme_price
        self.state.upper_last_snapshot_curr_extreme_cvd = decision.current_anchored_cvd
        self.state.upper_last_snapshot_price_extension_pct = decision.price_extension_pct
        self.state.upper_last_snapshot_cvd_recovery = decision.cvd_recovery
        self.state.upper_last_snapshot_divergence_confirmed = decision.confirmed
        self.state.upper_last_snapshot_divergence_reason = decision.reason
        self.state.upper_extreme_snapshot_pending = True

        if decision.confirmed:
            self.state.upper_anchored_divergence_confirmed = True
            self.state.upper_anchored_divergence_ts_ms = ts_ms
            self.state.upper_armed = True
            self.state.upper_armed_ts_ms = ts_ms
            self.state.upper_first_armed_ts_ms = ts_ms
            self.state.upper_cvd_divergence_confirmed = True
            # Save divergence extreme + reference band
            self.state.upper_divergence_extreme_price = snap.last_extreme_price
            self.state.upper_divergence_extreme_ts_ms = ts_ms
            self.state.upper_divergence_extreme_anchored_cvd = snap.last_extreme_anchored_cvd
            self.state.upper_divergence_ref_upper = boll.upper
            self.state.upper_divergence_ref_middle = boll.middle
            logger.info(
                "UPPER_ANCHORED_CVD_DIVERGENCE_CONFIRMED | "
                "prev_price=%.4f prev_cvd=%.4f curr_price=%.4f curr_cvd=%.4f "
                "cvd_recovery=%.4f price_ext_pct=%.6f ref_upper=%.4f ref_middle=%.4f ts_ms=%s",
                prev_price, prev_cvd,
                snap.last_extreme_price, snap.last_extreme_anchored_cvd,
                decision.cvd_recovery, decision.price_extension_pct,
                boll.upper, boll.middle, ts_ms,
            )
            logger.info("UPPER_ARMED | reason=anchored_divergence price=%.4f ts_ms=%s", price, ts_ms)
        else:
            self.state.upper_previous_extreme_price = snap.last_extreme_price
            self.state.upper_previous_extreme_ts_ms = ts_ms
            self.state.upper_previous_extreme_anchored_cvd = snap.last_extreme_anchored_cvd
            logger.debug(
                "UPPER_NEW_EXTREME_NO_DIVERGENCE | price=%.4f cum_cvd=%.4f "
                "reason=%s ts_ms=%s",
                snap.last_extreme_price, cum_cvd, decision.reason, ts_ms,
            )

        # ── Throttled extreme snapshot log (after divergence evaluation) ──
        self._maybe_log_reclaim_extreme_snapshot(
            side="UPPER", price=price, boll=boll, cvd=cvd, snap=snap,
        )

    # ── Reclaim V2 observability helpers ─────────────────────────────────

    def _maybe_log_reclaim_extreme_snapshot(
        self,
        *,
        side: str,
        price: float,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        snap: AnchoredOrderflowSnapshot,
    ) -> None:
        """Throttled log of latest extreme + anchored CVD after a new extreme.

        Only prints when a new extreme has been pending AND the interval has
        elapsed since the last snapshot.  If multiple new extremes occur within
        the interval only the latest is printed once.

        Uses the coherent snapshot cache so that prev/curr extreme and CVD
        values come from the SAME divergence evaluation.
        """
        interval_ms = self.config.reclaim_extreme_log_interval_seconds * 1000

        if side == "LOWER":
            if not self.state.lower_extreme_snapshot_pending:
                return
            if cvd.ts_ms - self.state.lower_last_extreme_snapshot_log_ts_ms < interval_ms:
                return

            logger.info(
                "LOWER_EXTREME_SNAPSHOT | "
                "prev_extreme=%.4f prev_cvd=%.4f "
                "curr_extreme=%.4f curr_cvd=%.4f "
                "price_ext_pct=%.6f cvd_recovery=%.4f "
                "new_extreme_count=%s divergence_confirmed=%s divergence_reason=%s "
                "price=%.4f lower=%.4f ts_ms=%s",
                self.state.lower_last_snapshot_prev_extreme_price or 0.0,
                self.state.lower_last_snapshot_prev_extreme_cvd or 0.0,
                self.state.lower_last_snapshot_curr_extreme_price or 0.0,
                self.state.lower_last_snapshot_curr_extreme_cvd or 0.0,
                self.state.lower_last_snapshot_price_extension_pct,
                self.state.lower_last_snapshot_cvd_recovery,
                snap.new_extreme_count,
                self.state.lower_last_snapshot_divergence_confirmed,
                self.state.lower_last_snapshot_divergence_reason or "",
                price,
                boll.lower,
                cvd.ts_ms,
            )

            self.state.lower_last_extreme_snapshot_log_ts_ms = cvd.ts_ms
            self.state.lower_extreme_snapshot_pending = False

        else:  # UPPER
            if not self.state.upper_extreme_snapshot_pending:
                return
            if cvd.ts_ms - self.state.upper_last_extreme_snapshot_log_ts_ms < interval_ms:
                return

            logger.info(
                "UPPER_EXTREME_SNAPSHOT | "
                "prev_extreme=%.4f prev_cvd=%.4f "
                "curr_extreme=%.4f curr_cvd=%.4f "
                "price_ext_pct=%.6f cvd_recovery=%.4f "
                "new_extreme_count=%s divergence_confirmed=%s divergence_reason=%s "
                "price=%.4f upper=%.4f ts_ms=%s",
                self.state.upper_last_snapshot_prev_extreme_price or 0.0,
                self.state.upper_last_snapshot_prev_extreme_cvd or 0.0,
                self.state.upper_last_snapshot_curr_extreme_price or 0.0,
                self.state.upper_last_snapshot_curr_extreme_cvd or 0.0,
                self.state.upper_last_snapshot_price_extension_pct,
                self.state.upper_last_snapshot_cvd_recovery,
                snap.new_extreme_count,
                self.state.upper_last_snapshot_divergence_confirmed,
                self.state.upper_last_snapshot_divergence_reason or "",
                price,
                boll.upper,
                cvd.ts_ms,
            )

            self.state.upper_last_extreme_snapshot_log_ts_ms = cvd.ts_ms
            self.state.upper_extreme_snapshot_pending = False

    def _log_reclaim_no_entry_reason(
        self,
        *,
        side: str,
        reason: str,
        price: float,
        boll: BollSnapshot,
        cvd: CvdSnapshot | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Throttled no-entry reason log — at most once per interval per side+reason."""
        interval_ms = self.config.reclaim_no_entry_log_interval_seconds * 1000

        if side == "LOWER":
            ts_ms = cvd.ts_ms if cvd else 0
            if (
                self.state.lower_last_no_entry_reason == reason
                and ts_ms - self.state.lower_last_no_entry_log_ts_ms < interval_ms
            ):
                return
            self.state.lower_last_no_entry_reason = reason
            self.state.lower_last_no_entry_log_ts_ms = ts_ms

            extra_str = ""
            if extra:
                extra_str = " " + " ".join(f"{k}={v}" for k, v in extra.items())

            logger.info(
                "LOWER_RECLAIM_NO_ENTRY | reason=%s price=%.4f lower=%.4f middle=%.4f "
                "extreme=%.4f divergence_confirmed=%s reclaim_seen=%s "
                "rejected_until_next_outside=%s "
                "anchored_cvd=%.4f div_cvd=%.4f ts_ms=%s%s",
                reason,
                price,
                boll.lower,
                boll.middle,
                self.state.lower_extreme_price or 0.0,
                self.state.lower_anchored_divergence_confirmed,
                self.state.lower_reclaim_seen,
                self.state.lower_reclaim_rejected_until_next_outside,
                self.state.lower_divergence_extreme_anchored_cvd or 0.0,
                self.state.lower_previous_extreme_anchored_cvd or 0.0,
                ts_ms,
                extra_str,
            )

        else:  # UPPER
            ts_ms = cvd.ts_ms if cvd else 0
            if (
                self.state.upper_last_no_entry_reason == reason
                and ts_ms - self.state.upper_last_no_entry_log_ts_ms < interval_ms
            ):
                return
            self.state.upper_last_no_entry_reason = reason
            self.state.upper_last_no_entry_log_ts_ms = ts_ms

            extra_str = ""
            if extra:
                extra_str = " " + " ".join(f"{k}={v}" for k, v in extra.items())

            logger.info(
                "UPPER_RECLAIM_NO_ENTRY | reason=%s price=%.4f upper=%.4f middle=%.4f "
                "extreme=%.4f divergence_confirmed=%s reclaim_seen=%s "
                "rejected_until_next_outside=%s "
                "anchored_cvd=%.4f div_cvd=%.4f ts_ms=%s%s",
                reason,
                price,
                boll.upper,
                boll.middle,
                self.state.upper_extreme_price or 0.0,
                self.state.upper_anchored_divergence_confirmed,
                self.state.upper_reclaim_seen,
                self.state.upper_reclaim_rejected_until_next_outside,
                self.state.upper_divergence_extreme_anchored_cvd or 0.0,
                self.state.upper_previous_extreme_anchored_cvd or 0.0,
                ts_ms,
                extra_str,
            )

    # ── Sweep profile helpers ────────────────────────────────────────────

    def _ensure_lower_sweep_profile(self) -> None:
        if self.state.lower_sweep_profile is None:
            self.state.lower_sweep_profile = SweepVolumeProfile(
                bucket_pct=self.config.entry_sweep_profile_bucket_pct,
            )

    def _ensure_upper_sweep_profile(self) -> None:
        if self.state.upper_sweep_profile is None:
            self.state.upper_sweep_profile = SweepVolumeProfile(
                bucket_pct=self.config.entry_sweep_profile_bucket_pct,
            )

    def _record_sweep_volume(
        self, side: str, price: float, cvd: CvdSnapshot
    ) -> None:
        """Record tick volume in the sweep volume profile."""
        if not self.config.entry_sweep_profile_enabled:
            return
        volume = max(float(cvd.size), 0.0)
        if volume <= 0:
            return
        if side == "LOWER":
            self._ensure_lower_sweep_profile()
            sp = self.state.lower_sweep_profile
        else:
            self._ensure_upper_sweep_profile()
            sp = self.state.upper_sweep_profile
        if sp is not None:
            sp.add(price, volume)  # type: ignore[union-attr]

    # ── Reclaim V2 CVD follow-through ─────────────────────────────────────

    def _check_lower_reclaim_v2_follow_through(
        self, price: float, cvd: CvdSnapshot, boll: BollSnapshot,
    ) -> bool:
        """Evaluate anchored CVD follow-through for LONG reclaim entry.

        Uses the divergence reference band (saved at divergence confirm)
        to define a shallow inside zone.  Returns True only when:
        - price is still in the shallow inside zone, AND
        - event-anchored cumulative CVD has continued to recover beyond
          the divergence-extreme CVD level.
        """
        ref_lower = self.state.lower_divergence_ref_lower or boll.lower
        ref_middle = self.state.lower_divergence_ref_middle or boll.middle
        band_width = ref_middle - ref_lower
        if band_width <= 0:
            return False
        max_entry_price = ref_lower + band_width * self.config.entry_reclaim_max_inside_depth_ratio

        # ── Shallow inside zone check ──────────────────────────────────
        if price > max_entry_price:
            # Price has moved too deep inside before CVD follow-through
            self.state.lower_reclaim_rejected_until_next_outside = True
            self.state.lower_reclaim_seen = False
            self.state.lower_reclaim_ts_ms = 0
            self.state.lower_reclaim_confirmed_logged = False
            self.state.lower_reclaim_cvd_follow_through_logged = False
            self.state.lower_reclaim_cycle_count += 1
            if self.state.lower_reclaim_cycle_count > self.config.entry_max_reclaim_cycles:
                logger.info(
                    "LOWER_SETUP_EXPIRED | reason=max_reclaim_cycles "
                    "cycles=%s max=%s ts_ms=%s",
                    self.state.lower_reclaim_cycle_count,
                    self.config.entry_max_reclaim_cycles,
                    cvd.ts_ms,
                )
                self._reset_lower_armed()
                return False
            logger.info(
                "LOWER_RECLAIM_ATTEMPT_REJECTED | reason=too_deep_inside_before_cvd_follow_through "
                "price=%.4f max_entry_price=%.4f ref_lower=%.4f ref_middle=%.4f band_width=%.4f "
                "cycle=%s ts_ms=%s",
                price, max_entry_price, ref_lower, ref_middle, band_width,
                self.state.lower_reclaim_cycle_count, cvd.ts_ms,
            )
            self._log_reclaim_no_entry_reason(
                side="LOWER",
                reason="too_deep_inside_before_cvd_follow_through",
                price=price,
                boll=boll,
                cvd=cvd,
            )
            return False

        # ── Anchored CVD follow-through ────────────────────────────────
        anchor_cvd = self.state.lower_anchor_cumulative_cvd
        div_cvd = self.state.lower_divergence_extreme_anchored_cvd
        if anchor_cvd is None or div_cvd is None:
            return False

        current_cumulative_cvd = self._cumulative_cvd(cvd)
        reclaim_anchored_cvd = current_cumulative_cvd - anchor_cvd

        if reclaim_anchored_cvd > div_cvd + self.config.entry_reclaim_min_cvd_follow_through:
            if not self.state.lower_reclaim_cvd_follow_through_logged:
                logger.info(
                    "LOWER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED | "
                    "reclaim_anchored_cvd=%.4f div_cvd=%.4f anchor_cvd=%.4f "
                    "price=%.4f ref_lower=%.4f max_entry_price=%.4f ts_ms=%s",
                    reclaim_anchored_cvd, div_cvd, anchor_cvd,
                    price, ref_lower, max_entry_price, cvd.ts_ms,
                )
                self.state.lower_reclaim_cvd_follow_through_logged = True
            return True

        # CVD not yet followed through, still in shallow zone → keep waiting
        logger.debug(
            "LOWER_RECLAIM_WAITING_CVD | reclaim_anchored_cvd=%.4f div_cvd=%.4f "
            "price=%.4f ref_lower=%.4f max_entry_price=%.4f",
            reclaim_anchored_cvd, div_cvd, price, ref_lower, max_entry_price,
        )
        self._log_reclaim_no_entry_reason(
            side="LOWER",
            reason="cvd_follow_through_not_met",
            price=price,
            boll=boll,
            cvd=cvd,
        )
        return False

    def _check_upper_reclaim_v2_follow_through(
        self, price: float, cvd: CvdSnapshot, boll: BollSnapshot,
    ) -> bool:
        """Evaluate anchored CVD follow-through for SHORT reclaim entry.

        Uses the divergence reference band (saved at divergence confirm)
        to define a shallow inside zone.  Returns True only when:
        - price is still in the shallow inside zone, AND
        - event-anchored cumulative CVD has continued to reverse down
          beyond the divergence-extreme CVD level.
        """
        ref_upper = self.state.upper_divergence_ref_upper or boll.upper
        ref_middle = self.state.upper_divergence_ref_middle or boll.middle
        band_width = ref_upper - ref_middle
        if band_width <= 0:
            return False
        min_entry_price = ref_upper - band_width * self.config.entry_reclaim_max_inside_depth_ratio

        # ── Shallow inside zone check ──────────────────────────────────
        if price < min_entry_price:
            # Price has moved too deep inside before CVD follow-through
            self.state.upper_reclaim_rejected_until_next_outside = True
            self.state.upper_reclaim_seen = False
            self.state.upper_reclaim_ts_ms = 0
            self.state.upper_reclaim_confirmed_logged = False
            self.state.upper_reclaim_cvd_follow_through_logged = False
            self.state.upper_reclaim_cycle_count += 1
            if self.state.upper_reclaim_cycle_count > self.config.entry_max_reclaim_cycles:
                logger.info(
                    "UPPER_SETUP_EXPIRED | reason=max_reclaim_cycles "
                    "cycles=%s max=%s ts_ms=%s",
                    self.state.upper_reclaim_cycle_count,
                    self.config.entry_max_reclaim_cycles,
                    cvd.ts_ms,
                )
                self._reset_upper_armed()
                return False
            logger.info(
                "UPPER_RECLAIM_ATTEMPT_REJECTED | reason=too_deep_inside_before_cvd_follow_through "
                "price=%.4f min_entry_price=%.4f ref_upper=%.4f ref_middle=%.4f band_width=%.4f "
                "cycle=%s ts_ms=%s",
                price, min_entry_price, ref_upper, ref_middle, band_width,
                self.state.upper_reclaim_cycle_count, cvd.ts_ms,
            )
            self._log_reclaim_no_entry_reason(
                side="UPPER",
                reason="too_deep_inside_before_cvd_follow_through",
                price=price,
                boll=boll,
                cvd=cvd,
            )
            return False

        # ── Anchored CVD follow-through ────────────────────────────────
        anchor_cvd = self.state.upper_anchor_cumulative_cvd
        div_cvd = self.state.upper_divergence_extreme_anchored_cvd
        if anchor_cvd is None or div_cvd is None:
            return False

        current_cumulative_cvd = self._cumulative_cvd(cvd)
        reclaim_anchored_cvd = current_cumulative_cvd - anchor_cvd

        if reclaim_anchored_cvd < div_cvd - self.config.entry_reclaim_min_cvd_follow_through:
            if not self.state.upper_reclaim_cvd_follow_through_logged:
                logger.info(
                    "UPPER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED | "
                    "reclaim_anchored_cvd=%.4f div_cvd=%.4f anchor_cvd=%.4f "
                    "price=%.4f ref_upper=%.4f min_entry_price=%.4f ts_ms=%s",
                    reclaim_anchored_cvd, div_cvd, anchor_cvd,
                    price, ref_upper, min_entry_price, cvd.ts_ms,
                )
                self.state.upper_reclaim_cvd_follow_through_logged = True
            return True

        # CVD not yet reversed down, still in shallow zone → keep waiting
        logger.debug(
            "UPPER_RECLAIM_WAITING_CVD | reclaim_anchored_cvd=%.4f div_cvd=%.4f "
            "price=%.4f ref_upper=%.4f min_entry_price=%.4f",
            reclaim_anchored_cvd, div_cvd, price, ref_upper, min_entry_price,
        )
        self._log_reclaim_no_entry_reason(
            side="UPPER",
            reason="cvd_follow_through_not_met",
            price=price,
            boll=boll,
            cvd=cvd,
        )
        return False

    def _check_upper_cvd_structure(self, cvd: CvdSnapshot, boll: BollSnapshot, ts_ms: int,
                                    new_extreme_detected: bool = False) -> None:
        """Evaluate both divergence and absorption during upper outside excursion.

        Divergence is only evaluated when new_extreme_detected=True — i.e. only
        on ticks that actually break a new price extreme.  Absorption is still
        evaluated on the first valid extreme tick (when extreme_fast_cvd is
        first recorded) regardless of the flag.
        """
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

        # ── Divergence: only evaluate when price broke a new extreme ──
        if new_extreme_detected and self.config.entry_cvd_divergence_enabled and not self.state.upper_cvd_divergence_confirmed:
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
        elif not new_extreme_detected and self.config.entry_cvd_divergence_enabled and not self.state.upper_cvd_divergence_confirmed:
            # CVD trend update on non-extreme ticks: update extreme_fast_cvd
            # only if CVD confirms (makes new high), to keep reference for
            # future divergence comparison.
            if cvd.fast_cvd > self.state.upper_extreme_fast_cvd:
                self.state.upper_extreme_fast_cvd = cvd.fast_cvd

        # ── Absorption: compare extreme_fast_cvd vs reference_fast_cvd ──
        self._check_upper_absorption(extreme, ts_ms)

    def _expire_armed_state(self, ts_ms: int) -> None:
        """Expire armed state using new-state-machine timeouts.

        When lower_extreme_ts_ms / upper_extreme_ts_ms is available, the
        extreme-to-reclaim timeout (entry_max_extreme_to_reclaim_seconds) is
        used so that a new price extreme resets the 15-minute reclaim window.
        The old max_armed_seconds is kept only as a fallback when NO extreme
        timestamp has been recorded yet (e.g. price went outside but never
        reached min_outside_pct depth).

        The total setup lifetime (entry_max_total_setup_seconds) is enforced
        separately in _update_armed_state() via first_armed_ts_ms.
        """
        _extreme_ms = self.config.entry_max_extreme_to_reclaim_seconds * 1000
        _fallback_ms = self.config.max_armed_seconds * 1000

        # ── Lower side ──────────────────────────────────────────────────
        if self.state.lower_armed:
            _extreme_ts = self.state.lower_extreme_ts_ms
            if _extreme_ts > 0 and ts_ms - _extreme_ts > _extreme_ms:
                logger.info(
                    "LOWER_ARMED_RESET | reason=extreme_to_reclaim_timeout "
                    "extreme_ts_ms=%s age_ms=%s max_ms=%s",
                    _extreme_ts, ts_ms - _extreme_ts, _extreme_ms,
                )
                self._reset_lower_armed()
            elif _extreme_ts <= 0 and ts_ms - self.state.lower_armed_ts_ms > _fallback_ms:
                logger.info(
                    "LOWER_ARMED_RESET | reason=expired_no_extreme age_ms=%s max_ms=%s",
                    ts_ms - self.state.lower_armed_ts_ms, _fallback_ms,
                )
                self._reset_lower_armed()

        # ── Upper side ──────────────────────────────────────────────────
        if self.state.upper_armed:
            _extreme_ts = self.state.upper_extreme_ts_ms
            if _extreme_ts > 0 and ts_ms - _extreme_ts > _extreme_ms:
                logger.info(
                    "UPPER_ARMED_RESET | reason=extreme_to_reclaim_timeout "
                    "extreme_ts_ms=%s age_ms=%s max_ms=%s",
                    _extreme_ts, ts_ms - _extreme_ts, _extreme_ms,
                )
                self._reset_upper_armed()
            elif _extreme_ts <= 0 and ts_ms - self.state.upper_armed_ts_ms > _fallback_ms:
                logger.info(
                    "UPPER_ARMED_RESET | reason=expired_no_extreme age_ms=%s max_ms=%s",
                    ts_ms - self.state.upper_armed_ts_ms, _fallback_ms,
                )
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
        self.state.lower_reclaim_confirmed_logged = False
        self.state.lower_reclaim_cvd_follow_through_logged = False
        self.state.lower_reclaim_rejected_until_next_outside = False
        # ── Clear Reclaim V2 state ──────────────────────────────────────
        self.state.lower_outside_observed = False
        self.state.lower_anchor_price = None
        self.state.lower_anchor_ts_ms = 0
        self.state.lower_anchor_cumulative_cvd = None
        self.state.lower_first_extreme_price = None
        self.state.lower_first_extreme_ts_ms = 0
        self.state.lower_first_extreme_anchored_cvd = None
        self.state.lower_previous_extreme_price = None
        self.state.lower_previous_extreme_ts_ms = 0
        self.state.lower_previous_extreme_anchored_cvd = None
        self.state.lower_anchored_divergence_confirmed = False
        self.state.lower_anchored_divergence_ts_ms = 0
        self.state.lower_divergence_extreme_price = None
        self.state.lower_divergence_extreme_ts_ms = 0
        self.state.lower_divergence_extreme_anchored_cvd = None
        self.state.lower_divergence_ref_lower = None
        self.state.lower_divergence_ref_middle = None
        # ── Clear Reclaim V2 observability state ───────────────────────
        self.state.lower_last_extreme_snapshot_log_ts_ms = 0
        self.state.lower_extreme_snapshot_pending = False
        self.state.lower_last_logged_extreme_price = None
        self.state.lower_last_extreme_divergence_reason = None
        self.state.lower_last_extreme_divergence_confirmed = False
        # ── Clear coherent snapshot cache ────────────────────────────────
        self.state.lower_last_snapshot_prev_extreme_price = None
        self.state.lower_last_snapshot_prev_extreme_cvd = None
        self.state.lower_last_snapshot_curr_extreme_price = None
        self.state.lower_last_snapshot_curr_extreme_cvd = None
        self.state.lower_last_snapshot_price_extension_pct = 0.0
        self.state.lower_last_snapshot_cvd_recovery = 0.0
        self.state.lower_last_snapshot_divergence_confirmed = False
        self.state.lower_last_snapshot_divergence_reason = None
        self.state.lower_last_no_entry_log_ts_ms = 0
        self.state.lower_last_no_entry_reason = None
        if self.state.lower_sweep_profile is not None:
            self.state.lower_sweep_profile.reset()  # type: ignore[union-attr]
            self.state.lower_sweep_profile = None
        self._lower_orderflow.reset()
        if hasattr(self, "_previous_lower_extreme_count"):
            delattr(self, "_previous_lower_extreme_count")

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
        self.state.upper_reclaim_confirmed_logged = False
        self.state.upper_reclaim_cvd_follow_through_logged = False
        self.state.upper_reclaim_rejected_until_next_outside = False
        # ── Clear Reclaim V2 state ──────────────────────────────────────
        self.state.upper_outside_observed = False
        self.state.upper_anchor_price = None
        self.state.upper_anchor_ts_ms = 0
        self.state.upper_anchor_cumulative_cvd = None
        self.state.upper_first_extreme_price = None
        self.state.upper_first_extreme_ts_ms = 0
        self.state.upper_first_extreme_anchored_cvd = None
        self.state.upper_previous_extreme_price = None
        self.state.upper_previous_extreme_ts_ms = 0
        self.state.upper_previous_extreme_anchored_cvd = None
        self.state.upper_anchored_divergence_confirmed = False
        self.state.upper_anchored_divergence_ts_ms = 0
        self.state.upper_divergence_extreme_price = None
        self.state.upper_divergence_extreme_ts_ms = 0
        self.state.upper_divergence_extreme_anchored_cvd = None
        self.state.upper_divergence_ref_upper = None
        self.state.upper_divergence_ref_middle = None
        # ── Clear Reclaim V2 observability state ───────────────────────
        self.state.upper_last_extreme_snapshot_log_ts_ms = 0
        self.state.upper_extreme_snapshot_pending = False
        self.state.upper_last_logged_extreme_price = None
        self.state.upper_last_extreme_divergence_reason = None
        self.state.upper_last_extreme_divergence_confirmed = False
        # ── Clear coherent snapshot cache ────────────────────────────────
        self.state.upper_last_snapshot_prev_extreme_price = None
        self.state.upper_last_snapshot_prev_extreme_cvd = None
        self.state.upper_last_snapshot_curr_extreme_price = None
        self.state.upper_last_snapshot_curr_extreme_cvd = None
        self.state.upper_last_snapshot_price_extension_pct = 0.0
        self.state.upper_last_snapshot_cvd_recovery = 0.0
        self.state.upper_last_snapshot_divergence_confirmed = False
        self.state.upper_last_snapshot_divergence_reason = None
        self.state.upper_last_no_entry_log_ts_ms = 0
        self.state.upper_last_no_entry_reason = None
        if self.state.upper_sweep_profile is not None:
            self.state.upper_sweep_profile.reset()  # type: ignore[union-attr]
            self.state.upper_sweep_profile = None
        self._upper_orderflow.reset()
        if hasattr(self, "_previous_upper_extreme_count"):
            delattr(self, "_previous_upper_extreme_count")

    # ── Reclaim V2 entry setup (immediate near-band CVD follow-through) ──

    def _long_setup_v2(self, price: float, cvd: CvdSnapshot, boll: BollSnapshot) -> bool:
        """Reclaim V2 LONG entry setup — immediate near-band CVD follow-through.

        Unlike the legacy path, V2 does NOT wait for ENTRY_RECLAIM_CONFIRM_SECONDS.
        The first tick where price reclaims inside the divergence reference band
        immediately checks anchored CVD follow-through.
        """
        if not self.state.lower_armed or self.state.lower_extreme_price is None:
            return False
        if not self.state.lower_deep_enough:
            return False
        if not self.state.lower_anchored_divergence_confirmed:
            return False
        if self.state.lower_reclaim_rejected_until_next_outside:
            return False

        ref_lower = self.state.lower_divergence_ref_lower or boll.lower

        # Not reclaimed yet — still outside reference band
        if price < ref_lower * (1 + self.config.entry_reclaim_buffer_pct):
            return False

        # Extreme-to-reclaim timeout
        if self.state.lower_extreme_ts_ms > 0:
            elapsed_ms = cvd.ts_ms - self.state.lower_extreme_ts_ms
            if elapsed_ms > self.config.entry_max_extreme_to_reclaim_seconds * 1000:
                logger.info(
                    "LOWER_SETUP_EXPIRED | reason=extreme_to_reclaim_timeout "
                    "elapsed_ms=%s max_ms=%s ts_ms=%s",
                    elapsed_ms,
                    self.config.entry_max_extreme_to_reclaim_seconds * 1000,
                    cvd.ts_ms,
                )
                self._reset_lower_armed()
                return False

        # First tick inside reference band → immediately check CVD follow-through
        return self._check_lower_reclaim_v2_follow_through(price, cvd, boll)

    def _short_setup_v2(self, price: float, cvd: CvdSnapshot, boll: BollSnapshot) -> bool:
        """Reclaim V2 SHORT entry setup — immediate near-band CVD follow-through.

        Unlike the legacy path, V2 does NOT wait for ENTRY_RECLAIM_CONFIRM_SECONDS.
        The first tick where price reclaims inside the divergence reference band
        immediately checks anchored CVD follow-through.
        """
        if not self.state.upper_armed or self.state.upper_extreme_price is None:
            return False
        if not self.state.upper_deep_enough:
            return False
        if not self.state.upper_anchored_divergence_confirmed:
            return False
        if self.state.upper_reclaim_rejected_until_next_outside:
            return False

        ref_upper = self.state.upper_divergence_ref_upper or boll.upper

        # Not reclaimed yet — still outside reference band
        if price > ref_upper * (1 - self.config.entry_reclaim_buffer_pct):
            return False

        # Extreme-to-reclaim timeout
        if self.state.upper_extreme_ts_ms > 0:
            elapsed_ms = cvd.ts_ms - self.state.upper_extreme_ts_ms
            if elapsed_ms > self.config.entry_max_extreme_to_reclaim_seconds * 1000:
                logger.info(
                    "UPPER_SETUP_EXPIRED | reason=extreme_to_reclaim_timeout "
                    "elapsed_ms=%s max_ms=%s ts_ms=%s",
                    elapsed_ms,
                    self.config.entry_max_extreme_to_reclaim_seconds * 1000,
                    cvd.ts_ms,
                )
                self._reset_upper_armed()
                return False

        # First tick inside reference band → immediately check CVD follow-through
        return self._check_upper_reclaim_v2_follow_through(price, cvd, boll)

    def _long_setup(self, price: float, cvd: CvdSnapshot, boll: BollSnapshot) -> bool:
        if not self.state.lower_armed or self.state.lower_extreme_price is None:
            return False
        if not self.state.lower_deep_enough:
            return False

        # ── CVD structure gate ──────────────────────────────────────────
        if not self._lower_cvd_structure_ok():
            return False

        # ── Reclaim V2: immediate near-band CVD follow-through ──────────
        if self.config.entry_reclaim_v2_enabled:
            return self._long_setup_v2(price, cvd, boll)

        # ── Legacy reclaim soft confirm state machine ───────────────────
        if self.config.entry_reclaim_confirm_seconds > 0:
            tolerance = self.config.entry_reclaim_outside_tolerance_pct

            # Check if price went back outside during confirmation
            if self.state.lower_reclaim_seen and self.state.lower_reclaim_ts_ms > 0:
                if price < boll.lower * (1 - tolerance):
                    # Outside band beyond tolerance → soft reset timer
                    # (the _update_armed_state handles new extreme vs minor breach)
                    # Here we just reset reclaim_ts_ms so the timer restarts
                    self.state.lower_reclaim_ts_ms = 0
                    self.state.lower_reclaim_confirmed_logged = False
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

            if not self.state.lower_reclaim_confirmed_logged:
                logger.info(
                    "LOWER_RECLAIM_CONFIRMED | reclaim_ts_ms=%s ts_ms=%s elapsed_ms=%s",
                    self.state.lower_reclaim_ts_ms, cvd.ts_ms, cvd.ts_ms - self.state.lower_reclaim_ts_ms,
                )
                self.state.lower_reclaim_confirmed_logged = True

        # ── Inside-band reclaim check ───────────────────────────────────
        if self.config.entry_reclaim_inside_band and price < boll.lower * (1 + self.config.entry_reclaim_buffer_pct):
            return False

        # ── Legacy CVD direction check at entry ─────────────────────────
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

        # ── Reclaim V2: immediate near-band CVD follow-through ──────────
        if self.config.entry_reclaim_v2_enabled:
            return self._short_setup_v2(price, cvd, boll)

        # ── Legacy reclaim soft confirm state machine ───────────────────
        if self.config.entry_reclaim_confirm_seconds > 0:
            tolerance = self.config.entry_reclaim_outside_tolerance_pct

            # Check if price went back outside during confirmation
            if self.state.upper_reclaim_seen and self.state.upper_reclaim_ts_ms > 0:
                if price > boll.upper * (1 + tolerance):
                    # Outside band beyond tolerance → soft reset timer
                    self.state.upper_reclaim_ts_ms = 0
                    self.state.upper_reclaim_confirmed_logged = False
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

            if not self.state.upper_reclaim_confirmed_logged:
                logger.info(
                    "UPPER_RECLAIM_CONFIRMED | reclaim_ts_ms=%s ts_ms=%s elapsed_ms=%s",
                    self.state.upper_reclaim_ts_ms, cvd.ts_ms, cvd.ts_ms - self.state.upper_reclaim_ts_ms,
                )
                self.state.upper_reclaim_confirmed_logged = True

        # ── Inside-band reclaim check ───────────────────────────────────
        if self.config.entry_reclaim_inside_band and price > boll.upper * (1 - self.config.entry_reclaim_buffer_pct):
            return False

        # ── Legacy CVD direction check at entry ─────────────────────────
        cvd_direction_ok = (
            (cvd.cross_negative or cvd.cvd_decreasing)
            and cvd.sell_ratio >= self.config.min_sell_ratio
            and cvd.no_new_high
        )
        return cvd_direction_ok


    def _entry_protective_sl_price(
        self, side: PositionSide, *, entry_price: float = 0.0
    ) -> float | None:
        """Return the entry protective stop-loss price.

        When Reclaim V2 is enabled, uses the adaptive POC / Extreme
        selection logic.  Otherwise falls back to the classic extreme *
        (1 ± buffer) formula.

        ``entry_price`` is used for tail-distance calculation in POC
        stop selection.  When 0 (e.g. called before entry price is
        known), the extreme price is used as a proxy.
        """
        if self.config.entry_reclaim_v2_enabled and self.config.entry_poc_stop_enabled:
            _ep = entry_price if entry_price > 0 else (
                self.state.lower_extreme_price if side == "LONG" and self.state.lower_extreme_price
                else self.state.upper_extreme_price if side == "SHORT" and self.state.upper_extreme_price
                else 0.0
            )
            if _ep > 0:
                sl, _mode = self._select_entry_stop_price(
                    side=side, entry_price=_ep,
                )
                if sl is not None:
                    return sl
            # Fall through to classic formula when adaptive returns None

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

    # ------------------------------------------------------------------
    # Trend Breakout wiring
    # ------------------------------------------------------------------

    def _get_trend_assessor(self) -> TrendBreakoutAssessor | None:
        """Lazy-initialise the TrendBreakoutAssessor."""
        if not self.config.trend_breakout_enabled:
            return None
        if self.trend_assessor is None:
            self.trend_assessor = TrendBreakoutAssessor(
                compression_valid_after_seconds=self.config.trend_compression_valid_after_seconds,
                confirm_min_seconds=self.config.trend_confirm_min_seconds,
                confirm_max_seconds=self.config.trend_confirm_max_seconds,
                range_expansion_ratio_min=self.config.trend_range_expansion_ratio_min,
                volume_expansion_ratio_min=self.config.trend_volume_expansion_ratio_min,
                outside_occupancy_min_ratio=self.config.trend_outside_occupancy_min_ratio,
                min_new_extreme_count=self.config.trend_min_new_extreme_count,
                max_inside_reclaim_seconds=self.config.trend_max_inside_reclaim_seconds,
                cvd_min_buy_ratio=self.config.trend_cvd_min_buy_ratio,
                cvd_min_sell_ratio=self.config.trend_cvd_min_sell_ratio,
                cvd_max_pullback_ratio=self.config.trend_cvd_max_pullback_ratio,
                trend_confirm_require_candle_close=self.config.trend_confirm_require_candle_close,
                trend_pre_breakout_pressure_enabled=self.config.trend_pre_breakout_pressure_enabled,
                trend_pre_breakout_min_cvd_ratio=self.config.trend_pre_breakout_min_cvd_ratio,
                trend_pre_breakout_max_pullback_ratio=self.config.trend_pre_breakout_max_pullback_ratio,
                trend_pre_breakout_min_observe_seconds=self.config.trend_pre_breakout_min_observe_seconds,
                trend_pre_breakout_pressure_min_score=self.config.trend_pre_breakout_pressure_min_score,
            )
        return self.trend_assessor

    def _get_trend_metrics_tracker(self) -> TrendBreakoutMetricsTracker | None:
        """Lazy-initialise the TrendBreakoutMetricsTracker."""
        if not self.config.trend_breakout_enabled:
            return None
        if self.trend_metrics_tracker is None:
            self.trend_metrics_tracker = TrendBreakoutMetricsTracker(
                range_expansion_ratio_min=self.config.trend_range_expansion_ratio_min,
                volume_expansion_ratio_min=self.config.trend_volume_expansion_ratio_min,
                outside_occupancy_min_ratio=self.config.trend_outside_occupancy_min_ratio,
                min_new_extreme_count=self.config.trend_min_new_extreme_count,
                max_inside_reclaim_seconds=self.config.trend_max_inside_reclaim_seconds,
                confirm_min_seconds=self.config.trend_confirm_min_seconds,
            )
        return self.trend_metrics_tracker

    def _route_regime(
        self,
        *,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        mr_long_allowed: bool,
        mr_short_allowed: bool,
    ) -> RegimeDecision | None:
        """Run trend detection and route through RegimeRouter.

        Returns:
            A ``RegimeDecision`` or ``None`` when trend_breakout is disabled
            (caller should fall through to existing MR-only logic).
        """
        if not self.config.trend_breakout_enabled:
            return None

        assessor = self._get_trend_assessor()
        if assessor is None:
            return None

        metrics_tracker = self._get_trend_metrics_tracker()
        if metrics_tracker is None:
            return None

        # Feed current BOLL band into trend assessor's ring buffer
        assessor.feed_band(BandSnapshot(
            upper=boll.upper,
            middle=boll.middle,
            lower=boll.lower,
            candle_ts_ms=boll.candle_ts_ms,
            source="closed_or_frozen",
        ))

        # ── Compute current breakout direction from price vs BOLL bands ──
        if price > boll.upper:
            current_direction = "UP"
        elif price < boll.lower:
            current_direction = "DOWN"
        else:
            current_direction = None

        # ── Compute baseline volume rate for sustained volume tracking ──
        # CVD baseline_volume is total volume over burst_baseline_seconds (default 60s)
        baseline_volume_rate = (
            cvd.baseline_volume / 60.0 if cvd.baseline_volume > 0 else 0.0
        )
        tick_volume = float(cvd.buy_volume + cvd.sell_volume)

        # ── Initialise or update metrics tracker based on breakout state ─
        if current_direction is not None and not metrics_tracker.initialised:
            metrics_tracker.anchor(
                ts_ms=ts_ms,
                price=price,
                fast_cvd=cvd.fast_cvd,
                cumulative_buy_volume=cvd.cumulative_buy_volume,
                cumulative_sell_volume=cvd.cumulative_sell_volume,
                direction=current_direction,
                boll_upper=boll.upper,
                boll_lower=boll.lower,
                pre_breakout_range=max(cvd.baseline_range_pct, 0.0),
                pre_breakout_volume=max(cvd.baseline_volume, 0.0),
            )
        elif current_direction is not None and metrics_tracker.initialised:
            if metrics_tracker.direction != current_direction:
                # Direction switched — reset old episode, anchor new
                logger.info(
                    "TREND_METRICS_DIRECTION_SWITCH | old=%s new=%s price=%.4f ts_ms=%s",
                    metrics_tracker.direction, current_direction, price, ts_ms,
                )
                metrics_tracker.reset()
                metrics_tracker.anchor(
                    ts_ms=ts_ms,
                    price=price,
                    fast_cvd=cvd.fast_cvd,
                    cumulative_buy_volume=cvd.cumulative_buy_volume,
                    cumulative_sell_volume=cvd.cumulative_sell_volume,
                    direction=current_direction,
                    boll_upper=boll.upper,
                    boll_lower=boll.lower,
                    pre_breakout_range=max(cvd.baseline_range_pct, 0.0),
                    pre_breakout_volume=max(cvd.baseline_volume, 0.0),
                )
            else:
                metrics_tracker.update(
                    ts_ms=ts_ms,
                    price=price,
                    fast_cvd=cvd.fast_cvd,
                    cumulative_buy_volume=cvd.cumulative_buy_volume,
                    cumulative_sell_volume=cvd.cumulative_sell_volume,
                    boll_upper=boll.upper,
                    boll_middle=boll.middle,
                    boll_lower=boll.lower,
                    burst_move_ratio=cvd.burst_move_ratio,
                    burst_volume_ratio=cvd.burst_volume_ratio,
                    baseline_range_pct=cvd.baseline_range_pct,
                    baseline_volume=cvd.baseline_volume,
                    baseline_volume_rate=baseline_volume_rate,
                    tick_volume=tick_volume,
                )
        elif current_direction is None and metrics_tracker.initialised:
            # Inside reclaim — do NOT reset, continue updating
            metrics_tracker.update(
                ts_ms=ts_ms,
                price=price,
                fast_cvd=cvd.fast_cvd,
                cumulative_buy_volume=cvd.cumulative_buy_volume,
                cumulative_sell_volume=cvd.cumulative_sell_volume,
                boll_upper=boll.upper,
                boll_middle=boll.middle,
                boll_lower=boll.lower,
                burst_move_ratio=cvd.burst_move_ratio,
                burst_volume_ratio=cvd.burst_volume_ratio,
                baseline_range_pct=cvd.baseline_range_pct,
                baseline_volume=cvd.baseline_volume,
                baseline_volume_rate=baseline_volume_rate,
                tick_volume=tick_volume,
            )

        # ── Get real metrics (NOT hardcoded True) ──────────────────────
        m = metrics_tracker.snapshot()
        episode_buy_volume = m.episode_buy_volume
        episode_sell_volume = m.episode_sell_volume
        episode_cvd_max = m.episode_cvd_max
        episode_cvd_min = m.episode_cvd_min

        # Metrics are considered "missing" when the tracker hasn't been fed
        # any real data yet (anchored CVD extremes are zero or equal to anchor).
        metrics_missing = (
            not metrics_tracker.initialised
            or (episode_buy_volume == 0.0 and episode_sell_volume == 0.0
                and episode_cvd_max == episode_cvd_min)
        )

        if metrics_missing and current_direction is not None:
            self._log_info_throttled(
                f"TREND_METRICS_MISSING:{current_direction}:episode_volume_cvd_not_accumulated",
                30_000,
                ts_ms,
                "TREND_METRICS_MISSING | reason=episode_volume_cvd_not_accumulated "
                "direction=%s price=%.4f ts_ms=%s",
                current_direction, price, ts_ms,
            )
            # Pass metrics as-is but mark not confirmed — TrendDetector will
            # not confirm without real CVD data.
            # Still call assess() so state machine can track candidate/expiry.

        # Run trend breakout assessment with REAL metrics
        trend_decision = assessor.assess(
            price=price,
            ts_ms=ts_ms,
            boll_upper=boll.upper,
            boll_middle=boll.middle,
            boll_lower=boll.lower,
            fast_cvd=cvd.fast_cvd,
            buy_ratio=cvd.buy_ratio,
            sell_ratio=cvd.sell_ratio,
            episode_buy_volume=episode_buy_volume,
            episode_sell_volume=episode_sell_volume,
            episode_cvd_max=episode_cvd_max,
            episode_cvd_min=episode_cvd_min,
            range_expansion_passed=m.range_expansion_passed,
            volume_expansion_passed=m.volume_expansion_passed,
            sustained_volume_passed=m.sustained_volume_passed,
            outside_occupancy_passed=m.outside_occupancy_passed,
            new_extreme_count=m.new_extreme_count,
            inside_reclaim_seconds=m.inside_reclaim_seconds,
            price_reclaimed_inside=m.price_reclaimed_inside,
            latest_candle_ts_ms=boll.candle_ts_ms,
            latest_candle_close=boll.close,
            latest_candle_live_mode=boll.live_mode,
        )

        # ── Build router input with REAL cooldown state ────────────────
        cooldown_side: str | None = self.state.post_entry_sl_cooldown_side  # type: ignore[assignment]
        router_input = RouterInput(
            trend_state=trend_decision.trend_state,
            trend_confirmed=trend_decision.is_trend_breakout,
            trend_confirmed_direction=trend_decision.direction,
            trend_candidate_active=(
                trend_decision.trend_assessment is not None
                and trend_decision.trend_assessment.is_candidate
            ),
            trend_candidate_direction=trend_decision.direction,
            trend_failed=(
                trend_decision.trend_assessment is not None
                and trend_decision.trend_assessment.is_failed
            ),
            trend_failure_reason=trend_decision.reason if (
                trend_decision.trend_assessment is not None
                and trend_decision.trend_assessment.is_failed
            ) else None,
            trend_blocks_mean_reversion=trend_decision.blocks_mean_reversion,
            mr_long_allowed=mr_long_allowed,
            mr_short_allowed=mr_short_allowed,
            cooldown_side=cooldown_side,
            cooldown_until_ts_ms=self.state.post_entry_sl_cooldown_until_ts_ms,
            cooldown_scope=self.config.post_entry_sl_cooldown_scope,  # type: ignore[arg-type]
            ts_ms=ts_ms,
        )

        # ── Throttled trend logging ────────────────────────────────────
        self._log_trend_assessment(trend_decision, ts_ms)

        return self.regime_router.route(router_input)

    def _log_trend_assessment(
        self,
        decision,  # TrendBreakoutDecision
        ts_ms: int,
    ) -> None:
        """Emit throttled log messages for trend assessment states."""
        assessment = decision.trend_assessment
        if assessment is None:
            return

        breakout_age = (ts_ms - (decision.breakout_ts_ms or ts_ms)) / 1000.0

        if assessment.is_confirmed:
            self._log_info_throttled(
                f"TREND_CONFIRMED:{decision.direction}",
                60_000,
                ts_ms,
                "TREND_CONFIRMED | direction=%s reason=%s breakout_age_seconds=%.0f "
                "pressure=%s pressure_score=%.2f",
                decision.direction,
                assessment.reason,
                breakout_age,
                assessment.pre_breakout_pressure_direction or "none",
                assessment.pre_breakout_pressure_score,
            )
        elif assessment.is_candidate:
            reason = assessment.reason
            if "waiting_candle_close" in reason or "candle_close" in reason:
                self._log_info_throttled(
                    f"TREND_CONFIRM_WAITING_CANDLE_CLOSE:{decision.direction}",
                    60_000,
                    ts_ms,
                    "TREND_CONFIRM_WAITING_CANDLE_CLOSE | direction=%s "
                    "breakout_age_seconds=%.0f "
                    "confirmed_candle_ts_ms=%d "
                    "pressure=%s pressure_score=%.2f",
                    decision.direction,
                    breakout_age,
                    assessment.confirmed_candle_ts_ms,
                    assessment.pre_breakout_pressure_direction or "none",
                    assessment.pre_breakout_pressure_score,
                )
            else:
                self._log_info_throttled(
                    f"TREND_CANDIDATE_OBSERVED:{decision.direction}",
                    60_000,
                    ts_ms,
                    "TREND_CANDIDATE_OBSERVED | direction=%s reason=%s "
                    "compression_valid=true "
                    "pre_breakout_pressure=%s pressure_score=%.2f",
                    decision.direction,
                    reason,
                    assessment.pre_breakout_pressure_direction or "none",
                    assessment.pre_breakout_pressure_score,
                )
        elif assessment.is_failed:
            self._log_info_throttled(
                f"TREND_CONFIRM_REJECTED:{decision.direction or 'none'}",
                60_000,
                ts_ms,
                "TREND_CONFIRM_REJECTED | reason=%s direction=%s",
                assessment.reason,
                decision.direction or "none",
            )

    def _maybe_trend_entry(
        self,
        side: PositionSide,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        regime_decision: RegimeDecision,
    ) -> TradeIntent | None:
        """Attempt a trend breakout entry.

        Calculates the trend middle SL, checks max stop distance,
        and delegates to EntryAddFlowCoordinator.open_trend_position().
        """
        # ── Post-entry SL cooldown gate ─────────────────────────────────
        if self._post_entry_sl_cooldown_blocks_side(side, ts_ms):
            self._log_post_entry_sl_cooldown_discard(side=side, ts_ms=ts_ms)
            return None

        # Calculate trend middle SL
        trend_sl = calculate_trend_middle_sl(
            boll_middle=boll.middle,
            buffer_pct=self.config.trend_middle_sl_buffer_pct,
            side=side,
        )

        # Check max stop distance
        if price > 0 and self.config.trend_max_stop_distance_pct > 0:
            if side == "LONG":
                stop_distance_pct = (price - trend_sl) / price
            else:
                stop_distance_pct = (trend_sl - price) / price
            if stop_distance_pct > self.config.trend_max_stop_distance_pct:
                logger.info(
                    "TREND_ENTRY_SKIPPED | reason=stop_distance_too_wide "
                    "side=%s price=%.4f sl=%.4f stop_pct=%.6f max_pct=%.6f",
                    side, price, trend_sl, stop_distance_pct,
                    self.config.trend_max_stop_distance_pct,
                )
                return None

        reason = (
            f"趋势入口: {regime_decision.reason} "
            f"方向={side} 置信度={regime_decision.confidence:.2f}"
        )
        return self._entry_add_flow().open_trend_position(
            side=side,
            price=price,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
            reason=reason,
            trend_sl_price=trend_sl,
        )

    def _maybe_trend_upgrade_addon(
        self,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
    ) -> TradeIntent | None:
        """Check for Trend Upgrade Add-on eligibility when holding a position.

        This method is called from ``on_tick`` when ``state.side is not None``.
        It uses the pure-logic ``TrendUpgradeAddonAssessor`` (via
        ``assess_trend_upgrade``) to evaluate:

        1. Whether the existing runner can switch to TREND_UPGRADE management.
        2. Whether an independent risk-sized add-on can be placed.

        Neither step uses legacy ADD_LONG / ADD_SHORT logic.
        """
        from src.strategies.trend_upgrade_addon import (
            TrendUpgradeAddonConfig,
            assess_trend_upgrade,
        )

        state = self.state

        if not self.config.trend_upgrade_addon_enabled:
            return None

        if state.side is None:
            return None

        # ── Check if Three-Stage runner was active ──────────────────────
        # Only positions with Three-Stage runner enabled (or trend_runner_active)
        # qualify for trend upgrade.
        has_three_stage_runner = (
            state.three_stage_runner_enabled_for_position
            or state.trend_runner_active
        )
        if not has_three_stage_runner:
            return None

        # ── Build trend upgrade config ──────────────────────────────────
        addon_config = TrendUpgradeAddonConfig(
            enabled=self.config.trend_upgrade_addon_enabled,
            profit_reinvest_ratio=self.config.trend_upgrade_profit_reinvest_ratio,
            max_addon_risk_pct=self.config.trend_upgrade_max_addon_risk_pct,
            max_total_notional_multiplier=self.config.trend_upgrade_max_total_notional_multiplier,
            require_tp1_consumed=self.config.trend_upgrade_require_tp1_consumed,
            require_tp2_consumed=self.config.trend_upgrade_require_tp2_consumed,
            min_runner_remaining_ratio=self.config.trend_upgrade_min_runner_remaining_ratio,
            min_trend_confidence=self.config.trend_upgrade_min_trend_confidence,
        )

        # ── Get trend assessment from assessor ──────────────────────────
        # Detectors are already updated by _maybe_update_trend_detectors
        # called just before this method from on_tick.
        assessor = self._get_trend_assessor()
        if assessor is None:
            return None

        metrics_tracker = self._get_trend_metrics_tracker()
        if metrics_tracker is None:
            return None

        m = metrics_tracker.snapshot()
        trend_decision = assessor.assess(
            price=price, ts_ms=ts_ms,
            boll_upper=boll.upper, boll_middle=boll.middle, boll_lower=boll.lower,
            fast_cvd=cvd.fast_cvd, buy_ratio=cvd.buy_ratio, sell_ratio=cvd.sell_ratio,
            episode_buy_volume=m.episode_buy_volume,
            episode_sell_volume=m.episode_sell_volume,
            episode_cvd_max=m.episode_cvd_max,
            episode_cvd_min=m.episode_cvd_min,
            range_expansion_passed=m.range_expansion_passed,
            volume_expansion_passed=m.volume_expansion_passed,
            sustained_volume_passed=m.sustained_volume_passed,
            outside_occupancy_passed=m.outside_occupancy_passed,
            new_extreme_count=m.new_extreme_count,
            inside_reclaim_seconds=m.inside_reclaim_seconds,
            price_reclaimed_inside=m.price_reclaimed_inside,
        )

        # ── Cooldown check (same-side) ──────────────────────────────────
        cooldown_active_same_side = (
            self.config.post_entry_sl_cooldown_enabled
            and state.post_entry_sl_cooldown_until_ts_ms > 0
            and ts_ms < state.post_entry_sl_cooldown_until_ts_ms
            and state.post_entry_sl_cooldown_side == state.side
        )

        current_total_notional = (
            state.total_entry_notional if state.total_entry_notional > 0
            else state.total_entry_qty * price
        )

        decision = assess_trend_upgrade(
            config=addon_config,
            has_position=True,
            position_side=state.side,
            entry_regime=state.entry_regime,
            three_stage_runner_enabled_for_position=state.three_stage_runner_enabled_for_position,
            three_stage_tp1_consumed=state.three_stage_tp1_consumed,
            three_stage_tp2_consumed=state.three_stage_tp2_consumed,
            three_stage_tp1_ratio=state.three_stage_tp1_ratio,
            three_stage_tp2_ratio=state.three_stage_tp2_ratio,
            three_stage_runner_ratio=state.three_stage_runner_ratio,
            trend_runner_active=state.trend_runner_active,
            trend_confirmed=trend_decision.is_trend_breakout,
            trend_direction=trend_decision.direction,
            trend_confidence=trend_decision.confidence,
            trend_state=trend_decision.trend_state.value,
            trend_blocks_mean_reversion=trend_decision.blocks_mean_reversion,
            post_entry_sl_cooldown_active_same_side=cooldown_active_same_side,
            delayed_market_exit_armed=state.delayed_market_exit_armed,
            avg_entry_price=state.avg_entry_price,
            total_entry_qty=state.total_entry_qty,
            three_stage_tp1_price=state.three_stage_tp1_price,
            three_stage_tp2_price=state.three_stage_tp2_price,
            equity_usdt=self.sizer.account_equity_usdt,
            leverage=float(self.sizer.config.leverage),
            fee_slippage_buffer_pct=self.config.entry_fee_slippage_buffer_pct,
            max_order_notional_usdt=self.sizer.config.max_order_notional_usdt,
            current_total_notional=current_total_notional,
            boll_middle=boll.middle,
            trend_middle_sl_buffer_pct=self.config.trend_middle_sl_buffer_pct,
            price=price,
            ts_ms=ts_ms,
        )

        if not decision.allowed:
            return None

        # ── Runner upgrade: switch management mode ──────────────────────
        if decision.runner_upgrade_allowed and not state.trend_upgrade_active:
            logger.warning(
                "TREND_UPGRADE_RUNNER_ACTIVATED | side=%s entry_regime=%s "
                "reason=%s confidence=%.2f trend_sl=%.4f",
                state.side, state.entry_regime,
                decision.reason, decision.confidence,
                decision.trend_sl_price if decision.trend_sl_price is not None else 0.0,
            )
            state.trend_upgrade_active = True
            state.position_management_mode = "TREND_UPGRADE"
            if state.entry_regime in (None, "MEAN_REVERSION"):
                state.entry_regime = "TREND_UPGRADE"
            if decision.trend_sl_price is not None:
                state.trend_trailing_sl_price = decision.trend_sl_price
            state.trend_upgrade_last_ts_ms = ts_ms

        # ── Add-on: place independent risk-sized entry ──────────────────
        if decision.addon_allowed and not state.trend_upgrade_addon_active:
            # Rate-limit add-on: at most one Trend Upgrade Add-on per position
            # per upgrade episode.  trend_upgrade_addon_active is set after
            # successful execution, so the decision gate prevents double-fire.
            pass  # Add-on is allowed, proceed below

        if not decision.addon_allowed:
            return None

        if state.trend_upgrade_addon_active:
            # Already have an active add-on — no duplicate
            return None

        if decision.trend_sl_price is None:
            return None

        # ── Calculate add-on size with independent risk budget ──────────
        try:
            addon_size = self.sizer.calculate_with_risk_budget(
                price=price,
                stop_price=decision.trend_sl_price,
                risk_budget_usdt=decision.risk_budget_usdt,
                layer_index=1,
            )
        except RuntimeError:
            logger.exception("TREND_UPGRADE_ADDON_SIZING_FAILED | side=%s", state.side)
            return None

        if addon_size.eth_qty <= 0 or addon_size.notional_usdt <= 0:
            return None

        # ── Calculate intent SL (used in intent only, NOT written to state) ──
        # State is committed ONLY after successful execution in _apply_entry_result.
        from src.strategies.trend_middle_trailing_sl import calculate_trend_middle_sl

        intent_sl = calculate_trend_middle_sl(
            boll_middle=boll.middle,
            buffer_pct=self.config.trend_middle_sl_buffer_pct,
            side=state.side,
        )

        reason = (
            f"趋势升级加仓: {decision.reason} "
            f"方向={state.side} 置信度={decision.confidence:.2f} "
            f"risk_budget={decision.risk_budget_usdt:.2f}"
        )

        intent = TradeIntent(
            intent_type="OPEN_LONG" if state.side == "LONG" else "OPEN_SHORT",
            side=state.side,
            price=price,
            layer_index=state.layers,
            tp_price=0.0,  # No fixed TP for trend upgrade add-on
            reason=reason,
            size=addon_size,
            fast_cvd=cvd.fast_cvd,
            previous_fast_cvd=cvd.previous_fast_cvd,
            buy_ratio=cvd.buy_ratio,
            sell_ratio=cvd.sell_ratio,
            boll_upper=boll.upper,
            boll_middle=boll.middle,
            boll_lower=boll.lower,
            ts_ms=ts_ms,
            avg_entry_price=state.avg_entry_price,
            breakeven_price=state.breakeven_price,
            tp_mode=state.tp_mode,
            tp_plan=state.tp_plan,
            entry_protective_sl_price=intent_sl,
            entry_regime="TREND_UPGRADE_ADDON",
            # ── Carry existing position state ──────────────────────────
            three_stage_tp1_price=state.three_stage_tp1_price,
            three_stage_tp2_price=state.three_stage_tp2_price,
            three_stage_tp1_consumed=state.three_stage_tp1_consumed,
            three_stage_tp2_consumed=state.three_stage_tp2_consumed,
            trend_runner_active=state.trend_runner_active,
        )

        logger.warning(
            "TREND_UPGRADE_ADDON_INTENT | side=%s price=%.4f sl=%.4f "
            "addon_qty=%.6f addon_notional=%.2f risk_budget=%.2f reason=%s",
            state.side, price, intent_sl,
            addon_size.eth_qty, addon_size.notional_usdt,
            decision.risk_budget_usdt, decision.reason,
        )

        return intent

    def _maybe_update_trend_detectors(
        self,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
    ) -> None:
        """Update trend detectors with current tick data.

        Called on every tick regardless of position to keep compression
        detector and metrics tracker current.  Does NOT produce decisions.
        """
        if not self.config.trend_breakout_enabled:
            return

        assessor = self._get_trend_assessor()
        if assessor is None:
            return

        metrics_tracker = self._get_trend_metrics_tracker()
        if metrics_tracker is None:
            return

        assessor.feed_band(BandSnapshot(
            upper=boll.upper, middle=boll.middle, lower=boll.lower,
            candle_ts_ms=boll.candle_ts_ms, source="closed_or_frozen",
        ))

        # ── Compute current breakout direction ──────────────────────────
        if price > boll.upper:
            current_direction = "UP"
        elif price < boll.lower:
            current_direction = "DOWN"
        else:
            current_direction = None

        baseline_volume_rate = (
            cvd.baseline_volume / 60.0 if cvd.baseline_volume > 0 else 0.0
        )
        tick_volume = float(cvd.buy_volume + cvd.sell_volume)

        # ── Initialise or update metrics tracker ────────────────────────
        if current_direction is not None and not metrics_tracker.initialised:
            metrics_tracker.anchor(
                ts_ms=ts_ms, price=price,
                fast_cvd=cvd.fast_cvd,
                cumulative_buy_volume=cvd.cumulative_buy_volume,
                cumulative_sell_volume=cvd.cumulative_sell_volume,
                direction=current_direction,
                boll_upper=boll.upper, boll_lower=boll.lower,
                pre_breakout_range=max(cvd.baseline_range_pct, 0.0),
                pre_breakout_volume=max(cvd.baseline_volume, 0.0),
            )
        elif current_direction is not None and metrics_tracker.initialised:
            if metrics_tracker.direction != current_direction:
                logger.info(
                    "TREND_METRICS_DIRECTION_SWITCH | old=%s new=%s price=%.4f ts_ms=%s",
                    metrics_tracker.direction, current_direction, price, ts_ms,
                )
                metrics_tracker.reset()
                metrics_tracker.anchor(
                    ts_ms=ts_ms, price=price,
                    fast_cvd=cvd.fast_cvd,
                    cumulative_buy_volume=cvd.cumulative_buy_volume,
                    cumulative_sell_volume=cvd.cumulative_sell_volume,
                    direction=current_direction,
                    boll_upper=boll.upper, boll_lower=boll.lower,
                    pre_breakout_range=max(cvd.baseline_range_pct, 0.0),
                    pre_breakout_volume=max(cvd.baseline_volume, 0.0),
                )
            else:
                metrics_tracker.update(
                    ts_ms=ts_ms, price=price,
                    fast_cvd=cvd.fast_cvd,
                    cumulative_buy_volume=cvd.cumulative_buy_volume,
                    cumulative_sell_volume=cvd.cumulative_sell_volume,
                    boll_upper=boll.upper, boll_middle=boll.middle,
                    boll_lower=boll.lower,
                    burst_move_ratio=cvd.burst_move_ratio,
                    burst_volume_ratio=cvd.burst_volume_ratio,
                    baseline_range_pct=cvd.baseline_range_pct,
                    baseline_volume=cvd.baseline_volume,
                    baseline_volume_rate=baseline_volume_rate,
                    tick_volume=tick_volume,
                )
        elif current_direction is None and metrics_tracker.initialised:
            metrics_tracker.update(
                ts_ms=ts_ms, price=price,
                fast_cvd=cvd.fast_cvd,
                cumulative_buy_volume=cvd.cumulative_buy_volume,
                cumulative_sell_volume=cvd.cumulative_sell_volume,
                boll_upper=boll.upper, boll_middle=boll.middle,
                boll_lower=boll.lower,
                burst_move_ratio=cvd.burst_move_ratio,
                burst_volume_ratio=cvd.burst_volume_ratio,
                baseline_range_pct=cvd.baseline_range_pct,
                baseline_volume=cvd.baseline_volume,
                baseline_volume_rate=baseline_volume_rate,
                tick_volume=tick_volume,
            )

    def reset_trend_state(self) -> None:
        """Reset all trend breakout internal state."""
        if self.trend_assessor is not None:
            self.trend_assessor.reset()
        if self.trend_metrics_tracker is not None:
            self.trend_metrics_tracker.reset()

    def feed_trend_band_snapshot(self, band: BandSnapshot) -> None:
        """Feed one historical/closed BOLL band into TrendBreakoutAssessor.

        Used by live startup warmup only.  No trading decision is made here.
        """
        assessor = self._get_trend_assessor()
        if assessor is not None:
            assessor.feed_band(band)

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
            *,
            sl_price_override: float | None = None,
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
            entry_protective_sl_price_override=sl_price_override,
        )

    def _managed_core_contracts_for_intent(self, intent_type: TradeIntentType) -> str | None:
        return self._intent_factory().managed_core_contracts_for_intent(intent_type)

    def _managed_core_eth_qty_for_intent(self, intent_type: TradeIntentType) -> float:
        return self._intent_factory().managed_core_eth_qty_for_intent(intent_type)

    def _protected_order_ids(self) -> tuple[str, ...]:
        return self._intent_factory().protected_order_ids()

    def _cooldown_ok(self, ts_ms: int) -> bool:
        return ts_ms - self.state.last_order_ts_ms >= self.config.order_cooldown_seconds * 1000

    def _post_entry_sl_cooldown_blocks_side(self, side: PositionSide, ts_ms: int) -> bool:
        """Return True when post-entry-SL cooldown blocks this entry side.

        Pure predicate: no active-state logging.
        It may clear expired cooldown once.
        """
        if not self.config.post_entry_sl_cooldown_enabled:
            return False
        until_ts_ms = int(self.state.post_entry_sl_cooldown_until_ts_ms or 0)
        if until_ts_ms <= 0:
            return False

        if ts_ms >= until_ts_ms:
            logger.info(
                "POST_ENTRY_SL_COOLDOWN_EXPIRED | until_ts_ms=%s ts_ms=%s side=%s reason=%s",
                until_ts_ms,
                ts_ms,
                self.state.post_entry_sl_cooldown_side,
                self.state.post_entry_sl_cooldown_reason,
            )
            self.state.post_entry_sl_cooldown_until_ts_ms = 0
            self.state.post_entry_sl_cooldown_side = None
            self.state.post_entry_sl_cooldown_reason = None
            return False

        scope = self.config.post_entry_sl_cooldown_scope
        if scope == "GLOBAL":
            return True
        if scope == "SIDE" and side == self.state.post_entry_sl_cooldown_side:
            return True
        return False

    def _post_entry_sl_cooldown_ok(self, side: PositionSide, ts_ms: int) -> bool:
        """Check whether post-entry-SL cooldown allows a new entry.

        Thin predicate wrapper — no active-state logging.
        Returns True if entry is allowed, False if blocked by cooldown.
        """
        return not self._post_entry_sl_cooldown_blocks_side(side, ts_ms)

    def _log_post_entry_sl_cooldown_discard(self, *, side: PositionSide, ts_ms: int) -> None:
        """Throttled log for setup discard due to post-entry-SL cooldown."""
        until_ts_ms = int(self.state.post_entry_sl_cooldown_until_ts_ms or 0)
        reason = self.state.post_entry_sl_cooldown_reason or ""
        scope = self.config.post_entry_sl_cooldown_scope
        cooldown_side = self.state.post_entry_sl_cooldown_side or ""
        remaining_ms = max(until_ts_ms - ts_ms, 0)

        self._log_info_throttled(
            "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED:"
            f"{side}:{scope}:{cooldown_side}:{until_ts_ms}:{reason}",
            60_000,
            ts_ms,
            "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED | side=%s scope=%s "
            "cooldown_side=%s until_ts_ms=%s remaining_ms=%s reason=%s",
            side,
            scope,
            cooldown_side,
            until_ts_ms,
            remaining_ms,
            reason,
        )

    def _discard_cooldown_blocked_setups(
        self,
        *,
        long_blocked: bool,
        short_blocked: bool,
        ts_ms: int,
    ) -> None:
        """Discard same-side MR setups while post-entry-SL cooldown is active.

        LONG setup maps to lower-side reclaim state.
        SHORT setup maps to upper-side reclaim state.
        """
        if long_blocked and self.state.lower_armed:
            self._log_post_entry_sl_cooldown_discard(side="LONG", ts_ms=ts_ms)
            self._reset_lower_armed()

        if short_blocked and self.state.upper_armed:
            self._log_post_entry_sl_cooldown_discard(side="SHORT", ts_ms=ts_ms)
            self._reset_upper_armed()

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
