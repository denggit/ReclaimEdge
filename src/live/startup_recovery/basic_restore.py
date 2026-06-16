from __future__ import annotations

from typing import Any

from src.execution.trader import PositionSnapshot
from src.live import time_utils as live_time_utils
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    StrategyPositionState,
)
from src.utils.log import get_logger

logger = get_logger(__name__)


def restore_strategy_from_position(
        strategy: BollCvdReclaimStrategy,
        position: PositionSnapshot,
        now_ms: int | None = None,
) -> None:
    if not position.has_position or position.side is None or position.avg_entry_price <= 0:
        return
    now_ms = int(now_ms or live_time_utils.utc_ms())
    strategy.state = StrategyPositionState(
        side=position.side,
        layers=1,
        last_entry_price=position.avg_entry_price,
        tp_price=None,
        last_order_ts_ms=now_ms,
        first_entry_ts_ms=now_ms,
        last_tp_update_ts_ms=0,
        total_entry_qty=position.eth_qty,
        total_entry_notional=position.avg_entry_price * position.eth_qty,
        avg_entry_price=position.avg_entry_price,
    )
    logger.warning(
        "Recovered existing position into strategy state | side=%s contracts=%s eth_qty=%.6f avg_entry=%.4f first_entry_ts_ms=%s last_order_ts_ms=%s",
        position.side,
        position.contracts,
        position.eth_qty,
        position.avg_entry_price,
        now_ms,
        now_ms,
    )


def restore_strategy_from_saved_state(
        strategy: BollCvdReclaimStrategy,
        saved_state: Any,
) -> None:
    tp_plan = getattr(saved_state, "tp_plan", "SINGLE")
    strategy.state = StrategyPositionState(
        side=saved_state.side,
        layers=saved_state.layers,
        last_entry_price=saved_state.last_entry_price,
        tp_price=saved_state.tp_price,
        tp_order_id=getattr(saved_state, "tp_order_id", None),
        tp_order_ids=list(getattr(saved_state, "tp_order_ids", []) or []),
        partial_tp_price=getattr(saved_state, "partial_tp_price", None),
        partial_tp_ratio=getattr(saved_state, "partial_tp_ratio", 0.0),
        tp_plan=tp_plan,
        partial_tp_consumed=getattr(saved_state, "partial_tp_consumed", False),
        last_order_ts_ms=saved_state.last_order_ts_ms,
        first_entry_ts_ms=getattr(saved_state, "first_entry_ts_ms", 0),
        add_freeze_until_ts_ms=getattr(saved_state, "add_freeze_until_ts_ms", 0),
        add_freeze_penalty_count=getattr(saved_state, "add_freeze_penalty_count", 0),
        last_tp_update_ts_ms=saved_state.last_tp_update_ts_ms,
        last_tp_update_candle_ts_ms=saved_state.last_tp_update_candle_ts_ms,
        total_entry_qty=saved_state.total_entry_qty,
        total_entry_notional=saved_state.total_entry_notional,
        avg_entry_price=saved_state.avg_entry_price,
        breakeven_price=saved_state.breakeven_price,
        position_cost_entry_notional=getattr(saved_state, "position_cost_entry_notional", 0.0),
        position_cost_exit_notional=getattr(saved_state, "position_cost_exit_notional", 0.0),
        position_cost_remaining_qty=getattr(saved_state, "position_cost_remaining_qty", 0.0),
        net_remaining_breakeven_price=getattr(saved_state, "net_remaining_breakeven_price", 0.0),
        tp_mode=saved_state.tp_mode,
        entry_protective_sl_price=getattr(saved_state, "entry_protective_sl_price", None),
        entry_protective_sl_order_id=getattr(saved_state, "entry_protective_sl_order_id", None),
        entry_protective_sl_protected=getattr(saved_state, "entry_protective_sl_protected", False),
        middle_runner_enabled_for_position=getattr(saved_state, "middle_runner_enabled_for_position", False),
        middle_runner_pending=getattr(saved_state, "middle_runner_pending", False),
        middle_runner_active=getattr(saved_state, "middle_runner_active", False),
        middle_runner_first_close_ratio=getattr(saved_state, "middle_runner_first_close_ratio", 0.0),
        middle_runner_keep_ratio=getattr(saved_state, "middle_runner_keep_ratio", 0.0),
        middle_runner_first_tp_price=getattr(saved_state, "middle_runner_first_tp_price", None),
        middle_runner_final_tp_price=getattr(saved_state, "middle_runner_final_tp_price", None),
        middle_runner_protective_sl_price=getattr(saved_state, "middle_runner_protective_sl_price", None),
        middle_runner_protective_sl_order_id=getattr(saved_state, "middle_runner_protective_sl_order_id", None),
        middle_runner_extension_triggered=getattr(saved_state, "middle_runner_extension_triggered", False),
        middle_runner_add_disabled=getattr(saved_state, "middle_runner_add_disabled", False),
        middle_runner_size_mismatch_protected=getattr(saved_state, "middle_runner_size_mismatch_protected", False),
        middle_runner_size_mismatch_warning_ts_ms=getattr(saved_state, "middle_runner_size_mismatch_warning_ts_ms", 0),
        middle_runner_sl_diag_last_signature=getattr(saved_state, "middle_runner_sl_diag_last_signature", None),
        middle_runner_sl_time_tighten_candle_count=getattr(saved_state, "middle_runner_sl_time_tighten_candle_count",
                                                           0),
        middle_runner_sl_time_tighten_last_candle_ts_ms=getattr(saved_state,
                                                                "middle_runner_sl_time_tighten_last_candle_ts_ms", 0),
        middle_runner_sl_time_tighten_log_candle_ts_ms=getattr(saved_state,
                                                               "middle_runner_sl_time_tighten_log_candle_ts_ms", 0),
        three_stage_runner_enabled_for_position=getattr(saved_state, "three_stage_runner_enabled_for_position", False),
        three_stage_tp1_price=getattr(saved_state, "three_stage_tp1_price", None),
        three_stage_tp2_price=getattr(saved_state, "three_stage_tp2_price", None),
        three_stage_runner_initial_tp_price=getattr(saved_state, "three_stage_runner_initial_tp_price", None),
        three_stage_tp1_ratio=getattr(saved_state, "three_stage_tp1_ratio", 0.0),
        three_stage_tp2_ratio=getattr(saved_state, "three_stage_tp2_ratio", 0.0),
        three_stage_runner_ratio=getattr(saved_state, "three_stage_runner_ratio", 0.0),
        three_stage_tp1_consumed=getattr(saved_state, "three_stage_tp1_consumed", False),
        three_stage_tp2_consumed=getattr(saved_state, "three_stage_tp2_consumed", False),
        three_stage_post_tp1_protective_sl_price=getattr(saved_state, "three_stage_post_tp1_protective_sl_price", None),
        three_stage_post_tp1_protective_sl_order_id=getattr(saved_state, "three_stage_post_tp1_protective_sl_order_id",
                                                            None),
        three_stage_post_tp1_sl_extension_triggered=getattr(saved_state, "three_stage_post_tp1_sl_extension_triggered",
                                                            False),
        three_stage_post_tp1_protected=getattr(saved_state, "three_stage_post_tp1_protected", False),
        three_stage_post_tp1_sl_diag_last_signature=getattr(saved_state, "three_stage_post_tp1_sl_diag_last_signature",
                                                            None),
        three_stage_post_tp1_sl_time_tighten_candle_count=getattr(saved_state,
                                                                  "three_stage_post_tp1_sl_time_tighten_candle_count",
                                                                  0),
        three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms=getattr(saved_state,
                                                                       "three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms",
                                                                       0),
        three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms=getattr(saved_state,
                                                                      "three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms",
                                                                      0),
        three_stage_pre_tp1_degrade_stage=getattr(saved_state, "three_stage_pre_tp1_degrade_stage", None),
        three_stage_pre_tp1_degraded_ts_ms=getattr(saved_state, "three_stage_pre_tp1_degraded_ts_ms", 0),
        trend_runner_active=getattr(saved_state, "trend_runner_active", False),
        trend_runner_trend_start_ts_ms=getattr(saved_state, "trend_runner_trend_start_ts_ms", 0),
        trend_runner_adjust_count=getattr(saved_state, "trend_runner_adjust_count", 0),
        trend_runner_last_update_candle_ts_ms=getattr(saved_state, "trend_runner_last_update_candle_ts_ms", 0),
        trend_runner_tp_price=getattr(saved_state, "trend_runner_tp_price", None),
        trend_runner_sl_price=getattr(saved_state, "trend_runner_sl_price", None),
        trend_runner_tp_order_id=getattr(saved_state, "trend_runner_tp_order_id", None),
        trend_runner_sl_order_id=getattr(saved_state, "trend_runner_sl_order_id", None),
        trend_runner_exit_reason=getattr(saved_state, "trend_runner_exit_reason", None),
        trend_runner_reverse_candidate=getattr(saved_state, "trend_runner_reverse_candidate", False),
        trend_runner_reverse_start_ts_ms=getattr(saved_state, "trend_runner_reverse_start_ts_ms", 0),
        trend_runner_reverse_start_price=getattr(saved_state, "trend_runner_reverse_start_price", None),
        trend_runner_reverse_extreme_price=getattr(saved_state, "trend_runner_reverse_extreme_price", None),
        trend_runner_reverse_fast_cvd_start=getattr(saved_state, "trend_runner_reverse_fast_cvd_start", 0.0),
        trend_runner_reverse_samples=getattr(saved_state, "trend_runner_reverse_samples", []) or [],
        last_add_skip_log_reason=getattr(saved_state, "last_add_skip_log_reason", None),
        last_add_skip_log_ts_ms=getattr(saved_state, "last_add_skip_log_ts_ms", 0),
        core_contracts=getattr(saved_state, "core_contracts", None),
        core_eth_qty=getattr(saved_state, "core_eth_qty", 0.0),
        startup_force_tp_reconcile=bool(getattr(saved_state, "startup_force_tp_reconcile", False)),
        # ── Middle Bucket Split fields ────────────────────────────────
        middle_bucket_split_active=bool(getattr(saved_state, "middle_bucket_split_active", False)),
        middle_bucket_split_fast_consumed=bool(getattr(saved_state, "middle_bucket_split_fast_consumed", False)),
        middle_bucket_split_slow_consumed=bool(getattr(saved_state, "middle_bucket_split_slow_consumed", False)),
        middle_bucket_split_fast_price=getattr(saved_state, "middle_bucket_split_fast_price", None),
        middle_bucket_split_slow_price=getattr(saved_state, "middle_bucket_split_slow_price", None),
        middle_bucket_split_effective_price=getattr(saved_state, "middle_bucket_split_effective_price", None),
        middle_bucket_split_middle_bucket_ratio=float(
            getattr(saved_state, "middle_bucket_split_middle_bucket_ratio", 0.0) or 0.0),
        middle_bucket_split_fast_ratio_of_bucket=float(
            getattr(saved_state, "middle_bucket_split_fast_ratio_of_bucket", 0.0) or 0.0),
        middle_bucket_split_slow_ratio_of_bucket=float(
            getattr(saved_state, "middle_bucket_split_slow_ratio_of_bucket", 0.0) or 0.0),
        middle_bucket_split_fast_total_ratio=float(
            getattr(saved_state, "middle_bucket_split_fast_total_ratio", 0.0) or 0.0),
        middle_bucket_split_slow_total_ratio=float(
            getattr(saved_state, "middle_bucket_split_slow_total_ratio", 0.0) or 0.0),
        middle_bucket_split_reason=getattr(saved_state, "middle_bucket_split_reason", None),
        middle_bucket_split_fast_sl_price=getattr(saved_state, "middle_bucket_split_fast_sl_price", None),
        middle_bucket_split_fast_sl_order_id=getattr(saved_state, "middle_bucket_split_fast_sl_order_id", None),
        middle_bucket_split_fast_sl_protected=bool(getattr(saved_state, "middle_bucket_split_fast_sl_protected", False)),
        middle_bucket_split_fast_sl_invalid_action_taken=getattr(
            saved_state, "middle_bucket_split_fast_sl_invalid_action_taken", None),
        middle_bucket_split_add_disabled=bool(getattr(saved_state, "middle_bucket_split_add_disabled", False)),
        # ── Post-Entry SL Cooldown ────────────────────────────────────
        post_entry_sl_cooldown_until_ts_ms=int(
            getattr(saved_state, "post_entry_sl_cooldown_until_ts_ms", 0) or 0),
        post_entry_sl_cooldown_side=getattr(saved_state, "post_entry_sl_cooldown_side", None),
        post_entry_sl_cooldown_reason=getattr(saved_state, "post_entry_sl_cooldown_reason", None),
    )
    logger.warning(
        "Recovered strategy state from local disk | position_id=%s side=%s layers=%s avg_entry=%.4f tp=%s partial_tp=%s tp_plan=%s partial_tp_consumed=%s",
        saved_state.position_id,
        saved_state.side,
        saved_state.layers,
        saved_state.avg_entry_price,
        saved_state.tp_price,
        getattr(saved_state, "partial_tp_price", None),
        tp_plan,
        getattr(saved_state, "partial_tp_consumed", False),
    )
