"""Middle Bucket Split state management — pure helper module.

Provides the canonical ``clear_middle_bucket_split_state`` helper and
disabled-reason constants so that every execution path clears split
state the same way and string literals are not scattered.

This module does NO I/O, NO OKX calls, NO logging, NO journal access,
NO state_store access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

# ── Disabled reason constants ────────────────────────────────────────────

MIDDLE_BUCKET_SPLIT_DISABLED_SUBLEG_TOO_SMALL = "subleg_too_small"
MIDDLE_BUCKET_SPLIT_DISABLED_ORDER_PLACEMENT_FAILED_FALLBACK_FINAL = (
    "split_order_placement_failed_fallback_final"
)
MIDDLE_BUCKET_SPLIT_DISABLED_SIZE_INVALID_RATIOS = "split_size_invalid_ratios"


# ── Actual order mode constants ──────────────────────────────────────────

MIDDLE_BUCKET_SPLIT_ACTUAL_ORDER_MODE_SPLIT_FAST_SLOW = "SPLIT_FAST_SLOW"
MIDDLE_BUCKET_SPLIT_ACTUAL_ORDER_MODE_UNSPLIT_MIDDLE_BUCKET = "UNSPLIT_MIDDLE_BUCKET"
MIDDLE_BUCKET_SPLIT_ACTUAL_ORDER_MODE_FINAL_FULL_SIZE = "FINAL_FULL_SIZE"
MIDDLE_BUCKET_SPLIT_ACTUAL_ORDER_MODE_POST_TP1_TP2_ONLY = "POST_TP1_TP2_ONLY"


# ── State clear helper ──────────────────────────────────────────────────


def clear_middle_bucket_split_state(
    state: StrategyPositionState,
    *,
    reason: str | None = None,
) -> None:
    """Clear every middle_bucket_split_* runtime field on the strategy state.

    Call this when actual exchange orders are no longer split fast/slow,
    for example:

    * split sub-leg too small → fallback to unsplit TP orders
    * split order placement failed → fallback to single final TP
    * flat reset / plan fallback

    The *reason* is stored in ``state.middle_bucket_split_reason`` so
    live_state snapshots can be audited later.
    """
    state.middle_bucket_split_active = False
    state.middle_bucket_split_fast_consumed = False
    state.middle_bucket_split_slow_consumed = False

    state.middle_bucket_split_fast_price = None
    state.middle_bucket_split_slow_price = None
    state.middle_bucket_split_effective_price = None

    state.middle_bucket_split_middle_bucket_ratio = 0.0
    state.middle_bucket_split_fast_ratio_of_bucket = 0.0
    state.middle_bucket_split_slow_ratio_of_bucket = 0.0
    state.middle_bucket_split_fast_total_ratio = 0.0
    state.middle_bucket_split_slow_total_ratio = 0.0

    state.middle_bucket_split_reason = reason

    state.middle_bucket_split_fast_sl_price = None
    state.middle_bucket_split_fast_sl_order_id = None
    state.middle_bucket_split_fast_sl_protected = False
    state.middle_bucket_split_fast_sl_invalid_action_taken = None

    state.middle_bucket_split_add_disabled = False


# ── State degrade helper ─────────────────────────────────────────────────


def degrade_middle_bucket_split_to_single_final(
    state: StrategyPositionState,
    *,
    reason: str | None = None,
) -> None:
    """Clear split state AND degrade the whole TP plan to SINGLE.

    Call this when the actual exchange order is a full-size final TP —
    for example, when split order placement failed and the fallback was
    a single ``("final", full_size, final_tp)`` order.

    This ensures ``state.tp_plan == "SINGLE"`` matches the actual
    OKX order structure.
    """
    # 1. Clear all middle_bucket_split_* fields first.
    clear_middle_bucket_split_state(state, reason=reason)

    # 2. Degrade TP plan to SINGLE.
    state.tp_plan = "SINGLE"

    # 3. Clear partial-TP / split-TP fields.
    state.partial_tp_price = None
    state.partial_tp_ratio = 0.0
    state.partial_tp_consumed = False

    # 4. Clear Three-Stage Runner runtime fields.
    if hasattr(state, "three_stage_tp1_price"):
        state.three_stage_tp1_price = None
    if hasattr(state, "three_stage_tp2_price"):
        state.three_stage_tp2_price = None
    if hasattr(state, "three_stage_tp1_consumed"):
        state.three_stage_tp1_consumed = False
    if hasattr(state, "three_stage_tp2_consumed"):
        state.three_stage_tp2_consumed = False
    if hasattr(state, "three_stage_post_tp1_protective_sl_price"):
        state.three_stage_post_tp1_protective_sl_price = None
    if hasattr(state, "three_stage_post_tp1_protective_sl_order_id"):
        state.three_stage_post_tp1_protective_sl_order_id = None
    if hasattr(state, "three_stage_post_tp1_protected"):
        state.three_stage_post_tp1_protected = False

    # 5. Clear Trend Runner runtime fields.
    if hasattr(state, "trend_runner_active"):
        state.trend_runner_active = False
    if hasattr(state, "trend_runner_tp_price"):
        state.trend_runner_tp_price = None
    if hasattr(state, "trend_runner_sl_price"):
        state.trend_runner_sl_price = None
    if hasattr(state, "trend_runner_tp_order_id"):
        state.trend_runner_tp_order_id = None
    if hasattr(state, "trend_runner_sl_order_id"):
        state.trend_runner_sl_order_id = None

    # 6. Clear Middle Runner runtime fields.
    if hasattr(state, "middle_runner_pending"):
        state.middle_runner_pending = False
    if hasattr(state, "middle_runner_active"):
        state.middle_runner_active = False
    if hasattr(state, "middle_runner_first_tp_price"):
        state.middle_runner_first_tp_price = None
    if hasattr(state, "middle_runner_final_tp_price"):
        state.middle_runner_final_tp_price = None
    if hasattr(state, "middle_runner_protective_sl_price"):
        state.middle_runner_protective_sl_price = None
    if hasattr(state, "middle_runner_protective_sl_order_id"):
        state.middle_runner_protective_sl_order_id = None
