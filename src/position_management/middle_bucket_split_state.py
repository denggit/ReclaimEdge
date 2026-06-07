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
