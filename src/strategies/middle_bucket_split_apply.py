"""Middle Bucket Split Apply — reusable helper for entry and TP-update paths.

This module provides the canonical apply functions that write middle-bucket-split
decisions into strategy state.  Both the entry/add flow (EntryAddFlowCoordinator)
and the TP-update flow (TpUpdateCoordinator) share these helpers, so split logic
is defined in exactly one place.

This module does NO I/O — no OKX calls, no email, no journal access.
It DOES write to strategy.state and emit logger warnings (same as before).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.position_management.middle_bucket_split_state import (
    clear_middle_bucket_split_state,
)
from src.strategies import middle_bucket_split as _mbs
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.monitors.boll_band_breakout_monitor import BollSnapshot
    from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy

logger = get_logger(__name__)


@dataclass(frozen=True)
class MiddleBucketSplitApplyResult:
    """Result of applying a middle-bucket-split decision to a TP branch.

    Callers MUST branch on ``action``, not on ``split_active`` or reason strings.
    """

    action: str
    split_active: bool
    partial_tp_price: float | None
    partial_tp_ratio: float
    tp_plan: str | None
    reason: str | None


def _preserve_middle_bucket_split_progress(state: object) -> tuple[bool, bool]:
    """Return existing split progress without mutating strategy state."""
    old_fast_consumed = bool(getattr(state, "middle_bucket_split_fast_consumed", False))
    old_slow_consumed = bool(getattr(state, "middle_bucket_split_slow_consumed", False))
    if old_fast_consumed or old_slow_consumed:
        logger.warning(
            "MIDDLE_BUCKET_SPLIT_PROGRESS_PRESERVED | "
            "fast_consumed=%s slow_consumed=%s",
            old_fast_consumed,
            old_slow_consumed,
        )
    return old_fast_consumed, old_slow_consumed


def _write_preserved_partial_split_state(
    *,
    state: object,
    decision: _mbs.MiddleBucketSplitDecision | None,
    old_fast_consumed: bool,
    old_slow_consumed: bool,
    middle_bucket_ratio: float,
    fast_ratio_of_bucket: float,
    effective_price_fallback: float | None,
) -> float | None:
    """Keep partial split progress alive when the new decision cannot fully split."""
    setattr(state, "middle_bucket_split_active", True)
    setattr(state, "middle_bucket_split_fast_consumed", old_fast_consumed)
    setattr(state, "middle_bucket_split_slow_consumed", old_slow_consumed)

    if decision is not None:
        if decision.action == "SPLIT":
            if decision.fast_price is not None:
                setattr(state, "middle_bucket_split_fast_price", decision.fast_price)
            if decision.slow_price is not None:
                setattr(state, "middle_bucket_split_slow_price", decision.slow_price)
        elif decision.action == "UNSPLIT_SLOW_MIDDLE":
            if not old_slow_consumed and decision.slow_price is not None:
                setattr(state, "middle_bucket_split_slow_price", decision.slow_price)

    current_middle_ratio = float(getattr(state, "middle_bucket_split_middle_bucket_ratio", 0.0) or 0.0)
    current_fast_ratio = float(getattr(state, "middle_bucket_split_fast_ratio_of_bucket", 0.0) or 0.0)
    current_slow_ratio = float(getattr(state, "middle_bucket_split_slow_ratio_of_bucket", 0.0) or 0.0)
    next_middle_ratio = (
        float(decision.middle_bucket_ratio)
        if decision is not None and decision.middle_bucket_ratio > 0.0
        else (middle_bucket_ratio if middle_bucket_ratio > 0.0 else current_middle_ratio)
    )
    next_fast_ratio = (
        float(decision.fast_ratio_of_bucket)
        if decision is not None and 0.0 < decision.fast_ratio_of_bucket < 1.0
        else (fast_ratio_of_bucket if 0.0 < fast_ratio_of_bucket < 1.0 else current_fast_ratio)
    )
    next_slow_ratio = (
        float(decision.slow_ratio_of_bucket)
        if decision is not None and 0.0 < decision.slow_ratio_of_bucket < 1.0
        else (1.0 - next_fast_ratio if 0.0 < next_fast_ratio < 1.0 else current_slow_ratio)
    )
    if next_middle_ratio > 0.0:
        setattr(state, "middle_bucket_split_middle_bucket_ratio", next_middle_ratio)
    if 0.0 < next_fast_ratio < 1.0:
        setattr(state, "middle_bucket_split_fast_ratio_of_bucket", next_fast_ratio)
    if 0.0 < next_slow_ratio < 1.0:
        setattr(state, "middle_bucket_split_slow_ratio_of_bucket", next_slow_ratio)

    current_fast_total = float(getattr(state, "middle_bucket_split_fast_total_ratio", 0.0) or 0.0)
    current_slow_total = float(getattr(state, "middle_bucket_split_slow_total_ratio", 0.0) or 0.0)
    next_fast_total = (
        float(decision.fast_total_ratio)
        if decision is not None and decision.fast_total_ratio > 0.0
        else current_fast_total
    )
    next_slow_total = (
        float(decision.slow_total_ratio)
        if decision is not None and decision.slow_total_ratio > 0.0
        else current_slow_total
    )
    if next_fast_total <= 0.0 and next_middle_ratio > 0.0 and 0.0 < next_fast_ratio < 1.0:
        next_fast_total = next_middle_ratio * next_fast_ratio
    if next_slow_total <= 0.0 and next_middle_ratio > 0.0 and 0.0 < next_slow_ratio < 1.0:
        next_slow_total = next_middle_ratio * next_slow_ratio
    if next_fast_total > 0.0:
        setattr(state, "middle_bucket_split_fast_total_ratio", next_fast_total)
    if next_slow_total > 0.0:
        setattr(state, "middle_bucket_split_slow_total_ratio", next_slow_total)

    effective_price = (
        (decision.effective_price if decision is not None else None)
        or getattr(state, "middle_bucket_split_effective_price", None)
        or effective_price_fallback
    )
    setattr(state, "middle_bucket_split_effective_price", effective_price)
    if decision is not None:
        setattr(state, "middle_bucket_split_reason", decision.reason)
    return effective_price


def _preserved_partial_split_result(
    *,
    state: object,
    decision: _mbs.MiddleBucketSplitDecision | None,
    old_fast_consumed: bool,
    old_slow_consumed: bool,
    middle_bucket_ratio: float,
    fast_ratio_of_bucket: float,
    effective_price_fallback: float | None,
    partial_tp_ratio: float,
    tp_plan: str,
) -> MiddleBucketSplitApplyResult:
    effective_price = _write_preserved_partial_split_state(
        state=state,
        decision=decision,
        old_fast_consumed=old_fast_consumed,
        old_slow_consumed=old_slow_consumed,
        middle_bucket_ratio=middle_bucket_ratio,
        fast_ratio_of_bucket=fast_ratio_of_bucket,
        effective_price_fallback=effective_price_fallback,
    )
    logger.warning(
        "MIDDLE_BUCKET_SPLIT_PARTIAL_PRESERVED | "
        "plan=%s action=%s reason=%s fast_consumed=%s slow_consumed=%s",
        tp_plan,
        decision.action if decision is not None else "DISABLED",
        decision.reason if decision is not None else "middle_bucket_split_config_disabled",
        old_fast_consumed,
        old_slow_consumed,
    )
    return MiddleBucketSplitApplyResult(
        action="SPLIT",
        split_active=True,
        partial_tp_price=effective_price,
        partial_tp_ratio=partial_tp_ratio,
        tp_plan=tp_plan,
        reason=decision.reason if decision is not None else "partial_split_progress_preserved",
    )


# ------------------------------------------------------------------
# Three-Stage Middle Bucket Split
# ------------------------------------------------------------------


def apply_three_stage_middle_bucket_split(
    *,
    strategy: BollCvdReclaimStrategy,
    boll: BollSnapshot,
) -> MiddleBucketSplitApplyResult:
    """Try to enable middle bucket split for the Three-Stage branch.

    Returns a MiddleBucketSplitApplyResult whose ``action`` field drives the
    caller's control flow.  The caller MUST NOT fall back to outer when
    action is ``UNSPLIT_SLOW_MIDDLE`` — it must use BOLL20 middle as the
    full unsplit middle bucket.

    Writes to strategy.state to record the split decision.
    """
    s = strategy
    old_fast_consumed, old_slow_consumed = _preserve_middle_bucket_split_progress(s.state)
    partial_progress_started = old_fast_consumed or old_slow_consumed
    if not s.config.middle_bucket_split_enabled:
        if partial_progress_started:
            tp1_ratio = float(s.state.three_stage_tp1_ratio or 0.0)
            result = _preserved_partial_split_result(
                state=s.state,
                decision=None,
                old_fast_consumed=old_fast_consumed,
                old_slow_consumed=old_slow_consumed,
                middle_bucket_ratio=float(getattr(s.state, "middle_bucket_split_middle_bucket_ratio", 0.0) or 0.0),
                fast_ratio_of_bucket=float(getattr(s.state, "middle_bucket_split_fast_ratio_of_bucket", 0.0) or 0.0),
                effective_price_fallback=getattr(s.state, "middle_bucket_split_effective_price", None),
                partial_tp_ratio=tp1_ratio,
                tp_plan="THREE_STAGE_RUNNER",
            )
            s.state.three_stage_tp1_price = result.partial_tp_price
            return result
        return MiddleBucketSplitApplyResult(
            action="DISABLED",
            split_active=False,
            partial_tp_price=None,
            partial_tp_ratio=0.0,
            tp_plan=None,
            reason="middle_bucket_split_config_disabled",
        )
    if s.state.side is None:
        if partial_progress_started:
            tp1_ratio = float(s.state.three_stage_tp1_ratio or 0.0)
            result = _preserved_partial_split_result(
                state=s.state,
                decision=None,
                old_fast_consumed=old_fast_consumed,
                old_slow_consumed=old_slow_consumed,
                middle_bucket_ratio=float(getattr(s.state, "middle_bucket_split_middle_bucket_ratio", 0.0) or 0.0),
                fast_ratio_of_bucket=float(getattr(s.state, "middle_bucket_split_fast_ratio_of_bucket", 0.0) or 0.0),
                effective_price_fallback=getattr(s.state, "middle_bucket_split_effective_price", None),
                partial_tp_ratio=tp1_ratio,
                tp_plan="THREE_STAGE_RUNNER",
            )
            s.state.three_stage_tp1_price = result.partial_tp_price
            return result
        return MiddleBucketSplitApplyResult(
            action="INVALID",
            split_active=False,
            partial_tp_price=None,
            partial_tp_ratio=0.0,
            tp_plan=None,
            reason="side_is_none",
        )

    middle_bucket_ratio = float(s.state.three_stage_tp1_ratio or 0.0)
    fast_middle_price = getattr(boll, "tp_middle", None)
    slow_middle_price = float(boll.middle) if boll.middle else None
    effective_be = s._effective_breakeven_for_tp_selection(s.state.side)
    fast_ratio_of_bucket = float(s.config.middle_bucket_split_fast_ratio)

    decision = _mbs.build_middle_bucket_split_decision(
        enabled=True,
        side=s.state.side,
        middle_bucket_ratio=middle_bucket_ratio,
        fast_ratio_of_bucket=fast_ratio_of_bucket,
        fast_middle_price=fast_middle_price,
        slow_middle_price=slow_middle_price,
        effective_breakeven=effective_be,
        min_net_profit_pct=s.config.tp_min_net_profit_pct,
    )

    if decision.action == "SPLIT":
        s.state.middle_bucket_split_active = True
        s.state.middle_bucket_split_fast_consumed = old_fast_consumed
        s.state.middle_bucket_split_slow_consumed = old_slow_consumed
        s.state.middle_bucket_split_fast_price = decision.fast_price
        s.state.middle_bucket_split_slow_price = decision.slow_price
        s.state.middle_bucket_split_effective_price = decision.effective_price
        s.state.middle_bucket_split_middle_bucket_ratio = decision.middle_bucket_ratio
        s.state.middle_bucket_split_fast_ratio_of_bucket = decision.fast_ratio_of_bucket
        s.state.middle_bucket_split_slow_ratio_of_bucket = decision.slow_ratio_of_bucket
        s.state.middle_bucket_split_fast_total_ratio = decision.fast_total_ratio
        s.state.middle_bucket_split_slow_total_ratio = decision.slow_total_ratio
        s.state.middle_bucket_split_reason = decision.reason
        s.state.three_stage_tp1_price = decision.effective_price
        candle_ts = getattr(boll, "candle_ts_ms", 0)
        logger.warning(
            "MIDDLE_BUCKET_SPLIT_SELECTED | "
            "plan=THREE_STAGE_RUNNER side=%s middle_bucket_ratio=%.4f "
            "fast_ratio_of_bucket=%.4f slow_ratio_of_bucket=%.4f "
            "fast_total_ratio=%.4f slow_total_ratio=%.4f "
            "fast_price=%.4f slow_price=%.4f effective_price=%.4f "
            "reason=%s candle_ts=%s",
            s.state.side,
            decision.middle_bucket_ratio,
            decision.fast_ratio_of_bucket,
            decision.slow_ratio_of_bucket,
            decision.fast_total_ratio,
            decision.slow_total_ratio,
            float(decision.fast_price or 0.0),
            float(decision.slow_price or 0.0),
            float(decision.effective_price or 0.0),
            decision.reason,
            candle_ts,
        )
        tp1_ratio = s.state.three_stage_tp1_ratio
        return MiddleBucketSplitApplyResult(
            action="SPLIT",
            split_active=True,
            partial_tp_price=decision.effective_price,
            partial_tp_ratio=tp1_ratio,
            tp_plan="THREE_STAGE_RUNNER",
            reason=decision.reason,
        )

    if decision.action == "UNSPLIT_SLOW_MIDDLE":
        if partial_progress_started:
            tp1_ratio = float(s.state.three_stage_tp1_ratio or 0.0)
            result = _preserved_partial_split_result(
                state=s.state,
                decision=decision,
                old_fast_consumed=old_fast_consumed,
                old_slow_consumed=old_slow_consumed,
                middle_bucket_ratio=middle_bucket_ratio,
                fast_ratio_of_bucket=fast_ratio_of_bucket,
                effective_price_fallback=slow_middle_price,
                partial_tp_ratio=tp1_ratio,
                tp_plan="THREE_STAGE_RUNNER",
            )
            s.state.three_stage_tp1_price = result.partial_tp_price
            return result
        clear_middle_bucket_split_state(s.state, reason=None)
        s.state.three_stage_tp1_price = slow_middle_price
        candle_ts = getattr(boll, "candle_ts_ms", 0)
        logger.warning(
            "MIDDLE_BUCKET_SPLIT_SKIPPED | "
            "plan=THREE_STAGE_RUNNER side=%s reason=%s action=%s "
            "fast_price=%s slow_price=%.4f required_price=%.4f "
            "using_full_middle_bucket_at_20 candle_ts=%s",
            s.state.side,
            decision.reason,
            decision.action,
            f"{float(decision.fast_price or 0.0):.4f}" if decision.fast_price is not None else "-",
            float(decision.slow_price or 0.0),
            float(decision.required_price or 0.0),
            candle_ts,
        )
        tp1_ratio = s.state.three_stage_tp1_ratio
        return MiddleBucketSplitApplyResult(
            action="UNSPLIT_SLOW_MIDDLE",
            split_active=False,
            partial_tp_price=slow_middle_price,
            partial_tp_ratio=tp1_ratio,
            tp_plan="THREE_STAGE_RUNNER",
            reason=decision.reason,
        )

    # FALLBACK_OUTER, INVALID, DISABLED — reset split state, return old behaviour
    if partial_progress_started:
        tp1_ratio = float(s.state.three_stage_tp1_ratio or 0.0)
        result = _preserved_partial_split_result(
            state=s.state,
            decision=decision,
            old_fast_consumed=old_fast_consumed,
            old_slow_consumed=old_slow_consumed,
            middle_bucket_ratio=middle_bucket_ratio,
            fast_ratio_of_bucket=fast_ratio_of_bucket,
            effective_price_fallback=getattr(s.state, "middle_bucket_split_effective_price", None) or slow_middle_price,
            partial_tp_ratio=tp1_ratio,
            tp_plan="THREE_STAGE_RUNNER",
        )
        s.state.three_stage_tp1_price = result.partial_tp_price
        return result
    clear_middle_bucket_split_state(s.state, reason=None)
    return MiddleBucketSplitApplyResult(
        action=decision.action,
        split_active=False,
        partial_tp_price=None,
        partial_tp_ratio=0.0,
        tp_plan=None,
        reason=decision.reason,
    )


# ------------------------------------------------------------------
# Middle Runner Bucket Split
# ------------------------------------------------------------------


def apply_middle_runner_bucket_split(
    *,
    strategy: BollCvdReclaimStrategy,
    boll: BollSnapshot,
) -> MiddleBucketSplitApplyResult:
    """Try to enable middle bucket split for the Middle Runner branch.

    Returns a MiddleBucketSplitApplyResult whose ``action`` field drives the
    caller's control flow.  The caller MUST NOT fall back to outer when
    action is ``UNSPLIT_SLOW_MIDDLE`` — it must use BOLL20 middle as the
    full unsplit middle bucket.

    Writes to strategy.state to record the split decision.
    """
    s = strategy
    old_fast_consumed, old_slow_consumed = _preserve_middle_bucket_split_progress(s.state)
    partial_progress_started = old_fast_consumed or old_slow_consumed
    if not s.config.middle_bucket_split_enabled:
        if partial_progress_started:
            partial_tp_ratio_val = s.state.middle_runner_first_close_ratio or min(
                max(s.config.middle_runner_first_close_ratio, 0.1), 0.95)
            result = _preserved_partial_split_result(
                state=s.state,
                decision=None,
                old_fast_consumed=old_fast_consumed,
                old_slow_consumed=old_slow_consumed,
                middle_bucket_ratio=float(getattr(s.state, "middle_bucket_split_middle_bucket_ratio", 0.0) or 0.0),
                fast_ratio_of_bucket=float(getattr(s.state, "middle_bucket_split_fast_ratio_of_bucket", 0.0) or 0.0),
                effective_price_fallback=getattr(s.state, "middle_bucket_split_effective_price", None),
                partial_tp_ratio=partial_tp_ratio_val,
                tp_plan="MIDDLE_RUNNER",
            )
            s.state.middle_runner_first_tp_price = result.partial_tp_price
            s.state.partial_tp_price = result.partial_tp_price
            return result
        return MiddleBucketSplitApplyResult(
            action="DISABLED",
            split_active=False,
            partial_tp_price=None,
            partial_tp_ratio=0.0,
            tp_plan=None,
            reason="middle_bucket_split_config_disabled",
        )
    if s.state.side is None:
        if partial_progress_started:
            partial_tp_ratio_val = s.state.middle_runner_first_close_ratio or min(
                max(s.config.middle_runner_first_close_ratio, 0.1), 0.95)
            result = _preserved_partial_split_result(
                state=s.state,
                decision=None,
                old_fast_consumed=old_fast_consumed,
                old_slow_consumed=old_slow_consumed,
                middle_bucket_ratio=float(getattr(s.state, "middle_bucket_split_middle_bucket_ratio", 0.0) or 0.0),
                fast_ratio_of_bucket=float(getattr(s.state, "middle_bucket_split_fast_ratio_of_bucket", 0.0) or 0.0),
                effective_price_fallback=getattr(s.state, "middle_bucket_split_effective_price", None),
                partial_tp_ratio=partial_tp_ratio_val,
                tp_plan="MIDDLE_RUNNER",
            )
            s.state.middle_runner_first_tp_price = result.partial_tp_price
            s.state.partial_tp_price = result.partial_tp_price
            return result
        return MiddleBucketSplitApplyResult(
            action="INVALID",
            split_active=False,
            partial_tp_price=None,
            partial_tp_ratio=0.0,
            tp_plan=None,
            reason="side_is_none",
        )

    middle_bucket_ratio = float(s.state.middle_runner_first_close_ratio or 0.0)
    if middle_bucket_ratio <= 0.0:
        middle_bucket_ratio = min(max(float(s.config.middle_runner_first_close_ratio), 0.1), 0.95)
    fast_middle_price = getattr(boll, "tp_middle", None)
    slow_middle_price = float(boll.middle) if boll.middle else None
    effective_be = s._effective_breakeven_for_tp_selection(s.state.side)
    fast_ratio_of_bucket = float(s.config.middle_bucket_split_fast_ratio)

    decision = _mbs.build_middle_bucket_split_decision(
        enabled=True,
        side=s.state.side,
        middle_bucket_ratio=middle_bucket_ratio,
        fast_ratio_of_bucket=fast_ratio_of_bucket,
        fast_middle_price=fast_middle_price,
        slow_middle_price=slow_middle_price,
        effective_breakeven=effective_be,
        min_net_profit_pct=s.config.tp_min_net_profit_pct,
    )

    if decision.action == "SPLIT":
        s.state.middle_bucket_split_active = True
        s.state.middle_bucket_split_fast_consumed = old_fast_consumed
        s.state.middle_bucket_split_slow_consumed = old_slow_consumed
        s.state.middle_bucket_split_fast_price = decision.fast_price
        s.state.middle_bucket_split_slow_price = decision.slow_price
        s.state.middle_bucket_split_effective_price = decision.effective_price
        s.state.middle_bucket_split_middle_bucket_ratio = decision.middle_bucket_ratio
        s.state.middle_bucket_split_fast_ratio_of_bucket = decision.fast_ratio_of_bucket
        s.state.middle_bucket_split_slow_ratio_of_bucket = decision.slow_ratio_of_bucket
        s.state.middle_bucket_split_fast_total_ratio = decision.fast_total_ratio
        s.state.middle_bucket_split_slow_total_ratio = decision.slow_total_ratio
        s.state.middle_bucket_split_reason = decision.reason
        s.state.middle_runner_first_tp_price = decision.effective_price
        s.state.partial_tp_price = decision.effective_price
        candle_ts = getattr(boll, "candle_ts_ms", 0)
        logger.warning(
            "MIDDLE_BUCKET_SPLIT_SELECTED | "
            "plan=MIDDLE_RUNNER side=%s middle_bucket_ratio=%.4f "
            "fast_ratio_of_bucket=%.4f slow_ratio_of_bucket=%.4f "
            "fast_total_ratio=%.4f slow_total_ratio=%.4f "
            "fast_price=%.4f slow_price=%.4f effective_price=%.4f "
            "reason=%s candle_ts=%s",
            s.state.side,
            decision.middle_bucket_ratio,
            decision.fast_ratio_of_bucket,
            decision.slow_ratio_of_bucket,
            decision.fast_total_ratio,
            decision.slow_total_ratio,
            float(decision.fast_price or 0.0),
            float(decision.slow_price or 0.0),
            float(decision.effective_price or 0.0),
            decision.reason,
            candle_ts,
        )
        partial_tp_ratio_val = s.state.middle_runner_first_close_ratio or min(
            max(s.config.middle_runner_first_close_ratio, 0.1), 0.95)
        return MiddleBucketSplitApplyResult(
            action="SPLIT",
            split_active=True,
            partial_tp_price=decision.effective_price,
            partial_tp_ratio=partial_tp_ratio_val,
            tp_plan="MIDDLE_RUNNER",
            reason=decision.reason,
        )

    if decision.action == "UNSPLIT_SLOW_MIDDLE":
        if partial_progress_started:
            partial_tp_ratio_val = s.state.middle_runner_first_close_ratio or min(
                max(s.config.middle_runner_first_close_ratio, 0.1), 0.95)
            result = _preserved_partial_split_result(
                state=s.state,
                decision=decision,
                old_fast_consumed=old_fast_consumed,
                old_slow_consumed=old_slow_consumed,
                middle_bucket_ratio=middle_bucket_ratio,
                fast_ratio_of_bucket=fast_ratio_of_bucket,
                effective_price_fallback=slow_middle_price,
                partial_tp_ratio=partial_tp_ratio_val,
                tp_plan="MIDDLE_RUNNER",
            )
            s.state.middle_runner_first_tp_price = result.partial_tp_price
            s.state.partial_tp_price = result.partial_tp_price
            return result
        clear_middle_bucket_split_state(s.state, reason=None)
        s.state.middle_runner_first_tp_price = slow_middle_price
        s.state.partial_tp_price = slow_middle_price
        candle_ts = getattr(boll, "candle_ts_ms", 0)
        logger.warning(
            "MIDDLE_BUCKET_SPLIT_SKIPPED | "
            "plan=MIDDLE_RUNNER side=%s reason=%s action=%s "
            "fast_price=%s slow_price=%.4f required_price=%.4f "
            "using_full_middle_bucket_at_20 candle_ts=%s",
            s.state.side,
            decision.reason,
            decision.action,
            f"{float(decision.fast_price or 0.0):.4f}" if decision.fast_price is not None else "-",
            float(decision.slow_price or 0.0),
            float(decision.required_price or 0.0),
            candle_ts,
        )
        partial_tp_ratio_val = s.state.middle_runner_first_close_ratio or min(
            max(s.config.middle_runner_first_close_ratio, 0.1), 0.95)
        return MiddleBucketSplitApplyResult(
            action="UNSPLIT_SLOW_MIDDLE",
            split_active=False,
            partial_tp_price=slow_middle_price,
            partial_tp_ratio=partial_tp_ratio_val,
            tp_plan="MIDDLE_RUNNER",
            reason=decision.reason,
        )

    # FALLBACK_OUTER, INVALID, DISABLED
    if partial_progress_started:
        partial_tp_ratio_val = s.state.middle_runner_first_close_ratio or min(
            max(s.config.middle_runner_first_close_ratio, 0.1), 0.95)
        result = _preserved_partial_split_result(
            state=s.state,
            decision=decision,
            old_fast_consumed=old_fast_consumed,
            old_slow_consumed=old_slow_consumed,
            middle_bucket_ratio=middle_bucket_ratio,
            fast_ratio_of_bucket=fast_ratio_of_bucket,
            effective_price_fallback=getattr(s.state, "middle_bucket_split_effective_price", None) or slow_middle_price,
            partial_tp_ratio=partial_tp_ratio_val,
            tp_plan="MIDDLE_RUNNER",
        )
        s.state.middle_runner_first_tp_price = result.partial_tp_price
        s.state.partial_tp_price = result.partial_tp_price
        return result
    clear_middle_bucket_split_state(s.state, reason=None)
    return MiddleBucketSplitApplyResult(
        action=decision.action,
        split_active=False,
        partial_tp_price=None,
        partial_tp_ratio=0.0,
        tp_plan=None,
        reason=decision.reason,
    )
