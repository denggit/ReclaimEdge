from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.execution.trader import PositionSnapshot
from src.position_management import cost_runtime as position_cost_runtime
from src.position_management.sidecar.model import sidecar_open_qty
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)

# Legacy split TP plans removed. Set kept as empty for backward compatibility
# with mark_partial_tp_consumed_if_position_reduced which now always returns False.
SPLIT_TP_PLANS: set[str] = set()


@dataclass(frozen=True, eq=False)
class MiddleBucketSplitProgressResult:
    event: str
    pre_split_tp_plan: str
    completed_leg: str | None = None
    full_completed: bool = False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.event == other
        if isinstance(other, MiddleBucketSplitProgressResult):
            return (
                self.event == other.event
                and self.pre_split_tp_plan == other.pre_split_tp_plan
                and self.completed_leg == other.completed_leg
                and self.full_completed == other.full_completed
            )
        return False


def middle_bucket_split_pending(state: object) -> bool:
    return (
        bool(getattr(state, "middle_bucket_split_active", False))
        and not (
            bool(getattr(state, "middle_bucket_split_fast_consumed", False))
            and bool(getattr(state, "middle_bucket_split_slow_consumed", False))
        )
    )


def mark_partial_tp_consumed_if_position_reduced(strategy: BollCvdReclaimStrategy, position: PositionSnapshot) -> bool:
    state = strategy.state
    original_plan = getattr(state, "tp_plan", "SINGLE")
    if original_plan not in SPLIT_TP_PLANS:
        return False
    if not position.has_position or position.side != state.side:
        return False
    total_entry_qty = float(getattr(state, "total_entry_qty", 0.0) or 0.0)
    if total_entry_qty <= 0:
        return False

    old_partial_tp_price = getattr(state, "partial_tp_price", None)
    partial_tp_ratio = float(getattr(state, "partial_tp_ratio", 0.0) or 0.0)
    reduction_ratio = 1 - (float(position.eth_qty) / total_entry_qty)
    required_ratio = max(0.05, partial_tp_ratio * 0.5)
    if reduction_ratio < required_ratio:
        return False

    position_cost_runtime.record_core_position_reduction_exit(
        state,
        position,
        exit_price=old_partial_tp_price,
        fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
    )
    state.partial_tp_consumed = True
    state.partial_tp_price = None
    state.partial_tp_ratio = 0.0
    state.tp_plan = "SINGLE"
    logger.warning(
        "SPLIT_TP_CONSUMED | side=%s original_plan=%s partial_tp_price=%s old_qty=%.8f new_qty=%.8f reduction_ratio=%.6f required_ratio=%.6f partial_ratio=%.4f",
        state.side,
        original_plan,
        old_partial_tp_price,
        total_entry_qty,
        position.eth_qty,
        reduction_ratio,
        required_ratio,
        partial_tp_ratio,
    )
    return True


def mark_middle_runner_active_if_position_reduced(strategy: BollCvdReclaimStrategy, position: PositionSnapshot) -> bool:
    state = strategy.state
    # Gate: when middle bucket split is active and any leg is still pending,
    # split progress owns the path — do NOT run old Middle Runner progress
    if middle_bucket_split_pending(state):
        return False
    if not getattr(state, "middle_runner_pending", False):
        return False
    if getattr(state, "middle_runner_active", False):
        return False
    if not position.has_position or position.side != state.side:
        return False
    total_entry_qty = float(getattr(state, "total_entry_qty", 0.0) or 0.0)
    keep_ratio = float(getattr(state, "middle_runner_keep_ratio", 0.0) or 0.0)
    if total_entry_qty <= 0 or keep_ratio <= 0 or keep_ratio >= 1:
        logger.warning(
            "MIDDLE_RUNNER_ORDER_WARNING | reason=activation_size_unknown side=%s total_entry_qty=%.8f keep_ratio=%.6f okx_eth_qty=%.8f",
            state.side,
            total_entry_qty,
            keep_ratio,
            position.eth_qty,
        )
        return False

    expected_qty = total_entry_qty * keep_ratio
    tolerance = max(total_entry_qty * 0.03, expected_qty * 0.10, 0.000001)
    if abs(float(position.eth_qty) - expected_qty) > tolerance:
        reduction_ratio = 1 - (float(position.eth_qty) / total_entry_qty)
        if reduction_ratio > 0.05:
            state.middle_runner_add_disabled = True
            now_ms = int(time.time() * 1000)
            last_warning_ms = int(getattr(state, "middle_runner_size_mismatch_warning_ts_ms", 0) or 0)
            if last_warning_ms <= 0 or now_ms - last_warning_ms >= 60_000:
                state.middle_runner_size_mismatch_warning_ts_ms = now_ms
                logger.warning(
                    "MIDDLE_RUNNER_ORDER_WARNING | reason=partial_size_mismatch_add_disabled side=%s old_qty=%.8f new_qty=%.8f expected_qty=%.8f tolerance=%.8f reduction_ratio=%.6f keep_ratio=%.6f",
                    state.side,
                    total_entry_qty,
                    position.eth_qty,
                    expected_qty,
                    tolerance,
                    reduction_ratio,
                    keep_ratio,
                )
        return False

    position_cost_runtime.record_core_position_reduction_exit(
        state,
        position,
        exit_price=getattr(state, "middle_runner_first_tp_price", None),
        fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
    )
    logger.warning(
        "MIDDLE_RUNNER_COST_BASIS_AFTER_FIRST_CLOSE | side=%s total_entry_qty=%.8f okx_core_eth_qty=%.8f sidecar_open_qty=%.8f position_cost_entry_notional=%.4f position_cost_exit_notional=%.4f position_cost_remaining_qty=%.8f net_remaining_breakeven_price=%.4f avg_entry_price=%.4f first_tp_price=%s first_close_ratio=%.4f keep_ratio=%.4f",
        state.side,
        total_entry_qty,
        float(position.eth_qty or 0.0),
        sidecar_open_qty(list(getattr(state, "sidecar_legs", []) or [])),
        float(getattr(state, "position_cost_entry_notional", 0.0) or 0.0),
        float(getattr(state, "position_cost_exit_notional", 0.0) or 0.0),
        float(getattr(state, "position_cost_remaining_qty", 0.0) or 0.0),
        float(getattr(state, "net_remaining_breakeven_price", 0.0) or 0.0),
        float(getattr(state, "avg_entry_price", 0.0) or 0.0),
        getattr(state, "middle_runner_first_tp_price", None),
        float(getattr(state, "middle_runner_first_close_ratio", 0.0) or 0.0),
        keep_ratio,
    )
    state.middle_runner_pending = False
    state.middle_runner_active = True
    state.middle_runner_add_disabled = True
    if hasattr(strategy, "_reset_middle_runner_sl_time_tighten_state"):
        strategy._reset_middle_runner_sl_time_tighten_state()
    if hasattr(strategy, "_seed_runner_sl_time_tighten_activation_candle"):
        strategy._seed_runner_sl_time_tighten_activation_candle(
            target="middle_runner",
            candle_ts_ms=int(getattr(strategy.state, "last_tp_update_candle_ts_ms", 0) or 0),
        )
    state.partial_tp_consumed = True
    state.partial_tp_price = None
    state.partial_tp_ratio = 0.0
    state.tp_plan = "SINGLE"
    logger.warning(
        "MIDDLE_RUNNER_ACTIVATED | side=%s old_qty=%.8f new_qty=%.8f expected_qty=%.8f first_close_ratio=%.4f keep_ratio=%.4f final_tp_price=%s add_disabled=true",
        state.side,
        total_entry_qty,
        position.eth_qty,
        expected_qty,
        getattr(state, "middle_runner_first_close_ratio", 0.0),
        keep_ratio,
        getattr(state, "middle_runner_final_tp_price", None),
    )
    return True


def mark_three_stage_progress_if_position_reduced(strategy: BollCvdReclaimStrategy, position: PositionSnapshot,
                                                  ts_ms: int) -> str | None:
    state = strategy.state
    # Gate: when middle bucket split is active and any leg is still pending,
    # split progress owns the path — do NOT run old Three-Stage progress
    if middle_bucket_split_pending(state):
        return None
    if not getattr(state, "three_stage_runner_enabled_for_position", False):
        return None
    if not position.has_position or position.side != state.side:
        return None
    total_entry_qty = float(getattr(state, "total_entry_qty", 0.0) or 0.0)
    if total_entry_qty <= 0:
        return None

    remaining_ratio = float(position.eth_qty) / total_entry_qty
    tp1_ratio = float(getattr(state, "three_stage_tp1_ratio", 0.0) or 0.0)
    tp2_ratio = float(getattr(state, "three_stage_tp2_ratio", 0.0) or 0.0)
    runner_ratio = float(getattr(state, "three_stage_runner_ratio", 0.0) or 0.0)
    after_tp1_ratio = max(0.0, 1.0 - tp1_ratio)
    after_tp2_ratio = max(0.0, runner_ratio)
    tp1_tolerance = max(0.02, tp1_ratio * 0.05, 0.000001)
    tp2_tolerance = max(0.01, runner_ratio * 0.10, 0.000001)
    event: str | None = None

    if not getattr(state, "three_stage_tp1_consumed", False) and remaining_ratio <= after_tp1_ratio + tp1_tolerance:
        expected_after_tp1_qty = total_entry_qty * after_tp1_ratio + sidecar_open_qty(
            list(getattr(state, "sidecar_legs", []) or []))
        will_mark_tp2_now = remaining_ratio <= after_tp2_ratio + tp2_tolerance
        position_cost_runtime.record_core_position_reduction_exit(
            state,
            position,
            exit_price=getattr(state, "three_stage_tp1_price", None),
            fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
            expected_remaining_qty=expected_after_tp1_qty if will_mark_tp2_now else None,
        )
        logger.warning(
            "THREE_STAGE_COST_BASIS_AFTER_TP1 | side=%s total_entry_qty=%.8f okx_core_eth_qty=%.8f sidecar_open_qty=%.8f position_cost_entry_notional=%.4f position_cost_exit_notional=%.4f position_cost_remaining_qty=%.8f net_remaining_breakeven_price=%.4f avg_entry_price=%.4f tp1_price=%s tp1_ratio=%.4f remaining_ratio=%.6f",
            state.side,
            total_entry_qty,
            float(position.eth_qty or 0.0),
            sidecar_open_qty(list(getattr(state, "sidecar_legs", []) or [])),
            float(getattr(state, "position_cost_entry_notional", 0.0) or 0.0),
            float(getattr(state, "position_cost_exit_notional", 0.0) or 0.0),
            float(getattr(state, "position_cost_remaining_qty", 0.0) or 0.0),
            float(getattr(state, "net_remaining_breakeven_price", 0.0) or 0.0),
            float(getattr(state, "avg_entry_price", 0.0) or 0.0),
            getattr(state, "three_stage_tp1_price", None),
            tp1_ratio,
            remaining_ratio,
        )
        state.three_stage_tp1_consumed = True
        if hasattr(strategy, "_reset_three_stage_post_tp1_sl_time_tighten_state"):
            strategy._reset_three_stage_post_tp1_sl_time_tighten_state()
        if hasattr(strategy, "_seed_runner_sl_time_tighten_activation_candle"):
            strategy._seed_runner_sl_time_tighten_activation_candle(
                target="three_stage_post_tp1",
                candle_ts_ms=int(getattr(strategy.state, "last_tp_update_candle_ts_ms", 0) or 0),
            )
        state.partial_tp_consumed = True
        event = "TP1"
        logger.warning(
            "THREE_STAGE_TP1_FILLED | side=%s old_qty=%.8f new_qty=%.8f remaining_ratio=%.6f expected_after_tp1=%.6f tp1_ratio=%.4f",
            state.side,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            after_tp1_ratio,
            tp1_ratio,
        )

    if (
            getattr(state, "three_stage_tp1_consumed", False)
            and not getattr(state, "three_stage_tp2_consumed", False)
            and remaining_ratio <= after_tp2_ratio + tp2_tolerance
    ):
        position_cost_runtime.record_core_position_reduction_exit(
            state,
            position,
            exit_price=getattr(state, "three_stage_tp2_price", None),
            fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
        )
        state.three_stage_tp2_consumed = True
        if hasattr(strategy, "_reset_three_stage_post_tp1_sl_time_tighten_state"):
            strategy._reset_three_stage_post_tp1_sl_time_tighten_state()
        state.trend_runner_active = True
        state.trend_runner_trend_start_ts_ms = ts_ms
        state.trend_runner_adjust_count = 0
        state.trend_runner_last_update_candle_ts_ms = 0
        state.trend_runner_tp_price = None
        state.trend_runner_sl_price = None
        state.trend_runner_tp_order_id = None
        state.trend_runner_sl_order_id = None
        state.tp_plan = "SINGLE"
        state.partial_tp_price = None
        state.partial_tp_ratio = 0.0
        logger.warning(
            "TREND_RUNNER_ACTIVATED | side=%s old_qty=%.8f new_qty=%.8f remaining_ratio=%.6f runner_ratio=%.6f tp2_ratio=%.4f runner_tp=%s runner_sl=%s trend_start_ts_ms=%s",
            state.side,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            runner_ratio,
            tp2_ratio,
            getattr(state, "trend_runner_tp_price", None),
            getattr(state, "trend_runner_sl_price", None),
            ts_ms,
        )
        return "TP1_TP2" if event == "TP1" else "TP2"
    return event


def append_three_stage_progress_journal_events(journal: Any, payload: dict[str, Any]) -> None:
    event = payload.get("event")
    position_id = payload.get("position_id")
    if event in {"TP1", "TP1_TP2"}:
        journal.append("THREE_STAGE_TP1_FILLED", dict(payload), position_id=position_id)
    if event in {"TP2", "TP1_TP2"}:
        journal.append("THREE_STAGE_TP2_FILLED", dict(payload), position_id=position_id)
        journal.append("TREND_RUNNER_ACTIVATED", dict(payload), position_id=position_id)


# ── Middle Bucket Split progress detection ───────────────────────────────


def _mark_middle_bucket_split_full_completed(
    *,
    strategy: BollCvdReclaimStrategy,
    position: PositionSnapshot,
    pre_split_tp_plan: str,
    total_entry_qty: float,
    remaining_ratio: float,
    ts_ms: int | None = None,
) -> None:
    state = strategy.state
    state.middle_bucket_split_fast_consumed = True
    state.middle_bucket_split_slow_consumed = True

    if pre_split_tp_plan == "THREE_STAGE_RUNNER":
        state.three_stage_tp1_consumed = True
        state.partial_tp_consumed = True
        if hasattr(strategy, "_reset_three_stage_post_tp1_sl_time_tighten_state"):
            strategy._reset_three_stage_post_tp1_sl_time_tighten_state()
        if hasattr(strategy, "_seed_runner_sl_time_tighten_activation_candle"):
            strategy._seed_runner_sl_time_tighten_activation_candle(
                target="three_stage_post_tp1",
                candle_ts_ms=int(getattr(strategy.state, "last_tp_update_candle_ts_ms", 0) or 0),
            )
        logger.warning(
            "MIDDLE_BUCKET_SPLIT_FULL_TP1_FILLED | side=%s plan=THREE_STAGE_RUNNER "
            "old_qty=%.8f new_qty=%.8f remaining_ratio=%.6f fast_price=%s slow_price=%s "
            "fast_sl_price=%s tp1_consumed=true",
            state.side,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            getattr(state, "middle_bucket_split_fast_price", None),
            getattr(state, "middle_bucket_split_slow_price", None),
            getattr(state, "middle_bucket_split_fast_sl_price", None),
        )
    elif pre_split_tp_plan == "MIDDLE_RUNNER":
        state.middle_runner_pending = False
        state.middle_runner_active = True
        state.middle_runner_add_disabled = True
        state.partial_tp_consumed = True
        state.partial_tp_price = None
        state.partial_tp_ratio = 0.0
        state.tp_plan = "SINGLE"
        if hasattr(strategy, "_reset_middle_runner_sl_time_tighten_state"):
            strategy._reset_middle_runner_sl_time_tighten_state()
        if hasattr(strategy, "_seed_runner_sl_time_tighten_activation_candle"):
            strategy._seed_runner_sl_time_tighten_activation_candle(
                target="middle_runner",
                candle_ts_ms=int(getattr(strategy.state, "last_tp_update_candle_ts_ms", 0) or 0),
            )
        logger.warning(
            "MIDDLE_BUCKET_SPLIT_FULL_TP1_FILLED | side=%s plan=MIDDLE_RUNNER "
            "old_qty=%.8f new_qty=%.8f remaining_ratio=%.6f fast_price=%s slow_price=%s "
            "fast_sl_price=%s middle_runner_active=true",
            state.side,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            getattr(state, "middle_bucket_split_fast_price", None),
            getattr(state, "middle_bucket_split_slow_price", None),
            getattr(state, "middle_bucket_split_fast_sl_price", None),
        )
    else:
        logger.warning(
            "MIDDLE_BUCKET_SPLIT_FULL_TP1_FILLED | side=%s plan=%s old_qty=%.8f new_qty=%.8f "
            "remaining_ratio=%.6f fast_price=%s slow_price=%s",
            state.side,
            pre_split_tp_plan,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            getattr(state, "middle_bucket_split_fast_price", None),
            getattr(state, "middle_bucket_split_slow_price", None),
        )


def mark_middle_bucket_split_progress_if_position_reduced(
    strategy: BollCvdReclaimStrategy,
    position: PositionSnapshot,
) -> MiddleBucketSplitProgressResult | None:
    """Detect fast/slow fills when middle bucket split is active.

    Returns:
        MiddleBucketSplitProgressResult for split progress, or None.
    """
    state = strategy.state
    if not getattr(state, "middle_bucket_split_active", False):
        return None
    if not position.has_position or position.side != state.side:
        return None
    total_entry_qty = float(getattr(state, "total_entry_qty", 0.0) or 0.0)
    if total_entry_qty <= 0:
        return None

    remaining_ratio = float(position.eth_qty) / total_entry_qty
    fast_total_ratio = float(getattr(state, "middle_bucket_split_fast_total_ratio", 0.0) or 0.0)
    slow_total_ratio = float(getattr(state, "middle_bucket_split_slow_total_ratio", 0.0) or 0.0)
    middle_bucket_ratio = float(getattr(state, "middle_bucket_split_middle_bucket_ratio", 0.0) or 0.0)
    if slow_total_ratio <= 0 and middle_bucket_ratio > fast_total_ratio:
        slow_total_ratio = middle_bucket_ratio - fast_total_ratio
    if middle_bucket_ratio <= 0:
        middle_bucket_ratio = fast_total_ratio + slow_total_ratio
    fast_consumed = bool(getattr(state, "middle_bucket_split_fast_consumed", False))
    slow_consumed = bool(getattr(state, "middle_bucket_split_slow_consumed", False))
    pre_split_tp_plan = getattr(state, "tp_plan", "SINGLE")

    after_fast_ratio = max(0.0, 1.0 - fast_total_ratio)
    after_slow_ratio = max(0.0, 1.0 - slow_total_ratio)
    after_middle_bucket_ratio = max(0.0, 1.0 - middle_bucket_ratio)
    fast_tolerance = max(0.02, fast_total_ratio * 0.05, 0.000001)
    slow_tolerance = max(0.02, slow_total_ratio * 0.05, 0.000001)
    full_tp1_tolerance = max(0.02, middle_bucket_ratio * 0.05, 0.000001)

    # Same-sync full: both legs already filled by this account snapshot.
    if (
        not fast_consumed
        and not slow_consumed
        and remaining_ratio <= after_middle_bucket_ratio + full_tp1_tolerance
    ):
        position_cost_runtime.record_core_position_reduction_exit(
            state,
            position,
            exit_price=getattr(state, "middle_bucket_split_effective_price", None),
            fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
        )
        _mark_middle_bucket_split_full_completed(
            strategy=strategy,
            position=position,
            pre_split_tp_plan=pre_split_tp_plan,
            total_entry_qty=total_entry_qty,
            remaining_ratio=remaining_ratio,
        )
        return MiddleBucketSplitProgressResult(
            event="MIDDLE_BUCKET_FULL",
            pre_split_tp_plan=pre_split_tp_plan,
            completed_leg=None,
            full_completed=True,
        )

    # ── Fast leg fill detection ───────────────────────────────────────
    if (
        not fast_consumed
        and not slow_consumed
        and remaining_ratio <= after_fast_ratio + fast_tolerance
        and remaining_ratio > after_middle_bucket_ratio + full_tp1_tolerance
    ):
        position_cost_runtime.record_core_position_reduction_exit(
            state,
            position,
            exit_price=getattr(state, "middle_bucket_split_fast_price", None),
            fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
        )
        state.middle_bucket_split_fast_consumed = True
        state.middle_bucket_split_add_disabled = True
        # Compute fast protective SL
        from src.strategies.middle_bucket_split import calculate_fast_protective_sl
        fast_sl = calculate_fast_protective_sl(
            side=state.side,
            avg_entry_price=float(state.avg_entry_price or 0.0),
            fee_buffer_pct=float(strategy.config.middle_bucket_split_fast_sl_fee_buffer_pct),
        )
        state.middle_bucket_split_fast_sl_price = fast_sl
        logger.warning(
            "MIDDLE_BUCKET_FAST_FILLED | side=%s old_qty=%.8f new_qty=%.8f remaining_ratio=%.6f "
            "fast_total_ratio=%.4f after_fast_ratio=%.6f tolerance=%.6f "
            "fast_price=%s fast_sl_price=%s avg_entry=%.4f add_disabled=true",
            state.side,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            fast_total_ratio,
            after_fast_ratio,
            fast_tolerance,
            getattr(state, "middle_bucket_split_fast_price", None),
            fast_sl,
            float(state.avg_entry_price or 0.0),
        )
        return MiddleBucketSplitProgressResult(
            event="MIDDLE_BUCKET_FAST",
            pre_split_tp_plan=pre_split_tp_plan,
            completed_leg="fast",
            full_completed=False,
        )

    # ── Slow-only fill detection ──────────────────────────────────────
    if (
        not fast_consumed
        and not slow_consumed
        and remaining_ratio <= after_slow_ratio + slow_tolerance
        and remaining_ratio > after_middle_bucket_ratio + full_tp1_tolerance
    ):
        position_cost_runtime.record_core_position_reduction_exit(
            state,
            position,
            exit_price=getattr(state, "middle_bucket_split_slow_price", None),
            fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
        )
        state.middle_bucket_split_slow_consumed = True
        state.middle_bucket_split_add_disabled = True
        logger.warning(
            "MIDDLE_BUCKET_SLOW_ONLY_FILLED | side=%s old_qty=%.8f new_qty=%.8f remaining_ratio=%.6f "
            "slow_total_ratio=%.4f after_slow_ratio=%.6f tolerance=%.6f slow_price=%s add_disabled=true",
            state.side,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            slow_total_ratio,
            after_slow_ratio,
            slow_tolerance,
            getattr(state, "middle_bucket_split_slow_price", None),
        )
        return MiddleBucketSplitProgressResult(
            event="MIDDLE_BUCKET_SLOW_ONLY",
            pre_split_tp_plan=pre_split_tp_plan,
            completed_leg="slow",
            full_completed=False,
        )

    # ── Fast-first full completion: slow leg fills now ────────────────
    if fast_consumed and not slow_consumed and remaining_ratio <= after_middle_bucket_ratio + full_tp1_tolerance:
        position_cost_runtime.record_core_position_reduction_exit(
            state,
            position,
            exit_price=getattr(state, "middle_bucket_split_slow_price", None),
            fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
        )
        logger.warning(
            "MIDDLE_BUCKET_SLOW_FILLED | side=%s old_qty=%.8f new_qty=%.8f remaining_ratio=%.6f "
            "middle_bucket_ratio=%.4f after_middle_bucket_ratio=%.6f tolerance=%.6f "
            "slow_price=%s",
            state.side,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            middle_bucket_ratio,
            after_middle_bucket_ratio,
            full_tp1_tolerance,
            getattr(state, "middle_bucket_split_slow_price", None),
        )
        _mark_middle_bucket_split_full_completed(
            strategy=strategy,
            position=position,
            pre_split_tp_plan=pre_split_tp_plan,
            total_entry_qty=total_entry_qty,
            remaining_ratio=remaining_ratio,
        )
        return MiddleBucketSplitProgressResult(
            event="MIDDLE_BUCKET_FULL",
            pre_split_tp_plan=pre_split_tp_plan,
            completed_leg="slow",
            full_completed=True,
        )

    # ── Slow-first full completion: fast leg fills now ────────────────
    if slow_consumed and not fast_consumed and remaining_ratio <= after_middle_bucket_ratio + full_tp1_tolerance:
        position_cost_runtime.record_core_position_reduction_exit(
            state,
            position,
            exit_price=getattr(state, "middle_bucket_split_fast_price", None),
            fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
        )
        logger.warning(
            "MIDDLE_BUCKET_FAST_FILLED | side=%s old_qty=%.8f new_qty=%.8f remaining_ratio=%.6f "
            "middle_bucket_ratio=%.4f after_middle_bucket_ratio=%.6f tolerance=%.6f fast_price=%s",
            state.side,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            middle_bucket_ratio,
            after_middle_bucket_ratio,
            full_tp1_tolerance,
            getattr(state, "middle_bucket_split_fast_price", None),
        )
        _mark_middle_bucket_split_full_completed(
            strategy=strategy,
            position=position,
            pre_split_tp_plan=pre_split_tp_plan,
            total_entry_qty=total_entry_qty,
            remaining_ratio=remaining_ratio,
        )
        return MiddleBucketSplitProgressResult(
            event="MIDDLE_BUCKET_FULL",
            pre_split_tp_plan=pre_split_tp_plan,
            completed_leg="fast",
            full_completed=True,
        )

    return None


def append_middle_bucket_split_journal_events(
    journal: Any,
    payload: dict[str, Any],
) -> None:
    """Append middle bucket split journal events from a progress payload."""
    event = payload.get("event")
    position_id = payload.get("position_id")
    if event == "MIDDLE_BUCKET_FAST":
        journal.append("MIDDLE_BUCKET_FAST_FILLED", dict(payload), position_id=position_id)
    if event == "MIDDLE_BUCKET_SLOW_ONLY":
        journal.append("MIDDLE_BUCKET_SLOW_ONLY_FILLED", dict(payload), position_id=position_id)
    if event == "MIDDLE_BUCKET_FULL":
        journal.append("MIDDLE_BUCKET_FULL_FILLED", dict(payload), position_id=position_id)
        journal.append("MIDDLE_BUCKET_SPLIT_COMPLETED", dict(payload), position_id=position_id)
    if event == "MIDDLE_BUCKET_SLOW":
        journal.append("MIDDLE_BUCKET_SLOW_FILLED", dict(payload), position_id=position_id)
        journal.append("MIDDLE_BUCKET_SPLIT_COMPLETED", dict(payload), position_id=position_id)
