"""TP Update Coordinator — extracted from BollCvdReclaimStrategy._maybe_update_tp.

This module owns the main TP-update orchestration loop so that the strategy
class itself only retains a thin wrapper.  The coordinator reads and writes
strategy.state freely and delegates to the strategy's existing helpers.

Phase 39 of the refactoring plan:
    Round 2, Item 39 — Extract Strategy TP Update Coordinator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.position_management.middle_bucket_split_state import (
    clear_middle_bucket_split_state,
)
from src.strategies.middle_bucket_split_apply import (
    MiddleBucketSplitApplyResult,
    apply_middle_runner_bucket_split,
    apply_three_stage_middle_bucket_split,
)
from src.strategies.pre_tp1_degrade_replan import (
    decide_pre_tp1_degrade_stage_for_replan,
)
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.strategies.boll_cvd_reclaim_strategy import (
        BollCvdReclaimStrategy,
        BollSnapshot,
        CvdSnapshot,
        TradeIntent,
        TpMode,
    )

logger = get_logger(__name__)


# MiddleBucketSplitApplyResult is imported from middle_bucket_split_apply.py
# (re-exported here for backward compatibility).

# Re-use the module-level helper from the strategy module so behaviour is
# byte-identical.
def _price_changed(old: float | None, new: float | None, threshold: float = 0.0001) -> bool:
    if old is None or new is None:
        return old is not None or new is not None
    if new == 0:
        return old != 0
    return abs(float(old) - float(new)) / abs(float(new)) >= threshold


class TpUpdateCoordinator:
    """Orchestrates every _maybe_update_tp call for BollCvdReclaimStrategy.

    The coordinator is deliberately NOT a pure function — it holds a reference to
    the owning strategy and can read/write ``strategy.state`` directly.
    """

    def __init__(self, strategy: BollCvdReclaimStrategy) -> None:
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def maybe_update_tp(
        self,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
    ) -> TradeIntent | None:
        """Full TP-update flow, moved verbatim from the strategy class."""
        s = self.strategy

        # ── early-return guards ──────────────────────────────────────
        if s.state.side is None or s.state.layers <= 0:
            return None

        trend_runner_needs_initial_orders = (
            s.state.trend_runner_active
            and (s.state.trend_runner_tp_price is None or s.state.trend_runner_sl_price is None)
        )
        force_reconcile = bool(getattr(s.state, "startup_force_tp_reconcile", False))
        if (
            s.state.last_tp_update_candle_ts_ms == boll.candle_ts_ms
            and not trend_runner_needs_initial_orders
            and not force_reconcile
        ):
            return None

        if force_reconcile:
            logger.warning(
                "STARTUP_FORCE_TP_RECONCILE_ARMED | side=%s layers=%s tp_plan=%s candle_ts=%s last_tp_update_candle_ts_ms=%s",
                s.state.side,
                s.state.layers,
                s.state.tp_plan,
                boll.candle_ts_ms,
                s.state.last_tp_update_candle_ts_ms,
            )
            # Refresh stale pre-TP1 degrade stage before reconcile so that
            # a saved SINGLE cap doesn't block recovery to THREE_STAGE_RUNNER
            # or MIDDLE_RUNNER when the position age actually permits it.
            self._refresh_pre_tp1_degrade_stage_before_startup_reconcile(ts_ms=ts_ms)

        # ── Three-Stage waiting-TP2 branch ────────────────────────────
        if s._three_stage_waiting_tp2():
            return self._maybe_update_three_stage_waiting_tp2(price, ts_ms, boll, cvd, force_reconcile)

        # ── snapshot old values for change-detection later ────────────
        old_runner_sl = s.state.middle_runner_protective_sl_price
        old_trend_runner_tp = s.state.trend_runner_tp_price
        old_trend_runner_sl = s.state.trend_runner_sl_price

        tp_price, tp_mode = s._select_tp_price(s.state.side, boll)
        middle_profit_fallback_locked = False
        reason_override: str | None = None

        # ── Middle-profit safety gate ─────────────────────────────────
        tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, \
            middle_profit_fallback_locked, reason_override = \
            self._apply_middle_profit_safety_gate(
                tp_price, tp_mode, boll, ts_ms,
                middle_profit_fallback_locked, reason_override,
            )

        # ── Three-Stage pre-TP1 degrade ───────────────────────────────
        degrade_applied = False
        if middle_profit_fallback_locked:
            pass
        else:
            tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, \
                reason_override, degrade_applied = self._apply_three_stage_pre_tp1_degrade(
                    tp_price, tp_mode, boll, ts_ms, reason_override,
                )

        # ── Trend Runner active branch ────────────────────────────────
        if not middle_profit_fallback_locked and not degrade_applied and s.state.trend_runner_active:
            tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan = \
                self._apply_trend_runner_active_branch(
                    tp_price, boll, old_trend_runner_tp, old_trend_runner_sl,
                )

        # ── Middle Runner active branch ───────────────────────────────
        elif not middle_profit_fallback_locked and not degrade_applied and s.state.middle_runner_active:
            tp_price, partial_tp_price, partial_tp_ratio, tp_plan = \
                self._apply_middle_runner_active_branch(price, boll, old_runner_sl)

        # ── Middle Runner pending branch ──────────────────────────────
        elif not middle_profit_fallback_locked and not degrade_applied and s.state.middle_runner_pending:
            tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, \
                reason_override = self._apply_middle_runner_pending_branch(
                    tp_price, boll, ts_ms, reason_override,
                )

        # ── Three-Stage enabled branch ────────────────────────────────
        elif (
            not middle_profit_fallback_locked
            and not degrade_applied
            and s.state.three_stage_runner_enabled_for_position
            and not s.state.trend_runner_active
        ):
            tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, \
                reason_override = self._apply_three_stage_enabled_branch(
                    tp_price, boll, ts_ms, reason_override,
                )

        # ── Near-TP protected / add_disabled branch ───────────────────
        elif not middle_profit_fallback_locked and not degrade_applied and (
            s.state.near_tp_protected or s.state.near_tp_add_disabled
        ):
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"

        # ── Normal _select_tp_plan branch ─────────────────────────────
        elif not middle_profit_fallback_locked and not degrade_applied:
            tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, \
                reason_override = self._apply_normal_plan_selection_branch(
                    tp_price, tp_mode, boll, ts_ms, reason_override,
                    force_reconcile=force_reconcile,
                )

        # ── Finalize state & emit intent (or skip) ────────────────────
        return self._finalize_state_and_maybe_emit_intent(
            price, ts_ms, boll, cvd,
            tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan,
            reason_override, force_reconcile,
            old_runner_sl, old_trend_runner_tp, old_trend_runner_sl,
        )

    # ------------------------------------------------------------------
    # Private sub-methods (one per logical branch)
    # ------------------------------------------------------------------

    def _maybe_update_three_stage_waiting_tp2(
        self,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        force_reconcile: bool,
    ) -> TradeIntent | None:
        """Three-Stage waiting-TP2 branch."""
        s = self.strategy

        old_post_tp1_sl = s.state.three_stage_post_tp1_protective_sl_price
        old_tp2_price = s.state.three_stage_tp2_price
        new_tp2_price, _tp2_src = s._select_three_stage_tp2_outer(s.state.side, boll)
        if s.config.three_stage_post_tp1_protective_sl_enabled:
            s._advance_runner_sl_time_tighten_candle_count(
                target="three_stage_post_tp1",
                candle_ts_ms=int(getattr(boll, "candle_ts_ms", 0) or 0),
            )
            calculated_sl = s._calculate_three_stage_post_tp1_protective_sl(s.state.side, price, boll)
            extension_sl = s._apply_three_stage_post_tp1_extension_trigger(
                s.state.side, price, boll, calculated_sl,
            )
            protective_sl = s._tighten_optional_three_stage_post_tp1_sl(
                s.state.side, old_post_tp1_sl, extension_sl,
            )
            s.state.three_stage_post_tp1_protective_sl_price = protective_sl
        else:
            protective_sl = old_post_tp1_sl

        s.state.three_stage_tp2_price = new_tp2_price
        s.state.tp_price = new_tp2_price
        s.state.tp_mode = "UPPER" if s.state.side == "LONG" else "LOWER"
        s.state.tp_plan = "THREE_STAGE_RUNNER"
        s.state.partial_tp_price = None
        s.state.partial_tp_ratio = 0.0
        s.state.last_tp_update_ts_ms = ts_ms
        s.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms

        post_tp1_sl_changed = protective_sl is not None and _price_changed(old_post_tp1_sl, protective_sl)
        tp2_changed = _price_changed(old_tp2_price, new_tp2_price)

        if post_tp1_sl_changed or tp2_changed or force_reconcile:
            s.state.startup_force_tp_reconcile = False
            size = s.sizer.calculate(price, layer_index=s.state.layers)
            reason_text = (
                "startup_force_tp_reconcile" if force_reconcile
                else "three_stage_post_tp1_dynamic_tp_sl_update"
            )
            logger.warning(
                "THREE_STAGE_TP1_PROTECTIVE_SL_UPDATE_SIGNAL | side=%s old_sl=%s new_sl=%s old_tp2=%s new_tp2=%.4f candle_ts=%s force_reconcile=%s",
                s.state.side,
                f"{old_post_tp1_sl:.4f}" if old_post_tp1_sl is not None else "-",
                f"{protective_sl:.4f}" if protective_sl is not None else "-",
                f"{old_tp2_price:.4f}" if old_tp2_price is not None else "-",
                new_tp2_price,
                boll.candle_ts_ms,
                force_reconcile,
            )
            s._log_tp_boll_price_selected(
                phase="waiting_tp2_dynamic",
                boll=boll,
                tp_price=new_tp2_price,
                tp_mode="UPPER" if s.state.side == "LONG" else "LOWER",
                tp_plan="THREE_STAGE_RUNNER",
                tp2_price=new_tp2_price,
            )
            return s._intent(
                "UPDATE_TP", s.state.side, price, s.state.layers,
                new_tp2_price, reason_text, size, boll, cvd, ts_ms,
            )

        s.state.startup_force_tp_reconcile = False
        logger.info(
            "TP_UPDATE_SKIPPED | reason=three_stage_waiting_tp2_plan_unchanged side=%s candle_ts=%s tp2_price=%s protective_sl=%s",
            s.state.side,
            boll.candle_ts_ms,
            s.state.three_stage_tp2_price,
            protective_sl,
        )
        return None

    def _apply_middle_profit_safety_gate(
        self,
        tp_price: float,
        tp_mode: TpMode,
        boll: BollSnapshot,
        ts_ms: int,
        middle_profit_fallback_locked: bool,
        reason_override: str | None,
    ) -> tuple[float, TpMode, float | None, float, str, bool, str | None]:
        """Before any complex TP mode is allowed, the middle band must offer
        sufficient net profit relative to the effective breakeven.

        Returns the (possibly overridden) tp_price, tp_mode, partial_tp_price,
        partial_tp_ratio, tp_plan, middle_profit_fallback_locked, reason_override.
        """
        s = self.strategy

        if tp_mode == "MIDDLE":
            return tp_price, tp_mode, None, 0.0, s.state.tp_plan, False, reason_override

        effective_be = s._effective_breakeven_for_tp_selection(s.state.side)
        required_middle = s._required_middle_for_profit(s.state.side, effective_be)

        partial_tp_price: float | None = None
        partial_tp_ratio: float = 0.0
        tp_plan: str = s.state.tp_plan

        # Three-Stage: only reset when TP1 has NOT been consumed and trend runner is NOT active
        if (
            s.state.three_stage_runner_enabled_for_position
            and not s.state.three_stage_tp1_consumed
            and not s.state.trend_runner_active
        ):
            old_tp1 = s.state.three_stage_tp1_price
            old_tp2 = s.state.three_stage_tp2_price
            outer, outer_src = s._select_three_stage_tp2_outer(s.state.side, boll)
            logger.warning(
                "THREE_STAGE_MIDDLE_PROFIT_INSUFFICIENT_SINGLE_OUTER | "
                "side=%s effective_breakeven=%.4f required_middle=%.4f "
                "tp_boll_middle=%s structure_middle=%.4f selected_outer=%.4f outer_source=%s "
                "old_tp1=%s old_tp2=%s candle_ts=%s",
                s.state.side,
                effective_be,
                required_middle,
                s._format_optional_price(getattr(boll, "tp_middle", None)),
                boll.middle,
                outer,
                outer_src,
                f"{old_tp1:.4f}" if old_tp1 is not None else "-",
                f"{old_tp2:.4f}" if old_tp2 is not None else "-",
                boll.candle_ts_ms,
            )
            tp_price, tp_mode = s._fallback_to_single_outer_due_middle_profit_insufficient(
                side=s.state.side,
                boll=boll,
                ts_ms=ts_ms,
                reason="three_stage_middle_profit_insufficient",
            )
            partial_tp_price = None
            partial_tp_ratio = 0.0
            tp_plan = "SINGLE"
            reason_override = "three_stage_middle_profit_insufficient_single_outer"
            middle_profit_fallback_locked = True

        # Middle Runner pending (first close NOT done): reset
        elif s.state.middle_runner_pending and not s.state.middle_runner_active:
            outer, outer_src = s._select_valid_tp_outer_with_profit_fallback(s.state.side, boll)
            logger.warning(
                "MIDDLE_RUNNER_MIDDLE_PROFIT_INSUFFICIENT_SINGLE_OUTER | "
                "side=%s effective_breakeven=%.4f required_middle=%.4f "
                "tp_boll_middle=%s structure_middle=%.4f selected_outer=%.4f outer_source=%s candle_ts=%s",
                s.state.side,
                effective_be,
                required_middle,
                s._format_optional_price(getattr(boll, "tp_middle", None)),
                boll.middle,
                outer,
                outer_src,
                boll.candle_ts_ms,
            )
            tp_price, tp_mode = s._fallback_to_single_outer_due_middle_profit_insufficient(
                side=s.state.side,
                boll=boll,
                ts_ms=ts_ms,
                reason="middle_runner_middle_profit_insufficient",
            )
            partial_tp_price = None
            partial_tp_ratio = 0.0
            tp_plan = "SINGLE"
            reason_override = "middle_runner_middle_profit_insufficient_single_outer"
            middle_profit_fallback_locked = True

        # SPLIT partial NOT consumed: fall back to SINGLE outer
        elif s.state.tp_plan == "SPLIT_PARTIAL_FINAL" and not s.state.partial_tp_consumed:
            outer, _outer_src = s._select_valid_tp_outer_with_profit_fallback(s.state.side, boll)
            outer_mode: TpMode = "UPPER" if s.state.side == "LONG" else "LOWER"
            logger.warning(
                "SPLIT_TP_DISABLED_MIDDLE_PROFIT_INSUFFICIENT | "
                "side=%s effective_breakeven=%.4f middle=%.4f required_middle=%.4f "
                "outer=%.4f outer_source=%s candle_ts=%s",
                s.state.side,
                effective_be,
                boll.middle,
                required_middle,
                outer,
                _outer_src,
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
            s.state.tp_plan != "SINGLE"
            and not s.state.trend_runner_active
            and not s.state.middle_runner_active
            and not s.state.three_stage_tp1_consumed
            and not s.state.three_stage_tp2_consumed
        ):
            outer, _outer_src = s._select_valid_tp_outer_with_profit_fallback(s.state.side, boll)
            outer_mode_t: TpMode = "UPPER" if s.state.side == "LONG" else "LOWER"
            logger.warning(
                "COMPLEX_TP_DISABLED_MIDDLE_PROFIT_INSUFFICIENT | "
                "side=%s effective_breakeven=%.4f middle=%.4f required_middle=%.4f "
                "outer=%.4f outer_source=%s old_plan=%s candle_ts=%s",
                s.state.side,
                effective_be,
                boll.middle,
                required_middle,
                outer,
                _outer_src,
                s.state.tp_plan,
                boll.candle_ts_ms,
            )
            tp_price = outer
            tp_mode = outer_mode_t
            partial_tp_price = None
            partial_tp_ratio = 0.0
            tp_plan = "SINGLE"
            middle_profit_fallback_locked = True

        return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, \
            middle_profit_fallback_locked, reason_override

    def _apply_three_stage_pre_tp1_degrade(
        self,
        tp_price: float,
        tp_mode: TpMode,
        boll: BollSnapshot,
        ts_ms: int,
        reason_override: str | None,
    ) -> tuple[float, TpMode, float | None, float, str, str | None, bool]:
        """Three-Stage pre-TP1 degrade logic.

        Returns (tp_price, tp_mode, partial_tp_price, partial_tp_ratio,
                 tp_plan, reason_override, degrade_applied).

        degrade_applied is True when a degrade was executed (SINGLE or
        MIDDLE_RUNNER).  Callers MUST skip all subsequent branch logic
        when degrade_applied is True to preserve the original if/elif
        exclusivity.
        """
        s = self.strategy

        degrade_target = s._three_stage_pre_tp1_degrade_target(ts_ms)

        if degrade_target == "SINGLE":
            tp_price, tp_mode = s._degrade_three_stage_pre_tp1_to_single(ts_ms, boll)
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
            reason_override = "three_stage_pre_tp1_degraded_to_single"
            return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override, True

        if degrade_target == "MIDDLE_RUNNER":
            s._degrade_three_stage_pre_tp1_to_middle_runner(ts_ms, boll)
            tp_price = s.state.tp_price or s._select_valid_tp_outer_with_profit_fallback(s.state.side, boll)[0]
            tp_mode = s.state.tp_mode
            partial_tp_price = s.state.partial_tp_price
            partial_tp_ratio = s.state.partial_tp_ratio
            tp_plan = s.state.tp_plan or "SINGLE"
            reason_override = (
                "three_stage_pre_tp1_degraded_to_middle_runner"
                if tp_plan == "MIDDLE_RUNNER"
                else "three_stage_pre_tp1_middle_degrade_middle_profit_insufficient"
            )
            return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override, True

        return tp_price, tp_mode, None, 0.0, s.state.tp_plan, reason_override, False

    def _apply_trend_runner_active_branch(
        self,
        tp_price: float,
        boll: BollSnapshot,
        old_trend_runner_tp: float | None,
        old_trend_runner_sl: float | None,
    ) -> tuple[float, TpMode, float | None, float, str]:
        """Trend Runner active branch."""
        s = self.strategy

        tp_mode: TpMode = "UPPER" if s.state.side == "LONG" else "LOWER"
        partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"

        if s.config.runner_dynamic_enabled:
            tp_price, runner_sl, tp_extra_pct, sl_distance_ratio = s._calculate_trend_runner_dynamic_orders(
                s.state.side,
                boll,
                s.state.trend_runner_adjust_count,
                s.state.trend_runner_sl_price,
            )
            s.state.trend_runner_tp_price = tp_price
            s.state.trend_runner_sl_price = runner_sl
            logger.warning(
                "TREND_RUNNER_UPDATE | side=%s old_tp=%s new_tp=%.4f old_sl=%s new_sl=%.4f adjust_count=%s tp_extra_pct=%.6f sl_distance_ratio=%.6f candle_ts=%s",
                s.state.side,
                f"{old_trend_runner_tp:.4f}" if old_trend_runner_tp is not None else "-",
                tp_price,
                f"{old_trend_runner_sl:.4f}" if old_trend_runner_sl is not None else "-",
                runner_sl,
                s.state.trend_runner_adjust_count,
                tp_extra_pct,
                sl_distance_ratio,
                boll.candle_ts_ms,
            )
        else:
            tp_price = s.state.trend_runner_tp_price or tp_price

        return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan

    def _apply_middle_runner_active_branch(
        self,
        price: float,
        boll: BollSnapshot,
        old_runner_sl: float | None,
    ) -> tuple[float, float | None, float, str]:
        """Middle Runner active branch."""
        s = self.strategy

        tp_price, _tp_src = s._select_valid_tp_outer_with_profit_fallback(s.state.side, boll)
        tp_mode: TpMode = "UPPER" if s.state.side == "LONG" else "LOWER"
        partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"

        s._advance_runner_sl_time_tighten_candle_count(
            target="middle_runner",
            candle_ts_ms=int(getattr(boll, "candle_ts_ms", 0) or 0),
        )
        calculated_sl = s._calculate_middle_runner_protective_sl(s.state.side, price, boll)
        extension_sl = s._apply_middle_runner_extension_trigger(s.state.side, price, boll, calculated_sl)
        protective_sl = s._tighten_optional_middle_runner_sl(s.state.side, old_runner_sl, extension_sl)
        s.state.middle_runner_final_tp_price = tp_price
        s.state.middle_runner_protective_sl_price = protective_sl

        return tp_price, partial_tp_price, partial_tp_ratio, tp_plan

    def _apply_middle_runner_pending_branch(
        self,
        tp_price: float,
        boll: BollSnapshot,
        ts_ms: int,
        reason_override: str | None,
    ) -> tuple[float, TpMode, float | None, float, str, str | None]:
        """Middle Runner pending branch."""
        s = self.strategy

        tp_price, _tp_src = s._select_valid_tp_outer_with_profit_fallback(s.state.side, boll)
        tp_mode: TpMode = "UPPER" if s.state.side == "LONG" else "LOWER"

        # ── Middle Bucket Split for Middle Runner ─────────────────────
        if s.config.middle_bucket_split_enabled and not s.state.middle_runner_active:
            split_result = self._apply_middle_bucket_split_for_middle_runner(boll)

            if split_result.action == "SPLIT":
                partial_tp_price = split_result.partial_tp_price
                partial_tp_ratio = split_result.partial_tp_ratio
                tp_plan = "MIDDLE_RUNNER"
                s.state.middle_runner_first_tp_price = partial_tp_price
                s.state.middle_runner_final_tp_price = tp_price
                s.state.middle_runner_pending = True
                return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override

            if split_result.action == "UNSPLIT_SLOW_MIDDLE":
                # BOLL15 insufficient, BOLL20 sufficient — use BOLL20 middle as
                # the full unsplit middle bucket.  MUST NOT fall back to outer.
                partial_tp_price = split_result.partial_tp_price  # boll.middle
                partial_tp_ratio = split_result.partial_tp_ratio
                tp_plan = "MIDDLE_RUNNER"
                s.state.middle_runner_first_tp_price = partial_tp_price
                s.state.middle_runner_final_tp_price = tp_price
                s.state.middle_runner_pending = True
                return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override

            # FALLBACK_OUTER / INVALID / DISABLED — fall through to old logic

        partial_tp_price, _ptp_src = s._select_valid_tp_middle_with_profit_fallback(s.state.side, boll)

        if partial_tp_price is None:
            effective_be = s._effective_breakeven_for_tp_selection(s.state.side)
            required_middle = s._required_middle_for_profit(s.state.side, effective_be)
            outer, outer_src = s._select_valid_tp_outer_with_profit_fallback(s.state.side, boll)
            logger.warning(
                "MIDDLE_RUNNER_MIDDLE_PROFIT_INSUFFICIENT_SINGLE_OUTER | "
                "side=%s effective_breakeven=%.4f required_middle=%.4f "
                "tp_boll_middle=%s structure_middle=%.4f selected_outer=%.4f outer_source=%s candle_ts=%s",
                s.state.side,
                effective_be,
                required_middle,
                s._format_optional_price(getattr(boll, "tp_middle", None)),
                boll.middle,
                outer,
                outer_src,
                boll.candle_ts_ms,
            )
            tp_price, tp_mode = s._fallback_to_single_outer_due_middle_profit_insufficient(
                side=s.state.side,
                boll=boll,
                ts_ms=ts_ms,
                reason="middle_runner_pending_middle_profit_insufficient",
            )
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
            reason_override = "middle_runner_middle_profit_insufficient_single_outer"
        else:
            partial_tp_ratio = s.state.middle_runner_first_close_ratio or min(
                max(s.config.middle_runner_first_close_ratio, 0.1), 0.95)
            tp_plan = "MIDDLE_RUNNER"
            s.state.middle_runner_first_tp_price = partial_tp_price
            s.state.middle_runner_final_tp_price = tp_price

        return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override

    def _apply_three_stage_enabled_branch(
        self,
        tp_price: float,
        boll: BollSnapshot,
        ts_ms: int,
        reason_override: str | None,
    ) -> tuple[float, TpMode, float | None, float, str, str | None]:
        """Three-Stage enabled branch."""
        s = self.strategy

        tp_price, _tp_src = s._select_three_stage_tp2_outer(s.state.side, boll)
        tp_mode: TpMode = "UPPER" if s.state.side == "LONG" else "LOWER"

        # ── Middle Bucket Split for Three-Stage ──────────────────────
        if (
            s.config.middle_bucket_split_enabled
            and not s.state.three_stage_tp1_consumed
            and not s.state.trend_runner_active
        ):
            split_result = self._apply_middle_bucket_split_for_three_stage(boll)

            if split_result.action == "SPLIT":
                return (
                    tp_price, tp_mode,
                    split_result.partial_tp_price, split_result.partial_tp_ratio,
                    split_result.tp_plan or "THREE_STAGE_RUNNER",
                    reason_override,
                )

            if split_result.action == "UNSPLIT_SLOW_MIDDLE":
                # BOLL15 insufficient, BOLL20 sufficient — use BOLL20 middle as
                # the full unsplit middle bucket.  MUST NOT fall back to outer.
                partial_tp_price = split_result.partial_tp_price  # boll.middle
                partial_tp_ratio = split_result.partial_tp_ratio
                tp_plan = "THREE_STAGE_RUNNER"
                # Set Three-Stage targets: tp1 = BOLL20 middle, tp2 = selected outer
                s.state.three_stage_tp1_price = partial_tp_price
                s.state.three_stage_tp2_price = tp_price
                return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override

            # FALLBACK_OUTER / INVALID / DISABLED — fall through to old logic

        partial_tp_price, _ptp_src = s._select_valid_tp_middle_with_profit_fallback(s.state.side, boll)
        updated = partial_tp_price is not None and s._update_three_stage_dynamic_targets_without_reset(
            s.state.side, boll)

        if not updated:
            old_tp1 = s.state.three_stage_tp1_price
            old_tp2 = s.state.three_stage_tp2_price
            effective_be = s._effective_breakeven_for_tp_selection(s.state.side)
            required_middle = s._required_middle_for_profit(s.state.side, effective_be)
            outer, outer_src = s._select_three_stage_tp2_outer(s.state.side, boll)
            logger.warning(
                "THREE_STAGE_MIDDLE_PROFIT_INSUFFICIENT_SINGLE_OUTER | "
                "side=%s effective_breakeven=%.4f required_middle=%.4f "
                "tp_boll_middle=%s structure_middle=%.4f selected_outer=%.4f outer_source=%s "
                "old_tp1=%s old_tp2=%s candle_ts=%s",
                s.state.side,
                effective_be,
                required_middle,
                s._format_optional_price(getattr(boll, "tp_middle", None)),
                boll.middle,
                outer,
                outer_src,
                f"{old_tp1:.4f}" if old_tp1 is not None else "-",
                f"{old_tp2:.4f}" if old_tp2 is not None else "-",
                boll.candle_ts_ms,
            )
            tp_price, tp_mode = s._fallback_to_single_outer_due_middle_profit_insufficient(
                side=s.state.side,
                boll=boll,
                ts_ms=ts_ms,
                reason="three_stage_dynamic_middle_profit_insufficient",
            )
            partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
            reason_override = "three_stage_middle_profit_insufficient_single_outer"
        else:
            tp1_ratio = s.state.three_stage_tp1_ratio
            partial_tp_ratio = tp1_ratio
            tp_plan = "THREE_STAGE_RUNNER"

        return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override

    def _apply_normal_plan_selection_branch(
        self,
        tp_price: float,
        tp_mode: TpMode,
        boll: BollSnapshot,
        ts_ms: int,
        reason_override: str | None,
        *,
        force_reconcile: bool = False,
    ) -> tuple[float, TpMode, float | None, float, str, str | None]:
        """Normal _select_tp_plan branch with middle/three-stage runner setup.

        When ``force_reconcile=True`` (startup path), the branch also
        attempts to re-apply Middle Bucket Split for THREE_STAGE_RUNNER
        and MIDDLE_RUNNER plans — this ensures startup reconcile can
        recover split state that was not persisted across restarts.
        """
        s = self.strategy

        partial_tp_price, partial_tp_ratio, tp_plan = s._select_tp_plan(
            s.state.side, tp_price, s.state.layers,
            tp_mode=tp_mode, boll=boll,
        )

        if tp_plan == "MIDDLE_RUNNER":
            tp_price, _tp_src = s._select_valid_tp_outer_with_profit_fallback(s.state.side, boll)
        if tp_plan == "THREE_STAGE_RUNNER":
            tp_price, _tp_src = s._select_three_stage_tp2_outer(s.state.side, boll)

        if tp_plan == "MIDDLE_RUNNER":
            # ── Startup reconcile: re-apply Middle Bucket Split ──────
            if force_reconcile and s.config.middle_bucket_split_enabled:
                split_result = self._apply_middle_bucket_split_for_middle_runner(boll)

                if split_result.action == "SPLIT":
                    partial_tp_price = split_result.partial_tp_price
                    partial_tp_ratio = split_result.partial_tp_ratio
                    tp_plan = split_result.tp_plan or "MIDDLE_RUNNER"
                    s.state.middle_runner_first_tp_price = partial_tp_price
                    s.state.middle_runner_final_tp_price = tp_price
                    s.state.middle_runner_pending = True
                    reason_override = reason_override or "startup_force_tp_reconcile"
                    return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override

                if split_result.action == "UNSPLIT_SLOW_MIDDLE":
                    partial_tp_price = split_result.partial_tp_price
                    partial_tp_ratio = split_result.partial_tp_ratio
                    tp_plan = "MIDDLE_RUNNER"
                    s.state.middle_runner_first_tp_price = partial_tp_price
                    s.state.middle_runner_final_tp_price = tp_price
                    s.state.middle_runner_pending = True
                    reason_override = reason_override or "startup_force_tp_reconcile"
                    return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override

                # FALLBACK_OUTER / INVALID / DISABLED → fall through

            s._set_middle_runner_planned(partial_tp_price, tp_price)
        elif tp_plan == "THREE_STAGE_RUNNER":
            # ── Startup reconcile: re-apply Middle Bucket Split ──────
            if force_reconcile and s.config.middle_bucket_split_enabled:
                # Ensure THREE_STAGE state fields are initialised before the
                # split helper reads them (tp1_ratio, etc. may be zero if the
                # THREE_STAGE plan was just recovered from a stale SINGLE cap).
                tp1_ratio, tp2_ratio, runner_ratio = s._normalized_three_stage_ratios()
                s.state.three_stage_tp1_ratio = tp1_ratio
                s.state.three_stage_tp2_ratio = tp2_ratio
                s.state.three_stage_runner_ratio = runner_ratio
                s.state.three_stage_runner_enabled_for_position = True

                split_result = self._apply_middle_bucket_split_for_three_stage(boll)

                if split_result.action == "SPLIT":
                    partial_tp_price = split_result.partial_tp_price
                    partial_tp_ratio = split_result.partial_tp_ratio
                    tp_plan = split_result.tp_plan or "THREE_STAGE_RUNNER"
                    reason_override = reason_override or "startup_force_tp_reconcile"
                    return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override

                if split_result.action == "UNSPLIT_SLOW_MIDDLE":
                    partial_tp_price = split_result.partial_tp_price
                    partial_tp_ratio = split_result.partial_tp_ratio
                    tp_plan = "THREE_STAGE_RUNNER"
                    s.state.three_stage_tp1_price = partial_tp_price
                    s.state.three_stage_tp2_price = tp_price
                    reason_override = reason_override or "startup_force_tp_reconcile"
                    return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override

                # FALLBACK_OUTER / INVALID / DISABLED → fall through

            if not s._update_three_stage_dynamic_targets_without_reset(s.state.side, boll):
                tp_price, tp_mode = s._fallback_to_single_outer_due_middle_profit_insufficient(
                    side=s.state.side,
                    boll=boll,
                    ts_ms=ts_ms,
                    reason="selected_three_stage_middle_profit_insufficient",
                )
                partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
                reason_override = "three_stage_middle_profit_insufficient_single_outer"
        elif s.state.middle_runner_pending and not s.state.middle_runner_active:
            s._reset_middle_runner_state()
        elif (
            s.state.three_stage_runner_enabled_for_position
            and not s.state.trend_runner_active
            and not s._three_stage_waiting_tp2()
        ):
            s._reset_three_stage_runner_state()

        return tp_price, tp_mode, partial_tp_price, partial_tp_ratio, tp_plan, reason_override

    def _finalize_state_and_maybe_emit_intent(
        self,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        tp_price: float,
        tp_mode: TpMode,
        partial_tp_price: float | None,
        partial_tp_ratio: float,
        tp_plan: str,
        reason_override: str | None,
        force_reconcile: bool,
        old_runner_sl: float | None,
        old_trend_runner_tp: float | None,
        old_trend_runner_sl: float | None,
    ) -> TradeIntent | None:
        """Write final state, detect changes, and emit UPDATE_TP or skip."""
        s = self.strategy

        s.state.last_tp_update_ts_ms = ts_ms
        s.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms

        runner_sl_changed = (
            s.state.middle_runner_active
            and s.state.middle_runner_protective_sl_price is not None
            and (
                old_runner_sl is None
                or abs(
                    s.state.middle_runner_protective_sl_price - old_runner_sl
                ) / s.state.middle_runner_protective_sl_price >= 0.0001
            )
        )
        trend_runner_orders_changed = (
            s.state.trend_runner_active
            and (
                old_trend_runner_tp is None
                or old_trend_runner_sl is None
                or s.state.trend_runner_tp_price is None
                or s.state.trend_runner_sl_price is None
                or abs(
                    s.state.trend_runner_tp_price - old_trend_runner_tp
                ) / s.state.trend_runner_tp_price >= 0.0001
                or abs(
                    s.state.trend_runner_sl_price - old_trend_runner_sl
                ) / s.state.trend_runner_sl_price >= 0.0001
            )
        )

        if (
            reason_override is None
            and s._tp_plan_unchanged(tp_price, partial_tp_price, partial_tp_ratio, tp_plan)
            and not runner_sl_changed
            and not trend_runner_orders_changed
            and not force_reconcile
        ):
            s.state.startup_force_tp_reconcile = False
            logger.info(
                "TP_UPDATE_SKIPPED | reason=plan_unchanged side=%s mode=%s plan=%s candle_ts=%s current_tp=%.4f target_tp=%.4f partial_tp=%s avg_entry=%.4f breakeven=%.4f",
                s.state.side,
                tp_mode,
                tp_plan,
                boll.candle_ts_ms,
                s.state.tp_price,
                tp_price,
                f"{partial_tp_price:.4f}" if partial_tp_price is not None else "-",
                s.state.avg_entry_price,
                s.state.breakeven_price,
            )
            return None

        s.state.startup_force_tp_reconcile = False
        s.state.tp_price = tp_price
        s.state.tp_mode = tp_mode
        s.state.partial_tp_price = partial_tp_price
        s.state.partial_tp_ratio = partial_tp_ratio
        s.state.tp_plan = tp_plan

        # Clear middle bucket split state if the selected plan does not use it
        self._clear_middle_bucket_split_if_not_active(tp_plan)

        if tp_plan == "MIDDLE_RUNNER":
            s._set_middle_runner_planned(partial_tp_price, tp_price)
        if tp_plan == "THREE_STAGE_RUNNER":
            # When middle bucket split is active, preserve the effective TP1 price
            # (weighted average of fast/slow) — do NOT let dynamic target update
            # overwrite it with plain BOLL15 middle.
            if getattr(s.state, "middle_bucket_split_active", False):
                # Only update TP2 (outer); keep split effective TP1 unchanged
                tp2_price, _tp2_src = s._select_three_stage_tp2_outer(s.state.side, boll)
                s.state.three_stage_tp2_price = tp2_price
                s.state.tp_price = tp2_price
            elif not s._update_three_stage_dynamic_targets_without_reset(s.state.side, boll):
                tp_price, tp_mode = s._fallback_to_single_outer_due_middle_profit_insufficient(
                    side=s.state.side,
                    boll=boll,
                    ts_ms=ts_ms,
                    reason="final_three_stage_middle_profit_insufficient",
                )
                partial_tp_price, partial_tp_ratio, tp_plan = None, 0.0, "SINGLE"
                s.state.tp_price = tp_price
                s.state.tp_mode = tp_mode
                s.state.partial_tp_price = None
                s.state.partial_tp_ratio = 0.0
                s.state.tp_plan = "SINGLE"
                reason_override = reason_override or "three_stage_middle_profit_insufficient_single_outer"

        if s.state.trend_runner_active and s.config.runner_dynamic_enabled:
            s.state.trend_runner_adjust_count += 1
            s.state.trend_runner_last_update_candle_ts_ms = boll.candle_ts_ms

        size = s.sizer.calculate(price, layer_index=s.state.layers)

        if reason_override is not None:
            reason_text = reason_override
        else:
            reason_text = (
                "startup_force_tp_reconcile" if force_reconcile
                else f"新15m K线更新止盈到{tp_mode}轨"
            )

        split_active = bool(getattr(s.state, "middle_bucket_split_active", False))
        split_fields = ""
        if split_active:
            split_fields = (
                f" middle_split_active=true"
                f" middle_split_fast_price={getattr(s.state, 'middle_bucket_split_fast_price', None)}"
                f" middle_split_slow_price={getattr(s.state, 'middle_bucket_split_slow_price', None)}"
                f" middle_split_effective_price={getattr(s.state, 'middle_bucket_split_effective_price', None)}"
                f" middle_split_fast_total_ratio={getattr(s.state, 'middle_bucket_split_fast_total_ratio', 0.0):.4f}"
                f" middle_split_slow_total_ratio={getattr(s.state, 'middle_bucket_split_slow_total_ratio', 0.0):.4f}"
                f" middle_split_reason={getattr(s.state, 'middle_bucket_split_reason', None)}"
            )
        logger.warning(
            "TP_SELECTED | reason=%s side=%s mode=%s plan=%s partial_tp=%s partial_ratio=%.2f avg_entry=%.4f breakeven=%.4f candle_ts=%s middle=%.4f upper=%.4f lower=%.4f final_tp=%.4f force_reconcile=%s%s",
            reason_text,
            s.state.side,
            tp_mode,
            tp_plan,
            f"{partial_tp_price:.4f}" if partial_tp_price is not None else "-",
            partial_tp_ratio,
            s.state.avg_entry_price,
            s.state.breakeven_price,
            boll.candle_ts_ms,
            boll.middle,
            boll.upper,
            boll.lower,
            tp_price,
            force_reconcile,
            split_fields,
        )
        s._log_tp_boll_price_selected(
            phase="waiting_tp2" if s._three_stage_waiting_tp2() else "update",
            boll=boll,
            tp_price=tp_price,
            tp_mode=tp_mode,
            tp_plan=tp_plan,
            partial_tp_price=partial_tp_price,
            tp1_price=s.state.three_stage_tp1_price if tp_plan == "THREE_STAGE_RUNNER" else None,
            tp2_price=s.state.three_stage_tp2_price if tp_plan == "THREE_STAGE_RUNNER" else None,
            first_tp_price=s.state.middle_runner_first_tp_price if tp_plan == "MIDDLE_RUNNER" else None,
            final_tp_price=s.state.middle_runner_final_tp_price if tp_plan == "MIDDLE_RUNNER" else None,
        )
        return s._intent(
            "UPDATE_TP", s.state.side, price, s.state.layers,
            tp_price, reason_text, size, boll, cvd, ts_ms,
        )

    # ------------------------------------------------------------------
    # Middle Bucket Split helpers
    # ------------------------------------------------------------------

    def _reset_middle_bucket_split_state(self) -> None:
        """Clear all middle-bucket-split state fields."""
        clear_middle_bucket_split_state(self.strategy.state, reason=None)

    # ------------------------------------------------------------------
    # Pre-TP1 degrade stage refresh (startup reconcile only)
    # ------------------------------------------------------------------

    def _refresh_pre_tp1_degrade_stage_before_startup_reconcile(
        self,
        *,
        ts_ms: int,
    ) -> None:
        """Re-compute the pre-TP1 degrade cap before startup TP reconcile.

        This is ONLY called when ``force_reconcile=True`` (startup path).
        It re-evaluates the stale degrade stage based on current position
        age so that a saved SINGLE cap doesn't permanently block recovery
        to THREE_STAGE_RUNNER or MIDDLE_RUNNER when market conditions
        actually permit it.

        **Post-TP1 guard**: if TP1 has already been consumed, or a
        trend/middle runner is active, or partial TP has been consumed,
        then pre-TP1 degrade stage is NOT refreshed — it is cleared
        instead to avoid polluting post-TP1 / runner state.

        Delegates to the shared pure helper
        ``decide_pre_tp1_degrade_stage_for_replan()``.
        """
        s = self.strategy
        old_stage = s.state.three_stage_pre_tp1_degrade_stage
        old_degraded_ts_ms = s.state.three_stage_pre_tp1_degraded_ts_ms
        pre_reconcile_tp_plan = s.state.tp_plan
        pre_reconcile_three_stage_enabled_for_position = (
            s.state.three_stage_runner_enabled_for_position
        )

        # ── Guard: only allow pre-TP1 degrade stage refresh when truly pre-TP1 ─
        is_pre_tp1 = not (
            s.state.three_stage_tp1_consumed
            or s.state.three_stage_tp2_consumed
            or s.state.trend_runner_active
            or s.state.middle_runner_active
            or s.state.partial_tp_consumed
        )

        three_stage_replan_cap_applicable = (
            s.config.three_stage_runner_enabled
            or pre_reconcile_three_stage_enabled_for_position
            or pre_reconcile_tp_plan == "THREE_STAGE_RUNNER"
        )

        decision = decide_pre_tp1_degrade_stage_for_replan(
            first_entry_ts_ms=s.state.first_entry_ts_ms,
            ts_ms=ts_ms,
            is_pre_tp1=is_pre_tp1,
            three_stage_replan_cap_applicable=three_stage_replan_cap_applicable,
            degrade_enabled=s.config.three_stage_pre_tp1_degrade_enabled,
            middle_runner_after_seconds=s.config.three_stage_pre_tp1_middle_runner_after_seconds,
            single_after_seconds=s.config.three_stage_pre_tp1_single_after_seconds,
        )

        # When not pre-TP1, clear degrade stage (post-TP1 should never hold
        # pre-TP1 cap).  Do NOT reset TP1/TP2/runner fields, tp_plan, or
        # protective SL.  Do NOT emit an intent solely because of this clear.
        if not is_pre_tp1:
            s.state.three_stage_pre_tp1_degrade_stage = None
            s.state.three_stage_pre_tp1_degraded_ts_ms = 0
        else:
            s.state.three_stage_pre_tp1_degrade_stage = decision.new_stage
            s.state.three_stage_pre_tp1_degraded_ts_ms = decision.degraded_ts_ms

        # Always log on startup reconcile — it's a low-frequency path and
        # the log entry is invaluable for diagnosing stale-stage issues.
        logger.warning(
            "STARTUP_PRE_TP1_DEGRADE_REFRESHED_BEFORE_TP_RECONCILE | "
            "old_stage=%s new_stage=%s old_degraded_ts_ms=%s "
            "new_degraded_ts_ms=%s age_seconds=%.1f first_entry_ts_ms=%s "
            "middle_after_seconds=%s single_after_seconds=%s "
            "is_pre_tp1=%s "
            "three_stage_replan_cap_applicable=%s "
            "three_stage_runner_enabled=%s "
            "pre_reconcile_tp_plan=%s "
            "pre_reconcile_three_stage_enabled_for_position=%s "
            "tp1_consumed=%s tp2_consumed=%s "
            "trend_runner_active=%s middle_runner_active=%s "
            "partial_tp_consumed=%s "
            "reason=%s",
            old_stage,
            decision.new_stage,
            old_degraded_ts_ms,
            decision.degraded_ts_ms,
            decision.age_seconds,
            s.state.first_entry_ts_ms,
            s.config.three_stage_pre_tp1_middle_runner_after_seconds,
            s.config.three_stage_pre_tp1_single_after_seconds,
            is_pre_tp1,
            three_stage_replan_cap_applicable,
            s.config.three_stage_runner_enabled,
            pre_reconcile_tp_plan,
            pre_reconcile_three_stage_enabled_for_position,
            s.state.three_stage_tp1_consumed,
            s.state.three_stage_tp2_consumed,
            s.state.trend_runner_active,
            s.state.middle_runner_active,
            s.state.partial_tp_consumed,
            decision.reason,
        )

    def _apply_middle_bucket_split_for_three_stage(
        self,
        boll: BollSnapshot,
    ) -> MiddleBucketSplitApplyResult:
        """Try to enable middle bucket split for the Three-Stage branch.

        Delegates to the shared helper in middle_bucket_split_apply.py so that
        the entry path and the TP-update path use the same logic.
        """
        return apply_three_stage_middle_bucket_split(
            strategy=self.strategy,
            boll=boll,
        )

    def _apply_middle_bucket_split_for_middle_runner(
        self,
        boll: BollSnapshot,
    ) -> MiddleBucketSplitApplyResult:
        """Try to enable middle bucket split for the Middle Runner branch.

        Delegates to the shared helper in middle_bucket_split_apply.py so that
        the entry path and the TP-update path use the same logic.
        """
        return apply_middle_runner_bucket_split(
            strategy=self.strategy,
            boll=boll,
        )

    def _clear_middle_bucket_split_if_not_active(self, tp_plan: str) -> None:
        """If the current TP plan is not using split, clear any stale split state."""
        s = self.strategy
        if not s.config.middle_bucket_split_enabled:
            self._reset_middle_bucket_split_state()
            return
        if tp_plan not in ("THREE_STAGE_RUNNER", "MIDDLE_RUNNER"):
            if s.state.middle_bucket_split_active:
                self._reset_middle_bucket_split_state()
