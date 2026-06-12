"""Entry / Add Flow Coordinator — extracted from BollCvdReclaimStrategy.

This module owns the entry/add orchestration methods so that the strategy
class itself only retains thin wrappers.  The coordinator reads and writes
strategy.state freely and delegates to the strategy's existing helpers.

Phase 41 of the refactoring plan:
    Round 2, Item 41 — Extract Strategy Entry/Add Flow Coordinator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.strategies.middle_bucket_split_apply import (
    apply_middle_runner_bucket_split,
    apply_three_stage_middle_bucket_split,
)
from src.strategies.tp_lifecycle import recover_pre_tp1_degrade_stage_after_add
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.indicators.cvd_tracker import CvdSnapshot
    from src.monitors.boll_band_breakout_monitor import BollSnapshot
    from src.strategies.boll_cvd_reclaim_strategy import (
        BollCvdReclaimStrategy,
        PositionSide,
        TradeIntent,
        TradeIntentType,
    )

logger = get_logger(__name__)


class EntryAddFlowCoordinator:
    """Orchestrates entry/open and add-to-position flows for BollCvdReclaimStrategy.

    The coordinator holds a reference to the owning strategy and can read/write
    ``strategy.state`` directly.  It delegates to the strategy's existing helper
    methods for gates, TP selection, state reset, etc.
    """

    def __init__(self, strategy: BollCvdReclaimStrategy) -> None:
        self.strategy = strategy

    # ------------------------------------------------------------------
    # maybe_open_or_add_long
    # ------------------------------------------------------------------

    def maybe_open_or_add_long(
        self,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
    ) -> TradeIntent | None:
        strategy = self.strategy
        if strategy.state.side is None:
            return strategy._open_position("LONG", "OPEN_LONG", price, ts_ms, boll, cvd,
                                           "下轨出轨深度达标 + 低点附近快速CVD回流/跌不动")
        if strategy.state.side != "LONG":
            return None
        if strategy.state.near_tp_add_disabled:
            strategy._log_add_skip_once_per_window(reason="near_tp_protected", side="LONG", price=price, ts_ms=ts_ms)
            return None
        if getattr(strategy.state, "middle_bucket_split_add_disabled", False):
            strategy._log_add_skip_once_per_window(reason="middle_bucket_fast_consumed", side="LONG", price=price,
                                                   ts_ms=ts_ms)
            return None
        if strategy.state.trend_runner_active:
            strategy._log_add_skip_once_per_window(reason="trend_runner_active", side="LONG", price=price, ts_ms=ts_ms)
            return None
        if (
                strategy.state.three_stage_runner_enabled_for_position
                and (strategy.state.three_stage_tp1_consumed or strategy.state.three_stage_tp2_consumed)
        ):
            strategy._log_add_skip_once_per_window(reason="three_stage_after_tp1", side="LONG", price=price, ts_ms=ts_ms)
            return None
        if strategy.state.middle_runner_add_disabled or strategy.state.middle_runner_active:
            strategy._log_add_skip_once_per_window(reason="middle_runner_active", side="LONG", price=price, ts_ms=ts_ms)
            return None
        if strategy.state.layers >= strategy.config.max_layers:
            return None
        if strategy.state.last_entry_price is None:
            return None
        target_layer = strategy.state.layers + 1
        timing_ok, timing_reason = strategy._add_timing_passed("LONG", price, ts_ms, target_layer)
        if not timing_ok:
            strategy._log_add_timing_skipped("LONG", timing_reason, price, ts_ms, target_layer)
            return None
        gap_ok, gap_pct, required_price = strategy._add_gap_passed("LONG", price, target_layer)
        if not gap_ok:
            logger.info(
                "ADD_SKIPPED | reason=add_gap side=LONG price=%.4f layers=%s target_layer=%s last_entry=%.4f required_price=%.4f gap_pct=%.4f%%",
                price,
                strategy.state.layers,
                target_layer,
                strategy.state.last_entry_price,
                required_price,
                gap_pct * 100,
            )
            return None
        logger.info(
            "ADD_GAP_PASSED | side=LONG price=%.4f layers=%s target_layer=%s last_entry=%.4f required_price=%.4f gap_pct=%.4f%%",
            price,
            strategy.state.layers,
            target_layer,
            strategy.state.last_entry_price,
            required_price,
            gap_pct * 100,
        )
        avg_improvement_ok, improvement_pct, projected_avg = strategy._add_avg_improvement_passed("LONG", price,
                                                                                                  target_layer)
        if not avg_improvement_ok:
            logger.info(
                "ADD_SKIPPED | reason=avg_improvement side=LONG price=%.4f layers=%s target_layer=%s avg_entry=%.4f projected_avg_entry=%.4f improvement_pct=%.6f required_improvement_pct=%.6f",
                price,
                strategy.state.layers,
                target_layer,
                strategy.state.avg_entry_price,
                projected_avg,
                improvement_pct,
                strategy.config.add_min_avg_improvement_pct,
            )
            return None
        logger.info(
            "ADD_AVG_IMPROVEMENT_PASSED | side=LONG price=%.4f layers=%s target_layer=%s avg_entry=%.4f projected_avg_entry=%.4f improvement_pct=%.6f required_improvement_pct=%.6f",
            price,
            strategy.state.layers,
            target_layer,
            strategy.state.avg_entry_price,
            projected_avg,
            improvement_pct,
            strategy.config.add_min_avg_improvement_pct,
        )
        return strategy._open_position(
            "LONG",
            "ADD_LONG",
            price,
            ts_ms,
            boll,
            cvd,
            f"距离上一多仓超过{gap_pct * 100:.2f}% + 补仓后均价改善{improvement_pct * 100:.2f}% + 新出轨深度达标后低点附近再次跌不动",
        )

    # ------------------------------------------------------------------
    # maybe_open_or_add_short
    # ------------------------------------------------------------------

    def maybe_open_or_add_short(
        self,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
    ) -> TradeIntent | None:
        strategy = self.strategy
        if strategy.state.side is None:
            return strategy._open_position("SHORT", "OPEN_SHORT", price, ts_ms, boll, cvd,
                                           "上轨出轨深度达标 + 高点附近快速CVD转弱/涨不动")
        if strategy.state.side != "SHORT":
            return None
        if strategy.state.near_tp_add_disabled:
            strategy._log_add_skip_once_per_window(reason="near_tp_protected", side="SHORT", price=price, ts_ms=ts_ms)
            return None
        if getattr(strategy.state, "middle_bucket_split_add_disabled", False):
            strategy._log_add_skip_once_per_window(reason="middle_bucket_fast_consumed", side="SHORT", price=price,
                                                   ts_ms=ts_ms)
            return None
        if strategy.state.trend_runner_active:
            strategy._log_add_skip_once_per_window(reason="trend_runner_active", side="SHORT", price=price, ts_ms=ts_ms)
            return None
        if (
                strategy.state.three_stage_runner_enabled_for_position
                and (strategy.state.three_stage_tp1_consumed or strategy.state.three_stage_tp2_consumed)
        ):
            strategy._log_add_skip_once_per_window(reason="three_stage_after_tp1", side="SHORT", price=price, ts_ms=ts_ms)
            return None
        if strategy.state.middle_runner_add_disabled or strategy.state.middle_runner_active:
            strategy._log_add_skip_once_per_window(reason="middle_runner_active", side="SHORT", price=price, ts_ms=ts_ms)
            return None
        if strategy.state.layers >= strategy.config.max_layers:
            return None
        if strategy.state.last_entry_price is None:
            return None
        target_layer = strategy.state.layers + 1
        timing_ok, timing_reason = strategy._add_timing_passed("SHORT", price, ts_ms, target_layer)
        if not timing_ok:
            strategy._log_add_timing_skipped("SHORT", timing_reason, price, ts_ms, target_layer)
            return None
        gap_ok, gap_pct, required_price = strategy._add_gap_passed("SHORT", price, target_layer)
        if not gap_ok:
            logger.info(
                "ADD_SKIPPED | reason=add_gap side=SHORT price=%.4f layers=%s target_layer=%s last_entry=%.4f required_price=%.4f gap_pct=%.4f%%",
                price,
                strategy.state.layers,
                target_layer,
                strategy.state.last_entry_price,
                required_price,
                gap_pct * 100,
            )
            return None
        logger.info(
            "ADD_GAP_PASSED | side=SHORT price=%.4f layers=%s target_layer=%s last_entry=%.4f required_price=%.4f gap_pct=%.4f%%",
            price,
            strategy.state.layers,
            target_layer,
            strategy.state.last_entry_price,
            required_price,
            gap_pct * 100,
        )
        avg_improvement_ok, improvement_pct, projected_avg = strategy._add_avg_improvement_passed("SHORT", price,
                                                                                                  target_layer)
        if not avg_improvement_ok:
            logger.info(
                "ADD_SKIPPED | reason=avg_improvement side=SHORT price=%.4f layers=%s target_layer=%s avg_entry=%.4f projected_avg_entry=%.4f improvement_pct=%.6f required_improvement_pct=%.6f",
                price,
                strategy.state.layers,
                target_layer,
                strategy.state.avg_entry_price,
                projected_avg,
                improvement_pct,
                strategy.config.add_min_avg_improvement_pct,
            )
            return None
        logger.info(
            "ADD_AVG_IMPROVEMENT_PASSED | side=SHORT price=%.4f layers=%s target_layer=%s avg_entry=%.4f projected_avg_entry=%.4f improvement_pct=%.6f required_improvement_pct=%.6f",
            price,
            strategy.state.layers,
            target_layer,
            strategy.state.avg_entry_price,
            projected_avg,
            improvement_pct,
            strategy.config.add_min_avg_improvement_pct,
        )
        return strategy._open_position(
            "SHORT",
            "ADD_SHORT",
            price,
            ts_ms,
            boll,
            cvd,
            f"距离上一空仓超过{gap_pct * 100:.2f}% + 补仓后均价改善{improvement_pct * 100:.2f}% + 新出轨深度达标后高点附近再次涨不动",
        )

    # ------------------------------------------------------------------
    # open_position
    # ------------------------------------------------------------------

    def open_position(
            self,
            side: PositionSide,
            intent_type: TradeIntentType,
            price: float,
            ts_ms: int,
            boll: BollSnapshot,
            cvd: CvdSnapshot,
            reason: str,
    ) -> TradeIntent:
        strategy = self.strategy
        next_layer = strategy.state.layers + 1
        size = strategy.sizer.calculate(price, layer_index=next_layer)
        if next_layer == 1:
            strategy.state.first_entry_ts_ms = ts_ms
            strategy.state.add_freeze_until_ts_ms = 0
            strategy.state.add_freeze_penalty_count = 0
            strategy.state.three_stage_pre_tp1_degrade_stage = None
            strategy.state.three_stage_pre_tp1_degraded_ts_ms = 0
            strategy.state.sidecar_enabled_for_position = bool(getattr(strategy.sizer.config, "sidecar_enabled", False))
            strategy.state.sidecar_margin_pct = (
                float(getattr(strategy.sizer.config, "sidecar_margin_pct", 0.0) or 0.0)
                if strategy.state.sidecar_enabled_for_position
                else 0.0
            )
            strategy.state.sidecar_tp_pct = (
                float(getattr(strategy.sizer.config, "sidecar_tp_pct", 0.0) or 0.0)
                if strategy.state.sidecar_enabled_for_position
                else 0.0
            )
            strategy.state.sidecar_total_qty = 0.0
            strategy.state.sidecar_open_qty = 0.0
            strategy.state.sidecar_total_notional = 0.0
            strategy.state.sidecar_realized_qty = 0.0
            strategy.state.sidecar_legs = []
            strategy.state.sidecar_dirty = False
            strategy.state.sidecar_halt_reason = None
            strategy.state.position_cost_entry_notional = 0.0
            strategy.state.position_cost_exit_notional = 0.0
            strategy.state.position_cost_remaining_qty = 0.0
            strategy.state.net_remaining_breakeven_price = 0.0
            strategy.state.last_add_skip_log_reason = None
            strategy.state.last_add_skip_log_ts_ms = 0
        strategy.state.side = side
        strategy._update_position_cost(price, size.eth_qty)
        if next_layer > 1:
            recover_pre_tp1_degrade_stage_after_add(
                state=strategy.state,
                position_age_seconds=strategy._three_stage_pre_tp1_age_seconds(ts_ms),
            )
        strategy.state.partial_tp_consumed = False
        strategy._reset_near_tp_state()
        strategy._reset_middle_runner_state()
        strategy._reset_three_stage_runner_state()
        tp_price, tp_mode = strategy._select_tp_price(side, boll)
        partial_tp_price, partial_tp_ratio, tp_plan = strategy._select_tp_plan(side, tp_price, next_layer, tp_mode=tp_mode,
                                                                               boll=boll)
        if tp_plan == "MIDDLE_RUNNER":
            tp_price, _tp_src = strategy._select_valid_tp_outer_with_profit_fallback(side, boll)
        if tp_plan == "THREE_STAGE_RUNNER":
            tp_price, _tp_src = strategy._select_three_stage_tp2_outer(side, boll)
        if tp_mode != "MIDDLE":
            reason = f"{reason} + 中轨净利润不足阈值，TP切换到{tp_mode}"
        if tp_plan == "SPLIT_PARTIAL_FINAL":
            reason = f"{reason} + 总层数>= {strategy.config.split_tp_min_layers}，启用分批止盈"
        if tp_plan == "MIDDLE_RUNNER":
            reason = f"{reason} + 中轨先平{partial_tp_ratio * 100:.0f}%，剩余runner到外轨"
        if tp_plan == "THREE_STAGE_RUNNER":
            reason = f"{reason} + 三段式趋势Runner：中轨{strategy.config.three_stage_tp1_ratio * 100:.0f}%/外轨{strategy.config.three_stage_tp2_ratio * 100:.0f}%/Runner{strategy.config.three_stage_runner_ratio * 100:.0f}%"
        strategy.state.layers = next_layer
        strategy.state.last_entry_price = price
        strategy.state.tp_price = tp_price
        strategy.state.tp_mode = tp_mode
        strategy.state.partial_tp_price = partial_tp_price
        strategy.state.partial_tp_ratio = partial_tp_ratio
        strategy.state.tp_plan = tp_plan
        if tp_plan == "MIDDLE_RUNNER":
            strategy._set_middle_runner_planned(partial_tp_price, tp_price)
        if tp_plan == "THREE_STAGE_RUNNER":
            strategy._set_three_stage_runner_planned(side, boll)

        # ── Middle Bucket Split on initial entry ────────────────────────────
        if tp_plan == "THREE_STAGE_RUNNER" and strategy.config.middle_bucket_split_enabled:
            split_result = apply_three_stage_middle_bucket_split(
                strategy=strategy, boll=boll,
            )
            if split_result.action == "SPLIT":
                partial_tp_price = split_result.partial_tp_price
                partial_tp_ratio = split_result.partial_tp_ratio
                strategy.state.partial_tp_price = partial_tp_price
                strategy.state.partial_tp_ratio = partial_tp_ratio
            elif split_result.action == "UNSPLIT_SLOW_MIDDLE":
                partial_tp_price = split_result.partial_tp_price
                partial_tp_ratio = split_result.partial_tp_ratio
                strategy.state.partial_tp_price = partial_tp_price
                strategy.state.partial_tp_ratio = partial_tp_ratio
            # FALLBACK_OUTER / INVALID / DISABLED — keep existing behaviour

        if tp_plan == "MIDDLE_RUNNER" and strategy.config.middle_bucket_split_enabled:
            split_result = apply_middle_runner_bucket_split(
                strategy=strategy, boll=boll,
            )
            if split_result.action == "SPLIT":
                partial_tp_price = split_result.partial_tp_price
                partial_tp_ratio = split_result.partial_tp_ratio
                strategy.state.partial_tp_price = partial_tp_price
                strategy.state.partial_tp_ratio = partial_tp_ratio
            elif split_result.action == "UNSPLIT_SLOW_MIDDLE":
                partial_tp_price = split_result.partial_tp_price
                partial_tp_ratio = split_result.partial_tp_ratio
                strategy.state.partial_tp_price = partial_tp_price
                strategy.state.partial_tp_ratio = partial_tp_ratio
            # FALLBACK_OUTER / INVALID / DISABLED — keep existing behaviour

        strategy.state.last_order_ts_ms = ts_ms
        strategy.state.last_tp_update_ts_ms = ts_ms
        strategy.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms
        logger.info(
            "TP_SELECTED | reason=entry side=%s mode=%s plan=%s partial_tp=%s partial_ratio=%.2f avg_entry=%.4f breakeven=%.4f candle_ts=%s middle=%.4f upper=%.4f lower=%.4f final_tp=%.4f",
            side,
            tp_mode,
            tp_plan,
            f"{partial_tp_price:.4f}" if partial_tp_price is not None else "-",
            partial_tp_ratio,
            strategy.state.avg_entry_price,
            strategy.state.breakeven_price,
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
                strategy.state.three_stage_tp1_price or 0.0,
                strategy.state.three_stage_tp1_ratio,
                strategy.state.three_stage_tp2_price or 0.0,
                strategy.state.three_stage_tp2_ratio,
                f"{strategy.state.trend_runner_tp_price:.4f}" if strategy.state.trend_runner_tp_price is not None else "-",
                f"{strategy.state.trend_runner_sl_price:.4f}" if strategy.state.trend_runner_sl_price is not None else "-",
                strategy.state.three_stage_runner_ratio,
            )
        strategy._log_tp_boll_price_selected(
            phase="initial",
            boll=boll,
            tp_price=tp_price,
            tp_mode=tp_mode,
            tp_plan=tp_plan,
            partial_tp_price=partial_tp_price,
            tp1_price=strategy.state.three_stage_tp1_price if tp_plan == "THREE_STAGE_RUNNER" else None,
            tp2_price=strategy.state.three_stage_tp2_price if tp_plan == "THREE_STAGE_RUNNER" else None,
            first_tp_price=strategy.state.middle_runner_first_tp_price if tp_plan == "MIDDLE_RUNNER" else None,
            final_tp_price=strategy.state.middle_runner_final_tp_price if tp_plan == "MIDDLE_RUNNER" else None,
        )
        return strategy._intent(intent_type, side, price, next_layer, tp_price, reason, size, boll, cvd, ts_ms)
