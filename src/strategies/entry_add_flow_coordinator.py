"""Entry / Add Flow Coordinator — extracted from BollCvdReclaimStrategy.

This module owns the entry/add orchestration methods so that the strategy
class itself only retains thin wrappers.  The coordinator reads and writes
strategy.state freely and delegates to the strategy's existing helpers.

Phase 41 of the refactoring plan:
    Round 2, Item 41 — Extract Strategy Entry/Add Flow Coordinator.
"""

from __future__ import annotations

import copy
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
        strategy._log_add_skip_once_per_window(reason="add_disabled", side="LONG", price=price, ts_ms=ts_ms)
        return None

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
        strategy._log_add_skip_once_per_window(reason="add_disabled", side="SHORT", price=price, ts_ms=ts_ms)
        return None

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
    ) -> TradeIntent | None:
        strategy = self.strategy
        if intent_type in {"ADD_LONG", "ADD_SHORT"}:
            strategy._log_add_skip_once_per_window(reason="add_disabled", side=side, price=price, ts_ms=ts_ms)
            return None
        if strategy.state.side is not None:
            return None
        previous_state = copy.deepcopy(strategy.state)
        next_layer = 1
        entry_sl_price = strategy._entry_protective_sl_price(side)
        if entry_sl_price is None:
            if side == "LONG" and boll.lower > 0:
                entry_sl_price = boll.lower * (1 - strategy.config.entry_sl_buffer_pct)
            elif side == "SHORT" and boll.upper > 0:
                entry_sl_price = boll.upper * (1 + strategy.config.entry_sl_buffer_pct)
        if entry_sl_price is None:
            logger.warning("ENTRY_SKIPPED | reason=missing_entry_protective_sl side=%s price=%.4f", side, price)
            return None
        size = strategy.sizer.calculate(price, layer_index=next_layer, stop_price=entry_sl_price)
        if next_layer == 1:
            strategy.state.first_entry_ts_ms = ts_ms
            strategy.state.add_freeze_until_ts_ms = 0
            strategy.state.add_freeze_penalty_count = 0
            strategy.state.three_stage_pre_tp1_degrade_stage = None
            strategy.state.three_stage_pre_tp1_degraded_ts_ms = 0
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
        if tp_plan == "MIDDLE_RUNNER":
            reason = f"{reason} + 中轨先平{partial_tp_ratio * 100:.0f}%，剩余runner到外轨"
        if tp_plan == "THREE_STAGE_RUNNER":
            reason = f"{reason} + 三段式趋势Runner：中轨{strategy.config.three_stage_tp1_ratio * 100:.0f}%/外轨{strategy.config.three_stage_tp2_ratio * 100:.0f}%/Runner{strategy.config.three_stage_runner_ratio * 100:.0f}%"
        rr_target_price, rr_target_source = strategy._entry_reward_risk_target_price(
            side=side,
            boll=boll,
            final_tp_price=tp_price,
        )
        rr_ok, rr_reason, stop_distance_pct, reward_pct, reward_risk = strategy._entry_reward_risk_check(
            side=side,
            entry_price=price,
            tp_price=rr_target_price,
            stop_price=entry_sl_price,
        )
        if not rr_ok:
            strategy.state = previous_state
            logger.info(
                "ENTRY_SKIPPED | reason=%s side=%s price=%.4f tp=%.4f rr_target=%.4f rr_target_source=%s sl=%.4f stop_pct=%.6f reward_pct=%.6f reward_risk=%.4f min_reward_risk=%.4f",
                rr_reason,
                side,
                price,
                tp_price,
                rr_target_price,
                rr_target_source,
                entry_sl_price,
                stop_distance_pct,
                reward_pct,
                reward_risk,
                strategy.config.entry_min_reward_risk,
            )
            # Throttled no-entry reason log for reclaim observability
            reclaim_side = "LOWER" if side == "LONG" else "UPPER"
            strategy._log_reclaim_no_entry_reason(
                side=reclaim_side,
                reason="reward_risk_not_met",
                price=price,
                boll=boll,
                cvd=cvd,
            )
            return None
        reason = (
            f"{reason} + risk_size stop={entry_sl_price:.4f} "
            f"stop={stop_distance_pct * 100:.3f}% reward={reward_pct * 100:.3f}% R={reward_risk:.2f}"
            f" rr_target_source={rr_target_source} rr_target={rr_target_price:.4f}"
        )
        strategy.state.entry_protective_sl_price = entry_sl_price
        strategy.state.entry_protective_sl_order_id = None
        strategy.state.entry_protective_sl_protected = False
        strategy.state.layers = next_layer
        strategy.state.last_entry_price = price
        strategy.state.tp_price = tp_price
        strategy.state.tp_mode = tp_mode
        strategy.state.partial_tp_price = partial_tp_price
        strategy.state.partial_tp_ratio = partial_tp_ratio
        strategy.state.tp_plan = tp_plan
        strategy.state.entry_regime = "MEAN_REVERSION"
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

    # ------------------------------------------------------------------
    # open_trend_position
    # ------------------------------------------------------------------

    def open_trend_position(
        self,
        side: PositionSide,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        reason: str,
        trend_sl_price: float,
    ) -> TradeIntent | None:
        """Open a new trend breakout position.

        Follows the same safety pattern as ``open_position()`` but:
        * Uses *trend_sl_price* as the entry protective SL (BOLL middle anchored).
        * Sets ``entry_regime = "TREND_BREAKOUT"``.
        * Forces ``tp_plan = "SINGLE"`` (no Three-Stage / Middle Runner for trend).
        * No take-profit orders — position is managed by trailing SL only.
        """
        strategy = self.strategy
        if strategy.state.side is not None:
            return None

        previous_state = copy.deepcopy(strategy.state)
        next_layer = 1

        # Validate trend SL
        entry_sl_price = trend_sl_price
        if entry_sl_price is None or entry_sl_price <= 0:
            logger.warning(
                "TREND_ENTRY_SKIPPED | reason=invalid_trend_sl side=%s price=%.4f sl=%s",
                side, price, entry_sl_price,
            )
            return None

        size = strategy.sizer.calculate(
            price, layer_index=next_layer, stop_price=entry_sl_price,
        )

        # Initialize position state (same as open_position)
        strategy.state.first_entry_ts_ms = ts_ms
        strategy.state.add_freeze_until_ts_ms = 0
        strategy.state.add_freeze_penalty_count = 0
        strategy.state.three_stage_pre_tp1_degrade_stage = None
        strategy.state.three_stage_pre_tp1_degraded_ts_ms = 0
        strategy.state.position_cost_entry_notional = 0.0
        strategy.state.position_cost_exit_notional = 0.0
        strategy.state.position_cost_remaining_qty = 0.0
        strategy.state.net_remaining_breakeven_price = 0.0
        strategy.state.last_add_skip_log_reason = None
        strategy.state.last_add_skip_log_ts_ms = 0
        strategy.state.side = side
        strategy._update_position_cost(price, size.eth_qty)
        strategy.state.partial_tp_consumed = False
        strategy._reset_middle_runner_state()
        strategy._reset_three_stage_runner_state()

        # Select TP: SINGLE outer for trend positions (informational only;
        # trend positions are managed by trailing SL, not fixed TP).
        tp_price, tp_mode = strategy._select_tp_price(side, boll)
        tp_plan = "SINGLE"
        partial_tp_price = None
        partial_tp_ratio = 0.0

        # Trend positions are managed by trailing SL — we do not apply a
        # fixed reward/risk check because the reward is trend continuation.
        # Stop-distance validation is done in _maybe_trend_entry().
        if price > 0:
            if side == "LONG":
                stop_distance_pct = (price - entry_sl_price) / price
            else:
                stop_distance_pct = (entry_sl_price - price) / price
        else:
            stop_distance_pct = 1.0
        reward_pct = 0.0  # trend reward is open-ended
        reward_risk = 0.0

        # Set trend-specific state
        strategy.state.entry_regime = "TREND_BREAKOUT"
        strategy.state.trend_breakout_active = True
        strategy.state.trend_trailing_sl_price = entry_sl_price
        strategy.state.trend_last_sl_update_ts_ms = ts_ms
        strategy.state.entry_protective_sl_price = entry_sl_price
        strategy.state.entry_protective_sl_order_id = None
        strategy.state.entry_protective_sl_protected = False
        strategy.state.layers = next_layer
        strategy.state.last_entry_price = price
        strategy.state.tp_price = tp_price
        strategy.state.tp_mode = tp_mode
        strategy.state.partial_tp_price = partial_tp_price
        strategy.state.partial_tp_ratio = partial_tp_ratio
        strategy.state.tp_plan = tp_plan
        strategy.state.last_order_ts_ms = ts_ms
        strategy.state.last_tp_update_ts_ms = ts_ms
        strategy.state.last_tp_update_candle_ts_ms = boll.candle_ts_ms

        trend_reason = (
            f"趋势突破入场 {side}: {reason} "
            f"risk_size stop={entry_sl_price:.4f} "
            f"stop={stop_distance_pct * 100:.3f}%"
        )
        logger.warning(
            "TREND_BREAKOUT_ENTRY | side=%s price=%.4f sl=%.4f tp=%.4f tp_mode=%s "
            "stop_pct=%.6f reward_pct=%.6f reward_risk=%.4f avg_entry=%.4f",
            side, price, entry_sl_price, tp_price, tp_mode,
            stop_distance_pct, reward_pct, reward_risk, strategy.state.avg_entry_price,
        )

        intent_type: TradeIntentType = "OPEN_LONG" if side == "LONG" else "OPEN_SHORT"
        return strategy._intent(
            intent_type, side, price, next_layer, tp_price,
            trend_reason, size, boll, cvd, ts_ms,
        )
