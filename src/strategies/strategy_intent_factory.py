from __future__ import annotations

from typing import TYPE_CHECKING

from src.position_management.sidecar.core_exit_safety import active_sidecar_tp_order_ids
from src.position_management.sidecar.model import trim_sidecar_legs_for_state

if TYPE_CHECKING:
    from src.indicators.cvd_tracker import CvdSnapshot
    from src.monitors.boll_band_breakout_monitor import BollSnapshot
    from src.risk.simple_position_sizer import PositionSize
    from src.strategies.boll_cvd_reclaim_strategy import (
        BollCvdReclaimStrategy,
        TradeIntent,
        TradeIntentType,
        PositionSide,
        TpMode,
    )


class StrategyIntentFactory:
    """Constructs TradeIntent payloads for BollCvdReclaimStrategy.

    Extracted from BollCvdReclaimStrategy to keep intent-construction
    logic in a single place without changing any trigger logic.
    """

    def __init__(self, strategy: BollCvdReclaimStrategy) -> None:
        self.strategy = strategy

    # ── helpers that are thin wrappers around strategy state ──────────────

    def managed_core_contracts_for_intent(self, intent_type: TradeIntentType) -> str | None:
        state = self.strategy.state
        if not state.sidecar_enabled_for_position:
            return None
        if intent_type in {"UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"}:
            return state.core_contracts
        return None

    def managed_core_eth_qty_for_intent(self, intent_type: TradeIntentType) -> float:
        state = self.strategy.state
        if not state.sidecar_enabled_for_position:
            return 0.0
        if intent_type in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT", "UPDATE_TP"}:
            return float(state.total_entry_qty or 0.0)
        return float(state.core_eth_qty or 0.0)

    def protected_order_ids(self) -> tuple[str, ...]:
        max_legs = int(getattr(self.strategy.sizer.config, "sidecar_max_legs", 10) or 10)
        ids: list[str] = list(
            active_sidecar_tp_order_ids(
                trim_sidecar_legs_for_state(self.strategy.state.sidecar_legs, max_legs)
            )
        )
        for order_id in (
                self.strategy.state.entry_protective_sl_order_id,
                self.strategy.state.near_tp_protective_sl_order_id,
                self.strategy.state.middle_runner_protective_sl_order_id,
                self.strategy.state.three_stage_post_tp1_protective_sl_order_id,
                self.strategy.state.trend_runner_sl_order_id,
        ):
            if order_id:
                ids.append(str(order_id))
        return tuple(dict.fromkeys(ids))

    # ── intent builders ───────────────────────────────────────────────────

    def build_intent(
            self,
            *,
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
        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

        state = self.strategy.state
        return TradeIntent(
            intent_type=intent_type,
            side=side,
            price=price,
            layer_index=layer_index,
            tp_price=tp_price,
            reason=reason,
            size=size,
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
            partial_tp_price=state.partial_tp_price,
            partial_tp_ratio=state.partial_tp_ratio,
            tp_plan=state.tp_plan,
            partial_tp_consumed=state.partial_tp_consumed,
            entry_protective_sl_price=getattr(state, "entry_protective_sl_price", None),
            entry_protective_sl_order_id=getattr(state, "entry_protective_sl_order_id", None),
            entry_protective_sl_protected=bool(getattr(state, "entry_protective_sl_protected", False)),
            middle_runner_enabled_for_position=state.middle_runner_enabled_for_position,
            middle_runner_pending=state.middle_runner_pending,
            middle_runner_active=state.middle_runner_active,
            middle_runner_first_close_ratio=state.middle_runner_first_close_ratio,
            middle_runner_keep_ratio=state.middle_runner_keep_ratio,
            middle_runner_first_tp_price=state.middle_runner_first_tp_price,
            middle_runner_final_tp_price=state.middle_runner_final_tp_price,
            middle_runner_protective_sl_price=state.middle_runner_protective_sl_price,
            middle_runner_protective_sl_order_id=state.middle_runner_protective_sl_order_id,
            middle_runner_extension_triggered=state.middle_runner_extension_triggered,
            middle_runner_add_disabled=state.middle_runner_add_disabled,
            three_stage_tp1_price=state.three_stage_tp1_price,
            three_stage_tp1_ratio=state.three_stage_tp1_ratio,
            three_stage_tp2_price=state.three_stage_tp2_price,
            three_stage_tp2_ratio=state.three_stage_tp2_ratio,
            three_stage_runner_tp_price=state.trend_runner_tp_price,
            three_stage_runner_ratio=state.three_stage_runner_ratio,
            three_stage_runner_sl_price=state.trend_runner_sl_price,
            three_stage_tp1_consumed=state.three_stage_tp1_consumed,
            three_stage_tp2_consumed=state.three_stage_tp2_consumed,
            three_stage_post_tp1_protective_sl_price=state.three_stage_post_tp1_protective_sl_price,
            three_stage_post_tp1_protective_sl_order_id=state.three_stage_post_tp1_protective_sl_order_id,
            three_stage_post_tp1_sl_extension_triggered=state.three_stage_post_tp1_sl_extension_triggered,
            three_stage_post_tp1_protected=state.three_stage_post_tp1_protected,
            trend_runner_active=state.trend_runner_active,
            trend_runner_tp_price=state.trend_runner_tp_price,
            trend_runner_sl_price=state.trend_runner_sl_price,
            trend_runner_tp_order_id=state.trend_runner_tp_order_id,
            trend_runner_sl_order_id=state.trend_runner_sl_order_id,
            trend_runner_exit_reason=state.trend_runner_exit_reason,
            trend_runner_adjust_count=state.trend_runner_adjust_count,
            protected_order_ids=self.protected_order_ids(),
            managed_core_contracts=self.managed_core_contracts_for_intent(intent_type),
            managed_core_eth_qty=self.managed_core_eth_qty_for_intent(intent_type),
            # ── Middle Bucket Split fields ────────────────────────────
            middle_bucket_split_active=bool(getattr(state, "middle_bucket_split_active", False)),
            middle_bucket_split_fast_consumed=bool(getattr(state, "middle_bucket_split_fast_consumed", False)),
            middle_bucket_split_slow_consumed=bool(getattr(state, "middle_bucket_split_slow_consumed", False)),
            middle_bucket_split_fast_price=getattr(state, "middle_bucket_split_fast_price", None),
            middle_bucket_split_slow_price=getattr(state, "middle_bucket_split_slow_price", None),
            middle_bucket_split_effective_price=getattr(state, "middle_bucket_split_effective_price", None),
            middle_bucket_split_middle_bucket_ratio=float(
                getattr(state, "middle_bucket_split_middle_bucket_ratio", 0.0) or 0.0),
            middle_bucket_split_fast_ratio_of_bucket=float(
                getattr(state, "middle_bucket_split_fast_ratio_of_bucket", 0.0) or 0.0),
            middle_bucket_split_slow_ratio_of_bucket=float(
                getattr(state, "middle_bucket_split_slow_ratio_of_bucket", 0.0) or 0.0),
            middle_bucket_split_fast_total_ratio=float(
                getattr(state, "middle_bucket_split_fast_total_ratio", 0.0) or 0.0),
            middle_bucket_split_slow_total_ratio=float(
                getattr(state, "middle_bucket_split_slow_total_ratio", 0.0) or 0.0),
            middle_bucket_split_reason=getattr(state, "middle_bucket_split_reason", None),
            middle_bucket_split_fast_sl_price=getattr(state, "middle_bucket_split_fast_sl_price", None),
            middle_bucket_split_fast_sl_order_id=getattr(state, "middle_bucket_split_fast_sl_order_id", None),
            middle_bucket_split_fast_sl_protected=bool(
                getattr(state, "middle_bucket_split_fast_sl_protected", False)),
        )

    def build_near_tp_reduce_intent(
            self,
            *,
            side: PositionSide,
            price: float,
            layer_index: int,
            tp_price: float,
            reason: str,
            size: PositionSize,
            boll: BollSnapshot,
            cvd: CvdSnapshot,
            ts_ms: int,
            progress: float,
            best: float,
            giveback: float,
            giveback_threshold: float,
            protective_sl: float,
    ) -> TradeIntent:
        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

        state = self.strategy.state
        return TradeIntent(
            intent_type="NEAR_TP_REDUCE",
            side=side,
            price=price,
            layer_index=layer_index,
            tp_price=tp_price,
            reason=reason,
            size=size,
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
            partial_tp_price=None,
            partial_tp_ratio=0.0,
            tp_plan="SINGLE",
            partial_tp_consumed=True,
            near_tp_progress_ratio=progress,
            near_tp_best_price=best,
            near_tp_giveback=giveback,
            near_tp_giveback_threshold=giveback_threshold,
            near_tp_reduce_ratio=self.strategy.config.near_tp_reduce_ratio,
            near_tp_protective_sl_price=protective_sl,
        )

    def build_runner_market_exit_intent(
            self,
            *,
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
        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

        state = self.strategy.state
        return TradeIntent(
            intent_type="MARKET_EXIT_RUNNER",
            side=side,
            price=price,
            layer_index=layer_index,
            tp_price=tp_price,
            reason=reason,
            size=size,
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
            partial_tp_price=None,
            partial_tp_ratio=0.0,
            tp_plan="SINGLE",
            partial_tp_consumed=True,
            trend_runner_active=True,
            trend_runner_tp_price=state.trend_runner_tp_price,
            trend_runner_sl_price=state.trend_runner_sl_price,
            trend_runner_tp_order_id=state.trend_runner_tp_order_id,
            trend_runner_sl_order_id=state.trend_runner_sl_order_id,
            trend_runner_exit_reason=reason,
            trend_runner_adjust_count=state.trend_runner_adjust_count,
        )
