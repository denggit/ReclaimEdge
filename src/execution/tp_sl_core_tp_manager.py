from __future__ import annotations

import os
from decimal import Decimal
from typing import TYPE_CHECKING

from src.execution import middle_bucket_split_size as _split_size
from src.execution import order_specs
from src.execution.trader import LiveTradeResult
from src.position_management.middle_bucket_split_state import (
    MIDDLE_BUCKET_SPLIT_DISABLED_ORDER_PLACEMENT_FAILED_FALLBACK_FINAL,
)
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader
    from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

logger = get_logger(__name__)


# ── Label-based classifier ──────────────────────────────────────────────


def _classify_middle_bucket_split_actual_order_mode(
    *,
    split_was_active: bool,
    specs: list[tuple[str, object, object]],
    split_disabled_reason: str | None,
) -> tuple[bool | None, str | None, str | None]:
    """Classify the actual order mode from the final specs labels.

    Returns:
        middle_bucket_split_executed
        middle_bucket_split_disabled_reason
        middle_bucket_split_actual_order_mode

    This classifier uses **labels** as the ground truth — not the
    ``split_disabled_reason`` string — so that order_specs fallbacks
    (e.g. TP2/runner too small) that produce a single ``"final"`` are
    correctly detected as ``FINAL_FULL_SIZE``.
    """
    if not split_was_active:
        return None, None, None

    labels = {label for label, _contracts, _price in specs}

    # ── SPLIT_FAST_SLOW: labels contain real fast/slow sub-labels ──────
    _three_stage_split_labels = {"tp1_middle_fast", "tp1_middle_slow"}
    _middle_runner_split_labels = {"middle_fast", "middle_slow"}

    if _three_stage_split_labels.issubset(labels) or _middle_runner_split_labels.issubset(labels):
        return True, None, "SPLIT_FAST_SLOW"

    # ── Legal partial/post-TP1 structures ─────────────────────────────
    if labels == {"tp2_outer"}:
        return True, None, "POST_TP1_TP2_ONLY"

    if "tp2_outer" in labels:
        if "tp1_middle_slow" in labels and "tp1_middle_fast" not in labels:
            return True, None, "PARTIAL_SPLIT_SLOW_PENDING"
        if "tp1_middle_fast" in labels and "tp1_middle_slow" not in labels:
            return True, None, "PARTIAL_SPLIT_FAST_PENDING"

    if labels == {"runner"}:
        return True, None, "POST_MIDDLE_BUCKET_RUNNER_ONLY"

    if "runner" in labels:
        if "middle_slow" in labels and "middle_fast" not in labels:
            return True, None, "PARTIAL_SPLIT_SLOW_PENDING"
        if "middle_fast" in labels and "middle_slow" not in labels:
            return True, None, "PARTIAL_SPLIT_FAST_PENDING"

    # ── FINAL_FULL_SIZE: only a single "final" label ───────────────────
    if labels == {"final"}:
        reason = split_disabled_reason or "split_fallback_final_order_structure"
        return False, reason, "FINAL_FULL_SIZE"

    # ── UNSPLIT_MIDDLE_BUCKET ──────────────────────────────────────────
    _three_stage_unsplit_labels = {"tp1_middle", "tp2_outer"}
    # Middle Runner unsplit: "middle" + "runner" (may also have "runner" only)
    _middle_runner_unsplit_label = "middle"

    if _three_stage_unsplit_labels.issubset(labels):
        reason = split_disabled_reason or "split_fallback_unsplit_middle_bucket"
        return False, reason, "UNSPLIT_MIDDLE_BUCKET"

    if _middle_runner_unsplit_label in labels and "runner" in labels:
        reason = split_disabled_reason or "split_fallback_unsplit_middle_bucket"
        return False, reason, "UNSPLIT_MIDDLE_BUCKET"

    # ── Unknown / safety fallback ──────────────────────────────────────
    logger.warning(
        "MIDDLE_BUCKET_SPLIT_UNKNOWN_ORDER_STRUCTURE | "
        "split_was_active=true labels=%s fallback_reason=%s "
        "action=degrade_to_single_final state_order_consistent=true",
        sorted(labels),
        split_disabled_reason or "n/a",
    )
    reason = split_disabled_reason or "split_unknown_order_structure_fallback_final"
    return False, reason, "FINAL_FULL_SIZE"


class CoreTakeProfitManager:
    def __init__(self, trader: Trader, protective_stops) -> None:
        self.trader = trader
        self.protective_stops = protective_stops

    async def replace_take_profit(self, intent: TradeIntent) -> LiveTradeResult:
        t = self.trader
        managed_core_contracts = t._managed_core_contracts_from_intent(intent)
        try:
            position = await t.fetch_position_snapshot()
            net_contracts_for_sl = position.contracts if position.has_position and position.side == intent.side else Decimal(
                "0")
            if managed_core_contracts is not None:
                core_contracts_for_tp = managed_core_contracts
            else:
                core_contracts_for_tp = net_contracts_for_sl
        except Exception:
            logger.exception("Failed to refresh position before replacing TP")
            if managed_core_contracts is not None and managed_core_contracts > 0:
                logger.error(
                    "REPLACE_TP_NET_POSITION_FETCH_FAILED | cannot determine net position for global SL; refusing to proceed with managed_core_contracts as fallback; sidecar may be unprotected manual_intervention_required=true core=%s",
                    t.decimal_to_str(managed_core_contracts),
                )
                raise RuntimeError(
                    "failed_to_fetch_net_position_for_global_sl: "
                    "cannot use managed_core_contracts as net position for SL in Sidecar mode"
                )
            return LiveTradeResult(
                False,
                intent.intent_type,
                None,
                None,
                "0",
                t.price_to_str(intent.tp_price),
                "failed to fetch position and no managed_core_contracts",
            )

        t.position_contracts = core_contracts_for_tp

        if net_contracts_for_sl <= 0:
            return LiveTradeResult(
                False,
                intent.intent_type,
                None,
                None,
                "0",
                t.price_to_str(intent.tp_price),
                "no position to protect",
            )

        # Sanity check: core cannot exceed net (only when there is a net position)
        if managed_core_contracts is not None and core_contracts_for_tp > net_contracts_for_sl:
            raise RuntimeError(
                f"managed_core_contracts_exceeds_net_position "
                f"core={t.decimal_to_str(core_contracts_for_tp)} net={t.decimal_to_str(net_contracts_for_sl)}"
            )

        if core_contracts_for_tp <= 0:
            return LiveTradeResult(
                False,
                intent.intent_type,
                None,
                None,
                "0",
                t.price_to_str(intent.tp_price),
                "no core position for TP",
            )

        await self.trader._cancel_existing_take_profit_orders_for_intent(intent)
        await self.trader._cancel_stale_runner_protective_stops_for_degrade(intent)

        specs, split_disabled_reason = self._build_take_profit_order_specs(intent)
        split_was_active = bool(getattr(intent, "middle_bucket_split_active", False))
        placed_order_ids: list[str] = []
        message = "take-profit replaced"
        try:
            placed_order_ids = await self.trader._place_reduce_only_take_profit_orders(intent, specs)
        except Exception:
            if len(specs) <= 1:
                raise
            if split_was_active:
                split_disabled_reason = MIDDLE_BUCKET_SPLIT_DISABLED_ORDER_PLACEMENT_FAILED_FALLBACK_FINAL
            logger.exception(
                "Failed to place split take-profit orders; falling back to one full-size final TP | "
                "split_was_active=%s state_must_clear_split=true "
                "fallback_order_labels=[final] "
                "reason=%s",
                split_was_active,
                split_disabled_reason or "n/a",
            )
            await self.trader._cancel_existing_take_profit_orders_for_intent(intent)
            fallback_specs = [("final", t.position_contracts, intent.tp_price)]
            placed_order_ids = await self.trader._place_reduce_only_take_profit_orders(intent, fallback_specs)
            specs = fallback_specs
            message = "split take-profit placement failed; fallback to single final TP"

        # ── Middle Bucket Split status (classified BEFORE protective SL so
        #    all early-return paths carry the fields) ────────────────────
        (
            middle_bucket_split_executed,
            middle_bucket_split_disabled_reason_val,
            middle_bucket_split_actual_order_mode_val,
        ) = _classify_middle_bucket_split_actual_order_mode(
            split_was_active=split_was_active,
            specs=specs,
            split_disabled_reason=split_disabled_reason,
        )

        tp_order_id = ",".join(placed_order_ids)
        t.tp_order_id = tp_order_id
        tp_price_text = self.trader._tp_price_summary(specs)
        protective_sl_order_id: str | None = None
        protective_sl_price_text = ""
        protective_sl_ok = False
        runner_sl_price = getattr(intent, "middle_runner_protective_sl_price", None)
        if getattr(intent, "middle_runner_active", False) and runner_sl_price is not None:
            old_sl_order_id = getattr(intent, "middle_runner_protective_sl_order_id",
                                      None) or t.middle_runner_protective_sl_order_id
            sl_ok, sl_order_id, sl_message = await self.trader.place_middle_runner_protective_stop_with_retries(
                intent.side,
                net_contracts_for_sl,
                float(runner_sl_price),
                retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
            )
            protective_sl_price_text = t.price_to_str(float(runner_sl_price))
            if not sl_ok:
                return LiveTradeResult(
                    False,
                    intent.intent_type,
                    None,
                    tp_order_id,
                    t.decimal_to_str(core_contracts_for_tp),
                    tp_price_text,
                    f"middle_runner_protective_sl_failed: {sl_message}",
                    entry_filled=False,
                    tp_ok=True,
                    tp_order_ids=tuple(placed_order_ids),
                    protective_sl_price=protective_sl_price_text,
                    protective_sl_ok=False,
                    middle_bucket_split_executed=middle_bucket_split_executed,
                    middle_bucket_split_disabled_reason=middle_bucket_split_disabled_reason_val,
                    middle_bucket_split_actual_order_mode=middle_bucket_split_actual_order_mode_val,
                )
            protective_sl_order_id = sl_order_id
            protective_sl_ok = True
            t.middle_runner_protective_sl_order_id = sl_order_id
            if old_sl_order_id and old_sl_order_id != sl_order_id:
                await self.trader.cancel_middle_runner_protective_stop(old_sl_order_id)
            logger.warning(
                "MIDDLE_RUNNER_SL_UPDATED | side=%s sl_contracts=%s core_contracts=%s net_contracts=%s protective_sl_price=%s old_sl_order_id=%s new_sl_order_id=%s",
                intent.side,
                t.decimal_to_str(net_contracts_for_sl),
                t.decimal_to_str(core_contracts_for_tp),
                t.decimal_to_str(net_contracts_for_sl),
                protective_sl_price_text,
                old_sl_order_id,
                sl_order_id,
            )
        trend_runner_sl_price = getattr(intent, "trend_runner_sl_price", None) or getattr(intent,
                                                                                          "three_stage_runner_sl_price",
                                                                                          None)
        if (
                getattr(intent, "trend_runner_active", False)
                and trend_runner_sl_price is not None
        ):
            old_sl_order_id = getattr(intent, "trend_runner_sl_order_id", None) or t.trend_runner_sl_order_id
            sl_contracts = self.trader._trend_runner_sl_contracts(intent, net_contracts_for_sl)
            sl_ok, sl_order_id, sl_message = await self.trader.place_trend_runner_protective_stop_with_retries(
                intent.side,
                sl_contracts,
                float(trend_runner_sl_price),
                retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
            )
            protective_sl_price_text = t.price_to_str(float(trend_runner_sl_price))
            if not sl_ok:
                return LiveTradeResult(
                    False,
                    intent.intent_type,
                    None,
                    tp_order_id,
                    t.decimal_to_str(core_contracts_for_tp),
                    tp_price_text,
                    f"trend_runner_protective_sl_failed: {sl_message}",
                    entry_filled=False,
                    tp_ok=True,
                    tp_order_ids=tuple(placed_order_ids),
                    protective_sl_price=protective_sl_price_text,
                    protective_sl_ok=False,
                    middle_bucket_split_executed=middle_bucket_split_executed,
                    middle_bucket_split_disabled_reason=middle_bucket_split_disabled_reason_val,
                    middle_bucket_split_actual_order_mode=middle_bucket_split_actual_order_mode_val,
                )
            protective_sl_order_id = sl_order_id
            protective_sl_ok = True
            t.trend_runner_sl_order_id = sl_order_id
            if old_sl_order_id and old_sl_order_id != sl_order_id:
                await self.trader.cancel_trend_runner_protective_stop(old_sl_order_id)
            logger.warning(
                "TREND_RUNNER_SL_UPDATED | side=%s sl_contracts=%s core_contracts=%s net_contracts=%s protective_sl_price=%s old_sl_order_id=%s new_sl_order_id=%s",
                intent.side,
                t.decimal_to_str(sl_contracts),
                t.decimal_to_str(core_contracts_for_tp),
                t.decimal_to_str(net_contracts_for_sl),
                protective_sl_price_text,
                old_sl_order_id,
                sl_order_id,
            )
        post_tp1_sl_price = getattr(intent, "three_stage_post_tp1_protective_sl_price", None)
        if (
                getattr(intent, "tp_plan", "SINGLE") == "THREE_STAGE_RUNNER"
                and getattr(intent, "three_stage_tp1_consumed", False)
                and not getattr(intent, "three_stage_tp2_consumed", False)
                and not getattr(intent, "trend_runner_active", False)
                and post_tp1_sl_price is not None
        ):
            old_sl_order_id = getattr(intent, "three_stage_post_tp1_protective_sl_order_id",
                                      None) or t.three_stage_post_tp1_protective_sl_order_id
            sl_ok, sl_order_id, sl_message = await self.trader.place_three_stage_post_tp1_protective_stop_with_retries(
                intent.side,
                net_contracts_for_sl,
                float(post_tp1_sl_price),
                retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
            )
            protective_sl_price_text = t.price_to_str(float(post_tp1_sl_price))
            if not sl_ok:
                return LiveTradeResult(
                    False,
                    intent.intent_type,
                    None,
                    tp_order_id,
                    t.decimal_to_str(core_contracts_for_tp),
                    tp_price_text,
                    f"three_stage_post_tp1_protective_sl_failed: {sl_message}",
                    entry_filled=False,
                    tp_ok=True,
                    tp_order_ids=tuple(placed_order_ids),
                    protective_sl_price=protective_sl_price_text,
                    protective_sl_ok=False,
                    middle_bucket_split_executed=middle_bucket_split_executed,
                    middle_bucket_split_disabled_reason=middle_bucket_split_disabled_reason_val,
                    middle_bucket_split_actual_order_mode=middle_bucket_split_actual_order_mode_val,
                )
            protective_sl_order_id = sl_order_id
            protective_sl_ok = True
            t.three_stage_post_tp1_protective_sl_order_id = sl_order_id
            if old_sl_order_id and old_sl_order_id != sl_order_id:
                await self.trader.cancel_three_stage_post_tp1_protective_stop(old_sl_order_id)
            logger.warning(
                "THREE_STAGE_TP1_PROTECTIVE_SL_UPDATED | side=%s sl_contracts=%s core_contracts=%s net_contracts=%s protective_sl_price=%s old_sl_order_id=%s new_sl_order_id=%s retry_config=near_tp",
                intent.side,
                t.decimal_to_str(net_contracts_for_sl),
                t.decimal_to_str(core_contracts_for_tp),
                t.decimal_to_str(net_contracts_for_sl),
                protective_sl_price_text,
                old_sl_order_id,
                sl_order_id,
            )
        return LiveTradeResult(
            True,
            intent.intent_type,
            None,
            tp_order_id,
            t.decimal_to_str(t.position_contracts),
            tp_price_text,
            message,
            entry_filled=False,
            tp_ok=True,
            tp_order_ids=tuple(placed_order_ids),
            protective_sl_order_id=protective_sl_order_id,
            protective_sl_price=protective_sl_price_text,
            protective_sl_ok=protective_sl_ok,
            middle_bucket_split_executed=middle_bucket_split_executed,
            middle_bucket_split_disabled_reason=middle_bucket_split_disabled_reason_val,
            middle_bucket_split_actual_order_mode=middle_bucket_split_actual_order_mode_val,
        )

    async def _cancel_existing_take_profit_orders_for_intent(self, intent: TradeIntent) -> None:
        t = self.trader
        t._protected_reduce_only_order_ids = self.trader._protected_order_ids_from_intent(intent)
        t._managed_reduce_only_order_ids = self._split_order_ids(t.tp_order_id)
        t._allow_cancel_unmanaged_reduce_only = False
        try:
            await self.trader.cancel_existing_reduce_only_orders()
        finally:
            t._protected_reduce_only_order_ids = set()
            t._managed_reduce_only_order_ids = set()
            t._allow_cancel_unmanaged_reduce_only = True

    async def _cancel_stale_runner_protective_stops_for_degrade(self, intent: TradeIntent) -> None:
        t = self.trader
        reason = str(getattr(intent, "reason", "") or "")
        if "three_stage_pre_tp1_degraded" not in reason:
            return
        middle_runner_sl_order_id = getattr(t, "middle_runner_protective_sl_order_id", None)
        if middle_runner_sl_order_id:
            await self.trader.cancel_middle_runner_protective_stop(middle_runner_sl_order_id)
        three_stage_post_tp1_sl_order_id = getattr(t, "three_stage_post_tp1_protective_sl_order_id", None)
        if three_stage_post_tp1_sl_order_id:
            await self.trader.cancel_three_stage_post_tp1_protective_stop(three_stage_post_tp1_sl_order_id)
        trend_runner_sl_order_id = getattr(t, "trend_runner_sl_order_id", None)
        if trend_runner_sl_order_id:
            await self.trader.cancel_trend_runner_protective_stop(trend_runner_sl_order_id)

    def _protected_order_ids_from_intent(self, intent: TradeIntent) -> set[str]:
        t = self.trader
        ids = set(getattr(intent, "protected_order_ids", ()) or ())
        for value in (
                getattr(intent, "near_tp_protective_sl_order_id", None),
                getattr(intent, "middle_runner_protective_sl_order_id", None),
                getattr(intent, "three_stage_post_tp1_protective_sl_order_id", None),
                getattr(intent, "trend_runner_sl_order_id", None),
                t.near_tp_protective_sl_order_id,
                t.middle_runner_protective_sl_order_id,
                t.three_stage_post_tp1_protective_sl_order_id,
                t.trend_runner_sl_order_id,
        ):
            if value:
                ids.add(str(value))
        return ids

    @staticmethod
    def _split_order_ids(value: str | None) -> set[str]:
        if not value:
            return set()
        return {item.strip() for item in str(value).split(",") if item.strip()}

    def _managed_core_contracts_from_intent(self, intent: TradeIntent) -> Decimal | None:
        t = self.trader
        raw = getattr(intent, "managed_core_contracts", None)
        if raw in (None, ""):
            return None
        try:
            contracts = Decimal(str(raw))
        except Exception:
            raise RuntimeError(f"invalid managed_core_contracts: {raw}")
        if contracts <= 0:
            return None
        contracts = t.round_contracts_down(contracts)
        if contracts < t.min_contracts:
            raise RuntimeError(
                f"managed_core_contracts below min_contracts contracts={t.decimal_to_str(contracts)} min_contracts={t.decimal_to_str(t.min_contracts)}"
            )
        return contracts

    def _build_take_profit_order_specs(
        self, intent: TradeIntent,
    ) -> tuple[list[tuple[str, Decimal, float]], str | None]:
        """Build take-profit order specs with middle-bucket-split size pre-check.

        Returns:
            (specs, split_disabled_reason)

            split_disabled_reason is None when split was either not requested or
            was successfully applied.  It is a non-empty string when split was
            active but had to be disabled due to sub-leg size constraints.
        """
        t = self.trader

        # ── Middle Bucket Split input ─────────────────────────────────
        split_active = bool(getattr(intent, "middle_bucket_split_active", False))
        split_disabled_reason: str | None = None
        middle_bucket_split_input = None

        if split_active:
            # ── Pre-check split sub-leg sizes BEFORE constructing input ──
            tp_plan = str(getattr(intent, "tp_plan", "SINGLE"))
            fast_ratio = Decimal(str(getattr(intent, "middle_bucket_split_fast_ratio_of_bucket", 0.0)))
            fast_consumed = bool(getattr(intent, "middle_bucket_split_fast_consumed", False))
            slow_consumed = bool(getattr(intent, "middle_bucket_split_slow_consumed", False))

            if tp_plan == "THREE_STAGE_RUNNER" and not (fast_consumed or slow_consumed):
                size_check = _split_size.check_three_stage_middle_bucket_split_size(
                    position_contracts=t.position_contracts,
                    min_contracts=t.min_contracts,
                    contract_precision=t.contract_precision,
                    three_stage_tp1_ratio=Decimal(str(getattr(intent, "three_stage_tp1_ratio", 0.0))),
                    fast_ratio_of_bucket=fast_ratio,
                )
            elif tp_plan == "MIDDLE_RUNNER" and not (fast_consumed or slow_consumed):
                size_check = _split_size.check_middle_runner_bucket_split_size(
                    position_contracts=t.position_contracts,
                    min_contracts=t.min_contracts,
                    contract_precision=t.contract_precision,
                    partial_tp_ratio=Decimal(str(getattr(intent, "partial_tp_ratio", 0.0))),
                    fast_ratio_of_bucket=fast_ratio,
                )
            else:
                size_check = None

            if size_check is not None and not size_check.ok:
                split_disabled_reason = "subleg_too_small"
                logger.warning(
                    "MIDDLE_BUCKET_SPLIT_DISABLED_ON_ORDER_BUILD | "
                    "reason=subleg_too_small tp_plan=%s "
                    "position_contracts=%s tp1_total=%s fast=%s slow=%s min=%s",
                    tp_plan,
                    t.decimal_to_str(t.position_contracts),
                    t.decimal_to_str(size_check.tp1_total_contracts),
                    t.decimal_to_str(size_check.fast_contracts),
                    t.decimal_to_str(size_check.slow_contracts),
                    t.decimal_to_str(size_check.min_contracts),
                )
            else:
                middle_bucket_split_input = order_specs.MiddleBucketSplitOrderInput(
                    active=True,
                    fast_price=getattr(intent, "middle_bucket_split_fast_price", None),
                    slow_price=getattr(intent, "middle_bucket_split_slow_price", None),
                    effective_price=getattr(intent, "middle_bucket_split_effective_price", None),
                    middle_bucket_ratio=Decimal(str(getattr(intent, "middle_bucket_split_middle_bucket_ratio", 0.0))),
                    fast_ratio_of_bucket=fast_ratio,
                    slow_ratio_of_bucket=Decimal(str(getattr(intent, "middle_bucket_split_slow_ratio_of_bucket", 0.0))),
                    fast_total_ratio=Decimal(str(getattr(intent, "middle_bucket_split_fast_total_ratio", 0.0))),
                    slow_total_ratio=Decimal(str(getattr(intent, "middle_bucket_split_slow_total_ratio", 0.0))),
                    fast_consumed=fast_consumed,
                    slow_consumed=slow_consumed,
                )

        decision = order_specs.build_take_profit_order_specs(
            position_contracts=t.position_contracts,
            min_contracts=t.min_contracts,
            contract_precision=t.contract_precision,
            tp_plan=getattr(intent, "tp_plan", "SINGLE"),
            final_tp_price=intent.tp_price,
            partial_tp_price=getattr(intent, "partial_tp_price", None),
            partial_tp_ratio=Decimal(str(getattr(intent, "partial_tp_ratio", 0.0))),
            partial_tp_consumed=bool(getattr(intent, "partial_tp_consumed", False)),
            middle_runner_active=bool(getattr(intent, "middle_runner_active", False)),
            three_stage_tp1_price=getattr(intent, "three_stage_tp1_price", None),
            three_stage_tp2_price=getattr(intent, "three_stage_tp2_price", None),
            three_stage_tp1_ratio=Decimal(str(getattr(intent, "three_stage_tp1_ratio", 0.0))),
            three_stage_tp2_ratio=Decimal(str(getattr(intent, "three_stage_tp2_ratio", 0.0))),
            three_stage_tp1_consumed=bool(getattr(intent, "three_stage_tp1_consumed", False)),
            three_stage_tp2_consumed=bool(getattr(intent, "three_stage_tp2_consumed", False)),
            three_stage_runner_ratio=Decimal(str(getattr(intent, "three_stage_runner_ratio", 0.0))),
            middle_bucket_split=middle_bucket_split_input,
        )
        reason = decision.fallback_reason
        ctx = decision.fallback_context
        if reason is not None and ctx is not None:
            if reason == "SPLIT_TP_FALLBACK_SINGLE_SIZE_TOO_SMALL":
                logger.warning(
                    "SPLIT_TP_FALLBACK_SINGLE | reason=size_too_small total_contracts=%s partial_contracts=%s final_contracts=%s min_contracts=%s",
                    ctx["total_contracts"],
                    ctx["partial_contracts"],
                    ctx["final_contracts"],
                    ctx["min_contracts"],
                )
            elif reason == "THREE_STAGE_TP2_AFTER_TP1_INVALID_RATIOS":
                logger.warning(
                    "THREE_STAGE_TP2_AFTER_TP1_FALLBACK | reason=invalid_ratios total_contracts=%s tp2_ratio=%s runner_ratio=%s tp2_price=%s",
                    ctx["total_contracts"],
                    ctx["tp2_ratio"],
                    ctx["runner_ratio"],
                    ctx["tp2_price"],
                )
            elif reason == "THREE_STAGE_TP2_AFTER_TP1_TP2_TOO_SMALL":
                logger.warning(
                    "THREE_STAGE_TP2_AFTER_TP1_FALLBACK | reason=tp2_too_small total_contracts=%s tp2_contracts=%s runner_contracts=%s min_contracts=%s",
                    ctx["total_contracts"],
                    ctx["tp2_contracts"],
                    ctx["runner_contracts"],
                    ctx["min_contracts"],
                )
            elif reason == "THREE_STAGE_TP2_AFTER_TP1_RUNNER_TOO_SMALL":
                logger.warning(
                    "THREE_STAGE_TP2_AFTER_TP1_FALLBACK | reason=runner_too_small total_contracts=%s tp2_contracts=%s runner_contracts=%s min_contracts=%s",
                    ctx["total_contracts"],
                    ctx["tp2_contracts"],
                    ctx["runner_contracts"],
                    ctx["min_contracts"],
                )
            elif reason == "THREE_STAGE_TP_FALLBACK_SINGLE_SIZE_TOO_SMALL":
                logger.warning(
                    "THREE_STAGE_TP_FALLBACK_SINGLE | reason=size_too_small total_contracts=%s tp1_contracts=%s tp2_contracts=%s runner_contracts=%s min_contracts=%s",
                    ctx["total_contracts"],
                    ctx["tp1_contracts"],
                    ctx["tp2_contracts"],
                    ctx["runner_contracts"],
                    ctx["min_contracts"],
                )
            elif reason == "MIDDLE_BUCKET_SPLIT_SUBLEG_TOO_SMALL_UNSPLIT":
                logger.warning(
                    "MIDDLE_BUCKET_SPLIT_FALLBACK_UNSPLIT | reason=subleg_too_small state_split_active=true "
                    "actual_order_labels=%s | state and order structure are now consistent (unsplit) "
                    "total_contracts=%s",
                    [s.label for s in decision.specs],
                    ctx.get("total_contracts", "?"),
                )
        # ── Map order_specs fallback_reason → split_disabled_reason ────
        # The pre-check above only catches sub-leg too small.  When the
        # pre-check passes but order_specs falls back to a single final
        # (e.g. TP2/runner too small), the reason must still be populated
        # so the label-based classifier can carry it through.
        if split_active and split_disabled_reason is None:
            _fr_map = {
                "THREE_STAGE_TP_SPLIT_FALLBACK_SINGLE_SIZE_TOO_SMALL",
                "MIDDLE_RUNNER_SPLIT_FALLBACK_RUNNER_TOO_SMALL",
            }
            if reason in _fr_map:
                split_disabled_reason = "split_fallback_final_order_structure"

        specs = [(spec.label, spec.contracts, spec.price) for spec in decision.specs]
        return specs, split_disabled_reason

    def _build_take_profit_order_specs_public(
        self, intent: TradeIntent,
    ) -> list[tuple[str, Decimal, float]]:
        """Public delegation: returns just the specs list (backward-compat)."""
        specs, _split_reason = self._build_take_profit_order_specs(intent)
        return specs

    def _build_three_stage_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        specs, _split_reason = self._build_take_profit_order_specs(intent)
        return specs

    def _build_three_stage_order_specs_public(
        self, intent: TradeIntent,
    ) -> list[tuple[str, Decimal, float]]:
        """Public delegation: returns just the specs list (backward-compat)."""
        return self._build_three_stage_order_specs(intent)

    def _trend_runner_sl_contracts(self, intent: TradeIntent, net_contracts_for_sl: Decimal) -> Decimal:
        t = self.trader
        return order_specs.trend_runner_sl_contracts(
            net_contracts_for_sl=net_contracts_for_sl,
            runner_ratio=Decimal(str(getattr(intent, "three_stage_runner_ratio", 0.0))),
            min_contracts=t.min_contracts,
            contract_precision=t.contract_precision,
            trend_runner_active=bool(getattr(intent, "trend_runner_active", False)),
        )

    async def _place_reduce_only_take_profit_orders(self, intent: TradeIntent,
                                                    specs: list[tuple[str, Decimal, float]]) -> list[str]:
        t = self.trader
        placed_order_ids: list[str] = []
        for label, contracts, price in specs:
            body = t._reduce_only_tp_order_body(intent.side, contracts, price)
            res = await t.request("POST", "/api/v5/trade/order", body)
            order_id = t.extract_order_id(res)
            placed_order_ids.append(order_id)
            logger.info(
                "TP_ORDER_PLACED | label=%s side=%s tp_contracts=%s core_contracts=%s price=%s ordId=%s",
                label,
                intent.side,
                t.decimal_to_str(contracts),
                t.decimal_to_str(t.position_contracts),
                t.price_to_str(price),
                order_id,
            )
        return placed_order_ids

    def _tp_price_summary(self, specs: list[tuple[str, Decimal, float]]) -> str:
        t = self.trader
        if len(specs) == 1:
            return t.price_to_str(specs[0][2])
        return ",".join(f"{label}:{t.price_to_str(price)}" for label, _contracts, price in specs)
