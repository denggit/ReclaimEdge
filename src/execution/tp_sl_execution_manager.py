from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from typing import Any, TYPE_CHECKING

from dataclasses import replace

from src.execution import order_specs
from src.execution.trader import LiveTradeResult, PositionSnapshot
from src.position_management.sidecar.model import sanitize_okx_client_order_id
from src.strategies.boll_cvd_reclaim_strategy import PositionSide, TradeIntent
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader

logger = get_logger(__name__)


class TpSlExecutionManager:
    def __init__(self, trader: Trader) -> None:
        self.trader = trader

    # ------------------------------------------------------------------
    # main TP / SL execution entry points
    # ------------------------------------------------------------------

    async def execute_near_tp_reduce(self, intent: TradeIntent) -> LiveTradeResult:
        t = self.trader
        action = "NEAR_TP_REDUCE"
        position = await t.fetch_position_snapshot()
        if not position.has_position:
            return LiveTradeResult(False, action, None, None, "0", t.price_to_str(intent.tp_price), "no position")
        if position.side != intent.side:
            return LiveTradeResult(False, action, None, None, "0", t.price_to_str(intent.tp_price),
                                   "position side mismatch")

        contracts_before = position.contracts
        reduce_ratio = Decimal(
            str(getattr(intent, "near_tp_reduce_ratio", 0.0) or os.getenv("NEAR_TP_REDUCE_RATIO", "0.5")))
        reduce_ratio = min(max(reduce_ratio, Decimal("0")), Decimal("1"))
        reduce_contracts = t.round_contracts_down(contracts_before * reduce_ratio)
        if reduce_contracts < t.min_contracts:
            return LiveTradeResult(
                False,
                action,
                None,
                None,
                t.decimal_to_str(contracts_before),
                t.price_to_str(intent.tp_price),
                "reduce size too small",
                contracts_before=t.decimal_to_str(contracts_before),
            )

        body = t._reduce_only_market_order_body(intent.side, reduce_contracts)
        res = await t.request("POST", "/api/v5/trade/order", body)
        order_id = t.extract_order_id(res)
        logger.warning(
            "NEAR_TP_REDUCE_ORDER_PLACED | side=%s contracts=%s ordId=%s",
            intent.side,
            t.decimal_to_str(reduce_contracts),
            order_id,
        )

        try:
            refreshed = await t.fetch_position_snapshot()
        except Exception:
            logger.exception("Failed to refresh position after Near-TP reduce")
            refreshed = PositionSnapshot(intent.side, contracts_before - reduce_contracts, position.avg_entry_price,
                                         0.0, Decimal("0"))
        contracts_after = refreshed.contracts if refreshed.has_position and refreshed.side == intent.side else Decimal(
            "0")
        t.position_contracts = contracts_after
        logger.warning(
            "NEAR_TP_REDUCE_FILLED | side=%s contracts_before=%s contracts_reduced=%s contracts_after=%s",
            intent.side,
            t.decimal_to_str(contracts_before),
            t.decimal_to_str(reduce_contracts),
            t.decimal_to_str(contracts_after),
        )

        base_result_kwargs = {
            "order_id": order_id,
            "contracts": t.decimal_to_str(reduce_contracts),
            "tp_price": t.price_to_str(intent.tp_price),
            "entry_filled": False,
            "reduce_filled": True,
            "contracts_before": t.decimal_to_str(contracts_before),
            "contracts_reduced": t.decimal_to_str(reduce_contracts),
            "contracts_after": t.decimal_to_str(contracts_after),
        }
        if contracts_after <= 0:
            return LiveTradeResult(
                True,
                action,
                tp_order_id=None,
                message="near_tp_reduce_closed_position",
                tp_ok=True,
                protective_sl_ok=True,
                near_tp_exit_all=True,
                **base_result_kwargs,
            )

        single_intent = replace(intent, partial_tp_price=None, partial_tp_ratio=0.0, tp_plan="SINGLE",
                                partial_tp_consumed=True)
        tp_order_id: str | None = None
        tp_order_ids: tuple[str, ...] = ()
        tp_ok = False
        final_tp_failure = ""
        try:
            tp = await self.trader.replace_take_profit(single_intent)
        except Exception as exc:
            logger.exception("Near-TP reduce filled but final TP replacement raised")
            final_tp_failure = f"final_tp_failed_exception: {exc}"
        else:
            tp_order_id = tp.tp_order_id
            tp_order_ids = tp.tp_order_ids
            tp_ok = bool(tp.ok)
            if tp.ok:
                logger.warning(
                    "NEAR_TP_FINAL_TP_REPLACED | side=%s contracts=%s tp_price=%s tp_order_id=%s",
                    intent.side,
                    tp.contracts,
                    tp.tp_price,
                    tp.tp_order_id,
                )
            else:
                final_tp_failure = f"final_tp_failed: {tp.message}"

        protective_sl_price = getattr(intent, "near_tp_protective_sl_price", None)
        if protective_sl_price is None:
            pct = float(os.getenv("NEAR_TP_PROTECTIVE_SL_PROFIT_PCT", "0.001"))
            protective_sl_price = intent.avg_entry_price * (
                        1 + pct) if intent.side == "LONG" else intent.avg_entry_price * (1 - pct)

        if os.getenv("NEAR_TP_PROTECTIVE_SL_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "y", "on"}:
            if not tp_ok:
                sl_ok = False
                sl_order_id = None
                sl_message = f"{final_tp_failure}; protective_sl_disabled"
            else:
                return LiveTradeResult(
                    True,
                    action,
                    tp_order_id=tp_order_id,
                    message="near_tp_reduce_done_protective_sl_disabled",
                    tp_ok=True,
                    tp_order_ids=tp_order_ids,
                    protective_sl_price=t.price_to_str(float(protective_sl_price)),
                    protective_sl_ok=True,
                    **base_result_kwargs,
                )
        else:
            sl_ok, sl_order_id, sl_message = await self.trader.place_near_tp_protective_stop_with_retries(
                intent.side,
                contracts_after,
                float(protective_sl_price),
                retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
            )
        message_prefix = f"{final_tp_failure}; " if final_tp_failure else ""
        if sl_ok:
            t.near_tp_protective_sl_order_id = sl_order_id
            logger.warning(
                "NEAR_TP_PROTECTIVE_SL_PLACED | side=%s contracts=%s stop_price=%s algoId=%s",
                intent.side,
                t.decimal_to_str(contracts_after),
                t.price_to_str(float(protective_sl_price)),
                sl_order_id,
            )
            return LiveTradeResult(
                True,
                action,
                tp_order_id=tp_order_id,
                message=f"{message_prefix}near_tp_reduce_done_final_tp_and_protective_sl_placed",
                tp_ok=tp_ok,
                tp_order_ids=tp_order_ids,
                protective_sl_order_id=sl_order_id,
                protective_sl_price=t.price_to_str(float(protective_sl_price)),
                protective_sl_ok=True,
                **base_result_kwargs,
            )

        fail_action = os.getenv("NEAR_TP_SL_FAIL_ACTION", "MARKET_EXIT").strip().upper()
        if fail_action == "MARKET_EXIT":
            logger.error("NEAR_TP_PROTECTIVE_SL_FAILED_MARKET_EXIT | side=%s message=%s", intent.side, sl_message)
            exit_ok, exit_message = await self.trader.market_exit_remaining_position_with_retries(
                intent.side,
                retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
            )
            if exit_ok:
                return LiveTradeResult(
                    True,
                    action,
                    tp_order_id=tp_order_id,
                    message=f"{message_prefix}protective_sl_failed_market_exit_success: {sl_message}; {exit_message}",
                    tp_ok=tp_ok,
                    tp_order_ids=tp_order_ids,
                    protective_sl_price=t.price_to_str(float(protective_sl_price)),
                    protective_sl_ok=False,
                    near_tp_exit_all=True,
                    contracts_after="0",
                    **{k: v for k, v in base_result_kwargs.items() if k != "contracts_after"},
                )
            return LiveTradeResult(
                False,
                action,
                tp_order_id=tp_order_id,
                message=f"{message_prefix}protective_sl_failed_and_market_exit_failed: {sl_message}; {exit_message}",
                tp_ok=tp_ok,
                tp_order_ids=tp_order_ids,
                protective_sl_price=t.price_to_str(float(protective_sl_price)),
                protective_sl_ok=False,
                **base_result_kwargs,
            )

        return LiveTradeResult(
            False,
            action,
            tp_order_id=tp_order_id,
            message=f"{message_prefix}protective_sl_failed_halt_only: {sl_message}",
            tp_ok=tp_ok,
            tp_order_ids=tp_order_ids,
            protective_sl_price=t.price_to_str(float(protective_sl_price)),
            protective_sl_ok=False,
            **base_result_kwargs,
        )

    async def execute_market_exit_runner(self, intent: TradeIntent) -> LiveTradeResult:
        t = self.trader
        action = "MARKET_EXIT_RUNNER"
        restored_trend_runner_sl_order_id = getattr(intent, "trend_runner_sl_order_id", None)
        if restored_trend_runner_sl_order_id and not t.trend_runner_sl_order_id:
            t.trend_runner_sl_order_id = restored_trend_runner_sl_order_id
        position = await t.fetch_position_snapshot()
        if not position.has_position:
            await self.trader._cleanup_after_near_tp_market_exit()
            return LiveTradeResult(True, action, None, None, "0", t.price_to_str(intent.tp_price),
                                   "runner_already_flat", near_tp_exit_all=True)
        if position.side != intent.side:
            await self.trader._cleanup_after_near_tp_market_exit()
            return LiveTradeResult(True, action, None, None, "0", t.price_to_str(intent.tp_price),
                                   "runner_side_absent", near_tp_exit_all=True)

        contracts_before = position.contracts
        ok, message = await self.trader.market_exit_remaining_position_with_retries(
            intent.side,
            retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
        )
        refreshed = await t.fetch_position_snapshot()
        contracts_after = refreshed.contracts if refreshed.has_position and refreshed.side == intent.side else Decimal(
            "0")
        t.position_contracts = contracts_after
        if ok:
            return LiveTradeResult(
                True,
                action,
                None,
                None,
                t.decimal_to_str(contracts_before),
                t.price_to_str(intent.tp_price),
                message,
                reduce_filled=True,
                near_tp_exit_all=True,
                contracts_before=t.decimal_to_str(contracts_before),
                contracts_reduced=t.decimal_to_str(contracts_before),
                contracts_after=t.decimal_to_str(contracts_after),
            )
        return LiveTradeResult(
            False,
            action,
            None,
            None,
            t.decimal_to_str(contracts_before),
            t.price_to_str(intent.tp_price),
            message,
            reduce_filled=True,
            near_tp_exit_all=False,
            contracts_before=t.decimal_to_str(contracts_before),
            contracts_reduced="",
            contracts_after=t.decimal_to_str(contracts_after),
        )

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

        specs = self.trader._build_take_profit_order_specs(intent)
        placed_order_ids: list[str] = []
        message = "take-profit replaced"
        try:
            placed_order_ids = await self.trader._place_reduce_only_take_profit_orders(intent, specs)
        except Exception:
            if len(specs) <= 1:
                raise
            logger.exception("Failed to place split take-profit orders; falling back to one full-size final TP")
            await self.trader._cancel_existing_take_profit_orders_for_intent(intent)
            fallback_specs = [("final", t.position_contracts, intent.tp_price)]
            placed_order_ids = await self.trader._place_reduce_only_take_profit_orders(intent, fallback_specs)
            specs = fallback_specs
            message = "split take-profit fallback to single final TP"

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
        )

    # ------------------------------------------------------------------
    # cancel / protect helpers for replace_take_profit
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # TP order spec building
    # ------------------------------------------------------------------

    def _build_take_profit_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        t = self.trader
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
        return [(spec.label, spec.contracts, spec.price) for spec in decision.specs]

    def _build_three_stage_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        return self.trader._build_take_profit_order_specs(intent)

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

    # ------------------------------------------------------------------
    # protective stop-loss placement with retries
    # ------------------------------------------------------------------

    async def place_near_tp_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal | str | int | float,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        t = self.trader
        contracts = t._to_decimal(contracts)
        retry_count = max(int(retry_count), 1)
        last_error = ""
        for attempt in range(1, retry_count + 1):
            try:
                body = t._near_tp_protective_sl_algo_body(side, contracts, stop_price)
                res = await t.request("POST", "/api/v5/trade/order-algo", body)
                algo_id = t.extract_algo_id(res)
                if await self.trader.verify_near_tp_protective_stop(algo_id, side, contracts, stop_price):
                    return True, algo_id, "protective_sl_placed"
                await self.trader._cancel_unverified_near_tp_algo(algo_id, phase="primary")
                last_error = f"protective_sl_verify_failed algoId={algo_id}"
                raise RuntimeError(last_error)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "NEAR_TP_PROTECTIVE_SL_RETRY | attempt=%s/%s side=%s contracts=%s stop_price=%s error=%s",
                    attempt,
                    retry_count,
                    side,
                    t.decimal_to_str(contracts),
                    t.price_to_str(stop_price),
                    exc,
                )
                if attempt < retry_count and retry_interval_seconds > 0:
                    await asyncio.sleep(retry_interval_seconds)

        for attempt in range(1, retry_count + 1):
            try:
                body = t._near_tp_fallback_conditional_close_body(side, contracts, stop_price)
                res = await t.request("POST", "/api/v5/trade/order-algo", body)
                algo_id = t.extract_algo_id(res)
                if await self.trader.verify_near_tp_protective_stop(algo_id, side, contracts, stop_price):
                    return True, algo_id, "fallback_conditional_close_placed"
                await self.trader._cancel_unverified_near_tp_algo(algo_id, phase="secondary")
                last_error = f"fallback_conditional_verify_failed algoId={algo_id}"
                raise RuntimeError(last_error)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "NEAR_TP_PROTECTIVE_SL_FALLBACK_RETRY | attempt=%s/%s side=%s contracts=%s stop_price=%s error=%s",
                    attempt,
                    retry_count,
                    side,
                    t.decimal_to_str(contracts),
                    t.price_to_str(stop_price),
                    exc,
                )
                if attempt < retry_count and retry_interval_seconds > 0:
                    await asyncio.sleep(retry_interval_seconds)
        return False, None, last_error or "protective_sl_retries_exhausted"

    async def place_middle_runner_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ok, order_id, message = await self.trader.place_near_tp_protective_stop_with_retries(
            side,
            contracts,
            stop_price,
            retry_count=retry_count,
            retry_interval_seconds=retry_interval_seconds,
        )
        if ok:
            self.trader.middle_runner_protective_sl_order_id = order_id
        return ok, order_id, message

    async def place_trend_runner_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ok, order_id, message = await self.trader.place_near_tp_protective_stop_with_retries(
            side,
            contracts,
            stop_price,
            retry_count=retry_count,
            retry_interval_seconds=retry_interval_seconds,
        )
        if ok:
            self.trader.trend_runner_sl_order_id = order_id
        return ok, order_id, message

    async def place_three_stage_post_tp1_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ok, order_id, message = await self.trader.place_near_tp_protective_stop_with_retries(
            side,
            contracts,
            stop_price,
            retry_count=retry_count,
            retry_interval_seconds=retry_interval_seconds,
        )
        if ok:
            self.trader.three_stage_post_tp1_protective_sl_order_id = order_id
        return ok, order_id, message

    async def _cancel_unverified_near_tp_algo(self, algo_id: str, *, phase: str) -> None:
        try:
            ok = await self.trader.cancel_near_tp_protective_stop(algo_id)
            logger.warning(
                "NEAR_TP_PROTECTIVE_SL_VERIFY_CANCELLED | phase=%s algoId=%s ok=%s",
                phase,
                algo_id,
                ok,
            )
        except Exception as exc:
            logger.warning(
                "NEAR_TP_PROTECTIVE_SL_VERIFY_CANCEL_FAILED | phase=%s algoId=%s error=%s",
                phase,
                algo_id,
                exc,
            )

    async def verify_near_tp_protective_stop(self, algo_id: str, side: PositionSide, contracts: Decimal,
                                             stop_price: float) -> bool:
        t = self.trader
        attempts = max(int(os.getenv("NEAR_TP_PROTECTIVE_SL_VERIFY_ATTEMPTS", "3")), 1)
        interval_seconds = float(os.getenv("NEAR_TP_PROTECTIVE_SL_VERIFY_INTERVAL_SECONDS", "0.2"))
        for attempt in range(1, attempts + 1):
            try:
                orders = await t.fetch_pending_algo_orders()
                for item in orders:
                    if self.trader._near_tp_protective_stop_matches(item, algo_id, side, contracts, stop_price):
                        return True
            except Exception as exc:
                logger.warning("NEAR_TP_PROTECTIVE_SL_VERIFY_FAILED | attempt=%s/%s algoId=%s error=%s", attempt,
                               attempts, algo_id, exc)
            if attempt < attempts and interval_seconds > 0:
                await asyncio.sleep(interval_seconds)
        logger.warning("NEAR_TP_PROTECTIVE_SL_VERIFY_MISSING | algoId=%s side=%s contracts=%s stop_price=%s", algo_id,
                       side, t.decimal_to_str(contracts), t.price_to_str(stop_price))
        return False

    def _near_tp_protective_stop_matches(self, item: dict[str, Any], algo_id: str, side: PositionSide,
                                         contracts: Decimal, stop_price: float) -> bool:
        t = self.trader
        item_algo_id = str(item.get("algoId") or item.get("ordId") or "")
        if item_algo_id != str(algo_id):
            return False
        if item.get("instId") != t.symbol:
            return False
        close_side = "sell" if side == "LONG" else "buy"
        if str(item.get("side", "")).lower() != close_side:
            return False
        try:
            item_contracts = Decimal(str(item.get("sz", "0")))
        except Exception:
            return False
        contract_tolerance = max(t.contract_precision, contracts.copy_abs() * Decimal("0.001"))
        if abs(item_contracts - contracts) > contract_tolerance:
            return False
        raw_trigger = item.get("slTriggerPx") or item.get("triggerPx")
        if raw_trigger is None:
            return False
        try:
            item_stop = Decimal(str(raw_trigger))
            expected_stop = Decimal(t.price_to_str(stop_price))
        except Exception:
            return False
        price_tolerance = max(Decimal("0.01"), expected_stop.copy_abs() * Decimal("0.0001"))
        return abs(item_stop - expected_stop) <= price_tolerance

    # ------------------------------------------------------------------
    # market exit
    # ------------------------------------------------------------------

    async def market_exit_remaining_position_with_retries(self, side: PositionSide, retry_count: int) -> tuple[
        bool, str]:
        t = self.trader
        retry_count = max(int(retry_count), 1)
        last_error = ""
        for attempt in range(1, retry_count + 1):
            try:
                position = await t.fetch_position_snapshot()
                if not position.has_position or position.contracts <= 0:
                    t.position_contracts = Decimal("0")
                    await self.trader._cleanup_after_near_tp_market_exit()
                    logger.warning("NEAR_TP_MARKET_EXIT_SUCCESS | reason=already_flat")
                    return True, "already_flat"
                if position.side != side:
                    t.position_contracts = Decimal("0")
                    await self.trader._cleanup_after_near_tp_market_exit()
                    logger.warning(
                        "NEAR_TP_MARKET_EXIT_SUCCESS | reason=target_side_absent expected_side=%s actual_side=%s contracts=%s",
                        side,
                        position.side,
                        t.decimal_to_str(position.contracts),
                    )
                    return True, "target_side_absent"
                if Decimal("0") < position.contracts < t.min_contracts:
                    last_error = (
                        f"dust_position_below_min_contracts contracts={t.decimal_to_str(position.contracts)} "
                        f"min_contracts={t.decimal_to_str(t.min_contracts)}"
                    )
                    logger.error("NEAR_TP_MARKET_EXIT_FAILED | reason=%s", last_error)
                    return False, last_error

                body = t._reduce_only_market_order_body(side, position.contracts)
                res = await t.request("POST", "/api/v5/trade/order", body)
                order_id = t.extract_order_id(res)
                refreshed = await t.fetch_position_snapshot()
                if not refreshed.has_position or refreshed.contracts <= 0:
                    t.position_contracts = Decimal("0")
                    await self.trader._cleanup_after_near_tp_market_exit()
                    logger.warning(
                        "NEAR_TP_MARKET_EXIT_SUCCESS | side=%s contracts=%s ordId=%s attempt=%s",
                        side,
                        t.decimal_to_str(position.contracts),
                        order_id,
                        attempt,
                    )
                    return True, f"market_exit_order_id={order_id}"
                if refreshed.side != side:
                    t.position_contracts = Decimal("0")
                    await self.trader._cleanup_after_near_tp_market_exit()
                    logger.warning(
                        "NEAR_TP_MARKET_EXIT_SUCCESS | reason=target_side_absent_after_order side=%s actual_side=%s ordId=%s attempt=%s",
                        side,
                        refreshed.side,
                        order_id,
                        attempt,
                    )
                    return True, f"market_exit_order_id={order_id};target_side_absent_after_order"
                if Decimal("0") < refreshed.contracts < t.min_contracts:
                    last_error = (
                        f"dust_position_below_min_contracts_after_order contracts={t.decimal_to_str(refreshed.contracts)} "
                        f"min_contracts={t.decimal_to_str(t.min_contracts)}"
                    )
                    logger.error(
                        "NEAR_TP_MARKET_EXIT_FAILED | reason=%s ordId=%s attempt=%s/%s",
                        last_error,
                        order_id,
                        attempt,
                        retry_count,
                    )
                    return False, last_error

                t.position_contracts = refreshed.contracts
                last_error = f"market_exit_not_flat_after_order contracts={t.decimal_to_str(refreshed.contracts)}"
                logger.error(
                    "NEAR_TP_MARKET_EXIT_FAILED | reason=not_flat_after_order attempt=%s/%s side=%s remaining_contracts=%s ordId=%s",
                    attempt,
                    retry_count,
                    side,
                    t.decimal_to_str(refreshed.contracts),
                    order_id,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.error("NEAR_TP_MARKET_EXIT_FAILED | attempt=%s/%s side=%s error=%s", attempt, retry_count, side,
                             exc)
        return False, last_error or "market_exit_failed"

    async def _cleanup_after_near_tp_market_exit(self) -> None:
        t = self.trader
        try:
            await self.trader.cancel_existing_reduce_only_orders()
        except Exception:
            logger.warning("NEAR_TP_MARKET_EXIT_SUCCESS | cleanup=cancel_reduce_only_tp_failed")
        if t.near_tp_protective_sl_order_id:
            await self.trader.cancel_near_tp_protective_stop(t.near_tp_protective_sl_order_id)
        middle_runner_sl_order_id = getattr(t, "middle_runner_protective_sl_order_id", None)
        if middle_runner_sl_order_id:
            await self.trader.cancel_middle_runner_protective_stop(middle_runner_sl_order_id)
        three_stage_post_tp1_sl_order_id = getattr(t, "three_stage_post_tp1_protective_sl_order_id", None)
        if three_stage_post_tp1_sl_order_id:
            await self.trader.cancel_three_stage_post_tp1_protective_stop(three_stage_post_tp1_sl_order_id)
        trend_runner_sl_order_id = getattr(t, "trend_runner_sl_order_id", None)
        if trend_runner_sl_order_id:
            await self.trader.cancel_trend_runner_protective_stop(trend_runner_sl_order_id)

    # ------------------------------------------------------------------
    # TP price helpers
    # ------------------------------------------------------------------

    def _tp_price_summary(self, specs: list[tuple[str, Decimal, float]]) -> str:
        t = self.trader
        if len(specs) == 1:
            return t.price_to_str(specs[0][2])
        return ",".join(f"{label}:{t.price_to_str(price)}" for label, _contracts, price in specs)

    # ------------------------------------------------------------------
    # cancel reduce-only orders
    # ------------------------------------------------------------------

    async def cancel_existing_reduce_only_orders(self) -> None:
        t = self.trader
        orders = await t.fetch_pending_orders()
        protected_order_ids = set(getattr(t, "_protected_reduce_only_order_ids", set()) or set())
        managed_order_ids = set(getattr(t, "_managed_reduce_only_order_ids", set()) or set())
        allow_unmanaged = bool(getattr(t, "_allow_cancel_unmanaged_reduce_only", True))
        for item in orders:
            if item.get("instId") != t.symbol:
                continue
            if str(item.get("reduceOnly", "")).lower() != "true":
                continue
            ord_id = item.get("ordId")
            if not ord_id:
                raise RuntimeError("reduce_only_order_identity_unknown")
            ord_id = str(ord_id)
            if ord_id in protected_order_ids:
                logger.info("Protected reduce-only order skipped | ordId=%s", ord_id)
                continue
            if managed_order_ids and ord_id not in managed_order_ids:
                raise RuntimeError("reduce_only_order_identity_unknown")
            if not managed_order_ids and not allow_unmanaged:
                raise RuntimeError("reduce_only_order_identity_unknown")
            try:
                await t.request("POST", "/api/v5/trade/cancel-order", order_specs.build_cancel_order_body(
                    inst_id=t.symbol,
                    order_id=ord_id,
                ))
                logger.info("Canceled existing reduce-only order | ordId=%s", ord_id)
            except Exception:
                logger.exception("Failed to cancel existing reduce-only order | ordId=%s", ord_id)

    # ------------------------------------------------------------------
    # sidecar fixed TP
    # ------------------------------------------------------------------

    async def place_sidecar_fixed_take_profit(
            self,
            *,
            side: PositionSide,
            contracts: str | Decimal,
            tp_price: float,
            client_order_id: str | None = None,
    ) -> str:
        t = self.trader
        sent_client_order_id = ""
        if client_order_id:
            sent_client_order_id = sanitize_okx_client_order_id(client_order_id)
        body = order_specs.build_reduce_only_tp_order_body(
            inst_id=t.symbol,
            td_mode=t.td_mode,
            side=side,
            contracts_text=t.decimal_to_str(Decimal(str(contracts))),
            price_text=t.price_to_str(float(tp_price)),
            pos_side_mode=t.pos_side_mode,
            client_order_id=sent_client_order_id or None,
        )
        res = await t.request("POST", "/api/v5/trade/order", body)
        order_id = t.extract_order_id(res)
        logger.warning(
            "SIDECAR_TP_PLACED | side=%s contracts=%s tp_price=%s sent_clOrdId=%s ordId=%s",
            side,
            t.decimal_to_str(Decimal(str(contracts))),
            t.price_to_str(float(tp_price)),
            sent_client_order_id or "-",
            order_id,
        )
        return order_id

    async def cancel_sidecar_take_profit(self, order_id: str | None) -> bool:
        t = self.trader
        if not order_id:
            return True
        try:
            await t.request("POST", "/api/v5/trade/cancel-order", order_specs.build_cancel_order_body(
                inst_id=t.symbol,
                order_id=order_id,
            ))
            logger.warning("SIDECAR_TP_CANCELLED | ordId=%s", order_id)
            return True
        except Exception as exc:
            text = str(exc).lower()
            if "not found" in text or "not exist" in text or "does not exist" in text or "already" in text:
                logger.info("SIDECAR_TP_CANCELLED | ordId=%s already_absent message=%s", order_id, exc)
                return True
            logger.error("SIDECAR_TP_CANCEL_FAILED | ordId=%s error=%s", order_id, exc)
            return False

    # ------------------------------------------------------------------
    # cancel protective stops
    # ------------------------------------------------------------------

    async def cancel_near_tp_protective_stop(self, order_id: str | None) -> bool:
        t = self.trader
        if not order_id:
            return True
        try:
            await t.request("POST", "/api/v5/trade/cancel-algos", order_specs.build_cancel_algo_body(
                inst_id=t.symbol,
                algo_id=order_id,
            ))
            if t.near_tp_protective_sl_order_id == order_id:
                t.near_tp_protective_sl_order_id = None
            logger.warning("NEAR_TP_PROTECTIVE_SL_CANCEL_ON_FLAT | algoId=%s", order_id)
            return True
        except Exception as exc:
            text = str(exc).lower()
            if "not found" in text or "not exist" in text or "does not exist" in text or "already" in text:
                logger.info("NEAR_TP_PROTECTIVE_SL_CANCEL_ON_FLAT | algoId=%s already_absent message=%s", order_id, exc)
                return True
            logger.warning("NEAR_TP_PROTECTIVE_SL_CANCEL_ON_FLAT | algoId=%s failed=%s", order_id, exc)
            return False

    async def cancel_middle_runner_protective_stop(self, order_id: str | None) -> bool:
        t = self.trader
        if not order_id:
            return True
        ok = await self.trader.cancel_near_tp_protective_stop(order_id)
        if ok and getattr(t, "middle_runner_protective_sl_order_id", None) == order_id:
            t.middle_runner_protective_sl_order_id = None
        if ok:
            logger.warning("MIDDLE_RUNNER_SL_CANCELLED | algoId=%s", order_id)
        return ok

    async def cancel_trend_runner_protective_stop(self, order_id: str | None) -> bool:
        t = self.trader
        if not order_id:
            return True
        ok = await self.trader.cancel_near_tp_protective_stop(order_id)
        if ok and getattr(t, "trend_runner_sl_order_id", None) == order_id:
            t.trend_runner_sl_order_id = None
        if ok:
            logger.warning("TREND_RUNNER_SL_CANCELLED | algoId=%s", order_id)
        return ok

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id: str | None) -> bool:
        t = self.trader
        if not order_id:
            return True
        ok = await self.trader.cancel_near_tp_protective_stop(order_id)
        if ok and getattr(t, "three_stage_post_tp1_protective_sl_order_id", None) == order_id:
            t.three_stage_post_tp1_protective_sl_order_id = None
        if ok:
            logger.warning("THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED | algoId=%s", order_id)
        return ok
