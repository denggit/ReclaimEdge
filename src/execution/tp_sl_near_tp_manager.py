from __future__ import annotations

import os
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING

from src.execution.trader import LiveTradeResult, PositionSnapshot
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader
    from src.strategies.boll_cvd_reclaim_strategy import PositionSide, TradeIntent

logger = get_logger(__name__)


class NearTpExecutionManager:
    def __init__(self, trader: Trader, core_tp, protective_stops, market_exit) -> None:
        self.trader = trader
        self.core_tp = core_tp
        self.protective_stops = protective_stops
        self.market_exit = market_exit

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
                context="near_tp_sl_failed",
                retry_interval_seconds=float(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_INTERVAL_SECONDS", "0.5")),
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
            await self.trader._cleanup_after_market_exit()
            return LiveTradeResult(True, action, None, None, "0", t.price_to_str(intent.tp_price),
                                   "runner_already_flat", near_tp_exit_all=True)
        if position.side != intent.side:
            await self.trader._cleanup_after_market_exit()
            return LiveTradeResult(True, action, None, None, "0", t.price_to_str(intent.tp_price),
                                   "runner_side_absent", near_tp_exit_all=True)

        contracts_before = position.contracts
        ok, message = await self.trader.market_exit_remaining_position_with_retries(
            intent.side,
            retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
            context="near_tp_market_exit_runner",
            retry_interval_seconds=float(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_INTERVAL_SECONDS", "0.5")),
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
