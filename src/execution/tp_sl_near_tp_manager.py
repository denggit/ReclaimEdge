from __future__ import annotations

import os
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING

from src.execution.broker_semantic_helpers import (
    broker_position_side,
    close_order_side,
    get_broker_semantic_executor,
)
from src.execution.trader import LiveTradeResult, PositionSnapshot
from src.exchanges.models import BrokerOrderSide, BrokerPositionSide, ExchangeName
from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest
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

        semantic_executor = get_broker_semantic_executor(t)
        if semantic_executor is not None:
            result = await semantic_executor.execute(
                BrokerSemanticRequest(
                    exchange=ExchangeName.OKX,
                    symbol=t.symbol,
                    action=BrokerSemanticAction.MARKET_EXIT,
                    role=BrokerSemanticOrderRole.MARKET_EXIT,
                    side=close_order_side(intent.side),
                    position_side=broker_position_side(intent.side),
                    quantity=reduce_contracts,
                    reduce_only=True,
                    close_position=False,
                    metadata={"context": "near_tp_reduce"},
                )
            )
            order_id = result.order_id or ""
            if not order_id:
                raise RuntimeError(f"Missing near TP reduce order_id in broker semantic result: {result}")
        else:
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

        # ── Protective SL failed → NO immediate market exit ─────────────
        # The caller (execution command processor) is responsible for arming
        # the delayed market exit.  We return a result that clearly indicates
        # the SL failure without triggering an immediate position liquidation.
        fail_action = os.getenv("NEAR_TP_SL_FAIL_ACTION", "HALT_ONLY").strip().upper()
        if fail_action == "MARKET_EXIT":
            logger.error(
                "NEAR_TP_PROTECTIVE_SL_FAILED_DELAYED_EXIT_ARM | side=%s message=%s "
                "delayed_market_exit_armed_by_caller=true no_immediate_market_exit=true",
                intent.side,
                sl_message,
            )
        else:
            logger.error(
                "NEAR_TP_PROTECTIVE_SL_FAILED_HALT_ONLY | side=%s message=%s",
                intent.side,
                sl_message,
            )

        return LiveTradeResult(
            False,
            action,
            tp_order_id=tp_order_id,
            message=f"{message_prefix}protective_sl_failed: {sl_message}",
            tp_ok=tp_ok,
            tp_order_ids=tp_order_ids,
            protective_sl_price=t.price_to_str(float(protective_sl_price)),
            protective_sl_ok=False,
            near_tp_exit_all=False,
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
