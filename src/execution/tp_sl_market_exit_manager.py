from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING

from src.execution.broker_semantic_helpers import (
    broker_position_side,
    close_order_side,
    get_broker_semantic_executor,
    require_semantic_order_id,
)
from src.exchanges.models import BrokerOrderSide, BrokerPositionSide, ExchangeName
from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader
    from src.strategies.boll_cvd_reclaim_strategy import PositionSide

logger = get_logger(__name__)


class MarketExitManager:
    def __init__(self, trader: Trader) -> None:
        self.trader = trader

    async def market_exit_remaining_position_with_retries(
        self,
        side: PositionSide,
        retry_count: int,
        *,
        context: str = "generic",
        retry_interval_seconds: float | None = None,
    ) -> tuple[bool, str]:
        t = self.trader
        retry_count = max(int(retry_count), 1)
        last_error = ""
        for attempt in range(1, retry_count + 1):
            if attempt > 1 and retry_interval_seconds is not None and retry_interval_seconds > 0:
                await asyncio.sleep(retry_interval_seconds)
            try:
                position = await t.fetch_position_snapshot()
                if not position.has_position or position.contracts <= 0:
                    t.position_contracts = Decimal("0")
                    await self.trader._cleanup_after_market_exit()
                    logger.warning(
                        "MARKET_EXIT_SUCCESS | context=%s side=%s reason=already_flat",
                        context,
                        side,
                    )
                    return True, "already_flat"
                if position.side != side:
                    t.position_contracts = Decimal("0")
                    await self.trader._cleanup_after_market_exit()
                    logger.warning(
                        "MARKET_EXIT_SUCCESS | context=%s side=%s reason=target_side_absent expected_side=%s actual_side=%s contracts=%s",
                        context,
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
                    logger.error(
                        "MARKET_EXIT_FAILED | context=%s side=%s reason=%s",
                        context,
                        side,
                        last_error,
                    )
                    return False, last_error

                semantic_executor = get_broker_semantic_executor(t)
                if semantic_executor is not None:
                    result = await semantic_executor.execute(
                        BrokerSemanticRequest(
                            exchange=ExchangeName.OKX,
                            symbol=t.symbol,
                            action=(
                                BrokerSemanticAction.MARKET_EXIT_RUNNER
                                if "runner" in str(context).lower()
                                else BrokerSemanticAction.MARKET_EXIT
                            ),
                            role=BrokerSemanticOrderRole.MARKET_EXIT,
                            side=close_order_side(side),
                            position_side=broker_position_side(side),
                            quantity=position.contracts,
                            reduce_only=True,
                            close_position=True,
                            metadata={"context": context},
                        )
                    )
                    order_id = require_semantic_order_id(result, action="MARKET_EXIT")
                else:
                    body = t._reduce_only_market_order_body(side, position.contracts)
                    res = await t.request("POST", "/api/v5/trade/order", body)
                    order_id = t.extract_order_id(res)
                refreshed = await t.fetch_position_snapshot()
                if not refreshed.has_position or refreshed.contracts <= 0:
                    t.position_contracts = Decimal("0")
                    await self.trader._cleanup_after_market_exit()
                    logger.warning(
                        "MARKET_EXIT_SUCCESS | context=%s side=%s contracts=%s ordId=%s attempt=%s/%s",
                        context,
                        side,
                        t.decimal_to_str(position.contracts),
                        order_id,
                        attempt,
                        retry_count,
                    )
                    return True, f"market_exit_order_id={order_id}"
                if refreshed.side != side:
                    t.position_contracts = Decimal("0")
                    await self.trader._cleanup_after_market_exit()
                    logger.warning(
                        "MARKET_EXIT_SUCCESS | context=%s side=%s reason=target_side_absent_after_order actual_side=%s ordId=%s attempt=%s/%s",
                        context,
                        side,
                        refreshed.side,
                        order_id,
                        attempt,
                        retry_count,
                    )
                    return True, f"market_exit_order_id={order_id};target_side_absent_after_order"
                if Decimal("0") < refreshed.contracts < t.min_contracts:
                    last_error = (
                        f"dust_position_below_min_contracts_after_order contracts={t.decimal_to_str(refreshed.contracts)} "
                        f"min_contracts={t.decimal_to_str(t.min_contracts)}"
                    )
                    logger.error(
                        "MARKET_EXIT_FAILED | context=%s side=%s reason=%s ordId=%s attempt=%s/%s",
                        context,
                        side,
                        last_error,
                        order_id,
                        attempt,
                        retry_count,
                    )
                    return False, last_error

                t.position_contracts = refreshed.contracts
                last_error = f"market_exit_not_flat_after_order contracts={t.decimal_to_str(refreshed.contracts)}"
                logger.error(
                    "MARKET_EXIT_FAILED | context=%s side=%s reason=not_flat_after_order attempt=%s/%s remaining_contracts=%s ordId=%s",
                    context,
                    side,
                    attempt,
                    retry_count,
                    t.decimal_to_str(refreshed.contracts),
                    order_id,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.error(
                    "MARKET_EXIT_FAILED | context=%s side=%s attempt=%s/%s error=%s",
                    context,
                    side,
                    attempt,
                    retry_count,
                    exc,
                )
        return False, last_error or "market_exit_failed"

    async def _cleanup_after_market_exit(self) -> None:
        t = self.trader
        try:
            try:
                await self.trader.cancel_existing_reduce_only_orders(phase="market_exit_runner")
            except TypeError:
                await self.trader.cancel_existing_reduce_only_orders()
        except Exception:
            logger.warning("MARKET_EXIT_CLEANUP | cleanup=cancel_reduce_only_tp_failed")
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

    # Backward-compat alias — new code must call _cleanup_after_market_exit
    _cleanup_after_near_tp_market_exit = _cleanup_after_market_exit
