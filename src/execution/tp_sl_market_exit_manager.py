from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING

from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader
    from src.execution.trading_client_port import TradingClientPort
    from src.strategies.boll_cvd_reclaim_strategy import PositionSide

logger = get_logger(__name__)


class MarketExitManager:
    def __init__(self, trader: Trader, trading_client: TradingClientPort) -> None:
        self.trader = trader
        self.trading_client = trading_client

    def _broker_semantic_market_exit_enabled(self) -> bool:
        import os

        value = os.getenv("BROKER_SEMANTIC_MARKET_EXIT_ENABLED", "false").strip().lower()
        return value in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _broker_position_side(side: str):
        from src.exchanges.models import BrokerPositionSide

        if side == "LONG":
            return BrokerPositionSide.LONG
        if side == "SHORT":
            return BrokerPositionSide.SHORT
        raise RuntimeError(f"unsupported_position_side_for_semantic_market_exit: {side}")

    @staticmethod
    def _is_runner_market_exit_context(context: str) -> bool:
        normalized = str(context or "").strip().lower()
        return "runner" in normalized

    async def _place_market_exit_order_semantic(
        self,
        *,
        side: str,
        contracts: Decimal,
        context: str,
    ) -> str:
        t = self.trader
        from src.exchanges.models import BrokerQuantityUnit

        executor = t.broker_semantic_executor
        common_kwargs = {
            "symbol": t.symbol,
            "side": self._broker_position_side(side),
            "quantity": contracts,
            "quantity_unit": BrokerQuantityUnit.CONTRACTS,
            "label": context,
        }
        if self._is_runner_market_exit_context(context):
            result = await executor.market_exit_runner(**common_kwargs)
        else:
            result = await executor.market_exit(**common_kwargs)
        if not result.ok or not result.order_id:
            raise RuntimeError(
                f"semantic_market_exit_order_failed context={context} side={side} "
                f"contracts={t.decimal_to_str(contracts)} message={result.message}"
            )
        return str(result.order_id)

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
                position = await self.trading_client.fetch_position()
                if not position.has_position or position.qty <= 0:
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
                        side,
                        position.side,
                        t.decimal_to_str(position.qty),
                    )
                    return True, "target_side_absent"
                if Decimal("0") < position.qty < t.min_contracts:
                    last_error = (
                        f"dust_position_below_min_contracts contracts={t.decimal_to_str(position.qty)} "
                        f"min_contracts={t.decimal_to_str(t.min_contracts)}"
                    )
                    logger.error(
                        "MARKET_EXIT_FAILED | context=%s side=%s reason=%s",
                        context,
                        side,
                        last_error,
                    )
                    return False, last_error

                if self._broker_semantic_market_exit_enabled():
                    order_id = await self._place_market_exit_order_semantic(
                        side=side,
                        contracts=position.qty,
                        context=context,
                    )
                else:
                    result = await self.trading_client.place_market_order(
                        side=side,
                        qty=position.qty,
                        reduce_only=True,
                        client_order_id="",
                    )
                    order_id = result.order_id
                    if order_id is None:
                        raise RuntimeError("reduce_only_market_exit_missing_order_id")
                refreshed = await self.trading_client.fetch_position()
                if not refreshed.has_position or refreshed.qty <= 0:
                    t.position_contracts = Decimal("0")
                    await self.trader._cleanup_after_market_exit()
                    logger.warning(
                        "MARKET_EXIT_SUCCESS | context=%s side=%s contracts=%s ordId=%s attempt=%s/%s",
                        context,
                        side,
                        t.decimal_to_str(position.qty),
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
                if Decimal("0") < refreshed.qty < t.min_contracts:
                    last_error = (
                        f"dust_position_below_min_contracts_after_order contracts={t.decimal_to_str(refreshed.qty)} "
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

                t.position_contracts = refreshed.qty
                last_error = f"market_exit_not_flat_after_order contracts={t.decimal_to_str(refreshed.qty)}"
                logger.error(
                    "MARKET_EXIT_FAILED | context=%s side=%s reason=not_flat_after_order attempt=%s/%s remaining_contracts=%s ordId=%s",
                    context,
                    side,
                    attempt,
                    retry_count,
                    t.decimal_to_str(refreshed.qty),
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
            await self.trader.cancel_existing_reduce_only_orders()
        except Exception:
            logger.warning("MARKET_EXIT_CLEANUP | cleanup=cancel_reduce_only_tp_failed")
        entry_sl_order_id = getattr(t, "entry_protective_sl_order_id", None)
        if entry_sl_order_id:
            await self.trader.cancel_near_tp_protective_stop(entry_sl_order_id)
            t.entry_protective_sl_order_id = None
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
