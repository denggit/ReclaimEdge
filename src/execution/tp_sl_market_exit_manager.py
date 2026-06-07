from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from src.execution import order_specs
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader
    from src.strategies.boll_cvd_reclaim_strategy import PositionSide

logger = get_logger(__name__)


class MarketExitManager:
    def __init__(self, trader: Trader) -> None:
        self.trader = trader

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
