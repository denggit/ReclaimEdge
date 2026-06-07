from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from typing import Any, TYPE_CHECKING

from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader
    from src.strategies.boll_cvd_reclaim_strategy import PositionSide

logger = get_logger(__name__)


class ProtectiveStopManager:
    def __init__(self, trader: Trader) -> None:
        self.trader = trader

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
