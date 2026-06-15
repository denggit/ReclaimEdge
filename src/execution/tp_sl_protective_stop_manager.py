from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from typing import Any, TYPE_CHECKING

from src.execution.trading_client_port import AlgoOrderSnapshot
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader
    from src.execution.trading_client_port import TradingClientPort
    from src.strategies.boll_cvd_reclaim_strategy import PositionSide

logger = get_logger(__name__)


class ProtectiveStopManager:
    def __init__(self, trader: Trader, trading_client: TradingClientPort) -> None:
        self.trader = trader
        self.trading_client = trading_client

    # ------------------------------------------------------------------
    # semantic protective SL placement switch (opt-in)
    # ------------------------------------------------------------------

    @staticmethod
    def _broker_semantic_protective_sl_placement_enabled() -> bool:
        value = os.getenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", "false").strip().lower()
        return value in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _broker_position_side(side: str):
        from src.exchanges.models import BrokerPositionSide

        if side == "LONG":
            return BrokerPositionSide.LONG
        if side == "SHORT":
            return BrokerPositionSide.SHORT
        raise RuntimeError(f"unsupported_position_side_for_semantic_protective_sl: {side}")

    async def _place_primary_protective_stop_semantic(
        self,
        *,
        side: str,
        contracts: Decimal,
        stop_price: float,
    ) -> str:
        t = self.trader
        from src.exchanges.models import BrokerQuantityUnit
        from src.exchanges.semantic_models import BrokerSemanticOrderRole

        result = await t.broker_semantic_executor.place_protective_stop(
            symbol=t.symbol,
            side=self._broker_position_side(side),
            quantity=contracts,
            trigger_price=Decimal(str(stop_price)),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
        )
        if not result.ok or not result.order_id:
            raise RuntimeError(
                f"semantic_protective_sl_order_failed side={side} "
                f"contracts={t.decimal_to_str(contracts)} stop_price={t.price_to_str(stop_price)} "
                f"message={result.message}"
            )
        return str(result.order_id)

    async def place_protective_stop_with_retries(
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
                if self._broker_semantic_protective_sl_placement_enabled():
                    algo_id = await self._place_primary_protective_stop_semantic(
                        side=side,
                        contracts=contracts,
                        stop_price=stop_price,
                    )
                else:
                    result = await self.trading_client.place_stop_market_order(
                        side=side,
                        qty=contracts,
                        trigger_price=Decimal(str(stop_price)),
                        reduce_only=True,
                        client_order_id="",
                    )
                    algo_id = result.order_id
                if await self.trader.verify_protective_stop(algo_id, side, contracts, stop_price):
                    return True, algo_id, "protective_sl_placed"
                await self.trader._cancel_unverified_algo(algo_id, phase="primary")
                last_error = f"protective_sl_verify_failed algoId={algo_id}"
                raise RuntimeError(last_error)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "PROTECTIVE_SL_RETRY | attempt=%s/%s side=%s contracts=%s stop_price=%s error=%s",
                    attempt,
                    retry_count,
                    side,
                    t.decimal_to_str(contracts),
                    t.price_to_str(stop_price),
                    exc,
                )
                if attempt < retry_count and retry_interval_seconds > 0:
                    await asyncio.sleep(retry_interval_seconds)

        # NOTE:
        # This fallback loop uses the same TradingClientPort.place_stop_market_order()
        # primitive as the primary path. The loop is retained for parity with the
        # original retry budget (2 × retry_count total attempts) and for distinct
        # logging / phase labelling.
        for attempt in range(1, retry_count + 1):
            try:
                result = await self.trading_client.place_stop_market_order(
                    side=side,
                    qty=contracts,
                    trigger_price=Decimal(str(stop_price)),
                    reduce_only=True,
                    client_order_id="",
                )
                algo_id = result.order_id
                if algo_id is None:
                    raise RuntimeError("protective_stop_fallback_missing_order_id")
                if await self.trader.verify_protective_stop(algo_id, side, contracts, stop_price):
                    return True, algo_id, "fallback_conditional_close_placed"
                await self.trader._cancel_unverified_algo(algo_id, phase="secondary")
                last_error = f"fallback_conditional_verify_failed algoId={algo_id}"
                raise RuntimeError(last_error)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "PROTECTIVE_SL_FALLBACK_RETRY | attempt=%s/%s side=%s contracts=%s stop_price=%s error=%s",
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

    async def place_entry_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal | str | int | float,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ok, order_id, message = await self.place_protective_stop_with_retries(
            side,
            contracts,
            stop_price,
            retry_count=retry_count,
            retry_interval_seconds=retry_interval_seconds,
        )
        if ok:
            self.trader.entry_protective_sl_order_id = order_id
        return ok, order_id, message

    async def place_middle_runner_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ok, order_id, message = await self.place_protective_stop_with_retries(
            side,
            contracts,
            stop_price,
            retry_count=retry_count,
            retry_interval_seconds=retry_interval_seconds,
        )
        if ok:
            self.trader.middle_runner_protective_sl_order_id = order_id
        return ok, order_id, message

    async def place_middle_bucket_fast_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ok, order_id, message = await self.place_protective_stop_with_retries(
            side,
            contracts,
            stop_price,
            retry_count=retry_count,
            retry_interval_seconds=retry_interval_seconds,
        )
        if ok:
            self.trader.middle_bucket_fast_sl_order_id = order_id
        return ok, order_id, message

    async def place_trend_runner_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ok, order_id, message = await self.place_protective_stop_with_retries(
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
        ok, order_id, message = await self.place_protective_stop_with_retries(
            side,
            contracts,
            stop_price,
            retry_count=retry_count,
            retry_interval_seconds=retry_interval_seconds,
        )
        if ok:
            self.trader.three_stage_post_tp1_protective_sl_order_id = order_id
        return ok, order_id, message

    async def _cancel_unverified_algo(self, algo_id: str, *, phase: str) -> None:
        try:
            ok = await self.trader.cancel_protective_stop(algo_id)
            logger.warning(
                "PROTECTIVE_SL_VERIFY_CANCELLED | phase=%s algoId=%s ok=%s",
                phase,
                algo_id,
                ok,
            )
        except Exception as exc:
            logger.warning(
                "PROTECTIVE_SL_VERIFY_CANCEL_FAILED | phase=%s algoId=%s error=%s",
                phase,
                algo_id,
                exc,
            )

    async def verify_protective_stop(self, algo_id: str, side: PositionSide, contracts: Decimal,
                                             stop_price: float) -> bool:
        t = self.trader
        attempts = max(int(os.getenv("PROTECTIVE_SL_VERIFY_ATTEMPTS", "3")), 1)
        interval_seconds = float(os.getenv("PROTECTIVE_SL_VERIFY_INTERVAL_SECONDS", "0.2"))
        for attempt in range(1, attempts + 1):
            try:
                orders = await self.trading_client.fetch_open_algo_orders()
                for item in orders:
                    if self._protective_stop_snapshot_matches(item, algo_id, side, contracts, stop_price):
                        return True
            except Exception as exc:
                logger.warning("PROTECTIVE_SL_VERIFY_FAILED | attempt=%s/%s algoId=%s error=%s", attempt,
                               attempts, algo_id, exc)
            if attempt < attempts and interval_seconds > 0:
                await asyncio.sleep(interval_seconds)
        logger.warning("PROTECTIVE_SL_VERIFY_MISSING | algoId=%s side=%s contracts=%s stop_price=%s", algo_id,
                       side, t.decimal_to_str(contracts), t.price_to_str(stop_price))
        return False

    def _protective_stop_snapshot_matches(
        self,
        item: AlgoOrderSnapshot,
        algo_id: str,
        side: PositionSide,
        contracts: Decimal,
        stop_price: float,
    ) -> bool:
        t = self.trader
        if item.order_id != str(algo_id):
            return False
        close_side = "sell" if side == "LONG" else "buy"
        if str(item.side or "").lower() != close_side:
            return False
        item_qty = item.qty
        if item_qty is None:
            return False
        contract_tolerance = max(t.contract_precision, contracts.copy_abs() * Decimal("0.001"))
        if abs(item_qty - contracts) > contract_tolerance:
            return False
        item_trigger = item.trigger_price
        if item_trigger is None:
            return False
        try:
            expected_stop = Decimal(t.price_to_str(stop_price))
        except Exception:
            return False
        price_tolerance = max(Decimal("0.01"), expected_stop.copy_abs() * Decimal("0.0001"))
        return abs(item_trigger - expected_stop) <= price_tolerance

    def _protective_stop_matches(self, item: dict[str, Any], algo_id: str, side: PositionSide,
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
