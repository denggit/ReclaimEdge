from __future__ import annotations

from decimal import Decimal
from typing import Any, TYPE_CHECKING

from src.execution import order_specs
from src.execution.broker_semantic_helpers import get_broker_semantic_executor
from src.exchanges.models import ExchangeName
from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest
from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager
from src.execution.tp_sl_market_exit_manager import MarketExitManager
from src.execution.tp_sl_near_tp_manager import NearTpExecutionManager
from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager
from src.execution.tp_sl_sidecar_manager import SidecarTpManager
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader, LiveTradeResult
    from src.strategies.boll_cvd_reclaim_strategy import PositionSide, TradeIntent

logger = get_logger(__name__)


class TpSlExecutionManager:
    def __init__(self, trader: Trader) -> None:
        self.trader = trader
        self.protective_stops = ProtectiveStopManager(trader)
        self.market_exit = MarketExitManager(trader)
        self.core_tp = CoreTakeProfitManager(trader, self.protective_stops)
        self.near_tp = NearTpExecutionManager(
            trader=trader,
            core_tp=self.core_tp,
            protective_stops=self.protective_stops,
            market_exit=self.market_exit,
        )
        self.sidecar = SidecarTpManager(trader)

    # ------------------------------------------------------------------
    # main TP / SL execution entry points
    # ------------------------------------------------------------------

    async def execute_near_tp_reduce(self, intent: TradeIntent) -> LiveTradeResult:
        return await self.near_tp.execute_near_tp_reduce(intent)

    async def execute_market_exit_runner(self, intent: TradeIntent) -> LiveTradeResult:
        return await self.near_tp.execute_market_exit_runner(intent)

    async def replace_take_profit(self, intent: TradeIntent) -> LiveTradeResult:
        return await self.core_tp.replace_take_profit(intent)

    # ------------------------------------------------------------------
    # cancel / protect helpers for replace_take_profit
    # ------------------------------------------------------------------

    async def _cancel_existing_take_profit_orders_for_intent(self, intent: TradeIntent) -> bool:
        return await self.core_tp._cancel_existing_take_profit_orders_for_intent(intent)

    async def _cancel_stale_runner_protective_stops_for_degrade(self, intent: TradeIntent) -> None:
        return await self.core_tp._cancel_stale_runner_protective_stops_for_degrade(intent)

    def _protected_order_ids_from_intent(self, intent: TradeIntent) -> set[str]:
        return self.core_tp._protected_order_ids_from_intent(intent)

    @staticmethod
    def _split_order_ids(value: str | None) -> set[str]:
        if not value:
            return set()
        return {item.strip() for item in str(value).split(",") if item.strip()}

    def _managed_core_contracts_from_intent(self, intent: TradeIntent) -> Decimal | None:
        return self.core_tp._managed_core_contracts_from_intent(intent)

    # ------------------------------------------------------------------
    # TP order spec building
    # ------------------------------------------------------------------

    def _build_take_profit_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        return self.core_tp._build_take_profit_order_specs_public(intent)

    def _build_three_stage_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        return self.core_tp._build_three_stage_order_specs_public(intent)

    def _trend_runner_sl_contracts(self, intent: TradeIntent, net_contracts_for_sl: Decimal) -> Decimal:
        return self.core_tp._trend_runner_sl_contracts(intent, net_contracts_for_sl)

    async def _place_reduce_only_take_profit_orders(self, intent: TradeIntent,
                                                    specs: list[tuple[str, Decimal, float]]) -> list[str]:
        return await self.core_tp._place_reduce_only_take_profit_orders(intent, specs)

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
        return await self.protective_stops.place_near_tp_protective_stop_with_retries(
            side, contracts, stop_price, retry_count, retry_interval_seconds)

    async def place_middle_runner_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        return await self.protective_stops.place_middle_runner_protective_stop_with_retries(
            side, contracts, stop_price, retry_count, retry_interval_seconds)

    async def place_trend_runner_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        return await self.protective_stops.place_trend_runner_protective_stop_with_retries(
            side, contracts, stop_price, retry_count, retry_interval_seconds)

    async def place_three_stage_post_tp1_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        return await self.protective_stops.place_three_stage_post_tp1_protective_stop_with_retries(
            side, contracts, stop_price, retry_count, retry_interval_seconds)

    async def place_middle_bucket_fast_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        return await self.protective_stops.place_middle_bucket_fast_protective_stop_with_retries(
            side, contracts, stop_price, retry_count, retry_interval_seconds)

    async def _cancel_unverified_near_tp_algo(self, algo_id: str, *, phase: str) -> None:
        return await self.protective_stops._cancel_unverified_near_tp_algo(algo_id, phase=phase)

    async def verify_near_tp_protective_stop(self, algo_id: str, side: PositionSide, contracts: Decimal,
                                             stop_price: float) -> bool:
        return await self.protective_stops.verify_near_tp_protective_stop(algo_id, side, contracts, stop_price)

    def _near_tp_protective_stop_matches(self, item: dict[str, Any], algo_id: str, side: PositionSide,
                                         contracts: Decimal, stop_price: float) -> bool:
        return self.protective_stops._near_tp_protective_stop_matches(item, algo_id, side, contracts, stop_price)

    # ------------------------------------------------------------------
    # market exit
    # ------------------------------------------------------------------

    async def market_exit_remaining_position_with_retries(
        self,
        side: PositionSide,
        retry_count: int,
        *,
        context: str = "generic",
        retry_interval_seconds: float | None = None,
    ) -> tuple[bool, str]:
        return await self.market_exit.market_exit_remaining_position_with_retries(
            side, retry_count, context=context, retry_interval_seconds=retry_interval_seconds,
        )

    async def _cleanup_after_market_exit(self) -> None:
        return await self.market_exit._cleanup_after_market_exit()

    # Backward-compat alias
    async def _cleanup_after_near_tp_market_exit(self) -> None:
        return await self.market_exit._cleanup_after_near_tp_market_exit()

    # ------------------------------------------------------------------
    # TP price helpers
    # ------------------------------------------------------------------

    def _tp_price_summary(self, specs: list[tuple[str, Decimal, float]]) -> str:
        return self.core_tp._tp_price_summary(specs)

    # ------------------------------------------------------------------
    # cancel reduce-only orders
    # ------------------------------------------------------------------

    async def cancel_existing_reduce_only_orders(self, *, phase: str = "normal_cancel") -> bool:
        t = self.trader
        orders = await t.fetch_pending_orders()
        protected_order_ids = set(getattr(t, "_protected_reduce_only_order_ids", set()) or set())
        managed_order_ids = set(getattr(t, "_managed_reduce_only_order_ids", set()) or set())
        allow_unmanaged = bool(getattr(t, "_allow_cancel_unmanaged_reduce_only", True))
        orders_to_cancel: list[str] = []
        unknown_action: str | None = None

        for item in orders:
            if item.get("instId") != t.symbol:
                continue
            if str(item.get("reduceOnly", "")).lower() != "true":
                continue
            ord_id = item.get("ordId")
            if not ord_id:
                unknown_action = unknown_action or "skip_risky_cancel_missing_order_id"
                continue
            ord_id = str(ord_id)
            if ord_id in protected_order_ids:
                continue
            if managed_order_ids and ord_id not in managed_order_ids:
                unknown_action = unknown_action or "skip_risky_cancel_unmanaged_order"
                continue
            if not managed_order_ids and not allow_unmanaged:
                unknown_action = unknown_action or "skip_risky_cancel_no_managed_identity"
                continue
            orders_to_cancel.append(ord_id)

        if unknown_action is not None:
            return self._handle_reduce_only_identity_unknown(
                phase=phase,
                orders=orders,
                action_taken=unknown_action,
            )

        for item in orders:
            if item.get("instId") != t.symbol:
                continue
            if str(item.get("reduceOnly", "")).lower() != "true":
                continue
            ord_id = item.get("ordId")
            if ord_id and str(ord_id) in protected_order_ids:
                logger.info("Protected reduce-only order skipped | ordId=%s", ord_id)
        for ord_id in orders_to_cancel:
            try:
                semantic_executor = get_broker_semantic_executor(t)
                if semantic_executor is not None:
                    await semantic_executor.execute(
                        BrokerSemanticRequest(
                            exchange=ExchangeName.OKX,
                            symbol=t.symbol,
                            action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
                            role=BrokerSemanticOrderRole.CORE_TP,
                            order_id=ord_id,
                        )
                    )
                else:
                    await t.request("POST", "/api/v5/trade/cancel-order", order_specs.build_cancel_order_body(
                        inst_id=t.symbol,
                        order_id=ord_id,
                    ))
                logger.info("Canceled existing reduce-only order | ordId=%s", ord_id)
            except Exception:
                logger.exception("Failed to cancel existing reduce-only order | ordId=%s", ord_id)
        return True

    def _handle_reduce_only_identity_unknown(
        self,
        *,
        phase: str,
        orders: list[dict[str, Any]],
        action_taken: str,
    ) -> bool:
        t = self.trader
        if phase in {"startup_reconcile", "update_tp", "market_exit_runner"}:
            known_order_ids = [
                str(item.get("ordId"))
                for item in orders
                if item.get("instId") == t.symbol and item.get("ordId")
            ]
            unknown_summary = [
                {
                    "instId": item.get("instId"),
                    "side": item.get("side"),
                    "ordId": item.get("ordId"),
                    "px": item.get("px"),
                    "sz": item.get("sz"),
                }
                for item in orders
                if item.get("instId") == t.symbol and str(item.get("reduceOnly", "")).lower() == "true"
            ]
            logger.warning(
                "REDUCE_ONLY_ORDER_IDENTITY_UNKNOWN_DEGRADED | symbol=%s phase=%s candidate_count=%s known_order_ids=%s unknown_order_summary=%s action_taken=%s no_halt=true",
                t.symbol,
                phase,
                len(unknown_summary),
                known_order_ids,
                unknown_summary,
                action_taken,
            )
            callback = getattr(t, "_on_reduce_only_identity_unknown_degraded", None)
            if callable(callback):
                try:
                    maybe_awaitable = callback(
                        phase=phase,
                        candidate_count=len(unknown_summary),
                        known_order_ids=tuple(known_order_ids),
                        unknown_order_summary=tuple(unknown_summary),
                        action_taken=action_taken,
                    )
                    if hasattr(maybe_awaitable, "send"):
                        logger.debug(
                            "REDUCE_ONLY_ORDER_IDENTITY_UNKNOWN_DEGRADED | async callback scheduled by caller"
                        )
                except Exception:
                    logger.exception(
                        "REDUCE_ONLY_ORDER_IDENTITY_UNKNOWN_DEGRADED_CALLBACK_FAILED | symbol=%s phase=%s",
                        t.symbol,
                        phase,
                    )
            return False
        raise RuntimeError("reduce_only_order_identity_unknown")

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
        return await self.sidecar.place_sidecar_fixed_take_profit(
            side=side,
            contracts=contracts,
            tp_price=tp_price,
            client_order_id=client_order_id,
        )

    async def cancel_sidecar_take_profit(self, order_id: str | None) -> bool:
        return await self.sidecar.cancel_sidecar_take_profit(order_id)

    async def fetch_sidecar_order_status(self, order_id: str) -> dict[str, Any]:
        return await self.sidecar.fetch_sidecar_order_status(order_id)

    # ------------------------------------------------------------------
    # cancel protective stops
    # ------------------------------------------------------------------

    async def cancel_near_tp_protective_stop(self, order_id: str | None) -> bool:
        t = self.trader
        if not order_id:
            return True
        try:
            semantic_executor = get_broker_semantic_executor(t)
            if semantic_executor is not None:
                await semantic_executor.execute(
                    BrokerSemanticRequest(
                        exchange=ExchangeName.OKX,
                        symbol=t.symbol,
                        action=BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
                        role=BrokerSemanticOrderRole.PROTECTIVE_SL,
                        order_id=order_id,
                    )
                )
            else:
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

    async def cancel_middle_bucket_fast_protective_stop(self, order_id: str | None) -> bool:
        t = self.trader
        if not order_id:
            return True
        ok = await self.trader.cancel_near_tp_protective_stop(order_id)
        if ok and getattr(t, "middle_bucket_fast_sl_order_id", None) == order_id:
            t.middle_bucket_fast_sl_order_id = None
        if ok:
            logger.warning("MIDDLE_BUCKET_FAST_SL_CANCELLED | algoId=%s", order_id)
        return ok
