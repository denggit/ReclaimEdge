from __future__ import annotations

from decimal import Decimal
from typing import Any, TYPE_CHECKING

from src.execution import order_specs
from src.execution.okx_trading_client import OkxTradingClient
from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager
from src.execution.tp_sl_market_exit_manager import MarketExitManager
from src.execution.tp_sl_near_tp_manager import NearTpExecutionManager
from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager
from src.execution.tp_sl_sidecar_manager import SidecarTpManager
from src.execution.trading_client_port import TradingClientPort
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader, LiveTradeResult
    from src.strategies.boll_cvd_reclaim_strategy import PositionSide, TradeIntent

logger = get_logger(__name__)


class TpSlExecutionManager:
    def __init__(self, trader: Trader) -> None:
        self.trader = trader
        self.trading_client: TradingClientPort = OkxTradingClient(trader)
        self.protective_stops = ProtectiveStopManager(trader, self.trading_client)
        self.market_exit = MarketExitManager(trader, self.trading_client)
        self.core_tp = CoreTakeProfitManager(trader, self.protective_stops, self.trading_client)
        self.near_tp = NearTpExecutionManager(
            trader=trader,
            core_tp=self.core_tp,
            protective_stops=self.protective_stops,
            market_exit=self.market_exit,
            trading_client=self.trading_client,
        )
        self.sidecar = SidecarTpManager(trader, self.trading_client)

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

    async def _cancel_existing_take_profit_orders_for_intent(self, intent: TradeIntent) -> None:
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

    def _broker_semantic_reduce_only_cancel_enabled(self) -> bool:
        import os

        value = os.getenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "false").strip().lower()
        return value in {"1", "true", "yes", "y", "on"}

    async def _cancel_reduce_only_order_semantic(self, order_id: str) -> None:
        t = self.trader
        result = await t.broker_semantic_executor.cancel_reduce_only_tp(
            symbol=t.symbol,
            order_id=order_id,
        )
        if not result.ok:
            raise RuntimeError(
                f"semantic_reduce_only_cancel_failed ordId={order_id} message={result.message}"
            )

    async def cancel_existing_reduce_only_orders(self) -> None:
        t = self.trader
        orders = await self.trading_client.fetch_open_orders()
        protected_order_ids = set(getattr(t, "_protected_reduce_only_order_ids", set()) or set())
        managed_order_ids = set(getattr(t, "_managed_reduce_only_order_ids", set()) or set())
        allow_unmanaged = bool(getattr(t, "_allow_cancel_unmanaged_reduce_only", True))
        for order in orders:
            if not order.reduce_only:
                continue
            ord_id = order.order_id
            if not ord_id:
                raise RuntimeError("reduce_only_order_identity_unknown")
            if ord_id in protected_order_ids:
                logger.info("Protected reduce-only order skipped | ordId=%s", ord_id)
                continue
            if managed_order_ids and ord_id not in managed_order_ids:
                raise RuntimeError("reduce_only_order_identity_unknown")
            if not managed_order_ids and not allow_unmanaged:
                raise RuntimeError("reduce_only_order_identity_unknown")
            if self._broker_semantic_reduce_only_cancel_enabled():
                await self._cancel_reduce_only_order_semantic(ord_id)
                logger.info("Canceled existing reduce-only order | ordId=%s (semantic)", ord_id)
            else:
                try:
                    await self.trading_client.cancel_order(order_id=ord_id)
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

    def _broker_semantic_protective_sl_cancel_enabled(self) -> bool:
        import os

        value = os.getenv(
            "BROKER_SEMANTIC_PROTECTIVE_SL_CANCEL_ENABLED",
            "false",
        ).strip().lower()
        return value in {"1", "true", "yes", "y", "on"}

    async def _cancel_protective_stop_semantic(self, order_id: str) -> bool:
        t = self.trader
        try:
            result = await t.broker_semantic_executor.cancel_protective_stop(
                symbol=t.symbol,
                order_id=order_id,
            )
            if result.ok:
                return True

            text = str(result.message or "").lower()
            if (
                "not found" in text
                or "not exist" in text
                or "does not exist" in text
                or "already" in text
            ):
                return True

            return False
        except Exception as exc:
            text = str(exc).lower()
            if (
                "not found" in text
                or "not exist" in text
                or "does not exist" in text
                or "already" in text
            ):
                return True
            logger.warning(
                "NEAR_TP_PROTECTIVE_SL_CANCEL_ON_FLAT | algoId=%s failed=%s",
                order_id,
                exc,
            )
            return False

    async def cancel_near_tp_protective_stop(self, order_id: str | None) -> bool:
        t = self.trader
        if not order_id:
            return True

        if self._broker_semantic_protective_sl_cancel_enabled():
            ok = await self._cancel_protective_stop_semantic(order_id)
            if ok:
                if t.near_tp_protective_sl_order_id == order_id:
                    t.near_tp_protective_sl_order_id = None
                logger.warning(
                    "NEAR_TP_PROTECTIVE_SL_CANCEL_ON_FLAT | algoId=%s semantic=true",
                    order_id,
                )
            return ok

        # NOTE: 20C-CLEAN-PORTS-05 — intentionally NOT routed through
        # TradingClientPort.cancel_order() here.  The caller knows this is
        # an algo order id, so the regular-cancel → algo-cancel fallback
        # inside cancel_order() would add unnecessary latency and a
        # spurious API error.  The legacy direct algo cancel is the
        # correct path for this specific call-site.
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
