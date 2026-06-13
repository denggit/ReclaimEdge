from __future__ import annotations

from decimal import Decimal
from typing import Any, TYPE_CHECKING

from src.execution import order_specs
from src.position_management.sidecar.model import sanitize_okx_client_order_id
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader
    from src.strategies.boll_cvd_reclaim_strategy import PositionSide

logger = get_logger(__name__)


class SidecarTpManager:
    def __init__(self, trader: Trader) -> None:
        self.trader = trader

    def _broker_semantic_sidecar_tp_placement_enabled(self) -> bool:
        import os

        value = os.getenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", "false").strip().lower()
        return value in {"1", "true", "yes", "y", "on"}

    def _broker_semantic_sidecar_tp_cancel_enabled(self) -> bool:
        import os

        value = os.getenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", "false").strip().lower()
        return value in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _broker_position_side(side: str):
        from src.exchanges.models import BrokerPositionSide

        if side == "LONG":
            return BrokerPositionSide.LONG
        if side == "SHORT":
            return BrokerPositionSide.SHORT
        raise RuntimeError(f"unsupported_position_side_for_semantic_sidecar_tp: {side}")

    async def _place_sidecar_take_profit_semantic(
        self,
        *,
        side: str,
        contracts: Decimal,
        tp_price: float,
        client_order_id: str | None,
    ) -> str:
        t = self.trader
        from src.exchanges.models import BrokerQuantityUnit

        result = await t.broker_semantic_executor.sidecar_tp(
            symbol=t.symbol,
            side=self._broker_position_side(side),
            quantity=contracts,
            trigger_price=Decimal(str(tp_price)),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            client_order_id=client_order_id,
            label="sidecar_tp",
        )
        if not result.ok or not result.order_id:
            raise RuntimeError(
                f"semantic_sidecar_tp_order_failed side={side} contracts={t.decimal_to_str(contracts)} "
                f"tp_price={t.price_to_str(float(tp_price))} message={result.message}"
            )
        return str(result.order_id)

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
        contracts_decimal = Decimal(str(contracts))
        if self._broker_semantic_sidecar_tp_placement_enabled():
            order_id = await self._place_sidecar_take_profit_semantic(
                side=side,
                contracts=contracts_decimal,
                tp_price=tp_price,
                client_order_id=sent_client_order_id or None,
            )
        else:
            body = order_specs.build_reduce_only_tp_order_body(
                inst_id=t.symbol,
                td_mode=t.td_mode,
                side=side,
                contracts_text=t.decimal_to_str(contracts_decimal),
                price_text=t.price_to_str(float(tp_price)),
                pos_side_mode=t.pos_side_mode,
                client_order_id=sent_client_order_id or None,
            )
            res = await t.request("POST", "/api/v5/trade/order", body)
            order_id = t.extract_order_id(res)
        logger.warning(
            "SIDECAR_TP_PLACED | side=%s contracts=%s tp_price=%s sent_clOrdId=%s ordId=%s",
            side,
            t.decimal_to_str(contracts_decimal),
            t.price_to_str(float(tp_price)),
            sent_client_order_id or "-",
            order_id,
        )
        return order_id

    async def _cancel_sidecar_take_profit_semantic(self, order_id: str) -> bool:
        t = self.trader
        from src.exchanges.semantic_models import BrokerSemanticOrderRole

        try:
            result = await t.broker_semantic_executor.cancel_reduce_only_tp(
                symbol=t.symbol,
                order_id=order_id,
                role=BrokerSemanticOrderRole.SIDECAR_TP,
                label="sidecar_tp",
            )
            if result.ok:
                return True

            text = str(result.message or "").lower()
            if "not found" in text or "not exist" in text or "does not exist" in text or "already" in text:
                return True

            return False
        except Exception as exc:
            text = str(exc).lower()
            if "not found" in text or "not exist" in text or "does not exist" in text or "already" in text:
                return True
            logger.error("SIDECAR_TP_CANCEL_FAILED | ordId=%s semantic=true error=%s", order_id, exc)
            return False

    async def cancel_sidecar_take_profit(self, order_id: str | None) -> bool:
        t = self.trader
        if not order_id:
            return True

        if self._broker_semantic_sidecar_tp_cancel_enabled():
            ok = await self._cancel_sidecar_take_profit_semantic(order_id)
            if ok:
                logger.warning("SIDECAR_TP_CANCELLED | ordId=%s semantic=true", order_id)
            return ok

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

    async def fetch_sidecar_order_status(self, order_id: str) -> dict[str, Any]:
        t = self.trader
        try:
            res = await t.request("GET", f"/api/v5/trade/order?instId={t.symbol}&ordId={order_id}")
        except Exception:
            return {"order_id": order_id, "status": "UNKNOWN", "filled_qty": None, "avg_fill_price": None}
        data = res.get("data", [])
        if not data:
            return {"order_id": order_id, "status": "NOT_FOUND", "filled_qty": None, "avg_fill_price": None}
        item = data[0]
        state = str(item.get("state") or "").lower()
        if state in {"live", "partially_filled"}:
            status = "OPEN"
        elif state == "filled":
            status = "FILLED"
        elif state in {"canceled", "cancelled"}:
            status = "CANCELED"
        else:
            status = "UNKNOWN"
        return {
            "order_id": order_id,
            "status": status,
            "filled_qty": _optional_float(item.get("accFillSz")),
            "avg_fill_price": _optional_float(item.get("avgPx")),
        }


def _optional_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except Exception:
        return None
