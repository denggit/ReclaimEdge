from __future__ import annotations

from decimal import Decimal
from typing import Any, TYPE_CHECKING

from src.execution import order_specs
from src.execution.broker_semantic_helpers import (
    broker_position_side,
    close_order_side,
    get_broker_semantic_executor,
    require_semantic_order_id,
    require_semantic_ok,
)
from src.exchanges.models import BrokerOrderSide, BrokerPositionSide, ExchangeName
from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest
from src.position_management.sidecar.model import sanitize_okx_client_order_id
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader
    from src.strategies.boll_cvd_reclaim_strategy import PositionSide

logger = get_logger(__name__)


class SidecarTpManager:
    def __init__(self, trader: Trader) -> None:
        self.trader = trader

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
        semantic_executor = get_broker_semantic_executor(t)
        if semantic_executor is not None:
            result = await semantic_executor.execute(
                BrokerSemanticRequest(
                    exchange=ExchangeName.OKX,
                    symbol=t.symbol,
                    action=BrokerSemanticAction.SIDECAR_TP,
                    role=BrokerSemanticOrderRole.SIDECAR_TP,
                    side=close_order_side(side),
                    position_side=broker_position_side(side),
                    quantity=Decimal(str(contracts)),
                    price=Decimal(str(tp_price)),
                    reduce_only=True,
                    client_order_id=sent_client_order_id or None,
                )
            )
            order_id = require_semantic_order_id(result, action="SIDECAR_TP")
        else:
            body = order_specs.build_reduce_only_tp_order_body(
                inst_id=t.symbol,
                td_mode=t.td_mode,
                side=side,
                contracts_text=t.decimal_to_str(Decimal(str(contracts))),
                price_text=t.price_to_str(float(tp_price)),
                pos_side_mode=t.pos_side_mode,
                client_order_id=sent_client_order_id or None,
            )
            res = await t.request("POST", "/api/v5/trade/order", body)
            order_id = t.extract_order_id(res)
        logger.warning(
            "SIDECAR_TP_PLACED | side=%s contracts=%s tp_price=%s sent_clOrdId=%s ordId=%s",
            side,
            t.decimal_to_str(Decimal(str(contracts))),
            t.price_to_str(float(tp_price)),
            sent_client_order_id or "-",
            order_id,
        )
        return order_id

    async def cancel_sidecar_take_profit(self, order_id: str | None) -> bool:
        t = self.trader
        if not order_id:
            return True
        try:
            semantic_executor = get_broker_semantic_executor(t)
            if semantic_executor is not None:
                result = await semantic_executor.execute(
                    BrokerSemanticRequest(
                        exchange=ExchangeName.OKX,
                        symbol=t.symbol,
                        action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
                        role=BrokerSemanticOrderRole.SIDECAR_TP,
                        order_id=order_id,
                    )
                )
                require_semantic_ok(result, action="SIDECAR_TP_CANCEL")
            else:
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
