from __future__ import annotations

from src.exchanges.base import BrokerClient
from src.exchanges.errors import ExchangeError, ExchangeErrorDetail, ExchangeErrorKind
from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderResult,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    ExchangeName,
)
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderQuery,
    BrokerSemanticRequest,
    BrokerSemanticResult,
)
from src.exchanges.semantics import (
    semantic_request_to_broker_order_request,
    validate_semantic_request,
)


class OkxBrokerSemanticExecutor:
    def __init__(self, broker: BrokerClient) -> None:
        self._broker = broker

    async def execute_semantic_order(
        self, request: BrokerSemanticRequest
    ) -> BrokerSemanticResult:
        validate_semantic_request(request)

        if request.action in {
            BrokerSemanticAction.OPEN_POSITION,
            BrokerSemanticAction.ADD_POSITION,
        }:
            broker_request = semantic_request_to_broker_order_request(
                request,
                BrokerOrderType.MARKET,
            )
            result = await self._broker.place_order(broker_request)
            return _semantic_result_from_order_result(request, result)

        if request.action in {
            BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            BrokerSemanticAction.SIDECAR_TP,
        }:
            broker_request = semantic_request_to_broker_order_request(
                request,
                BrokerOrderType.LIMIT,
            )
            result = await self._broker.place_order(broker_request)
            return _semantic_result_from_order_result(request, result)

        raise _unsupported(
            request.exchange,
            f"Unsupported semantic action {request.action.value!r}",
        )

    async def cancel_semantic_order(
        self, request: BrokerSemanticRequest, order_id: str
    ) -> BrokerSemanticResult:
        validate_semantic_request(request)
        await self._broker.cancel_order(request.symbol, order_id)
        return BrokerSemanticResult(
            exchange=request.exchange,
            symbol=request.symbol,
            action=request.action,
            role=request.role,
            ok=True,
            order_id=order_id,
        )

    async def cancel_semantic_orders_by_role(
        self, query: BrokerSemanticOrderQuery
    ) -> tuple[BrokerSemanticResult, ...]:
        raise _unsupported(
            query.exchange,
            "Cancelling semantic orders by role is not implemented",
        )

    async def fetch_semantic_orders(
        self, query: BrokerSemanticOrderQuery
    ) -> tuple[BrokerOrder, ...]:
        orders: list[BrokerOrder] = []
        if query.include_ordinary:
            orders.extend(await self._broker.fetch_open_orders(query.symbol))

        fetch_algo_orders = getattr(self._broker, "fetch_algo_orders", None)
        if query.include_algo and fetch_algo_orders is not None:
            orders.extend(await fetch_algo_orders(query.symbol))

        return tuple(orders)

    async def fetch_semantic_position(
        self, symbol: str, side: BrokerPositionSide | None = None
    ) -> BrokerPosition:
        return await self._broker.fetch_position(symbol, side)

    async def close(self) -> None:
        await self._broker.close()


def _semantic_result_from_order_result(
    request: BrokerSemanticRequest,
    result: BrokerOrderResult,
) -> BrokerSemanticResult:
    return BrokerSemanticResult(
        exchange=request.exchange,
        symbol=request.symbol,
        action=request.action,
        role=request.role,
        ok=True,
        order_id=result.order_id,
        status=result.status,
        filled_quantity=result.filled_quantity,
        avg_price=result.avg_fill_price,
        raw=result.raw,
    )


def _unsupported(exchange: ExchangeName, message: str) -> ExchangeError:
    return ExchangeError(
        ExchangeErrorDetail(
            exchange=exchange,
            kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
            message=message,
        )
    )
