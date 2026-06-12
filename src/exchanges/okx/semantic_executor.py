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
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
    BrokerSemanticResult,
)
from src.exchanges.semantics import (
    semantic_request_to_broker_order_request,
    validate_semantic_request,
)


class OkxBrokerSemanticExecutor:
    def __init__(self, broker: BrokerClient) -> None:
        if broker.exchange != ExchangeName.OKX:
            raise ExchangeError(
                ExchangeErrorDetail(
                    exchange=broker.exchange,
                    kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
                    message=(
                        "OKX semantic executor requires an OKX broker, "
                        f"got {broker.exchange.value!r}"
                    ),
                    raw={
                        "broker_exchange": broker.exchange.value,
                        "expected_exchange": ExchangeName.OKX.value,
                    },
                )
            )
        self._broker = broker

    async def execute_semantic_order(
        self, request: BrokerSemanticRequest
    ) -> BrokerSemanticResult:
        validate_semantic_request(request)
        _ensure_executor_exchange(self._broker, request.exchange)

        if request.action in {
            BrokerSemanticAction.OPEN_POSITION,
            BrokerSemanticAction.ADD_POSITION,
            BrokerSemanticAction.SIDECAR_ENTRY,
            BrokerSemanticAction.MARKET_EXIT,
            BrokerSemanticAction.CLOSE_POSITION,
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

        if request.action == BrokerSemanticAction.PLACE_PROTECTIVE_STOP:
            broker_request = semantic_request_to_broker_order_request(
                request,
                BrokerOrderType.STOP_MARKET,
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
        _ensure_executor_exchange(self._broker, request.exchange)
        _ensure_cancel_action(request)
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
        _ensure_executor_exchange(self._broker, query.exchange)
        _ensure_query_symbol(self._broker, query)
        if not query.roles:
            raise _unsupported(
                self._broker.exchange,
                "Cancelling semantic orders by role requires at least one role",
            )

        orders = await self.fetch_semantic_orders(query)
        wanted_roles = set(query.roles)
        results: list[BrokerSemanticResult] = []
        for order in orders:
            role = _semantic_role_from_order(order)
            if role not in wanted_roles:
                continue
            await self._broker.cancel_order(query.symbol, order.order_id)
            results.append(
                BrokerSemanticResult(
                    exchange=query.exchange,
                    symbol=query.symbol,
                    action=BrokerSemanticAction.CANCEL_ORDER,
                    role=role,
                    ok=True,
                    order_id=order.order_id,
                )
            )
        return tuple(results)

    async def fetch_semantic_orders(
        self, query: BrokerSemanticOrderQuery
    ) -> tuple[BrokerOrder, ...]:
        _ensure_executor_exchange(self._broker, query.exchange)
        _ensure_query_symbol(self._broker, query)

        orders: list[BrokerOrder] = []
        if query.include_ordinary:
            orders.extend(await self._broker.fetch_open_orders(query.symbol))

        fetch_algo_orders = getattr(self._broker, "fetch_algo_orders", None)
        if query.include_algo and fetch_algo_orders is not None:
            orders.extend(await fetch_algo_orders(query.symbol))

        for order in orders:
            _ensure_broker_order_matches(query, order)

        return tuple(orders)

    async def fetch_semantic_position(
        self, symbol: str, side: BrokerPositionSide | None = None
    ) -> BrokerPosition:
        if not symbol:
            raise ExchangeError(
                ExchangeErrorDetail(
                    exchange=self._broker.exchange,
                    kind=ExchangeErrorKind.INVALID_SYMBOL,
                    message="Semantic position symbol must not be empty",
                )
            )
        position = await self._broker.fetch_position(symbol, side)
        _ensure_broker_position_matches(self._broker.exchange, symbol, position)
        return position

    async def close(self) -> None:
        await self._broker.close()


def _semantic_result_from_order_result(
    request: BrokerSemanticRequest,
    result: BrokerOrderResult,
) -> BrokerSemanticResult:
    if result.exchange != request.exchange:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=request.exchange,
                kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
                message=(
                    "Broker order result exchange mismatch: "
                    f"result={result.exchange.value!r} request={request.exchange.value!r}"
                ),
                raw={
                    "result_exchange": result.exchange.value,
                    "request_exchange": request.exchange.value,
                    "order_id": result.order_id,
                },
            )
        )
    if result.symbol != request.symbol:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=request.exchange,
                kind=ExchangeErrorKind.INVALID_SYMBOL,
                message=(
                    "Broker order result symbol mismatch: "
                    f"result={result.symbol!r} request={request.symbol!r}"
                ),
                raw={
                    "result_symbol": result.symbol,
                    "request_symbol": request.symbol,
                    "order_id": result.order_id,
                },
            )
        )

    return BrokerSemanticResult(
        exchange=result.exchange,
        symbol=result.symbol,
        action=request.action,
        role=request.role,
        ok=True,
        order_id=result.order_id,
        status=result.status,
        filled_quantity=result.filled_quantity,
        avg_price=result.avg_fill_price,
        raw=result.raw,
    )


def _ensure_executor_exchange(broker: BrokerClient, exchange: ExchangeName) -> None:
    if exchange != broker.exchange:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=broker.exchange,
                kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
                message=(
                    "Semantic exchange does not match configured broker exchange: "
                    f"requested={exchange.value!r} broker={broker.exchange.value!r}"
                ),
                raw={
                    "requested_exchange": exchange.value,
                    "broker_exchange": broker.exchange.value,
                },
            )
        )


def _ensure_query_symbol(
    broker: BrokerClient,
    query: BrokerSemanticOrderQuery,
) -> None:
    if not query.symbol:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=broker.exchange,
                kind=ExchangeErrorKind.INVALID_SYMBOL,
                message="Semantic order query symbol must not be empty",
            )
        )


def _ensure_broker_order_matches(
    query: BrokerSemanticOrderQuery,
    order: BrokerOrder,
) -> None:
    if order.exchange != query.exchange:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=query.exchange,
                kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
                message=(
                    "Broker order exchange mismatch: "
                    f"order={order.exchange.value!r} query={query.exchange.value!r}"
                ),
                raw={
                    "order_exchange": order.exchange.value,
                    "query_exchange": query.exchange.value,
                    "order_id": order.order_id,
                },
            )
        )
    if order.symbol != query.symbol:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=query.exchange,
                kind=ExchangeErrorKind.INVALID_SYMBOL,
                message=(
                    "Broker order symbol mismatch: "
                    f"order={order.symbol!r} query={query.symbol!r}"
                ),
                raw={
                    "order_symbol": order.symbol,
                    "query_symbol": query.symbol,
                    "order_id": order.order_id,
                },
            )
        )


def _ensure_broker_position_matches(
    exchange: ExchangeName,
    symbol: str,
    position: BrokerPosition,
) -> None:
    if position.exchange != exchange:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=exchange,
                kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
                message=(
                    "Broker position exchange mismatch: "
                    f"position={position.exchange.value!r} broker={exchange.value!r}"
                ),
                raw={
                    "position_exchange": position.exchange.value,
                    "broker_exchange": exchange.value,
                },
            )
        )
    if position.symbol != symbol:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=exchange,
                kind=ExchangeErrorKind.INVALID_SYMBOL,
                message=(
                    "Broker position symbol mismatch: "
                    f"position={position.symbol!r} request={symbol!r}"
                ),
                raw={
                    "position_symbol": position.symbol,
                    "request_symbol": symbol,
                },
            )
        )


def _semantic_role_from_order(order: BrokerOrder) -> BrokerSemanticOrderRole:
    try:
        return BrokerSemanticOrderRole(order.label or "")
    except ValueError:
        return BrokerSemanticOrderRole.UNKNOWN


def _ensure_cancel_action(request: BrokerSemanticRequest) -> None:
    if request.action not in {
        BrokerSemanticAction.CANCEL_ORDER,
        BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
        BrokerSemanticAction.SIDECAR_CANCEL,
    }:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=request.exchange,
                kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
                message=(
                    f"Semantic action {request.action.value!r} cannot cancel an order"
                ),
                raw={"action": request.action.value},
            )
        )


def _unsupported(exchange: ExchangeName, message: str) -> ExchangeError:
    return ExchangeError(
        ExchangeErrorDetail(
            exchange=exchange,
            kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
            message=message,
        )
    )
