from __future__ import annotations

from dataclasses import replace

from src.exchanges.base import BrokerClient
from src.exchanges.errors import ExchangeError, ExchangeErrorDetail, ExchangeErrorKind
from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderResult,
    BrokerOrderSide,
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

    @property
    def exchange(self) -> ExchangeName:
        return ExchangeName.OKX

    async def execute(self, request: BrokerSemanticRequest) -> BrokerSemanticResult:
        validate_semantic_request(request)
        _ensure_executor_exchange(self._broker, request.exchange)

        if request.action in {
            BrokerSemanticAction.OPEN_POSITION,
            BrokerSemanticAction.ADD_POSITION,
            BrokerSemanticAction.SIDECAR_ENTRY,
            BrokerSemanticAction.MARKET_EXIT,
            BrokerSemanticAction.MARKET_EXIT_RUNNER,
            BrokerSemanticAction.CLOSE_POSITION,
            BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            BrokerSemanticAction.SIDECAR_TP,
            BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
        }:
            return await self.execute_semantic_order(request)

        if request.action in {
            BrokerSemanticAction.CANCEL_ORDER,
            BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
            BrokerSemanticAction.SIDECAR_CANCEL,
        }:
            order_id = _request_order_id(request)
            return await self.cancel_semantic_order(request, order_id)

        if request.action == BrokerSemanticAction.CANCEL_PROTECTIVE_STOP:
            order_id = _request_order_id(request)
            cancel_algo_order = getattr(self._broker, "cancel_algo_order", None)
            if callable(cancel_algo_order):
                await cancel_algo_order(request.symbol, order_id)
                return BrokerSemanticResult(
                    exchange=request.exchange,
                    symbol=request.symbol,
                    action=request.action,
                    role=request.role,
                    ok=True,
                    order_id=order_id,
                )
            return await self.cancel_semantic_order(request, order_id)

        if request.action in {
            BrokerSemanticAction.CANCEL_ALL_OPEN_ORDERS,
            BrokerSemanticAction.CANCEL_ALL_ORDINARY_ORDERS,
        }:
            await self._broker.cancel_all_open_orders(request.symbol)
            return BrokerSemanticResult(
                exchange=request.exchange,
                symbol=request.symbol,
                action=request.action,
                role=request.role,
                ok=True,
            )

        if request.action == BrokerSemanticAction.FETCH_OPEN_ORDERS:
            orders = await self.fetch_semantic_orders(
                BrokerSemanticOrderQuery(
                    exchange=request.exchange,
                    symbol=request.symbol,
                    include_ordinary=True,
                    include_algo=False,
                )
            )
            return _semantic_result_from_orders(request, orders)

        if request.action in {
            BrokerSemanticAction.FETCH_ALGO_ORDERS,
            BrokerSemanticAction.FETCH_PROTECTIVE_ORDERS,
        }:
            orders = await self.fetch_semantic_orders(
                BrokerSemanticOrderQuery(
                    exchange=request.exchange,
                    symbol=request.symbol,
                    include_ordinary=False,
                    include_algo=True,
                )
            )
            return _semantic_result_from_orders(request, orders)

        if request.action == BrokerSemanticAction.RECOVER_OPEN_ORDERS:
            orders = await self.fetch_semantic_orders(
                BrokerSemanticOrderQuery(
                    exchange=request.exchange,
                    symbol=request.symbol,
                    include_ordinary=True,
                    include_algo=True,
                )
            )
            return _semantic_result_from_orders(request, orders)

        if request.action in {
            BrokerSemanticAction.FETCH_POSITION,
            BrokerSemanticAction.SYNC_POSITION,
        }:
            position = await self.fetch_semantic_position(
                request.symbol,
                request.position_side,
            )
            return BrokerSemanticResult(
                exchange=request.exchange,
                symbol=request.symbol,
                action=request.action,
                role=request.role,
                ok=True,
                raw=position.raw,
            )

        raise _unsupported(
            request.exchange,
            f"Unsupported semantic action {request.action.value!r}",
        )

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
            BrokerSemanticAction.MARKET_EXIT_RUNNER,
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

        if request.action == BrokerSemanticAction.CANCEL_PROTECTIVE_STOP:
            cancel_algo_order = getattr(self._broker, "cancel_algo_order", None)
            if callable(cancel_algo_order):
                await cancel_algo_order(request.symbol, order_id)
                return BrokerSemanticResult(
                    exchange=request.exchange,
                    symbol=request.symbol,
                    action=request.action,
                    role=request.role,
                    ok=True,
                    order_id=order_id,
                )
            raise _unsupported(
                request.exchange,
                "Broker does not support cancel_algo_order; "
                "cannot cancel protective stop without algo cancel capability",
            )

        await self._broker.cancel_order(request.symbol, order_id)
        return BrokerSemanticResult(
            exchange=request.exchange,
            symbol=request.symbol,
            action=request.action,
            role=request.role,
            ok=True,
            order_id=order_id,
        )

    async def open_position(
        self,
        *,
        symbol: str,
        side: BrokerOrderSide,
        position_side: BrokerPositionSide,
        quantity,
        client_order_id: str | None = None,
        label: str | None = None,
    ) -> BrokerSemanticResult:
        return await self.execute(
            BrokerSemanticRequest(
                exchange=self.exchange,
                symbol=symbol,
                action=BrokerSemanticAction.OPEN_POSITION,
                role=BrokerSemanticOrderRole.ENTRY,
                side=side,
                position_side=position_side,
                quantity=quantity,
                client_order_id=client_order_id,
                label=label,
            )
        )

    async def add_position(
        self,
        *,
        symbol: str,
        side: BrokerOrderSide,
        position_side: BrokerPositionSide,
        quantity,
        client_order_id: str | None = None,
        label: str | None = None,
    ) -> BrokerSemanticResult:
        return await self.execute(
            BrokerSemanticRequest(
                exchange=self.exchange,
                symbol=symbol,
                action=BrokerSemanticAction.ADD_POSITION,
                role=BrokerSemanticOrderRole.ADD,
                side=side,
                position_side=position_side,
                quantity=quantity,
                client_order_id=client_order_id,
                label=label,
            )
        )

    async def place_reduce_only_tp(
        self,
        *,
        symbol: str,
        side: BrokerOrderSide,
        position_side: BrokerPositionSide,
        quantity,
        price,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.CORE_TP,
        client_order_id: str | None = None,
        label: str | None = None,
    ) -> BrokerSemanticResult:
        return await self.execute(
            BrokerSemanticRequest(
                exchange=self.exchange,
                symbol=symbol,
                action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
                role=role,
                side=side,
                position_side=position_side,
                quantity=quantity,
                price=price,
                client_order_id=client_order_id,
                label=label,
            )
        )

    async def place_protective_stop(
        self,
        *,
        symbol: str,
        side: BrokerOrderSide,
        position_side: BrokerPositionSide,
        quantity,
        trigger_price,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.PROTECTIVE_SL,
        client_order_id: str | None = None,
        label: str | None = None,
    ) -> BrokerSemanticResult:
        return await self.execute(
            BrokerSemanticRequest(
                exchange=self.exchange,
                symbol=symbol,
                action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
                role=role,
                side=side,
                position_side=position_side,
                quantity=quantity,
                trigger_price=trigger_price,
                reduce_only=True,
                client_order_id=client_order_id,
                label=label,
            )
        )

    async def cancel_order(
        self,
        *,
        symbol: str,
        order_id: str,
    ) -> BrokerSemanticResult:
        return await self.execute(
            BrokerSemanticRequest(
                exchange=self.exchange,
                symbol=symbol,
                action=BrokerSemanticAction.CANCEL_ORDER,
                order_id=order_id,
            )
        )

    async def cancel_protective_stop(
        self,
        *,
        symbol: str,
        order_id: str,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.PROTECTIVE_SL,
    ) -> BrokerSemanticResult:
        return await self.execute(
            BrokerSemanticRequest(
                exchange=self.exchange,
                symbol=symbol,
                action=BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
                role=role,
                order_id=order_id,
            )
        )

    async def cancel_reduce_only_tp(
        self,
        *,
        symbol: str,
        order_id: str,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.CORE_TP,
    ) -> BrokerSemanticResult:
        return await self.execute(
            BrokerSemanticRequest(
                exchange=self.exchange,
                symbol=symbol,
                action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
                role=role,
                order_id=order_id,
            )
        )

    async def market_exit(
        self,
        *,
        symbol: str,
        side: BrokerOrderSide,
        position_side: BrokerPositionSide,
        quantity,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.MARKET_EXIT,
    ) -> BrokerSemanticResult:
        action = (
            BrokerSemanticAction.MARKET_EXIT_RUNNER
            if role == BrokerSemanticOrderRole.RUNNER_TP
            else BrokerSemanticAction.MARKET_EXIT
        )
        return await self.execute(
            BrokerSemanticRequest(
                exchange=self.exchange,
                symbol=symbol,
                action=action,
                role=role,
                side=side,
                position_side=position_side,
                quantity=quantity,
                reduce_only=True,
                close_position=True,
            )
        )

    async def fetch_open_orders(self, *, symbol: str) -> BrokerSemanticResult:
        return await self.execute(
            BrokerSemanticRequest(
                exchange=self.exchange,
                symbol=symbol,
                action=BrokerSemanticAction.FETCH_OPEN_ORDERS,
            )
        )

    async def fetch_algo_orders(self, *, symbol: str) -> BrokerSemanticResult:
        return await self.execute(
            BrokerSemanticRequest(
                exchange=self.exchange,
                symbol=symbol,
                action=BrokerSemanticAction.FETCH_ALGO_ORDERS,
            )
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
            if _is_algo_cancel_order(order, role):
                cancel_algo_order = getattr(self._broker, "cancel_algo_order", None)
                if not callable(cancel_algo_order):
                    raise _unsupported(
                        query.exchange,
                        "Broker does not support cancel_algo_order; "
                        "cannot cancel algo order by role",
                    )
                await cancel_algo_order(query.symbol, order.order_id)
            else:
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
            orders.extend(
                _with_order_source(order, "ordinary")
                for order in await self._broker.fetch_open_orders(query.symbol)
            )

        fetch_algo_orders = getattr(self._broker, "fetch_algo_orders", None)
        if query.include_algo and fetch_algo_orders is not None:
            orders.extend(
                _with_order_source(order, "algo")
                for order in await fetch_algo_orders(query.symbol)
            )

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
        client_order_id=result.client_order_id,
        status=result.status,
        filled_quantity=result.filled_quantity,
        avg_price=result.avg_fill_price,
        raw=result.raw,
    )


def _semantic_result_from_orders(
    request: BrokerSemanticRequest,
    orders: tuple[BrokerOrder, ...],
) -> BrokerSemanticResult:
    return BrokerSemanticResult(
        exchange=request.exchange,
        symbol=request.symbol,
        action=request.action,
        role=request.role,
        ok=True,
        orders=orders,
        related_order_ids=tuple(order.order_id for order in orders if order.order_id),
        raw={
            "orders": [dict(order.raw) for order in orders],
        },
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


def _with_order_source(order: BrokerOrder, source: str) -> BrokerOrder:
    raw = dict(order.raw)
    raw.setdefault("source", source)
    return replace(order, raw=raw)


def _ensure_cancel_action(request: BrokerSemanticRequest) -> None:
    if request.action not in {
        BrokerSemanticAction.CANCEL_ORDER,
        BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
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


def _request_order_id(request: BrokerSemanticRequest) -> str:
    order_id = request.order_id or request.metadata.get("order_id")
    if order_id:
        return str(order_id)
    raise ExchangeError(
        ExchangeErrorDetail(
            exchange=request.exchange,
            kind=ExchangeErrorKind.BAD_REQUEST,
            message=f"Semantic action {request.action.value!r} requires order_id",
            raw={"action": request.action.value},
        )
    )


_PROTECTIVE_SL_ROLES: frozenset[BrokerSemanticOrderRole] = frozenset(
    {
        BrokerSemanticOrderRole.PROTECTIVE_SL,
        BrokerSemanticOrderRole.MIDDLE_RUNNER_SL,
        BrokerSemanticOrderRole.THREE_STAGE_SL,
        BrokerSemanticOrderRole.TREND_RUNNER_SL,
    }
)


def _is_algo_cancel_order(order: BrokerOrder, role: BrokerSemanticOrderRole) -> bool:
    """Determine whether *order* should be cancelled via the algo-cancel endpoint.

    Returns ``True`` when any of these conditions hold:

    1. The order type is ``STOP_MARKET`` (protective SL / algo orders).
    2. The semantic *role* is a known protective SL role.
    3. The raw order source (set by ``fetch_semantic_orders``) is ``"algo"``.
    """
    if order.order_type == BrokerOrderType.STOP_MARKET:
        return True
    if role in _PROTECTIVE_SL_ROLES:
        return True
    if order.raw.get("source") == "algo":
        return True
    return False


def _unsupported(exchange: ExchangeName, message: str) -> ExchangeError:
    return ExchangeError(
        ExchangeErrorDetail(
            exchange=exchange,
            kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
            message=message,
        )
    )
