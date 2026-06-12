from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from src.exchanges.errors import ExchangeError, ExchangeErrorDetail, ExchangeErrorKind
from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerTimeInForce,
    ExchangeName,
)
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderQuery,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
    BrokerSemanticResult,
)


class BrokerSemanticExecutor(Protocol):
    @property
    def exchange(self) -> ExchangeName: ...

    async def execute(self, request: BrokerSemanticRequest) -> BrokerSemanticResult: ...

    async def execute_semantic_order(
        self, request: BrokerSemanticRequest
    ) -> BrokerSemanticResult: ...

    async def cancel_semantic_order(
        self, request: BrokerSemanticRequest, order_id: str
    ) -> BrokerSemanticResult: ...

    async def cancel_semantic_orders_by_role(
        self, query: BrokerSemanticOrderQuery
    ) -> tuple[BrokerSemanticResult, ...]: ...

    async def fetch_semantic_orders(
        self, query: BrokerSemanticOrderQuery
    ) -> tuple[BrokerOrder, ...]: ...

    async def fetch_semantic_position(
        self, symbol: str, side: BrokerPositionSide | None = None
    ) -> BrokerPosition: ...

    async def close(self) -> None: ...

    async def open_position(
        self,
        *,
        symbol: str,
        side,
        position_side: BrokerPositionSide,
        quantity: Decimal,
        client_order_id: str | None = None,
        label: str | None = None,
    ) -> BrokerSemanticResult: ...

    async def add_position(
        self,
        *,
        symbol: str,
        side,
        position_side: BrokerPositionSide,
        quantity: Decimal,
        client_order_id: str | None = None,
        label: str | None = None,
    ) -> BrokerSemanticResult: ...

    async def place_reduce_only_tp(
        self,
        *,
        symbol: str,
        side,
        position_side: BrokerPositionSide,
        quantity: Decimal,
        price: Decimal,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.CORE_TP,
        client_order_id: str | None = None,
        label: str | None = None,
    ) -> BrokerSemanticResult: ...

    async def place_protective_stop(
        self,
        *,
        symbol: str,
        side,
        position_side: BrokerPositionSide,
        quantity: Decimal,
        trigger_price: Decimal,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.PROTECTIVE_SL,
        client_order_id: str | None = None,
        label: str | None = None,
    ) -> BrokerSemanticResult: ...

    async def cancel_order(
        self,
        *,
        symbol: str,
        order_id: str,
    ) -> BrokerSemanticResult: ...

    async def cancel_protective_stop(
        self,
        *,
        symbol: str,
        order_id: str,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.PROTECTIVE_SL,
    ) -> BrokerSemanticResult: ...

    async def cancel_reduce_only_tp(
        self,
        *,
        symbol: str,
        order_id: str,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.CORE_TP,
    ) -> BrokerSemanticResult: ...

    async def market_exit(
        self,
        *,
        symbol: str,
        side,
        position_side: BrokerPositionSide,
        quantity: Decimal,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.MARKET_EXIT,
    ) -> BrokerSemanticResult: ...

    async def fetch_open_orders(self, *, symbol: str) -> BrokerSemanticResult: ...

    async def fetch_algo_orders(self, *, symbol: str) -> BrokerSemanticResult: ...


def validate_semantic_request(request: BrokerSemanticRequest) -> None:
    if not request.symbol:
        raise _semantic_error(
            request.exchange,
            ExchangeErrorKind.INVALID_SYMBOL,
            "Semantic request symbol must not be empty",
        )
    if request.exchange not in {ExchangeName.OKX, ExchangeName.BINANCE}:
        raise _semantic_error(
            request.exchange,
            ExchangeErrorKind.UNSUPPORTED_OPERATION,
            f"Unsupported semantic exchange {request.exchange.value!r}",
        )
    if request.action == BrokerSemanticAction.UNKNOWN:
        raise _semantic_error(
            request.exchange,
            ExchangeErrorKind.UNSUPPORTED_OPERATION,
            "Unknown semantic action is not supported",
        )
    if request.quantity is not None and request.quantity <= Decimal("0"):
        raise _semantic_error(
            request.exchange,
            ExchangeErrorKind.INVALID_QUANTITY,
            "Semantic request quantity must be positive",
        )
    if request.price is not None and request.price <= Decimal("0"):
        raise _semantic_error(
            request.exchange,
            ExchangeErrorKind.INVALID_PRICE,
            "Semantic request price must be positive",
        )
    if request.trigger_price is not None and request.trigger_price <= Decimal("0"):
        raise _semantic_error(
            request.exchange,
            ExchangeErrorKind.INVALID_TRIGGER_PRICE,
            "Semantic request trigger price must be positive",
        )


def semantic_request_to_broker_order_request(
    request: BrokerSemanticRequest,
    order_type: BrokerOrderType,
    time_in_force: BrokerTimeInForce | None = None,
) -> BrokerOrderRequest:
    validate_semantic_request(request)

    if request.action in {
        BrokerSemanticAction.OPEN_POSITION,
        BrokerSemanticAction.ADD_POSITION,
        BrokerSemanticAction.SIDECAR_ENTRY,
    }:
        if order_type != BrokerOrderType.MARKET:
            raise _semantic_error(
                request.exchange,
                ExchangeErrorKind.UNSUPPORTED_OPERATION,
                f"{request.action.value} can only convert to MARKET order requests",
            )
        return _broker_order_request(
            request,
            order_type=BrokerOrderType.MARKET,
            reduce_only=False,
            time_in_force=time_in_force,
        )

    if request.action in {
        BrokerSemanticAction.MARKET_EXIT,
        BrokerSemanticAction.MARKET_EXIT_RUNNER,
        BrokerSemanticAction.CLOSE_POSITION,
    }:
        if order_type != BrokerOrderType.MARKET:
            raise _semantic_error(
                request.exchange,
                ExchangeErrorKind.UNSUPPORTED_OPERATION,
                f"{request.action.value} can only convert to MARKET order requests",
            )
        return _broker_order_request(
            request,
            order_type=BrokerOrderType.MARKET,
            reduce_only=True,
            close_position=True,
            time_in_force=time_in_force,
        )

    if request.action in {
        BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
        BrokerSemanticAction.SIDECAR_TP,
    }:
        if order_type != BrokerOrderType.LIMIT:
            raise _semantic_error(
                request.exchange,
                ExchangeErrorKind.UNSUPPORTED_OPERATION,
                f"{request.action.value} can only convert to LIMIT order requests",
            )
        if request.price is None:
            raise _semantic_error(
                request.exchange,
                ExchangeErrorKind.INVALID_PRICE,
                f"{request.action.value} requires a limit price",
            )
        return _broker_order_request(
            request,
            order_type=BrokerOrderType.LIMIT,
            reduce_only=True,
            time_in_force=time_in_force,
        )

    if request.action == BrokerSemanticAction.PLACE_PROTECTIVE_STOP:
        if order_type != BrokerOrderType.STOP_MARKET:
            raise _semantic_error(
                request.exchange,
                ExchangeErrorKind.UNSUPPORTED_OPERATION,
                f"{request.action.value} can only convert to STOP_MARKET order requests",
            )
        if request.trigger_price is None:
            raise _semantic_error(
                request.exchange,
                ExchangeErrorKind.INVALID_TRIGGER_PRICE,
                f"{request.action.value} requires a trigger price",
            )
        return _broker_order_request(
            request,
            order_type=BrokerOrderType.STOP_MARKET,
            reduce_only=True,
            time_in_force=time_in_force,
        )

    raise _semantic_error(
        request.exchange,
        ExchangeErrorKind.UNSUPPORTED_OPERATION,
        f"Semantic action {request.action.value} cannot convert to a broker order request",
    )


def _broker_order_request(
    request: BrokerSemanticRequest,
    *,
    order_type: BrokerOrderType,
    reduce_only: bool,
    time_in_force: BrokerTimeInForce | None,
    close_position: bool | None = None,
) -> BrokerOrderRequest:
    if request.side is None:
        raise _semantic_error(
            request.exchange,
            ExchangeErrorKind.BAD_REQUEST,
            "Semantic order conversion requires side",
        )
    if request.position_side is None:
        raise _semantic_error(
            request.exchange,
            ExchangeErrorKind.BAD_REQUEST,
            "Semantic order conversion requires position_side",
        )
    if request.quantity is None:
        raise _semantic_error(
            request.exchange,
            ExchangeErrorKind.INVALID_QUANTITY,
            "Semantic order conversion requires quantity",
        )

    return BrokerOrderRequest(
        exchange=request.exchange,
        symbol=request.symbol,
        side=request.side,
        position_side=request.position_side,
        order_type=order_type,
        quantity=request.quantity,
        price=request.price,
        trigger_price=request.trigger_price,
        reduce_only=reduce_only,
        close_position=request.close_position if close_position is None else close_position,
        time_in_force=time_in_force,
        client_order_id=request.client_order_id,
        label=request.label,
        metadata=request.metadata,
    )


def _semantic_error(
    exchange: ExchangeName,
    kind: ExchangeErrorKind,
    message: str,
) -> ExchangeError:
    return ExchangeError(
        ExchangeErrorDetail(
            exchange=exchange,
            kind=kind,
            message=message,
        )
    )
