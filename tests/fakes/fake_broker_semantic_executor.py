"""Reusable ``FakeBrokerSemanticExecutor`` for bridge tests.

This fake records all requests and can be configured with queued results or
errors, making it suitable for both positive-path and error-path testing
without accessing a real exchange.
"""

from __future__ import annotations

from typing import Any

from src.exchanges.models import BrokerOrderStatus, ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderQuery,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
    BrokerSemanticResult,
)


class FakeBrokerSemanticExecutor:
    """Configurable fake that records ``execute()`` calls.

    Usage::

        fake = FakeBrokerSemanticExecutor()
        fake.queue_result(order_id="fake-1", ok=True)
        result = await fake.execute(request)
        assert fake.requests[0].action == BrokerSemanticAction.PLACE_REDUCE_ONLY_TP
    """

    def __init__(
        self,
        *,
        exchange: ExchangeName = ExchangeName.OKX,
    ) -> None:
        self._exchange = exchange
        self.requests: list[BrokerSemanticRequest] = []
        self._results: list[BrokerSemanticResult | Exception] = []
        self.cancel_semantic_order_calls: list[tuple[BrokerSemanticRequest, str]] = []
        self.cancel_semantic_orders_by_role_calls: list[BrokerSemanticOrderQuery] = []
        self.fetch_semantic_orders_calls: list[BrokerSemanticOrderQuery] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def exchange(self) -> ExchangeName:
        return self._exchange

    def queue_result(
        self,
        *,
        order_id: str | None = None,
        ok: bool = True,
        message: str = "",
        action: BrokerSemanticAction | None = None,
        role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.UNKNOWN,
        symbol: str = "ETH-USDT-SWAP",
        status: BrokerOrderStatus | None = None,
    ) -> None:
        """Push a successful result onto the queue."""
        self._results.append(
            BrokerSemanticResult(
                exchange=self._exchange,
                symbol=symbol,
                action=action or BrokerSemanticAction.UNKNOWN,
                role=role,
                ok=ok,
                message=message,
                order_id=order_id,
                status=status,
            )
        )

    def queue_error(self, exc: Exception) -> None:
        """Push an exception onto the queue. It will be raised during ``execute()``."""
        self._results.append(exc)

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    async def execute(self, request: BrokerSemanticRequest) -> BrokerSemanticResult:
        self.requests.append(request)

        if not self._results:
            # Default: return a success result with the request's order_id
            return BrokerSemanticResult(
                exchange=self._exchange,
                symbol=request.symbol,
                action=request.action,
                role=request.role,
                ok=True,
                order_id=request.order_id or "default-order-id",
            )

        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def execute_semantic_order(
        self, request: BrokerSemanticRequest
    ) -> BrokerSemanticResult:
        return await self.execute(request)

    async def cancel_semantic_order(
        self, request: BrokerSemanticRequest, order_id: str
    ) -> BrokerSemanticResult:
        self.cancel_semantic_order_calls.append((request, order_id))
        return BrokerSemanticResult(
            exchange=self._exchange,
            symbol=request.symbol,
            action=request.action,
            role=request.role,
            ok=True,
            order_id=order_id,
        )

    async def cancel_semantic_orders_by_role(
        self, query: BrokerSemanticOrderQuery
    ) -> tuple[BrokerSemanticResult, ...]:
        self.cancel_semantic_orders_by_role_calls.append(query)
        return ()

    async def fetch_semantic_orders(
        self, query: BrokerSemanticOrderQuery
    ) -> tuple:
        self.fetch_semantic_orders_calls.append(query)
        return ()

    async def fetch_open_orders(self, *, symbol: str) -> BrokerSemanticResult:
        return BrokerSemanticResult(
            exchange=self._exchange,
            symbol=symbol,
            action=BrokerSemanticAction.FETCH_OPEN_ORDERS,
            ok=True,
        )

    async def fetch_algo_orders(self, *, symbol: str) -> BrokerSemanticResult:
        return BrokerSemanticResult(
            exchange=self._exchange,
            symbol=symbol,
            action=BrokerSemanticAction.FETCH_ALGO_ORDERS,
            ok=True,
        )

    async def close(self) -> None:
        pass
