#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_okx_semantic_executor.py
@Description: Tests for the OKX broker semantic executor adapter.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerCancelResult,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.exchanges.okx.semantic_executor import OkxBrokerSemanticExecutor
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
)


SYMBOL = "ETH-USDT-SWAP"


class FakeBrokerClient:
    def __init__(self) -> None:
        self.place_order_requests: list[BrokerOrderRequest] = []
        self.protective_stop_requests: list[BrokerOrderRequest] = []
        self.cancel_order_calls: list[tuple[str, str]] = []
        self.cancel_algo_calls: list[tuple[str, str]] = []
        self.open_orders: list[BrokerOrder] = []
        self.algo_orders: list[BrokerOrder] = []
        self.position: BrokerPosition | None = None

    @property
    def exchange(self) -> ExchangeName:
        return ExchangeName.OKX

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        self.place_order_requests.append(request)
        return BrokerOrderResult(
            exchange=ExchangeName.OKX,
            symbol=request.symbol,
            ok=True,
            order_id="order-1",
            client_order_id=request.client_order_id,
            raw={"fake": "place_order"},
        )

    async def place_protective_stop_order(
        self,
        request: BrokerOrderRequest,
    ) -> BrokerOrderResult:
        self.protective_stop_requests.append(request)
        return BrokerOrderResult(
            exchange=ExchangeName.OKX,
            symbol=request.symbol,
            ok=True,
            order_id="algo-1",
            client_order_id=request.client_order_id,
            raw={"fake": "protective_stop"},
        )

    async def cancel_order(self, symbol: str, order_id: str) -> BrokerCancelResult:
        self.cancel_order_calls.append((symbol, order_id))
        return BrokerCancelResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            ok=True,
            order_id=order_id,
            raw={"fake": "cancel_order"},
        )

    async def cancel_algo_order(
        self,
        symbol: str,
        algo_id: str,
    ) -> BrokerCancelResult:
        self.cancel_algo_calls.append((symbol, algo_id))
        return BrokerCancelResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            ok=True,
            order_id=algo_id,
            raw={"fake": "cancel_algo"},
        )

    async def fetch_open_orders(self, symbol: str) -> tuple[BrokerOrder, ...]:
        return tuple(self.open_orders)

    async def fetch_algo_orders(self, symbol: str) -> tuple[BrokerOrder, ...]:
        return tuple(self.algo_orders)

    async def fetch_position(self, symbol: str) -> BrokerPosition | None:
        return self.position


@pytest.fixture
def fake() -> FakeBrokerClient:
    return FakeBrokerClient()


@pytest.fixture
def executor(fake: FakeBrokerClient) -> OkxBrokerSemanticExecutor:
    return OkxBrokerSemanticExecutor(fake)


def _request(
    action: BrokerSemanticAction,
    role: BrokerSemanticOrderRole,
    *,
    side: BrokerPositionSide | None = None,
    quantity: Decimal | None = None,
    quantity_unit: BrokerQuantityUnit | None = BrokerQuantityUnit.CONTRACTS,
    price: Decimal | None = None,
    trigger_price: Decimal | None = None,
    order_id: str | None = None,
) -> BrokerSemanticRequest:
    return BrokerSemanticRequest(
        exchange=ExchangeName.OKX,
        symbol=SYMBOL,
        action=action,
        role=role,
        side=side,
        quantity=quantity,
        quantity_unit=quantity_unit,
        price=price,
        trigger_price=trigger_price,
        order_id=order_id,
    )


def _order(
    order_id: str,
    *,
    metadata: dict[str, str] | None = None,
    raw: dict[str, str] | None = None,
) -> BrokerOrder:
    return BrokerOrder(
        exchange=ExchangeName.OKX,
        symbol=SYMBOL,
        order_id=order_id,
        client_order_id=None,
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.LIMIT,
        status=BrokerOrderStatus.OPEN,
        price=Decimal("3500"),
        quantity=Decimal("1"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        reduce_only=True,
        raw=raw or {"order_id": order_id},
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_open_position_places_market_non_reduce_only_entry(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    result = await executor.execute(
        _request(
            BrokerSemanticAction.OPEN_POSITION,
            BrokerSemanticOrderRole.ENTRY,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("12"),
        )
    )

    assert result.ok is True
    assert result.order_id == "order-1"
    assert len(fake.place_order_requests) == 1
    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.MARKET
    assert order_request.reduce_only is False
    assert order_request.side == BrokerOrderSide.BUY
    assert order_request.position_side == BrokerPositionSide.LONG


@pytest.mark.asyncio
async def test_add_position_places_market_non_reduce_only_short(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    await executor.execute(
        _request(
            BrokerSemanticAction.ADD_POSITION,
            BrokerSemanticOrderRole.ADD,
            side=BrokerPositionSide.SHORT,
            quantity=Decimal("4"),
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.side == BrokerOrderSide.SELL
    assert order_request.reduce_only is False


@pytest.mark.asyncio
async def test_place_reduce_only_tp_places_limit_close_order(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    await executor.execute(
        _request(
            BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            BrokerSemanticOrderRole.TP1,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("10"),
            price=Decimal("3500"),
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.LIMIT
    assert order_request.reduce_only is True
    assert order_request.side == BrokerOrderSide.SELL
    assert order_request.position_side == BrokerPositionSide.LONG
    assert order_request.price == Decimal("3500")


@pytest.mark.asyncio
async def test_sidecar_tp_places_limit_close_order_for_short(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    await executor.execute(
        _request(
            BrokerSemanticAction.SIDECAR_TP,
            BrokerSemanticOrderRole.SIDECAR_TP,
            side=BrokerPositionSide.SHORT,
            quantity=Decimal("2"),
            price=Decimal("3300"),
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.LIMIT
    assert order_request.reduce_only is True
    assert order_request.side == BrokerOrderSide.BUY
    assert order_request.position_side == BrokerPositionSide.SHORT


@pytest.mark.asyncio
async def test_place_protective_stop_delegates_to_algo_stop_client_method(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    result = await executor.execute(
        _request(
            BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            BrokerSemanticOrderRole.MIDDLE_RUNNER_SL,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("5"),
            trigger_price=Decimal("3400"),
        )
    )

    assert len(fake.protective_stop_requests) == 1
    order_request = fake.protective_stop_requests[0]
    assert order_request.order_type == BrokerOrderType.STOP_MARKET
    assert order_request.reduce_only is True
    assert order_request.side == BrokerOrderSide.SELL
    assert order_request.trigger_price == Decimal("3400")
    assert result.order_id == "algo-1"


@pytest.mark.asyncio
async def test_market_exit_runner_places_reduce_only_market_close(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    await executor.execute(
        _request(
            BrokerSemanticAction.MARKET_EXIT_RUNNER,
            BrokerSemanticOrderRole.MARKET_EXIT,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("3"),
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.MARKET
    assert order_request.reduce_only is True
    assert order_request.side == BrokerOrderSide.SELL


@pytest.mark.asyncio
async def test_sidecar_entry_places_market_non_reduce_only(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    await executor.execute(
        _request(
            BrokerSemanticAction.SIDECAR_ENTRY,
            BrokerSemanticOrderRole.SIDECAR_ENTRY,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.MARKET
    assert order_request.reduce_only is False
    assert order_request.side == BrokerOrderSide.BUY


@pytest.mark.asyncio
async def test_cancel_reduce_only_tp_uses_cancel_order(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    result = await executor.execute(
        _request(
            BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
            BrokerSemanticOrderRole.TP1,
            order_id="tp-1",
        )
    )

    assert fake.cancel_order_calls == [(SYMBOL, "tp-1")]
    assert result.ok is True
    assert result.order_id == "tp-1"


@pytest.mark.asyncio
async def test_cancel_protective_stop_uses_cancel_algo_order(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    result = await executor.execute(
        _request(
            BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
            BrokerSemanticOrderRole.PROTECTIVE_SL,
            order_id="algo-1",
        )
    )

    assert fake.cancel_algo_calls == [(SYMBOL, "algo-1")]
    assert result.ok is True
    assert result.order_id == "algo-1"


@pytest.mark.asyncio
async def test_fetch_open_orders_returns_tuple(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    fake.open_orders.append(_order("ord-1"))

    result = await executor.execute(
        _request(
            BrokerSemanticAction.FETCH_OPEN_ORDERS,
            BrokerSemanticOrderRole.UNKNOWN,
        )
    )

    assert result.ok is True
    assert result.orders == tuple(fake.open_orders)


@pytest.mark.asyncio
async def test_fetch_algo_orders_returns_tuple(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    fake.algo_orders.append(_order("algo-1", metadata={"source": "algo"}))

    result = await executor.execute(
        _request(
            BrokerSemanticAction.FETCH_ALGO_ORDERS,
            BrokerSemanticOrderRole.UNKNOWN,
        )
    )

    assert result.ok is True
    assert result.orders == tuple(fake.algo_orders)


@pytest.mark.asyncio
async def test_recover_open_orders_combines_ordinary_and_algo_sources(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    ordinary = _order(
        "ord-1",
        metadata={"source": "ordinary"},
        raw={"raw": "ordinary"},
    )
    algo = _order("algo-1", metadata={"source": "algo"}, raw={"raw": "algo"})
    fake.open_orders.append(ordinary)
    fake.algo_orders.append(algo)

    result = await executor.execute(
        _request(
            BrokerSemanticAction.RECOVER_OPEN_ORDERS,
            BrokerSemanticOrderRole.RECOVERY,
        )
    )

    assert len(result.orders) == 2
    assert result.orders[0].metadata["source"] == "ordinary"
    assert result.orders[1].metadata["source"] == "algo"
    assert result.orders[0].raw == {"raw": "ordinary"}
    assert result.orders[1].raw == {"raw": "algo"}
    assert result.message == "recovered_open_orders"


@pytest.mark.asyncio
async def test_fetch_position_returns_position(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    fake.position = BrokerPosition(
        exchange=ExchangeName.OKX,
        symbol=SYMBOL,
        position_side=BrokerPositionSide.LONG,
        quantity=Decimal("5"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    result = await executor.execute(
        _request(BrokerSemanticAction.FETCH_POSITION, BrokerSemanticOrderRole.UNKNOWN)
    )

    assert result.ok is True
    assert result.position == fake.position


@pytest.mark.asyncio
async def test_cancel_all_open_orders_is_unsupported_until_identity_safety_exists(
    executor: OkxBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    fake.open_orders.extend([_order("ord-1"), _order("ord-2")])

    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(
            _request(
                BrokerSemanticAction.CANCEL_ALL_OPEN_ORDERS,
                BrokerSemanticOrderRole.RECOVERY,
            )
        )

    assert exc_info.value.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert "reduce-only identity safety" in exc_info.value.message
    assert fake.cancel_order_calls == []


@pytest.mark.parametrize(
    ("semantic_request", "expected_kind"),
    [
        (
            _request(
                BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
                BrokerSemanticOrderRole.TP1,
                side=BrokerPositionSide.LONG,
                quantity=Decimal("1"),
            ),
            ExchangeErrorKind.INVALID_PRICE,
        ),
        (
            _request(
                BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
                BrokerSemanticOrderRole.PROTECTIVE_SL,
                side=BrokerPositionSide.LONG,
                quantity=Decimal("1"),
            ),
            ExchangeErrorKind.INVALID_PRICE,
        ),
        (
            _request(
                BrokerSemanticAction.MARKET_EXIT_RUNNER,
                BrokerSemanticOrderRole.MARKET_EXIT,
                side=BrokerPositionSide.LONG,
            ),
            ExchangeErrorKind.INVALID_ORDER_SIZE,
        ),
        (
            _request(
                BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
                BrokerSemanticOrderRole.PROTECTIVE_SL,
            ),
            ExchangeErrorKind.EXCHANGE_REJECTED,
        ),
        (
            _request(
                BrokerSemanticAction.OPEN_POSITION,
                BrokerSemanticOrderRole.ENTRY,
                side=BrokerPositionSide.UNKNOWN,
                quantity=Decimal("1"),
            ),
            ExchangeErrorKind.UNSUPPORTED_OPERATION,
        ),
    ],
)
@pytest.mark.asyncio
async def test_missing_or_invalid_required_fields_raise_exchange_error(
    executor: OkxBrokerSemanticExecutor,
    semantic_request: BrokerSemanticRequest,
    expected_kind: ExchangeErrorKind,
) -> None:
    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(semantic_request)

    assert exc_info.value.kind == expected_kind


@pytest.mark.asyncio
async def test_unsupported_action_raises_exchange_error(
    executor: OkxBrokerSemanticExecutor,
) -> None:
    request = BrokerSemanticRequest(
        exchange=ExchangeName.OKX,
        symbol=SYMBOL,
        action="UNKNOWN_ACTION",  # type: ignore[arg-type]
        role=BrokerSemanticOrderRole.UNKNOWN,
    )

    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(request)

    assert exc_info.value.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
