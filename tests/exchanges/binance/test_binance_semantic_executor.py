#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_semantic_executor.py
@Description: Tests for the Binance broker semantic executor adapter.
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
from src.exchanges.binance.semantic_executor import BinanceBrokerSemanticExecutor
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
)

SYMBOL = "ETHUSDT"


class FakeBrokerClient:
    def __init__(self) -> None:
        self.place_order_requests: list[BrokerOrderRequest] = []
        self.cancel_order_calls: list[tuple[str, str]] = []
        self.open_orders: list[BrokerOrder] = []
        self.position: BrokerPosition | None = None
        self.next_order_result: BrokerOrderResult | None = None
        self.next_cancel_result: BrokerCancelResult | None = None

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        self.place_order_requests.append(request)
        if self.next_order_result is not None:
            return self.next_order_result
        order = BrokerOrder(
            exchange=ExchangeName.BINANCE,
            symbol=request.symbol,
            order_id="order-1",
            client_order_id=request.client_order_id,
            side=request.side,
            position_side=request.position_side,
            order_type=request.order_type,
            status=BrokerOrderStatus.OPEN,
            price=request.price,
            quantity=request.quantity,
            quantity_unit=request.quantity_unit,
            reduce_only=request.reduce_only,
            trigger_price=request.trigger_price,
        )
        return BrokerOrderResult(
            exchange=ExchangeName.BINANCE,
            symbol=request.symbol,
            ok=True,
            order_id="order-1",
            client_order_id=request.client_order_id,
            order=order,
            raw={"fake": True},
        )

    async def cancel_order(self, symbol: str, order_id: str) -> BrokerCancelResult:
        self.cancel_order_calls.append((symbol, order_id))
        if self.next_cancel_result is not None:
            return self.next_cancel_result
        return BrokerCancelResult(
            exchange=ExchangeName.BINANCE,
            symbol=symbol,
            ok=True,
            order_id=order_id,
            raw={"fake": True},
        )

    async def fetch_position(self, symbol: str) -> BrokerPosition | None:
        return self.position

    async def fetch_open_orders(self, symbol: str) -> tuple[BrokerOrder, ...]:
        return tuple(self.open_orders)


@pytest.fixture
def fake() -> FakeBrokerClient:
    return FakeBrokerClient()


@pytest.fixture
def executor(fake: FakeBrokerClient) -> BinanceBrokerSemanticExecutor:
    return BinanceBrokerSemanticExecutor(fake)


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
    client_order_id: str | None = None,
) -> BrokerSemanticRequest:
    return BrokerSemanticRequest(
        exchange=ExchangeName.BINANCE,
        symbol=SYMBOL,
        action=action,
        role=role,
        side=side,
        quantity=quantity,
        quantity_unit=quantity_unit,
        price=price,
        trigger_price=trigger_price,
        order_id=order_id,
        client_order_id=client_order_id,
    )


def _order(
    order_id: str,
    *,
    metadata: dict[str, str] | None = None,
    raw: dict[str, str] | None = None,
) -> BrokerOrder:
    return BrokerOrder(
        exchange=ExchangeName.BINANCE,
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


# ---------------------------------------------------------------------------
# exchange identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_identity_is_binance(
    executor: BinanceBrokerSemanticExecutor,
) -> None:
    assert executor.exchange == ExchangeName.BINANCE


# ---------------------------------------------------------------------------
# exchange mismatch guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_mismatch_raises_unsupported_operation(
    executor: BinanceBrokerSemanticExecutor,
) -> None:
    request = BrokerSemanticRequest(
        exchange=ExchangeName.OKX,
        symbol=SYMBOL,
        action=BrokerSemanticAction.FETCH_POSITION,
        role=BrokerSemanticOrderRole.UNKNOWN,
    )

    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(request)

    assert exc_info.value.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert "BinanceBrokerSemanticExecutor cannot execute okx" in exc_info.value.message


# ---------------------------------------------------------------------------
# open_position LONG -> BUY LONG MARKET reduce_only=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_position_long_places_buy_market_non_reduce_only(
    executor: BinanceBrokerSemanticExecutor,
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
    assert order_request.exchange == ExchangeName.BINANCE


# ---------------------------------------------------------------------------
# open_position SHORT -> SELL SHORT MARKET reduce_only=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_position_short_places_sell_market_non_reduce_only(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    result = await executor.execute(
        _request(
            BrokerSemanticAction.OPEN_POSITION,
            BrokerSemanticOrderRole.ENTRY,
            side=BrokerPositionSide.SHORT,
            quantity=Decimal("4"),
        )
    )

    assert result.ok is True
    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.MARKET
    assert order_request.reduce_only is False
    assert order_request.side == BrokerOrderSide.SELL
    assert order_request.position_side == BrokerPositionSide.SHORT


# ---------------------------------------------------------------------------
# add_position maps same as open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_position_places_market_non_reduce_only_short(
    executor: BinanceBrokerSemanticExecutor,
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
    assert order_request.order_type == BrokerOrderType.MARKET
    assert order_request.reduce_only is False
    assert order_request.side == BrokerOrderSide.SELL


# ---------------------------------------------------------------------------
# sidecar_entry maps same as open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sidecar_entry_places_market_non_reduce_only(
    executor: BinanceBrokerSemanticExecutor,
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


# ---------------------------------------------------------------------------
# place_reduce_only_tp LONG -> SELL LONG LIMIT reduce_only=True price set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_reduce_only_tp_long_places_sell_limit_reduce_only(
    executor: BinanceBrokerSemanticExecutor,
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


# ---------------------------------------------------------------------------
# place_reduce_only_tp SHORT -> BUY SHORT LIMIT reduce_only=True price set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_reduce_only_tp_short_places_buy_limit_reduce_only(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    await executor.execute(
        _request(
            BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            BrokerSemanticOrderRole.TP1,
            side=BrokerPositionSide.SHORT,
            quantity=Decimal("3"),
            price=Decimal("3200"),
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.LIMIT
    assert order_request.reduce_only is True
    assert order_request.side == BrokerOrderSide.BUY
    assert order_request.position_side == BrokerPositionSide.SHORT
    assert order_request.price == Decimal("3200")


# ---------------------------------------------------------------------------
# sidecar_tp maps same as reduce-only TP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sidecar_tp_places_limit_close_order_for_short(
    executor: BinanceBrokerSemanticExecutor,
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


# ---------------------------------------------------------------------------
# place_protective_stop LONG -> SELL LONG STOP_MARKET reduce_only=True trigger_price set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_protective_stop_long_places_sell_stop_market(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    result = await executor.execute(
        _request(
            BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("5"),
            trigger_price=Decimal("3400"),
        )
    )

    assert len(fake.place_order_requests) == 1
    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.STOP_MARKET
    assert order_request.reduce_only is True
    assert order_request.side == BrokerOrderSide.SELL
    assert order_request.position_side == BrokerPositionSide.LONG
    assert order_request.trigger_price == Decimal("3400")
    assert result.order_id == "order-1"


# ---------------------------------------------------------------------------
# place_protective_stop SHORT -> BUY SHORT STOP_MARKET reduce_only=True trigger_price set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_protective_stop_short_places_buy_stop_market(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    result = await executor.execute(
        _request(
            BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerPositionSide.SHORT,
            quantity=Decimal("5"),
            trigger_price=Decimal("3600"),
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.STOP_MARKET
    assert order_request.reduce_only is True
    assert order_request.side == BrokerOrderSide.BUY
    assert order_request.position_side == BrokerPositionSide.SHORT
    assert order_request.trigger_price == Decimal("3600")
    assert result.ok is True


# ---------------------------------------------------------------------------
# market_exit LONG -> SELL LONG MARKET reduce_only=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_exit_long_places_sell_market_reduce_only(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    await executor.execute(
        _request(
            BrokerSemanticAction.MARKET_EXIT,
            BrokerSemanticOrderRole.MARKET_EXIT,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("3"),
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.MARKET
    assert order_request.reduce_only is True
    assert order_request.side == BrokerOrderSide.SELL
    assert order_request.position_side == BrokerPositionSide.LONG


# ---------------------------------------------------------------------------
# market_exit SHORT -> BUY SHORT MARKET reduce_only=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_exit_short_places_buy_market_reduce_only(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    await executor.execute(
        _request(
            BrokerSemanticAction.MARKET_EXIT,
            BrokerSemanticOrderRole.MARKET_EXIT,
            side=BrokerPositionSide.SHORT,
            quantity=Decimal("2"),
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.order_type == BrokerOrderType.MARKET
    assert order_request.reduce_only is True
    assert order_request.side == BrokerOrderSide.BUY
    assert order_request.position_side == BrokerPositionSide.SHORT


# ---------------------------------------------------------------------------
# market_exit_runner maps same as market_exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_exit_runner_places_reduce_only_market_close(
    executor: BinanceBrokerSemanticExecutor,
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


# ---------------------------------------------------------------------------
# cancel_order calls broker_client.cancel_order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_order_calls_broker_client_cancel_order(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    result = await executor.execute(
        _request(
            BrokerSemanticAction.CANCEL_ORDER,
            BrokerSemanticOrderRole.UNKNOWN,
            order_id="ord-1",
        )
    )

    assert fake.cancel_order_calls == [(SYMBOL, "ord-1")]
    assert result.ok is True
    assert result.order_id == "ord-1"


# ---------------------------------------------------------------------------
# cancel_reduce_only_tp calls broker_client.cancel_order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_reduce_only_tp_calls_broker_client_cancel_order(
    executor: BinanceBrokerSemanticExecutor,
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


# ---------------------------------------------------------------------------
# cancel_protective_stop calls broker_client.cancel_order (not cancel_algo_order)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_protective_stop_calls_broker_client_cancel_order(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    result = await executor.execute(
        _request(
            BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
            BrokerSemanticOrderRole.PROTECTIVE_SL,
            order_id="sl-1",
        )
    )

    assert fake.cancel_order_calls == [(SYMBOL, "sl-1")]
    assert result.ok is True
    assert result.order_id == "sl-1"


# ---------------------------------------------------------------------------
# cancel_all_open_orders raises UNSUPPORTED_OPERATION
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_all_open_orders_is_unsupported_until_identity_safety_exists(
    executor: BinanceBrokerSemanticExecutor,
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


# ---------------------------------------------------------------------------
# fetch_position returns semantic result with position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_position_returns_position(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    fake.position = BrokerPosition(
        exchange=ExchangeName.BINANCE,
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


# ---------------------------------------------------------------------------
# fetch_position no position message == no_position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_position_no_position_returns_no_position_message(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    fake.position = None

    result = await executor.execute(
        _request(BrokerSemanticAction.FETCH_POSITION, BrokerSemanticOrderRole.UNKNOWN)
    )

    assert result.ok is True
    assert result.position is None
    assert result.message == "no_position"


# ---------------------------------------------------------------------------
# fetch_open_orders returns orders tuple
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_open_orders_returns_tuple(
    executor: BinanceBrokerSemanticExecutor,
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


# ---------------------------------------------------------------------------
# fetch_algo_orders returns ok=True orders=() message=binance_algo_orders_are_regular_open_orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_algo_orders_returns_empty_with_explanatory_message(
    executor: BinanceBrokerSemanticExecutor,
) -> None:
    result = await executor.execute(
        _request(
            BrokerSemanticAction.FETCH_ALGO_ORDERS,
            BrokerSemanticOrderRole.UNKNOWN,
        )
    )

    assert result.ok is True
    assert result.orders == ()
    assert result.message == "binance_algo_orders_are_regular_open_orders"


# ---------------------------------------------------------------------------
# recover_open_orders uses fetch_open_orders and metadata source=ordinary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_open_orders_uses_only_ordinary_orders(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    ordinary = _order(
        "ord-1",
        metadata={"original": "value"},
        raw={"raw": "ordinary"},
    )
    fake.open_orders.append(ordinary)

    result = await executor.execute(
        _request(
            BrokerSemanticAction.RECOVER_OPEN_ORDERS,
            BrokerSemanticOrderRole.RECOVERY,
        )
    )

    assert len(result.orders) == 1
    assert result.orders[0].metadata["source"] == "ordinary"
    assert result.orders[0].metadata["original"] == "value"
    assert result.orders[0].raw == {"raw": "ordinary"}
    assert result.message == "recovered_open_orders"


# ---------------------------------------------------------------------------
# missing field validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_side_raises_exchange_rejected(
    executor: BinanceBrokerSemanticExecutor,
) -> None:
    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(
            _request(
                BrokerSemanticAction.OPEN_POSITION,
                BrokerSemanticOrderRole.ENTRY,
                quantity=Decimal("1"),
            )
        )

    assert exc_info.value.kind == ExchangeErrorKind.EXCHANGE_REJECTED
    assert "side" in exc_info.value.message


@pytest.mark.asyncio
async def test_missing_quantity_raises_invalid_order_size(
    executor: BinanceBrokerSemanticExecutor,
) -> None:
    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(
            _request(
                BrokerSemanticAction.OPEN_POSITION,
                BrokerSemanticOrderRole.ENTRY,
                side=BrokerPositionSide.LONG,
            )
        )

    assert exc_info.value.kind == ExchangeErrorKind.INVALID_ORDER_SIZE
    assert "positive quantity" in exc_info.value.message


@pytest.mark.asyncio
async def test_missing_price_for_tp_raises_invalid_price(
    executor: BinanceBrokerSemanticExecutor,
) -> None:
    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(
            _request(
                BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
                BrokerSemanticOrderRole.TP1,
                side=BrokerPositionSide.LONG,
                quantity=Decimal("1"),
            )
        )

    assert exc_info.value.kind == ExchangeErrorKind.INVALID_PRICE
    assert "price" in exc_info.value.message


@pytest.mark.asyncio
async def test_missing_trigger_price_for_sl_raises_invalid_price(
    executor: BinanceBrokerSemanticExecutor,
) -> None:
    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(
            _request(
                BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
                BrokerSemanticOrderRole.PROTECTIVE_SL,
                side=BrokerPositionSide.LONG,
                quantity=Decimal("1"),
            )
        )

    assert exc_info.value.kind == ExchangeErrorKind.INVALID_PRICE
    assert "trigger_price" in exc_info.value.message


@pytest.mark.asyncio
async def test_missing_order_id_for_cancel_raises_exchange_rejected(
    executor: BinanceBrokerSemanticExecutor,
) -> None:
    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(
            _request(
                BrokerSemanticAction.CANCEL_ORDER,
                BrokerSemanticOrderRole.UNKNOWN,
            )
        )

    assert exc_info.value.kind == ExchangeErrorKind.EXCHANGE_REJECTED
    assert "order_id" in exc_info.value.message


# ---------------------------------------------------------------------------
# unsupported action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_action_raises_exchange_error(
    executor: BinanceBrokerSemanticExecutor,
) -> None:
    request = BrokerSemanticRequest(
        exchange=ExchangeName.BINANCE,
        symbol=SYMBOL,
        action="UNKNOWN_ACTION",  # type: ignore[arg-type]
        role=BrokerSemanticOrderRole.UNKNOWN,
    )

    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(request)

    assert exc_info.value.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert "Unsupported Binance" in exc_info.value.message


# ---------------------------------------------------------------------------
# client_order_id propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_order_id_is_propagated_to_broker_order_request(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    await executor.execute(
        _request(
            BrokerSemanticAction.OPEN_POSITION,
            BrokerSemanticOrderRole.ENTRY,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            client_order_id="my-client-id",
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.client_order_id == "my-client-id"


# ---------------------------------------------------------------------------
# quantity_unit defaults to CONTRACTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quantity_unit_defaults_to_contracts(
    executor: BinanceBrokerSemanticExecutor,
    fake: FakeBrokerClient,
) -> None:
    await executor.execute(
        _request(
            BrokerSemanticAction.OPEN_POSITION,
            BrokerSemanticOrderRole.ENTRY,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
        )
    )

    order_request = fake.place_order_requests[0]
    assert order_request.quantity_unit == BrokerQuantityUnit.CONTRACTS


# ---------------------------------------------------------------------------
# UNKNOWN position_side in _open_order_side / _close_order_side
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("position_side", "expected_kind"),
    [
        (BrokerPositionSide.UNKNOWN, ExchangeErrorKind.UNSUPPORTED_OPERATION),
        (BrokerPositionSide.NET, ExchangeErrorKind.UNSUPPORTED_OPERATION),
    ],
)
async def test_invalid_position_side_for_open_order_raises_unsupported(
    executor: BinanceBrokerSemanticExecutor,
    position_side: BrokerPositionSide,
    expected_kind: ExchangeErrorKind,
) -> None:
    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(
            _request(
                BrokerSemanticAction.OPEN_POSITION,
                BrokerSemanticOrderRole.ENTRY,
                side=position_side,
                quantity=Decimal("1"),
            )
        )

    assert exc_info.value.kind == expected_kind


# ---------------------------------------------------------------------------
# zero and negative quantity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("quantity", [Decimal("0"), Decimal("-1")])
async def test_non_positive_quantity_raises_invalid_order_size(
    executor: BinanceBrokerSemanticExecutor,
    quantity: Decimal,
) -> None:
    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute(
            _request(
                BrokerSemanticAction.OPEN_POSITION,
                BrokerSemanticOrderRole.ENTRY,
                side=BrokerPositionSide.LONG,
                quantity=quantity,
            )
        )

    assert exc_info.value.kind == ExchangeErrorKind.INVALID_ORDER_SIZE
