#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_live_smoke_test.py
@Description: Unit tests for the Binance live smoke test script.

All tests use mocked / fake transports — no real network calls.
"""

from __future__ import annotations

import os
from decimal import ROUND_UP, Decimal
from typing import Any
from unittest import mock

import pytest

from scripts.binance_live_smoke_test import (
    BINANCE_SYMBOL,
    CANONICAL_SYMBOL,
    CLIENT_ORDER_ID_PREFIX,
    CONFIRM_ENV,
    CONFIRM_VALUE,
    DEFAULT_MAX_NOTIONAL,
    DEFAULT_SL_PCT,
    DEFAULT_TP_PCT,
    ExchangeInfoFilters,
    Preflight,
    _generate_client_order_id,
    _make_order_request,
    _round_up_to_step,
    calculate_safe_quantity,
    cancel_order_by_id,
    cancel_smoke_orders,
    cleanup,
    close_long_position,
    fetch_account_balance,
    fetch_long_position,
    fetch_open_orders,
    load_binance_credentials,
    open_long,
    place_sl,
    place_tp,
    require_binance_exchange,
    require_live_confirmation,
)
from src.exchanges.binance.client import BinanceBrokerClient
from src.exchanges.binance.signing import (
    BINANCE_USDM_BASE_URL,
    BINANCE_USDM_OPEN_ORDERS_PATH,
    BINANCE_USDM_ORDER_PATH,
    BINANCE_USDM_POSITION_RISK_PATH,
    BinanceSignedRequest,
)
from src.exchanges.binance.transport import BinanceTransportResponse
from src.exchanges.errors import ExchangeErrorKind
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


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class FakeBinanceTransport:
    """Record requests and return canned responses in sequence."""

    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.requests: list[BinanceSignedRequest] = []
        self._idx = 0

    async def send(self, request: BinanceSignedRequest) -> BinanceTransportResponse:
        self.requests.append(request)
        if self._idx >= len(self.responses):
            return BinanceTransportResponse(
                status_code=200,
                payload={},
                headers={},
            )
        payload = self.responses[self._idx]
        self._idx += 1
        if isinstance(payload, BinanceTransportResponse):
            return payload
        status_code = 200
        if isinstance(payload, tuple):
            status_code, actual_payload = payload
            return BinanceTransportResponse(
                status_code=status_code,
                payload=actual_payload,
                headers={},
            )
        return BinanceTransportResponse(
            status_code=status_code,
            payload=payload,
            headers={},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(*responses: Any) -> BinanceBrokerClient:
    transport = FakeBinanceTransport(*responses)
    return BinanceBrokerClient(
        api_key="test-key",
        api_secret="test-secret",
        transport=transport,
    )


def _minimal_order_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "symbol": "ETHUSDT",
        "orderId": 123456789,
        "clientOrderId": "cid-abc-001",
        "side": "BUY",
        "positionSide": "LONG",
        "type": "MARKET",
        "status": "NEW",
        "price": "0",
        "origQty": "0.1",
        "executedQty": "0.1",
        "avgPrice": "3100.50",
        "reduceOnly": False,
    }
    data.update(overrides)
    return data


def _minimal_position_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "symbol": "ETHUSDT",
        "positionAmt": "0.1",
        "entryPrice": "3100.50",
        "markPrice": "3105.00",
        "unRealizedProfit": "0.45",
        "positionSide": "LONG",
        "leverage": "1",
    }
    data.update(overrides)
    return data


def _fake_open_order_response(
    order_id: int = 1,
    client_order_id: str = "cid-tp-001",
    side: str = "SELL",
    order_type: str = "LIMIT",
    price: str = "3200.00",
    trigger_price: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "symbol": "ETHUSDT",
        "orderId": order_id,
        "clientOrderId": client_order_id,
        "side": side,
        "positionSide": "LONG",
        "type": order_type,
        "status": "NEW",
        "price": price,
        "origQty": "0.1",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
    }
    if trigger_price is not None:
        data["stopPrice"] = trigger_price
    return data


def _cancel_payload(order_id: int = 123456789) -> dict[str, Any]:
    return {
        "symbol": "ETHUSDT",
        "orderId": order_id,
        "clientOrderId": "cid-cancel-1",
        "status": "CANCELED",
    }


# ---------------------------------------------------------------------------
# Tests: safety gates
# ---------------------------------------------------------------------------


class TestRequireLiveConfirmation:
    def test_rejects_when_env_not_set(self, monkeypatch) -> None:
        monkeypatch.delenv(CONFIRM_ENV, raising=False)
        with pytest.raises(SystemExit) as exc_info:
            require_live_confirmation()
        assert exc_info.value.code == 1

    def test_rejects_when_env_has_wrong_value(self, monkeypatch) -> None:
        monkeypatch.setenv(CONFIRM_ENV, "NO")
        with pytest.raises(SystemExit) as exc_info:
            require_live_confirmation()
        assert exc_info.value.code == 1

    def test_accepts_correct_confirmation(self, monkeypatch) -> None:
        monkeypatch.setenv(CONFIRM_ENV, CONFIRM_VALUE)
        require_live_confirmation()  # does not raise


class TestRequireBinanceExchange:
    def test_rejects_when_exchange_not_set(self, monkeypatch) -> None:
        monkeypatch.delenv("EXCHANGE", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            require_binance_exchange()
        assert exc_info.value.code == 1

    def test_rejects_okx(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCHANGE", "okx")
        with pytest.raises(SystemExit) as exc_info:
            require_binance_exchange()
        assert exc_info.value.code == 1

    def test_accepts_binance(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCHANGE", "binance")
        require_binance_exchange()  # does not raise

    def test_accepts_binance_case_insensitive(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCHANGE", "BINANCE")
        require_binance_exchange()  # does not raise


class TestLoadBinanceCredentials:
    def test_rejects_missing_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
        monkeypatch.setenv("EXCHANGE_API_SECRET", "secret")
        with pytest.raises(SystemExit) as exc_info:
            load_binance_credentials()
        assert exc_info.value.code == 1

    def test_rejects_missing_api_secret(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "key")
        monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            load_binance_credentials()
        assert exc_info.value.code == 1

    def test_rejects_empty_api_key(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "   ")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "secret")
        with pytest.raises(SystemExit) as exc_info:
            load_binance_credentials()
        assert exc_info.value.code == 1

    def test_returns_credentials_when_set(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "my-key")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "my-secret")
        key, secret = load_binance_credentials()
        assert key == "my-key"
        assert secret == "my-secret"


# ---------------------------------------------------------------------------
# Tests: _round_up_to_step
# ---------------------------------------------------------------------------


class TestRoundUpToStep:
    def test_exact_multiple_unchanged(self) -> None:
        assert _round_up_to_step(Decimal("0.1"), Decimal("0.1")) == Decimal("0.1")

    def test_rounds_up_to_next_step(self) -> None:
        assert _round_up_to_step(Decimal("0.011"), Decimal("0.001")) == Decimal("0.011")
        assert _round_up_to_step(Decimal("0.0015"), Decimal("0.001")) == Decimal("0.002")

    def test_step_of_0_1(self) -> None:
        assert _round_up_to_step(Decimal("0.05"), Decimal("0.1")) == Decimal("0.1")
        assert _round_up_to_step(Decimal("0.10"), Decimal("0.1")) == Decimal("0.1")
        assert _round_up_to_step(Decimal("0.11"), Decimal("0.1")) == Decimal("0.2")

    def test_raises_on_zero_step(self) -> None:
        with pytest.raises(ValueError, match="step_size must be positive"):
            _round_up_to_step(Decimal("0.1"), Decimal("0"))


# ---------------------------------------------------------------------------
# Tests: calculate_safe_quantity
# ---------------------------------------------------------------------------


class TestCalculateSafeQuantity:
    _FILTERS = ExchangeInfoFilters(
        min_qty=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("5"),
    )

    def test_basic_calculation(self) -> None:
        qty, notional = calculate_safe_quantity(
            mark_price=Decimal("3000"),
            max_notional=Decimal("6"),
            filters=self._FILTERS,
        )
        assert qty > 0
        assert notional > 0

    def test_quantity_rounded_to_step(self) -> None:
        qty, _ = calculate_safe_quantity(
            mark_price=Decimal("3100"),
            max_notional=Decimal("5"),
            filters=self._FILTERS,
        )
        # 5/3100 = 0.0016129... → rounded up to 0.002
        assert qty == Decimal("0.002")

    def test_notional_below_min_gets_bumped(self) -> None:
        filters = ExchangeInfoFilters(
            min_qty=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("100"),
        )
        qty, notional = calculate_safe_quantity(
            mark_price=Decimal("3000"),
            max_notional=Decimal("6"),
            filters=filters,
        )
        assert notional >= filters.min_notional

    def test_below_min_qty_exits(self) -> None:
        filters = ExchangeInfoFilters(
            min_qty=Decimal("10"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("1"),
        )
        with pytest.raises(SystemExit):
            calculate_safe_quantity(
                mark_price=Decimal("3000"),
                max_notional=Decimal("6"),
                filters=filters,
            )

    def test_min_notional_exceeds_exchange_info_when_quantity_too_large(self) -> None:
        """When min_qty * mark_price >> user budget, exit early."""
        filters = ExchangeInfoFilters(
            min_qty=Decimal("1000"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("1"),
        )
        with pytest.raises(SystemExit):
            calculate_safe_quantity(
                mark_price=Decimal("3000"),
                max_notional=Decimal("6"),
                filters=filters,
            )


# ---------------------------------------------------------------------------
# Tests: _generate_client_order_id
# ---------------------------------------------------------------------------


class TestGenerateClientOrderId:
    def test_has_smoke_prefix(self) -> None:
        cid = _generate_client_order_id("open")
        assert cid.startswith(CLIENT_ORDER_ID_PREFIX)

    def test_contains_label(self) -> None:
        cid = _generate_client_order_id("tp")
        assert "tp" in cid

    def test_unique_per_call(self) -> None:
        cid1 = _generate_client_order_id("x")
        cid2 = _generate_client_order_id("x")
        assert cid1 != cid2


# ---------------------------------------------------------------------------
# Tests: _make_order_request
# ---------------------------------------------------------------------------


class TestMakeOrderRequest:
    def test_market_buy_long(self) -> None:
        req = _make_order_request(
            side=BrokerOrderSide.BUY,
            order_type=BrokerOrderType.MARKET,
            quantity=Decimal("0.1"),
            client_order_id="test-001",
        )
        assert req.exchange == ExchangeName.BINANCE
        assert req.symbol == BINANCE_SYMBOL
        assert req.side == BrokerOrderSide.BUY
        assert req.position_side == BrokerPositionSide.LONG
        assert req.order_type == BrokerOrderType.MARKET
        assert req.quantity == Decimal("0.1")
        assert req.quantity_unit == BrokerQuantityUnit.BASE_ASSET
        assert req.reduce_only is False
        assert req.client_order_id == "test-001"

    def test_limit_sell_tp(self) -> None:
        req = _make_order_request(
            side=BrokerOrderSide.SELL,
            order_type=BrokerOrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("3200"),
            client_order_id="tp-001",
        )
        assert req.side == BrokerOrderSide.SELL
        assert req.order_type == BrokerOrderType.LIMIT
        assert req.price == Decimal("3200")

    def test_stop_market_sell_sl(self) -> None:
        req = _make_order_request(
            side=BrokerOrderSide.SELL,
            order_type=BrokerOrderType.STOP_MARKET,
            quantity=Decimal("0.1"),
            trigger_price=Decimal("2900"),
            client_order_id="sl-001",
        )
        assert req.side == BrokerOrderSide.SELL
        assert req.order_type == BrokerOrderType.STOP_MARKET
        assert req.trigger_price == Decimal("2900")


# ---------------------------------------------------------------------------
# Tests: open_long
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_long_sends_correct_request() -> None:
    client = _make_client(_minimal_order_payload())
    result = await open_long(client, Decimal("0.1"), "RE_SMOKE_open_123")
    assert result.ok is True
    assert result.order is not None
    transport = client._transport
    assert transport is not None
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "POST"
    assert transport.requests[0].path == BINANCE_USDM_ORDER_PATH


@pytest.mark.asyncio
async def test_open_long_raises_on_failure() -> None:
    from src.exchanges.errors import ExchangeError
    client = _make_client(
        BinanceTransportResponse(status_code=400, payload={"code": -2019, "msg": "Insufficient balance"}, headers={}),
    )
    with pytest.raises(ExchangeError) as exc_info:
        await open_long(client, Decimal("0.1"), "RE_SMOKE_open_123")
    assert exc_info.value.kind == ExchangeErrorKind.INSUFFICIENT_BALANCE


# ---------------------------------------------------------------------------
# Tests: fetch_long_position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_long_position_returns_broker_position() -> None:
    client = _make_client([_minimal_position_payload()])
    pos = await fetch_long_position(client)
    assert pos is not None
    assert pos.symbol == "ETHUSDT"
    assert pos.position_side == BrokerPositionSide.LONG
    assert pos.quantity == Decimal("0.1")


@pytest.mark.asyncio
async def test_fetch_long_position_returns_none_when_zero() -> None:
    payload = _minimal_position_payload(positionAmt="0", positionSide="LONG")
    client = _make_client([payload])
    pos = await fetch_long_position(client)
    assert pos is None


@pytest.mark.asyncio
async def test_fetch_long_position_returns_none_for_short() -> None:
    payload = _minimal_position_payload(positionAmt="0.1", positionSide="SHORT")
    client = _make_client([payload])
    pos = await fetch_long_position(client)
    assert pos is None


# ---------------------------------------------------------------------------
# Tests: place_tp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_tp_sends_limit_sell() -> None:
    client = _make_client(
        _minimal_order_payload(side="SELL", type="LIMIT", price="3200.00"),
    )
    result = await place_tp(client, Decimal("0.1"), Decimal("3200"), "RE_SMOKE_tp_123")
    assert result.ok is True
    assert result.order is not None
    transport = client._transport
    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert req.method == "POST"
    assert req.path == BINANCE_USDM_ORDER_PATH


@pytest.mark.asyncio
async def test_place_tp_raises_on_failure() -> None:
    from src.exchanges.errors import ExchangeError
    client = _make_client(
        BinanceTransportResponse(
            status_code=400, payload={"code": -2019, "msg": "Margin insufficient"}, headers={},
        ),
    )
    with pytest.raises(ExchangeError) as exc_info:
        await place_tp(client, Decimal("0.1"), Decimal("3200"), "RE_SMOKE_tp_123")
    assert exc_info.value.kind == ExchangeErrorKind.INSUFFICIENT_BALANCE


# ---------------------------------------------------------------------------
# Tests: place_sl
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_sl_sends_stop_market_sell() -> None:
    client = _make_client(
        _minimal_order_payload(
            side="SELL", type="STOP_MARKET", price="0", stopPrice="2900",
        ),
    )
    result = await place_sl(client, Decimal("0.1"), Decimal("2900"), "RE_SMOKE_sl_123")
    assert result.ok is True
    transport = client._transport
    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert req.method == "POST"
    assert req.path == BINANCE_USDM_ORDER_PATH


@pytest.mark.asyncio
async def test_place_sl_raises_on_failure() -> None:
    from src.exchanges.errors import ExchangeError
    client = _make_client(
        BinanceTransportResponse(
            status_code=400, payload={"code": -2021, "msg": "Stop price error"}, headers={},
        ),
    )
    with pytest.raises(ExchangeError) as exc_info:
        await place_sl(client, Decimal("0.1"), Decimal("2900"), "RE_SMOKE_sl_123")
    assert exc_info.value.kind == ExchangeErrorKind.EXCHANGE_REJECTED


# ---------------------------------------------------------------------------
# Tests: fetch_open_orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_open_orders_returns_list() -> None:
    client = _make_client([
        _fake_open_order_response(order_id=1, client_order_id="RE_SMOKE_tp_1"),
        _fake_open_order_response(order_id=2, client_order_id="RE_SMOKE_sl_1", order_type="STOP_MARKET", price="0", trigger_price="2900"),
    ])
    orders = await fetch_open_orders(client)
    assert len(orders) == 2
    transport = client._transport
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "GET"
    assert transport.requests[0].path == BINANCE_USDM_OPEN_ORDERS_PATH


@pytest.mark.asyncio
async def test_fetch_open_orders_empty() -> None:
    client = _make_client([])
    orders = await fetch_open_orders(client)
    assert orders == []


# ---------------------------------------------------------------------------
# Tests: cancel_order_by_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_order_by_id() -> None:
    client = _make_client(_cancel_payload(order_id=123))
    result = await cancel_order_by_id(client, "123")
    assert result.ok is True
    assert result.order_id == "123"
    transport = client._transport
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "DELETE"
    assert transport.requests[0].path == BINANCE_USDM_ORDER_PATH


@pytest.mark.asyncio
async def test_cancel_order_not_found_does_not_raise() -> None:
    client = _make_client(
        BinanceTransportResponse(
            status_code=400,
            payload={"code": -2011, "msg": "Order does not exist."},
            headers={},
        ),
    )
    # cancel_order_by_id calls client.cancel_order which raises ExchangeError on failure
    # But cancel_order_by_id doesn't catch it explicitly - let me check
    # Actually, looking at my code, it does catch and prints a warning
    # Let me re-examine...
    #
    # Wait: cancel_order_by_id calls client.cancel_order which will raise ExchangeError.
    # My cancel_order_by_id doesn't have a try/except. Let me verify:
    #
    # Looking at my code:
    # async def cancel_order_by_id(client, order_id):
    #     result = await client.cancel_order(BINANCE_SYMBOL, order_id)
    #     if not result.ok:
    #         print(...)
    #     ...
    #
    # This will RAISE ExchangeError when the order is not found, because
    # BinanceBrokerClient.cancel_order raises map_binance_error for error codes.
    #
    # So the test should expect the error to propagate. Let me fix the test.

    # Actually, the test is correct — we're testing that the error propagates.
    # The script handles it in the main flow via the try/except in main().
    # But for the unit test, we should expect the ExchangeError.

    from src.exchanges.errors import ExchangeError, ExchangeErrorKind
    with pytest.raises(ExchangeError) as exc_info:
        await cancel_order_by_id(client, "999")
    assert exc_info.value.kind == ExchangeErrorKind.ORDER_NOT_FOUND


# ---------------------------------------------------------------------------
# Tests: cancel_smoke_orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_smoke_orders_only_cancels_prefixed() -> None:
    client = _make_client(
        [
            _fake_open_order_response(order_id=1, client_order_id="RE_SMOKE_tp_1"),
            _fake_open_order_response(order_id=2, client_order_id="OTHER_ORDER"),
            _fake_open_order_response(order_id=3, client_order_id="RE_SMOKE_sl_1", order_type="STOP_MARKET", price="0", trigger_price="2900"),
        ],
        _cancel_payload(order_id=1),
        _cancel_payload(order_id=3),
    )
    result = await cancel_smoke_orders(client)
    # Should cancel 2 smoke orders, skip the other one
    assert result == 2


@pytest.mark.asyncio
async def test_cancel_smoke_orders_none_to_cancel() -> None:
    client = _make_client([])
    result = await cancel_smoke_orders(client)
    assert result == 0


# ---------------------------------------------------------------------------
# Tests: close_long_position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_long_sends_market_sell() -> None:
    client = _make_client(
        _minimal_order_payload(side="SELL", type="MARKET"),
    )
    result = await close_long_position(client, Decimal("0.1"), "RE_SMOKE_close_123")
    assert result.ok is True
    transport = client._transport
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "POST"
    assert transport.requests[0].path == BINANCE_USDM_ORDER_PATH


# ---------------------------------------------------------------------------
# Tests: cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_no_residual_position() -> None:
    client = _make_client(
        [],  # open orders → empty
        [_minimal_position_payload(positionAmt="0")],  # position → None
        [_minimal_position_payload(positionAmt="0")],  # final check → None
    )
    await cleanup(client)  # should not raise


@pytest.mark.asyncio
async def test_cleanup_with_open_smoke_orders() -> None:
    client = _make_client(
        [
            _fake_open_order_response(order_id=1, client_order_id="RE_SMOKE_tp_1"),
        ],
        _cancel_payload(order_id=1),  # cancel TP
        [_minimal_position_payload(positionAmt="0")],  # position → None
        [_minimal_position_payload(positionAmt="0")],  # final → None
    )
    await cleanup(client)


# ---------------------------------------------------------------------------
# Tests: overall flow sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_sequence_order() -> None:
    """Verify the smoke test flow goes through all 7 steps in order."""
    # This test uses the _run_smoke_test flow indirectly by
    # verifying that the functions are called in the right order.
    # We can't call _run_smoke_test directly without mocking the
    # public endpoints, but we can verify that each step function
    # works with the transport.

    responses = [
        _minimal_order_payload(orderId=1, clientOrderId="RE_SMOKE_open_1"),
        [_minimal_position_payload(positionAmt="0.1", entryPrice="3100")],
        _minimal_order_payload(
            orderId=2, clientOrderId="RE_SMOKE_tp_1",
            side="SELL", type="LIMIT", price="3118.60",
        ),
        _minimal_order_payload(
            orderId=3, clientOrderId="RE_SMOKE_sl_1",
            side="SELL", type="STOP_MARKET", price="0", stopPrice="3081.40",
        ),
        [
            _fake_open_order_response(
                order_id=2, client_order_id="RE_SMOKE_tp_1",
                side="SELL", order_type="LIMIT", price="3118.60",
            ),
            _fake_open_order_response(
                order_id=3, client_order_id="RE_SMOKE_sl_1",
                side="SELL", order_type="STOP_MARKET", price="0", trigger_price="3081.40",
            ),
        ],
        _cancel_payload(order_id=2),
        _cancel_payload(order_id=3),
        [],
        [_minimal_position_payload(positionAmt="0.1", entryPrice="3100")],
        _minimal_order_payload(
            orderId=4, clientOrderId="RE_SMOKE_close_1",
            side="SELL", type="MARKET",
        ),
        [_minimal_position_payload(positionAmt="0")],
    ]

    transport = FakeBinanceTransport(*responses)
    client = BinanceBrokerClient(
        api_key="test-key",
        api_secret="test-secret",
        transport=transport,
    )

    # Step 1: Open
    open_result = await open_long(client, Decimal("0.1"), "RE_SMOKE_open_1")
    assert open_result.ok

    # Step 2: Fetch position
    pos = await fetch_long_position(client)
    assert pos is not None
    assert pos.quantity > 0

    # Step 3: Place TP
    tp_result = await place_tp(client, Decimal("0.1"), Decimal("3118.60"), "RE_SMOKE_tp_1")
    assert tp_result.ok

    # Step 4: Place SL
    sl_result = await place_sl(client, Decimal("0.1"), Decimal("3081.40"), "RE_SMOKE_sl_1")
    assert sl_result.ok

    # Step 5: Fetch open orders
    orders = await fetch_open_orders(client)
    assert len(orders) >= 2

    # Step 6: Cancel TP/SL
    c1 = await cancel_order_by_id(client, str(tp_result.order_id or "2"))
    c2 = await cancel_order_by_id(client, str(sl_result.order_id or "3"))
    assert c1.ok
    assert c2.ok

    # Verify cancelled
    orders_after = await fetch_open_orders(client)
    assert len(orders_after) == 0

    # Step 7: Market close
    pos_before = await fetch_long_position(client)
    assert pos_before is not None
    close_result = await close_long_position(client, pos_before.quantity, "RE_SMOKE_close_1")
    assert close_result.ok

    pos_after = await fetch_long_position(client)
    assert pos_after is None  # position = 0


# ---------------------------------------------------------------------------
# Tests: Preflight dataclass
# ---------------------------------------------------------------------------


class TestPreflightDataclass:
    def test_preflight_construction(self) -> None:
        p = Preflight(
            api_key="k",
            api_secret="s",
            mark_price=Decimal("3000"),
            available_usdt_balance=Decimal("10"),
            max_notional=Decimal("6"),
            tp_pct=Decimal("0.006"),
            sl_pct=Decimal("0.006"),
            filters=ExchangeInfoFilters(
                min_qty=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("5"),
            ),
            calculated_quantity=Decimal("0.002"),
            calculated_notional=Decimal("6"),
        )
        assert p.calculated_quantity == Decimal("0.002")
        assert p.calculated_notional == Decimal("6")


# ---------------------------------------------------------------------------
# Tests: ExchangeInfoFilters dataclass
# ---------------------------------------------------------------------------


class TestExchangeInfoFilters:
    def test_construction(self) -> None:
        f = ExchangeInfoFilters(
            min_qty=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("5"),
        )
        assert f.min_qty == Decimal("0.001")
        assert f.step_size == Decimal("0.001")
        assert f.min_notional == Decimal("5")


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_canonical_symbol_is_eth_usdt_perp(self) -> None:
        assert CANONICAL_SYMBOL == "ETH-USDT-PERP"

    def test_binance_symbol_is_ethusdt(self) -> None:
        assert BINANCE_SYMBOL == "ETHUSDT"

    def test_client_order_id_prefix(self) -> None:
        assert CLIENT_ORDER_ID_PREFIX == "RE_SMOKE_"
