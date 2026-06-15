#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_binance_trading_client_port.py
@Description: Tests for BinanceTradingClient with an injected fake transport.
              Covers all 11 TradingClientPort methods.  No real Binance access.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.binance.private_client import BinancePrivateClient
from src.exchanges.binance.trading_client import BinanceTradingClient
from src.exchanges.binance.transport import BinanceTransportResponse
from src.exchanges.errors import ExchangeError, ExchangeErrorKind


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class FakeTransport:
    """Records every request and returns a canned response."""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.requests: list = []

    async def send(self, request):
        self.requests.append(request)
        return BinanceTransportResponse(
            status_code=self.status_code,
            payload=self.payload,
            headers={},
        )

    async def close(self) -> None:
        pass


class SequenceTransport:
    """Returns a sequence of responses, one per request."""

    def __init__(self, responses: list):
        self.responses = responses
        self.requests: list = []
        self._idx = 0

    async def send(self, request):
        self.requests.append(request)
        if self._idx >= len(self.responses):
            return BinanceTransportResponse(
                status_code=200,
                payload={},
                headers={},
            )
        status, payload = self.responses[self._idx]
        self._idx += 1
        return BinanceTransportResponse(
            status_code=status,
            payload=payload,
            headers={},
        )

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trading_client(
    transport=None,
    *,
    symbol="ETHUSDT",
    margin_asset="USDT",
    api_key="test-key",
    api_secret="test-secret",
    leverage=20,
    margin_mode="isolated",
    position_mode="net",
) -> BinanceTradingClient:
    private_client = BinancePrivateClient(
        api_key=api_key,
        api_secret=api_secret,
        transport=transport,
    )
    return BinanceTradingClient(
        symbol=symbol,
        margin_asset=margin_asset,
        api_key=api_key,
        api_secret=api_secret,
        leverage=leverage,
        margin_mode=margin_mode,
        position_mode=position_mode,
        private_client=private_client,
    )


# ---------------------------------------------------------------------------
# configure_instrument
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_configure_instrument_calls_leverage_and_margin() -> None:
    seq = SequenceTransport([
        (200, {"leverage": 20}),       # set leverage
        (200, {"code": 200, "msg": "success"}),  # set margin type
    ])
    client = _make_trading_client(transport=seq)

    await client.configure_instrument()

    assert len(seq.requests) == 2
    assert seq.requests[0].path == "/fapi/v1/leverage"
    assert seq.requests[1].path == "/fapi/v1/marginType"


@pytest.mark.asyncio
async def test_configure_instrument_leverage_already_set_non_fatal() -> None:
    """Error -4047 (leverage already set) must not propagate."""
    seq = SequenceTransport([
        (200, {"code": -4047, "msg": "leverage already set"}),  # leverage — already set
        (200, {"code": 200, "msg": "success"}),                  # margin type
    ])
    client = _make_trading_client(transport=seq)
    # Should not raise
    await client.configure_instrument()
    assert len(seq.requests) == 2


@pytest.mark.asyncio
async def test_configure_instrument_margin_already_set_non_fatal() -> None:
    """Error -4046 (margin type already set) must not propagate."""
    seq = SequenceTransport([
        (200, {"leverage": 20}),                                  # leverage
        (200, {"code": -4046, "msg": "No need to change margin type"}),  # margin — already set
    ])
    client = _make_trading_client(transport=seq)
    await client.configure_instrument()
    assert len(seq.requests) == 2


@pytest.mark.asyncio
async def test_configure_instrument_auth_error_propagates() -> None:
    seq = SequenceTransport([
        (401, {"code": -2015, "msg": "Invalid API-key"}),
    ])
    client = _make_trading_client(transport=seq)
    with pytest.raises(ExchangeError) as exc_info:
        await client.configure_instrument()
    assert exc_info.value.kind == ExchangeErrorKind.AUTH_ERROR


# ---------------------------------------------------------------------------
# fetch_balance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_balance_returns_snapshot() -> None:
    fake = FakeTransport([
        {"asset": "USDT", "balance": "5000.00", "crossWalletBalance": "5000.00",
         "crossUnPnl": "0.00", "availableBalance": "4800.00", "maxWithdrawAmount": "4800.00"},
    ])
    client = _make_trading_client(transport=fake)
    snap = await client.fetch_balance()
    assert snap.asset == "USDT"
    assert snap.total == Decimal("5000.00")
    assert snap.available == Decimal("4800.00")


@pytest.mark.asyncio
async def test_fetch_balance_no_asset_returns_zero() -> None:
    fake = FakeTransport([
        {"asset": "ETH", "balance": "10", "availableBalance": "10"},
    ])
    client = _make_trading_client(transport=fake)
    snap = await client.fetch_balance()
    assert snap.asset == "USDT"
    assert snap.total == Decimal("0")
    assert snap.available == Decimal("0")


@pytest.mark.asyncio
async def test_fetch_balance_empty_list_returns_zero() -> None:
    fake = FakeTransport([])
    client = _make_trading_client(transport=fake)
    snap = await client.fetch_balance()
    assert snap.total == Decimal("0")


# ---------------------------------------------------------------------------
# fetch_position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_position_long() -> None:
    fake = FakeTransport([
        {"symbol": "ETHUSDT", "positionAmt": "2.5", "entryPrice": "3000.00",
         "markPrice": "3050.00", "unRealizedProfit": "125.00", "positionSide": "BOTH"},
    ])
    client = _make_trading_client(transport=fake)
    snap = await client.fetch_position()
    assert snap.side == "LONG"
    assert snap.qty == Decimal("2.5")
    assert snap.avg_entry_price == Decimal("3000.00")


@pytest.mark.asyncio
async def test_fetch_position_short() -> None:
    fake = FakeTransport([
        {"symbol": "ETHUSDT", "positionAmt": "-1.0", "entryPrice": "3100.00", "positionSide": "BOTH"},
    ])
    client = _make_trading_client(transport=fake)
    snap = await client.fetch_position()
    assert snap.side == "SHORT"
    assert snap.qty == Decimal("1.0")


@pytest.mark.asyncio
async def test_fetch_position_none() -> None:
    fake = FakeTransport([
        {"symbol": "ETHUSDT", "positionAmt": "0", "entryPrice": "0", "positionSide": "BOTH"},
    ])
    client = _make_trading_client(transport=fake)
    snap = await client.fetch_position()
    assert snap.side is None
    assert snap.qty == Decimal("0")


@pytest.mark.asyncio
async def test_fetch_position_empty_list_returns_none() -> None:
    fake = FakeTransport([])
    client = _make_trading_client(transport=fake)
    snap = await client.fetch_position()
    assert snap.side is None
    assert snap.qty == Decimal("0")


# ---------------------------------------------------------------------------
# fetch_open_orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_open_orders() -> None:
    fake = FakeTransport([
        {"symbol": "ETHUSDT", "orderId": 1, "clientOrderId": "c1",
         "side": "BUY", "type": "LIMIT", "price": "3000", "origQty": "1.0",
         "reduceOnly": False, "status": "NEW"},
        {"symbol": "ETHUSDT", "orderId": 2, "clientOrderId": "c2",
         "side": "SELL", "type": "STOP_MARKET", "stopPrice": "2800", "origQty": "0.5",
         "reduceOnly": True, "status": "NEW"},
    ])
    client = _make_trading_client(transport=fake)
    orders = await client.fetch_open_orders()
    assert len(orders) == 2
    assert orders[0].order_id == "1"
    assert orders[0].side == "BUY"
    assert orders[1].trigger_price == Decimal("2800")


@pytest.mark.asyncio
async def test_fetch_open_orders_empty() -> None:
    fake = FakeTransport([])
    client = _make_trading_client(transport=fake)
    orders = await client.fetch_open_orders()
    assert orders == []


# ---------------------------------------------------------------------------
# fetch_order_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_order_status_by_order_id() -> None:
    fake = FakeTransport({
        "symbol": "ETHUSDT", "orderId": 123, "clientOrderId": "cid-1",
        "status": "FILLED", "executedQty": "1.0", "avgPrice": "3000.00",
    })
    client = _make_trading_client(transport=fake)
    snap = await client.fetch_order_status(order_id="123")
    assert snap.status == "FILLED"
    assert snap.order_id == "123"
    assert snap.filled_qty == Decimal("1.0")


@pytest.mark.asyncio
async def test_fetch_order_status_by_client_order_id() -> None:
    fake = FakeTransport({
        "symbol": "ETHUSDT", "orderId": 456, "clientOrderId": "my-cid",
        "status": "NEW",
    })
    client = _make_trading_client(transport=fake)
    snap = await client.fetch_order_status(client_order_id="my-cid")
    assert snap.status == "OPEN"


@pytest.mark.asyncio
async def test_fetch_order_status_no_id_raises() -> None:
    client = _make_trading_client(transport=FakeTransport({}))
    with pytest.raises(ValueError, match="at least one of"):
        await client.fetch_order_status()


# ---------------------------------------------------------------------------
# fetch_open_algo_orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_open_algo_orders_filters_stop_orders() -> None:
    fake = FakeTransport([
        {"symbol": "ETHUSDT", "orderId": 1, "side": "BUY", "type": "LIMIT",
         "origQty": "1", "status": "NEW"},
        {"symbol": "ETHUSDT", "orderId": 2, "side": "SELL", "type": "STOP_MARKET",
         "stopPrice": "2800", "origQty": "0.5", "status": "NEW"},
        {"symbol": "ETHUSDT", "orderId": 3, "side": "SELL", "type": "TAKE_PROFIT_MARKET",
         "stopPrice": "3200", "origQty": "1.0", "status": "NEW"},
    ])
    client = _make_trading_client(transport=fake)
    algos = await client.fetch_open_algo_orders()
    assert len(algos) == 2
    assert algos[0].order_id == "2"
    assert algos[0].trigger_price == Decimal("2800")
    assert algos[1].order_id == "3"
    assert algos[1].trigger_price == Decimal("3200")


@pytest.mark.asyncio
async def test_fetch_open_algo_orders_empty() -> None:
    fake = FakeTransport([])
    client = _make_trading_client(transport=fake)
    algos = await client.fetch_open_algo_orders()
    assert algos == ()


# ---------------------------------------------------------------------------
# place_market_order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_market_order_buy() -> None:
    fake = FakeTransport({
        "symbol": "ETHUSDT", "orderId": 999, "clientOrderId": "cid-mkt",
        "side": "BUY", "type": "MARKET", "status": "FILLED",
    })
    client = _make_trading_client(transport=fake)
    result = await client.place_market_order(
        side="BUY", qty=Decimal("1.0"), reduce_only=False, client_order_id="cid-mkt",
    )
    assert result.ok is True
    assert result.order_id == "999"
    assert result.client_order_id == "cid-mkt"

    # Verify params sent to Binance
    query = fake.requests[0].query_string
    assert "type=MARKET" in query
    assert "side=BUY" in query
    assert "quantity=1" in query
    assert "reduceOnly=false" in query
    assert "newClientOrderId=cid-mkt" in query


@pytest.mark.asyncio
async def test_place_market_order_long_side_maps_to_buy() -> None:
    fake = FakeTransport({"orderId": 1, "clientOrderId": "cid", "status": "NEW"})
    client = _make_trading_client(transport=fake)
    result = await client.place_market_order(
        side="LONG", qty=Decimal("1"), reduce_only=False, client_order_id="cid",
    )
    assert result.ok is True
    assert "side=BUY" in fake.requests[0].query_string


@pytest.mark.asyncio
async def test_place_market_order_long_reduce_only_maps_to_sell() -> None:
    fake = FakeTransport({"orderId": 2, "clientOrderId": "cid", "status": "NEW"})
    client = _make_trading_client(transport=fake)
    result = await client.place_market_order(
        side="LONG", qty=Decimal("1"), reduce_only=True, client_order_id="cid",
    )
    assert result.ok is True
    assert "side=SELL" in fake.requests[0].query_string


@pytest.mark.asyncio
async def test_place_market_order_quantity_is_string() -> None:
    fake = FakeTransport({"orderId": 1, "clientOrderId": "cid", "status": "NEW"})
    client = _make_trading_client(transport=fake)
    await client.place_market_order(
        side="BUY", qty=Decimal("1.5"), reduce_only=False, client_order_id="cid",
    )
    assert "quantity=1.5" in fake.requests[0].query_string


@pytest.mark.asyncio
async def test_place_market_order_empty_client_order_id_raises() -> None:
    client = _make_trading_client(transport=FakeTransport({}))
    with pytest.raises(ValueError, match="client_order_id"):
        await client.place_market_order(
            side="BUY", qty=Decimal("1"), reduce_only=False, client_order_id="",
        )


@pytest.mark.asyncio
async def test_place_market_order_business_rejection_returns_ok_false() -> None:
    fake = FakeTransport(
        {"code": -2019, "msg": "Margin is insufficient."},
        status_code=200,
    )
    client = _make_trading_client(transport=fake)
    result = await client.place_market_order(
        side="BUY", qty=Decimal("1"), reduce_only=False, client_order_id="cid",
    )
    assert result.ok is False
    assert "Margin is insufficient" in result.message


# ---------------------------------------------------------------------------
# place_limit_order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_limit_order() -> None:
    fake = FakeTransport({
        "symbol": "ETHUSDT", "orderId": 100, "clientOrderId": "cid-limit",
        "side": "SELL", "type": "LIMIT", "status": "NEW",
    })
    client = _make_trading_client(transport=fake)
    result = await client.place_limit_order(
        side="SELL", qty=Decimal("1.0"), price=Decimal("3100.00"),
        reduce_only=True, client_order_id="cid-limit",
    )
    assert result.ok is True
    assert result.order_id == "100"

    query = fake.requests[0].query_string
    assert "type=LIMIT" in query
    assert "side=SELL" in query
    assert "price=3100" in query
    assert "quantity=1" in query
    assert "timeInForce=GTC" in query
    assert "reduceOnly=true" in query


@pytest.mark.asyncio
async def test_place_limit_order_short_reduce_only_maps_to_buy() -> None:
    fake = FakeTransport({"orderId": 3, "clientOrderId": "cid", "status": "NEW"})
    client = _make_trading_client(transport=fake)
    await client.place_limit_order(
        side="SHORT", qty=Decimal("1"), price=Decimal("3000"),
        reduce_only=True, client_order_id="cid",
    )
    assert "side=BUY" in fake.requests[0].query_string


@pytest.mark.asyncio
async def test_place_limit_order_price_is_string() -> None:
    fake = FakeTransport({"orderId": 1, "clientOrderId": "cid", "status": "NEW"})
    client = _make_trading_client(transport=fake)
    await client.place_limit_order(
        side="SELL", qty=Decimal("0.5"), price=Decimal("3100.50"),
        reduce_only=True, client_order_id="cid",
    )
    assert "price=3100.5" in fake.requests[0].query_string


# ---------------------------------------------------------------------------
# place_stop_market_order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_stop_market_order() -> None:
    fake = FakeTransport({
        "symbol": "ETHUSDT", "orderId": 200, "clientOrderId": "cid-stop",
        "side": "SELL", "type": "STOP_MARKET", "status": "NEW",
    })
    client = _make_trading_client(transport=fake)
    result = await client.place_stop_market_order(
        side="SELL", qty=Decimal("1.0"), trigger_price=Decimal("2800.00"),
        reduce_only=True, client_order_id="cid-stop",
    )
    assert result.ok is True

    query = fake.requests[0].query_string
    assert "type=STOP_MARKET" in query
    assert "stopPrice=2800" in query
    assert "quantity=1" in query
    assert "reduceOnly=true" in query


@pytest.mark.asyncio
async def test_place_stop_market_order_qty_none_raises() -> None:
    client = _make_trading_client(transport=FakeTransport({}))
    with pytest.raises(ValueError, match="qty=None"):
        await client.place_stop_market_order(
            side="SELL", qty=None, trigger_price=Decimal("2800"),
            reduce_only=True, client_order_id="cid",
        )


@pytest.mark.asyncio
async def test_place_stop_market_order_trigger_price_string() -> None:
    fake = FakeTransport({"orderId": 1, "clientOrderId": "cid", "status": "NEW"})
    client = _make_trading_client(transport=fake)
    await client.place_stop_market_order(
        side="SELL", qty=Decimal("1"), trigger_price=Decimal("2800.50"),
        reduce_only=True, client_order_id="cid",
    )
    assert "stopPrice=2800.5" in fake.requests[0].query_string


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_order_by_order_id() -> None:
    fake = FakeTransport({
        "symbol": "ETHUSDT", "orderId": 999, "clientOrderId": "cid-cancel",
        "status": "CANCELED",
    })
    client = _make_trading_client(transport=fake)
    result = await client.cancel_order(order_id="999")
    assert result.ok is True
    assert result.order_id == "999"

    query = fake.requests[0].query_string
    assert "orderId=999" in query


@pytest.mark.asyncio
async def test_cancel_order_by_client_order_id() -> None:
    fake = FakeTransport({
        "symbol": "ETHUSDT", "orderId": 888, "clientOrderId": "my-cid",
        "status": "CANCELED",
    })
    client = _make_trading_client(transport=fake)
    result = await client.cancel_order(client_order_id="my-cid")
    assert result.ok is True
    assert "origClientOrderId=my-cid" in fake.requests[0].query_string


@pytest.mark.asyncio
async def test_cancel_order_no_args_raises() -> None:
    client = _make_trading_client(transport=FakeTransport({}))
    with pytest.raises(ValueError, match="at least one of"):
        await client.cancel_order()


@pytest.mark.asyncio
async def test_cancel_order_not_found_returns_ok_false() -> None:
    fake = FakeTransport(
        {"code": -2011, "msg": "Unknown order sent."},
        status_code=200,
    )
    client = _make_trading_client(transport=fake)
    result = await client.cancel_order(order_id="999")
    assert result.ok is False
    assert "Unknown order" in result.message


# ---------------------------------------------------------------------------
# cancel_algo_order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_algo_order_by_order_id() -> None:
    fake = FakeTransport({
        "symbol": "ETHUSDT", "orderId": 555, "status": "CANCELED",
    })
    client = _make_trading_client(transport=fake)
    result = await client.cancel_algo_order(order_id="555")
    assert result.ok is True
    assert result.order_id == "555"


@pytest.mark.asyncio
async def test_cancel_algo_order_by_client_order_id() -> None:
    fake = FakeTransport({
        "symbol": "ETHUSDT", "orderId": 666, "clientOrderId": "algo-cid",
        "status": "CANCELED",
    })
    client = _make_trading_client(transport=fake)
    result = await client.cancel_algo_order(client_order_id="algo-cid")
    assert result.ok is True


@pytest.mark.asyncio
async def test_cancel_algo_order_no_args_raises() -> None:
    client = _make_trading_client(transport=FakeTransport({}))
    with pytest.raises(ValueError, match="at least one of"):
        await client.cancel_algo_order()


# ---------------------------------------------------------------------------
# Decimal precision — no float anywhere
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_order_decimal_to_string_no_precision_loss() -> None:
    fake = FakeTransport({"orderId": 1, "clientOrderId": "cid", "status": "NEW"})
    client = _make_trading_client(transport=fake)
    await client.place_market_order(
        side="BUY", qty=Decimal("0.12345678"), reduce_only=False,
        client_order_id="cid",
    )
    assert "quantity=0.12345678" in fake.requests[0].query_string


@pytest.mark.asyncio
async def test_limit_order_decimal_price_no_precision_loss() -> None:
    fake = FakeTransport({"orderId": 1, "clientOrderId": "cid", "status": "NEW"})
    client = _make_trading_client(transport=fake)
    await client.place_limit_order(
        side="SELL", qty=Decimal("1"), price=Decimal("3125.87"),
        reduce_only=True, client_order_id="cid",
    )
    assert "price=3125.87" in fake.requests[0].query_string


# ---------------------------------------------------------------------------
# Symbol, margin_asset on all endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_balance_uses_correct_endpoint() -> None:
    fake = FakeTransport([{"asset": "USDT", "balance": "1000", "availableBalance": "900"}])
    client = _make_trading_client(transport=fake)
    await client.fetch_balance()
    assert fake.requests[0].path == "/fapi/v2/balance"


@pytest.mark.asyncio
async def test_fetch_position_uses_correct_endpoint_and_symbol() -> None:
    fake = FakeTransport([{"symbol": "ETHUSDT", "positionAmt": "0"}])
    client = _make_trading_client(transport=fake)
    await client.fetch_position()
    assert fake.requests[0].path == "/fapi/v2/positionRisk"
    assert "symbol=ETHUSDT" in fake.requests[0].query_string
