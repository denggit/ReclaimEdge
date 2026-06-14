#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_algo_orders.py
@Description: Tests for BinanceAlgoOrderClient — Algo Order API wrapper.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.binance.algo_orders import BinanceAlgoOrderClient
from src.exchanges.binance.signing import (
    BINANCE_USDM_ALGO_ORDER_PATH,
    BINANCE_USDM_BASE_URL,
    BINANCE_USDM_OPEN_ALGO_ORDERS_PATH,
    BinanceSignedRequest,
)
from src.exchanges.binance.transport import BinanceTransportResponse
from src.exchanges.errors import ExchangeError
from src.exchanges.models import ExchangeName


# ======================================================================
# Helpers
# ======================================================================


class FakeTransport:
    """In-memory transport that records requests and returns canned responses."""

    def __init__(self) -> None:
        self.requests: list[BinanceSignedRequest] = []
        self.next_response: BinanceTransportResponse | None = None

    async def send(self, request: BinanceSignedRequest) -> BinanceTransportResponse:
        self.requests.append(request)
        if self.next_response is not None:
            return self.next_response
        return BinanceTransportResponse(status_code=200, payload={})


def _make_client(transport: FakeTransport | None = None) -> BinanceAlgoOrderClient:
    if transport is None:
        transport = FakeTransport()
    return BinanceAlgoOrderClient(
        api_key="test_key",
        api_secret="test_secret",
        transport=transport,
    )


# ======================================================================
# Construction
# ======================================================================


class TestConstruction:
    """BinanceAlgoOrderClient construction tests."""

    def test_no_transport_raises_on_use(self) -> None:
        """Without transport, methods raise UnsupportedOperation."""
        client = BinanceAlgoOrderClient()
        with pytest.raises(ExchangeError, match="transport not wired"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                client.place_stop_loss(
                    symbol="ETHUSDT",
                    side="SELL",
                    quantity=Decimal("0.1"),
                    trigger_price=Decimal("3000"),
                    client_algo_id="RE_MAIN_sl",
                )
            )


# ======================================================================
# place_stop_loss
# ======================================================================


class TestPlaceStopLoss:
    """place_stop_loss sends correct params to Algo Order API."""

    def test_place_stop_loss_path_and_params(self) -> None:
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=200,
            payload={"algoId": 12345, "clientAlgoId": "RE_MAIN_sl"},
        )
        client = _make_client(transport)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            client.place_stop_loss(
                symbol="ETHUSDT",
                side="SELL",
                quantity=Decimal("0.1"),
                trigger_price=Decimal("3000"),
                client_algo_id="RE_MAIN_sl",
            )
        )

        assert result.ok
        # Fix 6: order_id is now clientAlgoId (unified for cancel)
        assert result.order_id == "RE_MAIN_sl"
        assert result.client_order_id == "RE_MAIN_sl"
        assert result.exchange == ExchangeName.BINANCE

        req = transport.requests[0]
        assert req.path == BINANCE_USDM_ALGO_ORDER_PATH
        assert req.method == "POST"

        params = dict(req.params)
        assert params.get("algoType") == "CONDITIONAL"
        assert params.get("type") == "STOP_MARKET"
        assert params.get("reduceOnly") == "true"
        assert params.get("symbol") == "ETHUSDT"
        assert params.get("side") == "SELL"
        assert params.get("clientAlgoId") == "RE_MAIN_sl"

    def test_place_stop_loss_no_position_side(self) -> None:
        """Raw params must NOT contain positionSide."""
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=200,
            payload={"algoId": 1, "clientAlgoId": "RE_MAIN_sl"},
        )
        client = _make_client(transport)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_stop_loss(
                symbol="ETHUSDT",
                side="BUY",
                quantity=Decimal("0.2"),
                trigger_price=Decimal("3500"),
                client_algo_id="RE_MAIN_sl_close_short",
            )
        )

        req = transport.requests[0]
        params = dict(req.params)
        assert "positionSide" not in params

    def test_place_stop_loss_client_algo_id_prefix(self) -> None:
        """clientAlgoId must follow RE_MAIN_ prefix convention."""
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=200,
            payload={"algoId": 2, "clientAlgoId": "RE_MAIN_sl_test"},
        )
        client = _make_client(transport)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            client.place_stop_loss(
                symbol="ETHUSDT",
                side="SELL",
                quantity=Decimal("0.1"),
                trigger_price=Decimal("3000"),
                client_algo_id="RE_MAIN_sl_test",
            )
        )

        assert result.client_order_id == "RE_MAIN_sl_test"

    def test_place_stop_loss_reduce_only_enforced(self) -> None:
        """reduceOnly must be 'true' in all SL orders."""
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=200,
            payload={"algoId": 3},
        )
        client = _make_client(transport)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_stop_loss(
                symbol="ETHUSDT",
                side="SELL",
                quantity=Decimal("0.1"),
                trigger_price=Decimal("3000"),
                client_algo_id="RE_MAIN_sl",
            )
        )

        req = transport.requests[0]
        assert dict(req.params).get("reduceOnly") == "true"

    def test_place_stop_loss_http_error(self) -> None:
        """HTTP error responses raise ExchangeError."""
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=400,
            payload={"code": -1102, "msg": "Mandatory parameter missing"},
        )
        client = _make_client(transport)

        import asyncio
        with pytest.raises(ExchangeError):
            asyncio.get_event_loop().run_until_complete(
                client.place_stop_loss(
                    symbol="ETHUSDT",
                    side="SELL",
                    quantity=Decimal("0.1"),
                    trigger_price=Decimal("3000"),
                    client_algo_id="RE_MAIN_sl",
                )
            )


# ======================================================================
# cancel_algo_order
# ======================================================================


class TestCancelAlgoOrder:
    """cancel_algo_order sends correct DELETE request."""

    def test_cancel_by_client_algo_id(self) -> None:
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=200,
            payload={"code": 200, "msg": "OK"},
        )
        client = _make_client(transport)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            client.cancel_algo_order(
                symbol="ETHUSDT",
                client_algo_id="RE_MAIN_sl",
            )
        )

        assert result.ok
        req = transport.requests[0]
        assert req.path == BINANCE_USDM_ALGO_ORDER_PATH
        assert req.method == "DELETE"
        assert dict(req.params).get("clientAlgoId") == "RE_MAIN_sl"

    def test_cancel_http_error_returns_not_ok(self) -> None:
        """HTTP error on cancel returns ok=False, does NOT raise."""
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=500,
            payload={},
        )
        client = _make_client(transport)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            client.cancel_algo_order(
                symbol="ETHUSDT",
                client_algo_id="RE_MAIN_sl",
            )
        )

        assert not result.ok

    def test_cancel_empty_client_id_raises(self) -> None:
        transport = FakeTransport()
        client = _make_client(transport)

        import asyncio
        with pytest.raises(ValueError, match="client_algo_id"):
            asyncio.get_event_loop().run_until_complete(
                client.cancel_algo_order(symbol="ETHUSDT", client_algo_id="")
            )


# ======================================================================
# fetch_open_algo_orders
# ======================================================================


class TestFetchOpenAlgoOrders:
    """fetch_open_algo_orders uses GET /openAlgoOrders."""

    def test_fetch_returns_list(self) -> None:
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=200,
            payload=[
                {"algoId": 1, "clientAlgoId": "RE_MAIN_sl", "orderType": "STOP_MARKET",
                 "side": "SELL", "quantity": "0.1", "triggerPrice": "3000"},
            ],
        )
        client = _make_client(transport)

        import asyncio
        orders = asyncio.get_event_loop().run_until_complete(
            client.fetch_open_algo_orders(symbol="ETHUSDT")
        )

        assert len(orders) == 1
        assert orders[0]["clientAlgoId"] == "RE_MAIN_sl"

        req = transport.requests[0]
        assert req.path == BINANCE_USDM_OPEN_ALGO_ORDERS_PATH
        assert req.method == "GET"

    def test_fetch_empty_when_empty_list(self) -> None:
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=200, payload=[]
        )
        client = _make_client(transport)

        import asyncio
        orders = asyncio.get_event_loop().run_until_complete(
            client.fetch_open_algo_orders(symbol="ETHUSDT")
        )
        assert orders == []

    def test_fetch_http_error_raises(self) -> None:
        """Fix 8: HTTP errors now raise ExchangeError instead of returning []."""
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=500, payload={}
        )
        client = _make_client(transport)

        import asyncio
        from src.exchanges.errors import ExchangeError
        with pytest.raises(ExchangeError):
            asyncio.get_event_loop().run_until_complete(
                client.fetch_open_algo_orders(symbol="ETHUSDT")
            )

    def test_fetch_payload_not_list_raises(self) -> None:
        """Fix 8: non-list payload raises ExchangeError."""
        transport = FakeTransport()
        transport.next_response = BinanceTransportResponse(
            status_code=200, payload={"not": "a list"}
        )
        client = _make_client(transport)

        import asyncio
        from src.exchanges.errors import ExchangeError
        with pytest.raises(ExchangeError):
            asyncio.get_event_loop().run_until_complete(
                client.fetch_open_algo_orders(symbol="ETHUSDT")
            )


# ======================================================================
# No positionSide in any params
# ======================================================================


class TestNoPositionSide:
    """All algo order operations must NOT include positionSide."""

    def test_no_position_side_in_any_request(self) -> None:
        transport = FakeTransport()

        # place_stop_loss
        transport.next_response = BinanceTransportResponse(
            status_code=200,
            payload={"algoId": 1, "clientAlgoId": "RE_MAIN_sl"},
        )
        client = _make_client(transport)

        import asyncio
        loop = asyncio.get_event_loop()

        loop.run_until_complete(
            client.place_stop_loss(
                symbol="ETHUSDT",
                side="SELL",
                quantity=Decimal("0.1"),
                trigger_price=Decimal("3000"),
                client_algo_id="RE_MAIN_sl",
            )
        )
        assert "positionSide" not in dict(transport.requests[-1].params)

        # cancel
        transport.next_response = BinanceTransportResponse(
            status_code=200, payload={}
        )
        loop.run_until_complete(
            client.cancel_algo_order(
                symbol="ETHUSDT",
                client_algo_id="RE_MAIN_sl",
            )
        )
        assert "positionSide" not in dict(transport.requests[-1].params)

        # fetch
        transport.next_response = BinanceTransportResponse(
            status_code=200, payload=[]
        )
        loop.run_until_complete(
            client.fetch_open_algo_orders(symbol="ETHUSDT")
        )
        assert "positionSide" not in dict(transport.requests[-1].params)
