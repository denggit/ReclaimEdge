#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_okx_trading_client_extended_reads.py
@Description: Functional tests for OkxTradingClient extended read methods:
              fetch_order_status, fetch_open_algo_orders, configure_instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

import pytest

from src.execution.okx_trading_client import OkxTradingClient
from src.execution.trading_client_port import (
    AlgoOrderSnapshot,
    OrderStatusSnapshot,
)


# ======================================================================
# Fake Trader
# ======================================================================


@dataclass(frozen=True)
class FakePositionSnapshot:
    side: Optional[str]
    contracts: Decimal
    avg_entry_price: float
    eth_qty: float
    raw_pos: Decimal

    @property
    def has_position(self) -> bool:
        return self.side is not None and self.contracts > 0


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    td_mode = "isolated"
    pos_side_mode = "net"
    leverage = "50"

    def __init__(self) -> None:
        self._equity: float = 1234.56
        self._position: FakePositionSnapshot = FakePositionSnapshot(
            side="LONG",
            contracts=Decimal("0.5"),
            avg_entry_price=3100.0,
            eth_qty=0.05,
            raw_pos=Decimal("0.5"),
        )
        self._algo_orders: list[dict[str, Any]] = []
        self._request_responses: list[dict[str, Any]] = []
        self._request_calls: list[tuple[str, str, Any]] = []
        self._set_leverage_called = False
        self._client = self  # OkxTradingClient uses self._trader._client.request()

    def set_request_responses(self, responses: list[dict[str, Any]]) -> None:
        self._request_responses = list(responses)

    def set_algo_orders(self, orders: list[dict[str, Any]]) -> None:
        self._algo_orders = list(orders)

    async def fetch_usdt_equity(self) -> float:
        return self._equity

    async def fetch_position_snapshot(self) -> FakePositionSnapshot:
        return self._position

    async def fetch_pending_algo_orders(self) -> list[dict[str, Any]]:
        return list(self._algo_orders)

    async def request(self, method: str, endpoint: str, payload: Any = None) -> dict[str, Any]:
        self._request_calls.append((method, endpoint, payload))
        if self._request_responses:
            return self._request_responses.pop(0)
        return {"code": "0", "msg": "", "data": []}

    async def set_leverage(self) -> None:
        self._set_leverage_called = True

    @staticmethod
    def extract_order_id(res: dict[str, Any]) -> str:
        data = res.get("data", [])
        if not data or not data[0].get("ordId"):
            raise RuntimeError(f"Missing ordId in response: {res}")
        return str(data[0]["ordId"])

    @staticmethod
    def extract_algo_id(res: dict[str, Any]) -> str:
        data = res.get("data", [])
        if not data:
            raise RuntimeError(f"Missing algoId in response: {res}")
        algo_id = data[0].get("algoId") or data[0].get("ordId")
        if not algo_id:
            raise RuntimeError(f"Missing algoId in response: {res}")
        return str(algo_id)

    @staticmethod
    def decimal_to_str(value: Decimal | str | int | float) -> str:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        import math
        if not math.isfinite(price):
            raise RuntimeError(f"Invalid price: {price}")
        return f"{price:.2f}"


# ======================================================================
# Tests: configure_instrument
# ======================================================================


class TestConfigureInstrument:
    @pytest.mark.asyncio
    async def test_calls_okx_set_leverage_rest(self) -> None:
        """configure_instrument() now calls OKX REST directly via _client."""
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": []},
            {"code": "0", "msg": "", "data": []},
        ])
        client = OkxTradingClient(trader)

        await client.configure_instrument()

        # Verify REST calls were made (2 calls: 1 per posSide mode)
        assert len(trader._request_calls) >= 1
        method, endpoint, body = trader._request_calls[0]
        assert method == "POST"
        assert endpoint == "/api/v5/account/set-leverage"


# ======================================================================
# Tests: fetch_order_status
# ======================================================================


class TestFetchOrderStatus:
    @pytest.mark.asyncio
    async def test_status_open(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [{
                    "ordId": "ord-open-1",
                    "clOrdId": "cid-open-1",
                    "state": "live",
                    "accFillSz": "0.1",
                    "avgPx": "3100.50",
                }],
            },
        ])
        client = OkxTradingClient(trader)

        result = await client.fetch_order_status(order_id="ord-open-1")

        assert isinstance(result, OrderStatusSnapshot)
        assert result.order_id == "ord-open-1"
        assert result.client_order_id == "cid-open-1"
        assert result.status == "OPEN"
        assert result.filled_qty == Decimal("0.1")
        assert result.avg_fill_price == Decimal("3100.50")

    @pytest.mark.asyncio
    async def test_status_partially_filled(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [{
                    "ordId": "ord-partial",
                    "state": "partially_filled",
                    "accFillSz": "0.3",
                    "avgPx": "3050.00",
                }],
            },
        ])
        client = OkxTradingClient(trader)

        result = await client.fetch_order_status(order_id="ord-partial")

        assert result.status == "OPEN"

    @pytest.mark.asyncio
    async def test_status_filled(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [{
                    "ordId": "ord-filled",
                    "state": "filled",
                    "accFillSz": "1.0",
                    "avgPx": "3200.00",
                }],
            },
        ])
        client = OkxTradingClient(trader)

        result = await client.fetch_order_status(order_id="ord-filled")

        assert result.status == "FILLED"
        assert result.filled_qty == Decimal("1.0")
        assert result.avg_fill_price == Decimal("3200.00")

    @pytest.mark.asyncio
    async def test_status_canceled(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [{
                    "ordId": "ord-cancel",
                    "state": "canceled",
                    "accFillSz": "0",
                    "avgPx": "",
                }],
            },
        ])
        client = OkxTradingClient(trader)

        result = await client.fetch_order_status(order_id="ord-cancel")

        assert result.status == "CANCELED"
        assert result.filled_qty == Decimal("0")

    @pytest.mark.asyncio
    async def test_status_cancelled_uk_spelling(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [{
                    "ordId": "ord-cancelled",
                    "state": "cancelled",
                    "accFillSz": "0",
                    "avgPx": "",
                }],
            },
        ])
        client = OkxTradingClient(trader)

        result = await client.fetch_order_status(order_id="ord-cancelled")

        assert result.status == "CANCELED"

    @pytest.mark.asyncio
    async def test_status_not_found(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": []},
        ])
        client = OkxTradingClient(trader)

        result = await client.fetch_order_status(order_id="ord-missing")

        assert result.status == "NOT_FOUND"
        assert result.filled_qty is None
        assert result.avg_fill_price is None

    @pytest.mark.asyncio
    async def test_status_exception_returns_unknown(self) -> None:
        trader = FakeTrader()

        async def failing_request(method: str, endpoint: str, payload: Any = None) -> dict[str, Any]:
            raise RuntimeError("network failure")

        trader.request = failing_request  # type: ignore[method-assign]
        client = OkxTradingClient(trader)

        result = await client.fetch_order_status(order_id="ord-err")

        assert result.status == "UNKNOWN"
        assert result.order_id == "ord-err"

    @pytest.mark.asyncio
    async def test_status_unknown_state(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [{
                    "ordId": "ord-weird",
                    "state": "weird_state",
                    "accFillSz": "0",
                    "avgPx": "",
                }],
            },
        ])
        client = OkxTradingClient(trader)

        result = await client.fetch_order_status(order_id="ord-weird")

        assert result.status == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_by_client_order_id(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [{
                    "ordId": "ord-by-cid",
                    "clOrdId": "my-cid",
                    "state": "live",
                    "accFillSz": "0",
                    "avgPx": "",
                }],
            },
        ])
        client = OkxTradingClient(trader)

        result = await client.fetch_order_status(client_order_id="my-cid")

        assert result.status == "OPEN"
        assert result.order_id == "ord-by-cid"
        # Verify the endpoint used clOrdId
        _method, endpoint, _payload = trader._request_calls[0]
        assert "clOrdId=my-cid" in endpoint

    @pytest.mark.asyncio
    async def test_no_identifier_raises(self) -> None:
        trader = FakeTrader()
        client = OkxTradingClient(trader)

        with pytest.raises(ValueError, match="at least one of order_id or client_order_id"):
            await client.fetch_order_status()


# ======================================================================
# Tests: fetch_open_algo_orders
# ======================================================================


class TestFetchOpenAlgoOrders:
    @pytest.mark.asyncio
    async def test_parses_algo_order_basic(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "algoId": "algo-001",
                        "clOrdId": "cid-algo-001",
                        "side": "sell",
                        "sz": "1.5",
                        "slTriggerPx": "2900.00",
                        "state": "live",
                    },
                ],
            },
        ])
        client = OkxTradingClient(trader)

        results = await client.fetch_open_algo_orders()

        assert isinstance(results, tuple)
        assert len(results) == 1
        snap = results[0]
        assert isinstance(snap, AlgoOrderSnapshot)
        assert snap.order_id == "algo-001"
        assert snap.client_order_id == "cid-algo-001"
        assert snap.side == "sell"
        assert snap.qty == Decimal("1.5")
        assert snap.trigger_price == Decimal("2900.00")
        assert snap.status == "live"

    @pytest.mark.asyncio
    async def test_parses_algo_order_with_ord_id_fallback(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "ordId": "ord-fallback",
                        "side": "buy",
                        "sz": "2.0",
                        "triggerPx": "3200.00",
                    },
                ],
            },
        ])
        client = OkxTradingClient(trader)

        results = await client.fetch_open_algo_orders()

        assert len(results) == 1
        snap = results[0]
        assert snap.order_id == "ord-fallback"
        assert snap.side == "buy"
        assert snap.qty == Decimal("2.0")
        assert snap.trigger_price == Decimal("3200.00")
        assert snap.status == "OPEN"  # default when state is missing

    @pytest.mark.asyncio
    async def test_parses_multiple_algo_orders(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "algoId": "algo-001",
                        "side": "sell",
                        "sz": "1.0",
                        "slTriggerPx": "2900.00",
                        "state": "live",
                    },
                    {
                        "algoId": "algo-002",
                        "side": "buy",
                        "sz": "2.0",
                        "slTriggerPx": "3100.00",
                        "state": "live",
                    },
                ],
            },
        ])
        client = OkxTradingClient(trader)

        results = await client.fetch_open_algo_orders()

        assert len(results) == 2
        assert results[0].order_id == "algo-001"
        assert results[1].order_id == "algo-002"
        assert results[0].trigger_price == Decimal("2900.00")
        assert results[1].trigger_price == Decimal("3100.00")

    @pytest.mark.asyncio
    async def test_empty_algo_orders(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": []},
        ])
        client = OkxTradingClient(trader)

        results = await client.fetch_open_algo_orders()

        assert results == ()

    @pytest.mark.asyncio
    async def test_missing_optional_fields(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "algoId": "",
                        "side": "",
                        "sz": "",
                        "slTriggerPx": None,
                        "triggerPx": None,
                    },
                ],
            },
        ])
        client = OkxTradingClient(trader)

        results = await client.fetch_open_algo_orders()

        assert len(results) == 1
        snap = results[0]
        assert snap.order_id is None  # empty string → None
        assert snap.side is None
        assert snap.qty is None
        assert snap.trigger_price is None

    @pytest.mark.asyncio
    async def test_invalid_sz_returns_none(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "algoId": "algo-bad-sz",
                        "side": "sell",
                        "sz": "not_a_number",
                    },
                ],
            },
        ])
        client = OkxTradingClient(trader)

        results = await client.fetch_open_algo_orders()

        assert len(results) == 1
        assert results[0].qty is None
