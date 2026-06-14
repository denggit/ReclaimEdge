#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_okx_trading_client.py
@Description: Functional tests for OkxTradingClient using a FakeTrader.

No real API calls.  No env reads.  No production wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

import pytest

from src.execution.okx_trading_client import OkxTradingClient, _normalise_client_order_id, _normalise_position_side
from src.execution.trading_client_port import (
    BalanceSnapshot,
    CancelResult,
    OrderResult,
    OrderSnapshot,
    PositionSnapshot,
)
from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)


# ======================================================================
# Fake Trader
# ======================================================================


@dataclass(frozen=True)
class FakePositionSnapshot:
    """Mimics Trader.PositionSnapshot for the FakeTrader."""
    side: Optional[str]
    contracts: Decimal
    avg_entry_price: float
    eth_qty: float
    raw_pos: Decimal

    @property
    def has_position(self) -> bool:
        return self.side is not None and self.contracts > 0


class FakeTrader:
    """A fake Trader that records calls and returns canned responses.

    Does NOT call any real API.  Does NOT read env.
    Can be passed as ``private_client`` to OkxTradingClient.
    """

    symbol = "ETH-USDT-SWAP"
    td_mode = "isolated"
    pos_side_mode = "net"
    leverage = "50"
    contract_multiplier = Decimal("0.1")

    def __init__(self) -> None:
        self._equity: float = 1234.56
        self._position: FakePositionSnapshot = FakePositionSnapshot(
            side="LONG",
            contracts=Decimal("0.5"),
            avg_entry_price=3100.0,
            eth_qty=0.05,
            raw_pos=Decimal("0.5"),
        )
        self._open_orders: tuple[BrokerOrder, ...] = ()
        self._algo_orders: list[dict[str, Any]] = []
        self._request_responses: list[dict[str, Any]] = []
        self._request_calls: list[tuple[str, str, Any]] = []
        self._execute_intent_called: bool = False

    # -- canned response setters ----------------------------------------

    def set_equity(self, value: float) -> None:
        self._equity = value

    def set_position(self, snap: FakePositionSnapshot) -> None:
        self._position = snap

    def set_open_orders(self, orders: tuple[BrokerOrder, ...]) -> None:
        self._open_orders = orders

    def set_request_responses(self, responses: list[dict[str, Any]]) -> None:
        self._request_responses = list(responses)

    # -- Trader methods -------------------------------------------------

    async def fetch_usdt_equity(self) -> float:
        return self._equity

    async def fetch_position_snapshot(self) -> FakePositionSnapshot:
        return self._position

    async def fetch_broker_open_orders(self) -> tuple[BrokerOrder, ...]:
        return self._open_orders

    async def request(self, method: str, endpoint: str, payload: Any = None) -> dict[str, Any]:
        self._request_calls.append((method, endpoint, payload))
        if self._request_responses:
            return self._request_responses.pop(0)

        # Handle balance queries with canned equity
        if "/api/v5/account/balance" in endpoint:
            return {
                "code": "0",
                "msg": "",
                "data": [{"totalEq": str(self._equity), "details": [
                    {"ccy": "USDT", "eq": str(self._equity)}]}],
            }

        # Handle position queries with canned position
        if "/api/v5/account/positions" in endpoint:
            if self._position.side is not None and self._position.contracts > 0:
                return {
                    "code": "0",
                    "msg": "",
                    "data": [{
                        "instId": self.symbol,
                        "pos": str(self._position.raw_pos),
                        "avgPx": str(self._position.avg_entry_price),
                    }],
                }
            return {"code": "0", "msg": "", "data": []}

        # Handle open orders queries
        if "/api/v5/trade/orders-pending" in endpoint:
            return {"code": "0", "msg": "", "data": []}

        # Handle algo orders queries
        if "/api/v5/trade/orders-algo-pending" in endpoint:
            return {"code": "0", "msg": "", "data": self._algo_orders}

        # Default success response
        return {
            "code": "0",
            "msg": "",
            "data": [{"ordId": "fake-order-001", "algoId": "fake-algo-001"}],
        }

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
# Helpers
# ======================================================================


def _make_broker_order(
    order_id: str = "ord-1",
    client_order_id: str = "cid-1",
    side: BrokerOrderSide = BrokerOrderSide.BUY,
    quantity: Decimal = Decimal("0.5"),
    price: Decimal | None = None,
    trigger_price: Decimal | None = None,
    reduce_only: bool = False,
) -> BrokerOrder:
    return BrokerOrder(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        order_id=order_id,
        client_order_id=client_order_id,
        side=side,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.LIMIT,
        status=BrokerOrderStatus.OPEN,
        price=price,
        quantity=quantity,
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        reduce_only=reduce_only,
        trigger_price=trigger_price,
        raw={"ordId": order_id, "clOrdId": client_order_id},
    )


def _make_client(trader: FakeTrader) -> OkxTradingClient:
    """Create an OkxTradingClient with the FakeTrader acting as private_client."""
    return OkxTradingClient(trader, private_client=trader)


# ======================================================================
# Tests: _normalise_position_side
# ======================================================================


class TestNormalisePositionSide:
    def test_long(self) -> None:
        assert _normalise_position_side("LONG") == "LONG"

    def test_short(self) -> None:
        assert _normalise_position_side("SHORT") == "SHORT"

    def test_lowercase_long(self) -> None:
        assert _normalise_position_side("long") == "LONG"

    def test_lowercase_short(self) -> None:
        assert _normalise_position_side("short") == "SHORT"

    def test_whitespace(self) -> None:
        assert _normalise_position_side("  LONG  ") == "LONG"

    def test_invalid_side_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported position side"):
            _normalise_position_side("BUY")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            _normalise_position_side("")


# ======================================================================
# Tests: fetch_balance
# ======================================================================


class TestFetchBalance:
    @pytest.mark.asyncio
    async def test_maps_equity_to_balance_snapshot(self) -> None:
        trader = FakeTrader()
        trader.set_equity(999.88)
        client = _make_client(trader)

        result = await client.fetch_balance()

        assert isinstance(result, BalanceSnapshot)
        assert result.asset == "USDT"
        assert result.total == Decimal("999.88")
        assert result.available is None
        assert result.raw == {"account_equity_usdt": 999.88}

    @pytest.mark.asyncio
    async def test_zero_equity(self) -> None:
        trader = FakeTrader()
        trader.set_equity(0.0)
        client = _make_client(trader)

        result = await client.fetch_balance()

        assert result.total == Decimal("0")
        assert result.raw == {"account_equity_usdt": 0.0}


# ======================================================================
# Tests: fetch_position
# ======================================================================


class TestFetchPosition:
    @pytest.mark.asyncio
    async def test_maps_contracts_and_avg(self) -> None:
        trader = FakeTrader()
        trader.set_position(FakePositionSnapshot(
            side="LONG",
            contracts=Decimal("1.5"),
            avg_entry_price=3200.0,
            eth_qty=0.15,
            raw_pos=Decimal("1.5"),
        ))
        client = _make_client(trader)

        result = await client.fetch_position()

        assert isinstance(result, PositionSnapshot)
        assert result.side == "LONG"
        assert result.qty == Decimal("1.5")
        assert result.avg_entry_price == Decimal("3200.0")
        assert result.raw["contracts"] == "1.5"
        assert result.raw["eth_qty"] == 0.15
        assert result.raw["raw_pos"] == "1.5"

    @pytest.mark.asyncio
    async def test_no_position(self) -> None:
        trader = FakeTrader()
        trader.set_position(FakePositionSnapshot(
            side=None,
            contracts=Decimal("0"),
            avg_entry_price=0.0,
            eth_qty=0.0,
            raw_pos=Decimal("0"),
        ))
        client = _make_client(trader)

        result = await client.fetch_position()

        assert result.side is None
        assert result.qty == Decimal("0")
        assert result.avg_entry_price is None  # 0.0 is falsy → None
        assert result.has_position is False

    @pytest.mark.asyncio
    async def test_short_position(self) -> None:
        trader = FakeTrader()
        trader.set_position(FakePositionSnapshot(
            side="SHORT",
            contracts=Decimal("2.0"),
            avg_entry_price=3000.0,
            eth_qty=0.2,
            raw_pos=Decimal("-2.0"),
        ))
        client = _make_client(trader)

        result = await client.fetch_position()

        assert result.side == "SHORT"
        assert result.qty == Decimal("2.0")

    @pytest.mark.asyncio
    async def test_avg_entry_price_none(self) -> None:
        trader = FakeTrader()
        trader.set_position(FakePositionSnapshot(
            side="LONG",
            contracts=Decimal("0.5"),
            avg_entry_price=0.0,
            eth_qty=0.05,
            raw_pos=Decimal("0.5"),
        ))
        client = _make_client(trader)

        result = await client.fetch_position()

        # avg_entry_price is 0.0 (falsy), so it maps to None
        assert result.avg_entry_price is None


# ======================================================================
# Tests: fetch_open_orders (now calls OKX REST directly)
# ======================================================================


class TestFetchOpenOrders:
    @pytest.mark.asyncio
    async def test_maps_raw_orders_to_snapshots(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "instId": "ETH-USDT-SWAP",
                        "ordId": "ord-1",
                        "clOrdId": "cid-1",
                        "side": "buy",
                        "sz": "0.5",
                        "reduceOnly": "false",
                    },
                    {
                        "instId": "ETH-USDT-SWAP",
                        "ordId": "ord-2",
                        "clOrdId": "cid-2",
                        "side": "sell",
                        "sz": "1",
                        "px": "3200.00",
                        "reduceOnly": "true",
                    },
                ],
            },
        ])
        client = _make_client(trader)

        results = await client.fetch_open_orders()

        assert len(results) == 2
        assert all(isinstance(o, OrderSnapshot) for o in results)

        assert results[0].order_id == "ord-1"
        assert results[0].client_order_id == "cid-1"
        assert results[0].side == "BUY"
        assert results[0].qty == Decimal("0.5")
        assert results[0].reduce_only is False

        assert results[1].order_id == "ord-2"
        assert results[1].reduce_only is True
        assert results[1].price == Decimal("3200.00")

    @pytest.mark.asyncio
    async def test_empty_orders(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": []},
        ])
        client = _make_client(trader)

        results = await client.fetch_open_orders()

        assert results == []

    @pytest.mark.asyncio
    async def test_filters_by_inst_id(self) -> None:
        """Orders with a different instId are filtered out."""
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "instId": "BTC-USDT-SWAP",
                        "ordId": "btc-ord",
                        "side": "buy",
                        "sz": "1",
                        "reduceOnly": "false",
                    },
                    {
                        "instId": "ETH-USDT-SWAP",
                        "ordId": "eth-ord",
                        "side": "sell",
                        "sz": "0.5",
                        "reduceOnly": "true",
                    },
                ],
            },
        ])
        client = _make_client(trader)

        results = await client.fetch_open_orders()

        assert len(results) == 1
        assert results[0].order_id == "eth-ord"

    @pytest.mark.asyncio
    async def test_order_quantity_empty_returns_zero(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "instId": "ETH-USDT-SWAP",
                        "ordId": "ord-x",
                        "side": "buy",
                        "sz": "",
                        "reduceOnly": "false",
                    },
                ],
            },
        ])
        client = _make_client(trader)

        results = await client.fetch_open_orders()

        assert results[0].qty == Decimal("0")


# ======================================================================
# Tests: place_market_order
# ======================================================================


class TestPlaceMarketOrder:
    @pytest.mark.asyncio
    async def test_open_long_uses_buy_side(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "entry-001"}]},
        ])
        client = _make_client(trader)

        result = await client.place_market_order(
            side="LONG",
            qty=Decimal("0.5"),
            reduce_only=False,
            client_order_id="my-clord-1",
        )

        assert result.ok is True
        assert result.order_id == "entry-001"
        assert result.client_order_id == "my-clord-1"

        # Verify the request body
        assert len(trader._request_calls) == 1
        _method, endpoint, body = trader._request_calls[0]
        assert endpoint == "/api/v5/trade/order"
        assert body["ordType"] == "market"
        assert body["side"] == "buy"
        assert body["sz"] == "0.5"
        assert body["clOrdId"] == "my-clord-1"
        assert body.get("reduceOnly") != "true"

    @pytest.mark.asyncio
    async def test_open_short_uses_sell_side(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "short-001"}]},
        ])
        client = _make_client(trader)

        result = await client.place_market_order(
            side="SHORT",
            qty=Decimal("1.0"),
            reduce_only=False,
            client_order_id="cid-short",
        )

        assert result.ok is True
        assert result.order_id == "short-001"

        _method, _endpoint, body = trader._request_calls[0]
        assert body["side"] == "sell"
        assert body["sz"] == "1"  # decimal_to_str normalises "1.0" → "1"

    @pytest.mark.asyncio
    async def test_reduce_only_long_uses_sell_side(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "reduce-001"}]},
        ])
        client = _make_client(trader)

        result = await client.place_market_order(
            side="LONG",
            qty=Decimal("0.5"),
            reduce_only=True,
            client_order_id="cid-reduce",
        )

        assert result.ok is True

        _method, _endpoint, body = trader._request_calls[0]
        assert body["side"] == "sell"
        assert body["reduceOnly"] == "true"
        assert body["clOrdId"] == "cid-reduce"

    @pytest.mark.asyncio
    async def test_reduce_only_short_uses_buy_side(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "reduce-s-001"}]},
        ])
        client = _make_client(trader)

        result = await client.place_market_order(
            side="SHORT",
            qty=Decimal("1.0"),
            reduce_only=True,
            client_order_id="cid-reduce-s",
        )

        assert result.ok is True

        _method, _endpoint, body = trader._request_calls[0]
        assert body["side"] == "buy"
        assert body["reduceOnly"] == "true"

    @pytest.mark.asyncio
    async def test_invalid_side_raises(self) -> None:
        trader = FakeTrader()
        client = _make_client(trader)

        with pytest.raises(ValueError, match="Unsupported position side"):
            await client.place_market_order(
                side="BUY",
                qty=Decimal("1"),
                reduce_only=False,
                client_order_id="x",
            )

    @pytest.mark.asyncio
    async def test_does_not_call_execute_intent(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "no-intent-1"}]},
        ])
        client = _make_client(trader)

        await client.place_market_order(
            side="LONG",
            qty=Decimal("0.5"),
            reduce_only=False,
            client_order_id="x",
        )

        assert not trader._execute_intent_called


# ======================================================================
# Tests: place_limit_order
# ======================================================================


class TestPlaceLimitOrder:
    @pytest.mark.asyncio
    async def test_reduce_only_places_tp_limit(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "tp-001"}]},
        ])
        client = _make_client(trader)

        result = await client.place_limit_order(
            side="LONG",
            qty=Decimal("0.5"),
            price=Decimal("3200.50"),
            reduce_only=True,
            client_order_id="tp-cid-1",
        )

        assert result.ok is True
        assert result.order_id == "tp-001"
        assert result.client_order_id == "tp-cid-1"

        _method, endpoint, body = trader._request_calls[0]
        assert endpoint == "/api/v5/trade/order"
        assert body["ordType"] == "limit"
        assert body["side"] == "sell"  # LONG close = sell
        assert body["reduceOnly"] == "true"
        assert body["px"] == "3200.50"
        assert body["clOrdId"] == "tp-cid-1"

    @pytest.mark.asyncio
    async def test_reduce_only_false_raises(self) -> None:
        trader = FakeTrader()
        client = _make_client(trader)

        with pytest.raises(ValueError, match="reduce_only=True only"):
            await client.place_limit_order(
                side="LONG",
                qty=Decimal("0.5"),
                price=Decimal("3200"),
                reduce_only=False,
                client_order_id="x",
            )

    @pytest.mark.asyncio
    async def test_short_reduce_only_uses_buy_side(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "tp-short"}]},
        ])
        client = _make_client(trader)

        await client.place_limit_order(
            side="SHORT",
            qty=Decimal("1.0"),
            price=Decimal("2900.00"),
            reduce_only=True,
            client_order_id="tp-short-cid",
        )

        _method, _endpoint, body = trader._request_calls[0]
        assert body["side"] == "buy"  # SHORT close = buy


# ======================================================================
# Tests: place_stop_market_order
# ======================================================================


class TestPlaceStopMarketOrder:
    @pytest.mark.asyncio
    async def test_reduce_only_places_algo_order(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"algoId": "sl-algo-001"}]},
        ])
        client = _make_client(trader)

        result = await client.place_stop_market_order(
            side="LONG",
            qty=Decimal("0.5"),
            trigger_price=Decimal("2900.00"),
            reduce_only=True,
            client_order_id="sl-cid-1",
        )

        assert result.ok is True
        assert result.order_id == "sl-algo-001"
        assert result.client_order_id == "sl-cid-1"

        _method, endpoint, body = trader._request_calls[0]
        assert endpoint == "/api/v5/trade/order-algo"
        assert body["ordType"] == "conditional"
        assert body["side"] == "sell"  # LONG close = sell
        assert body["reduceOnly"] == "true"
        assert body["slTriggerPx"] == "2900.00"
        assert body["algoClOrdId"] == "sl-cid-1"

    @pytest.mark.asyncio
    async def test_qty_none_uses_current_position(self) -> None:
        trader = FakeTrader()
        trader.set_position(FakePositionSnapshot(
            side="LONG",
            contracts=Decimal("2.5"),
            avg_entry_price=3100.0,
            eth_qty=0.25,
            raw_pos=Decimal("2.5"),
        ))
        # First response: position fetch (triggered by qty=None)
        # Second response: algo order placement
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [
                {"instId": "ETH-USDT-SWAP", "pos": "2.5", "avgPx": "3100.0"}
            ]},
            {"code": "0", "msg": "", "data": [{"algoId": "sl-pos-qty"}]},
        ])
        client = _make_client(trader)

        result = await client.place_stop_market_order(
            side="LONG",
            qty=None,
            trigger_price=Decimal("2900.00"),
            reduce_only=True,
            client_order_id="sl-auto-qty",
        )

        assert result.ok is True

        # The second call is the algo order (first was position fetch)
        _method, _endpoint, body = trader._request_calls[1]
        assert body["sz"] == "2.5"

    @pytest.mark.asyncio
    async def test_qty_none_zero_position_raises(self) -> None:
        trader = FakeTrader()
        trader.set_position(FakePositionSnapshot(
            side=None,
            contracts=Decimal("0"),
            avg_entry_price=0.0,
            eth_qty=0.0,
            raw_pos=Decimal("0"),
        ))
        client = _make_client(trader)

        with pytest.raises(RuntimeError, match="qty > 0"):
            await client.place_stop_market_order(
                side="LONG",
                qty=None,
                trigger_price=Decimal("2900.00"),
                reduce_only=True,
                client_order_id="x",
            )

    @pytest.mark.asyncio
    async def test_reduce_only_false_raises(self) -> None:
        trader = FakeTrader()
        client = _make_client(trader)

        with pytest.raises(ValueError, match="reduce_only=True only"):
            await client.place_stop_market_order(
                side="LONG",
                qty=Decimal("0.5"),
                trigger_price=Decimal("2900.00"),
                reduce_only=False,
                client_order_id="x",
            )

    @pytest.mark.asyncio
    async def test_short_reduce_only_uses_buy_side(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"algoId": "sl-short"}]},
        ])
        client = _make_client(trader)

        await client.place_stop_market_order(
            side="SHORT",
            qty=Decimal("1.0"),
            trigger_price=Decimal("3200.00"),
            reduce_only=True,
            client_order_id="sl-short-cid",
        )

        _method, _endpoint, body = trader._request_calls[0]
        assert body["side"] == "buy"  # SHORT close = buy
        assert body["slTriggerPx"] == "3200.00"


# ======================================================================
# Tests: cancel_order
# ======================================================================


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_by_order_id(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "cancel-me", "sCode": "0"}]},
        ])
        client = _make_client(trader)

        result = await client.cancel_order(order_id="cancel-me")

        assert result.ok is True
        assert result.order_id == "cancel-me"

        _method, endpoint, body = trader._request_calls[0]
        assert endpoint == "/api/v5/trade/cancel-order"
        assert body["ordId"] == "cancel-me"
        assert body["instId"] == "ETH-USDT-SWAP"

    @pytest.mark.asyncio
    async def test_cancel_by_client_order_id(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "by-cid", "sCode": "0"}]},
        ])
        client = _make_client(trader)

        result = await client.cancel_order(client_order_id="my-clord")

        assert result.ok is True
        assert result.client_order_id == "my-clord"

        _method, _endpoint, body = trader._request_calls[0]
        assert _endpoint == "/api/v5/trade/cancel-order"
        assert body["clOrdId"] == "my-clord"
        assert "ordId" not in body

    @pytest.mark.asyncio
    async def test_cancel_by_both_ids_uses_order_id(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "by-ord", "sCode": "0"}]},
        ])
        client = _make_client(trader)

        result = await client.cancel_order(order_id="by-ord", client_order_id="also-cid")

        assert result.ok is True
        assert result.order_id == "by-ord"

        _method, _endpoint, body = trader._request_calls[0]
        assert body["ordId"] == "by-ord"

    @pytest.mark.asyncio
    async def test_cancel_no_id_raises(self) -> None:
        trader = FakeTrader()
        client = _make_client(trader)

        with pytest.raises(ValueError, match="at least one of order_id or client_order_id"):
            await client.cancel_order()

    @pytest.mark.asyncio
    async def test_cancel_fallback_to_algo(self) -> None:
        """When regular cancel fails with an order_id, fallback to algo cancel."""
        trader = FakeTrader()

        call_count = [0]

        async def request_mock(method: str, endpoint: str, payload: Any = None) -> dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("regular cancel failed")
            # Second call (algo cancel) succeeds
            return {"code": "0", "msg": "", "data": [{"algoId": "algo-x", "sCode": "0"}]}

        trader.request = request_mock  # type: ignore[method-assign]
        client = _make_client(trader)

        result = await client.cancel_order(order_id="algo-x")

        assert result.ok is True
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_cancel_by_cid_no_fallback(self) -> None:
        """When regular cancel fails with only client_order_id, no algo fallback."""
        trader = FakeTrader()

        async def request_mock(method: str, endpoint: str, payload: Any = None) -> dict[str, Any]:
            raise RuntimeError("regular cancel failed")

        trader.request = request_mock  # type: ignore[method-assign]
        client = _make_client(trader)

        with pytest.raises(RuntimeError, match="regular cancel failed"):
            await client.cancel_order(client_order_id="only-cid")


# ======================================================================
# Tests: _normalise_client_order_id
# ======================================================================


class TestNormaliseClientOrderId:
    def test_none_returns_none(self) -> None:
        assert _normalise_client_order_id(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _normalise_client_order_id("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert _normalise_client_order_id("   ") is None
        assert _normalise_client_order_id("\t") is None
        assert _normalise_client_order_id("\n") is None

    def test_non_empty_preserved(self) -> None:
        assert _normalise_client_order_id("my-id") == "my-id"

    def test_padded_stripped(self) -> None:
        assert _normalise_client_order_id("  my-id  ") == "my-id"


# ======================================================================
# Tests: empty client_order_id does NOT write body key
# ======================================================================


class TestEmptyClientOrderIdOmittedFromBody:
    """Ensure empty/whitespace client_order_id is NOT written to the OKX
    request body — preventing ``clOrdId=""`` or ``algoClOrdId=""``."""

    @pytest.mark.asyncio
    async def test_market_order_empty_cid_no_cl_ord_id(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "entry-001"}]},
        ])
        client = _make_client(trader)

        result = await client.place_market_order(
            side="LONG",
            qty=Decimal("0.5"),
            reduce_only=False,
            client_order_id="",
        )

        assert result.ok is True
        assert result.order_id == "entry-001"
        assert result.client_order_id is None

        _method, _endpoint, body = trader._request_calls[0]
        assert "clOrdId" not in body

    @pytest.mark.asyncio
    async def test_market_order_whitespace_cid_no_cl_ord_id(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "entry-002"}]},
        ])
        client = _make_client(trader)

        result = await client.place_market_order(
            side="SHORT",
            qty=Decimal("1.0"),
            reduce_only=True,
            client_order_id="   ",
        )

        assert result.ok is True
        assert result.client_order_id is None

        _method, _endpoint, body = trader._request_calls[0]
        assert "clOrdId" not in body

    @pytest.mark.asyncio
    async def test_limit_order_empty_cid_no_cl_ord_id(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "tp-001"}]},
        ])
        client = _make_client(trader)

        result = await client.place_limit_order(
            side="LONG",
            qty=Decimal("0.5"),
            price=Decimal("3200.50"),
            reduce_only=True,
            client_order_id="",
        )

        assert result.ok is True
        assert result.order_id == "tp-001"
        assert result.client_order_id is None

        _method, _endpoint, body = trader._request_calls[0]
        assert "clOrdId" not in body

    @pytest.mark.asyncio
    async def test_limit_order_whitespace_cid_no_cl_ord_id(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "tp-002"}]},
        ])
        client = _make_client(trader)

        result = await client.place_limit_order(
            side="SHORT",
            qty=Decimal("1.0"),
            price=Decimal("2900.00"),
            reduce_only=True,
            client_order_id="\t",
        )

        assert result.ok is True
        assert result.client_order_id is None

        _method, _endpoint, body = trader._request_calls[0]
        assert "clOrdId" not in body

    @pytest.mark.asyncio
    async def test_stop_market_order_empty_cid_no_algo_cl_ord_id(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"algoId": "sl-001"}]},
        ])
        client = _make_client(trader)

        result = await client.place_stop_market_order(
            side="LONG",
            qty=Decimal("0.5"),
            trigger_price=Decimal("2900.00"),
            reduce_only=True,
            client_order_id="",
        )

        assert result.ok is True
        assert result.order_id == "sl-001"
        assert result.client_order_id is None

        _method, _endpoint, body = trader._request_calls[0]
        assert "algoClOrdId" not in body

    @pytest.mark.asyncio
    async def test_stop_market_order_whitespace_cid_no_algo_cl_ord_id(self) -> None:
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"algoId": "sl-002"}]},
        ])
        client = _make_client(trader)

        result = await client.place_stop_market_order(
            side="SHORT",
            qty=Decimal("1.0"),
            trigger_price=Decimal("3200.00"),
            reduce_only=True,
            client_order_id="   ",
        )

        assert result.ok is True
        assert result.client_order_id is None

        _method, _endpoint, body = trader._request_calls[0]
        assert "algoClOrdId" not in body


# ======================================================================
# Tests: cancellation with empty/whitespace client_order_id
# ======================================================================


class TestCancelOrderEmptyClientOrderId:
    @pytest.mark.asyncio
    async def test_cancel_empty_cid_no_order_id_raises(self) -> None:
        """cancel_order(client_order_id="") with no order_id raises ValueError
        because the empty string normalises to None → no identifier provided."""
        trader = FakeTrader()
        client = _make_client(trader)

        with pytest.raises(ValueError, match="at least one of order_id or client_order_id"):
            await client.cancel_order(client_order_id="")

    @pytest.mark.asyncio
    async def test_cancel_whitespace_cid_no_order_id_raises(self) -> None:
        trader = FakeTrader()
        client = _make_client(trader)

        with pytest.raises(ValueError, match="at least one of order_id or client_order_id"):
            await client.cancel_order(client_order_id="   ")

    @pytest.mark.asyncio
    async def test_cancel_empty_cid_with_order_id_uses_order_id(self) -> None:
        """When order_id is provided, empty client_order_id is harmlessly
        normalised to None — the cancel proceeds by order_id."""
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "by-ord", "sCode": "0"}]},
        ])
        client = _make_client(trader)

        result = await client.cancel_order(order_id="by-ord", client_order_id="")

        assert result.ok is True
        assert result.order_id == "by-ord"
        assert result.client_order_id is None

        _method, _endpoint, body = trader._request_calls[0]
        assert body["ordId"] == "by-ord"

    @pytest.mark.asyncio
    async def test_cancel_non_empty_cid_still_works(self) -> None:
        """Non-empty client_order_id with no order_id still works — the
        existing behavior is preserved."""
        trader = FakeTrader()
        trader.set_request_responses([
            {"code": "0", "msg": "", "data": [{"ordId": "by-cid", "sCode": "0"}]},
        ])
        client = _make_client(trader)

        result = await client.cancel_order(client_order_id="my-clord")

        assert result.ok is True
        assert result.client_order_id == "my-clord"

        _method, _endpoint, body = trader._request_calls[0]
        assert body["clOrdId"] == "my-clord"
