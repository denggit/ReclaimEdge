"""Tests for src.exchanges.okx.client.OkxBrokerClient — using FakeTrader."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerExecutionAction,
    BrokerExecutionResult,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerTimeInForce,
    ExchangeName,
)
from src.exchanges.okx.client import OkxBrokerClient


# ---------------------------------------------------------------------------
# FakeTrader
# ---------------------------------------------------------------------------


@dataclass
class _FakeMetadata:
    contract_multiplier: Decimal = Decimal("0.1")
    contract_precision: Decimal = Decimal("0.01")
    min_contracts: Decimal = Decimal("0.01")


@dataclass
class _FakePositionSnapshot:
    side: str | None
    contracts: Decimal
    avg_entry_price: float
    eth_qty: float
    raw_pos: Decimal


class FakeTrader:
    """Lightweight fake implementing the _OkxTraderLike protocol."""

    def __init__(
        self,
        *,
        symbol: str = "ETH-USDT-SWAP",
        td_mode: str = "isolated",
        pos_side_mode: str = "net",
        equity: float = 100.0,
        position_snapshot: _FakePositionSnapshot | None = None,
        pending_orders: list[dict[str, Any]] | None = None,
        pending_algo_orders: list[dict[str, Any]] | None = None,
        order_response: dict[str, Any] | None = None,
        algo_order_response: dict[str, Any] | None = None,
        cancel_response: dict[str, Any] | None = None,
        cancel_algo_response: dict[str, Any] | None = None,
    ) -> None:
        self.symbol = symbol
        self.td_mode = td_mode
        self.pos_side_mode = pos_side_mode
        self.instrument_metadata = _FakeMetadata()
        self._equity = equity
        self._position_snapshot = position_snapshot or _FakePositionSnapshot(
            side=None, contracts=Decimal("0"), avg_entry_price=0.0, eth_qty=0.0, raw_pos=Decimal("0")
        )
        self._pending_orders = pending_orders or []
        self._pending_algo_orders = pending_algo_orders or []
        self._order_response = order_response
        self._algo_order_response = algo_order_response
        self._cancel_response = cancel_response
        self._cancel_algo_response = cancel_algo_response
        self.requests: list[dict[str, Any]] = []
        self._next_order_id = 1
        self.closed = False

    async def fetch_usdt_equity(self) -> float:
        return self._equity

    async def fetch_position_snapshot(self) -> _FakePositionSnapshot:
        return self._position_snapshot

    async def fetch_pending_orders(self) -> list[dict[str, Any]]:
        return list(self._pending_orders)

    async def fetch_pending_algo_orders(self) -> list[dict[str, Any]]:
        return list(self._pending_algo_orders)

    async def request(
        self, method: str, endpoint: str, payload: Any | None = None
    ) -> dict[str, Any]:
        self.requests.append(
            {"method": method, "endpoint": endpoint, "payload": payload}
        )
        # Return a fake order response
        if endpoint == "/api/v5/trade/order":
            if self._order_response is not None:
                return self._order_response
            ord_id = str(self._next_order_id)
            self._next_order_id += 1
            return {"code": "0", "data": [{"ordId": ord_id, "clOrdId": payload.get("clOrdId", "") if payload else "", "sCode": "0"}]}
        if endpoint == "/api/v5/trade/order-algo":
            if self._algo_order_response is not None:
                return self._algo_order_response
            algo_id = f"algo-{self._next_order_id}"
            self._next_order_id += 1
            return {"code": "0", "data": [{"algoId": algo_id, "sCode": "0"}]}
        if endpoint == "/api/v5/trade/cancel-order":
            if self._cancel_response is not None:
                return self._cancel_response
            return {"code": "0", "data": [{"ordId": payload.get("ordId", ""), "sCode": "0"}]}
        if endpoint == "/api/v5/trade/cancel-algos":
            if self._cancel_algo_response is not None:
                return self._cancel_algo_response
            return {"code": "0", "data": [{"algoId": payload[0].get("algoId", "") if payload else "", "sCode": "0"}]}
        return {"code": "0", "data": []}

    @staticmethod
    def extract_order_id(res: dict[str, Any]) -> str:
        data = res.get("data", [])
        if not data or not data[0].get("ordId"):
            raise RuntimeError(f"Missing ordId in response: {res}")
        return str(data[0]["ordId"])

    @staticmethod
    def extract_algo_id(res: dict[str, Any]) -> str:
        data = res.get("data", [])
        if not data or not (data[0].get("algoId") or data[0].get("ordId")):
            raise RuntimeError(f"Missing algoId in response: {res}")
        return str(data[0].get("algoId") or data[0].get("ordId"))

    @staticmethod
    def decimal_to_str(value: Any) -> str:
        from decimal import Decimal as D

        if isinstance(value, D):
            return format(value.normalize(), "f")
        return format(D(str(value)).normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        return f"{price:.2f}"

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broker_order_request(
    *,
    exchange: ExchangeName = ExchangeName.OKX,
    symbol: str = "ETH-USDT-SWAP",
    side: BrokerOrderSide = BrokerOrderSide.BUY,
    position_side: BrokerPositionSide = BrokerPositionSide.LONG,
    order_type: BrokerOrderType = BrokerOrderType.MARKET,
    quantity: Decimal = Decimal("1"),
    price: Decimal | None = None,
    reduce_only: bool = False,
    close_position: bool = False,
    time_in_force: BrokerTimeInForce | None = None,
    client_order_id: str | None = None,
    trigger_price: Decimal | None = None,
) -> BrokerOrderRequest:
    return BrokerOrderRequest(
        exchange=exchange,
        symbol=symbol,
        side=side,
        position_side=position_side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        trigger_price=trigger_price,
        reduce_only=reduce_only,
        close_position=close_position,
        time_in_force=time_in_force,
        client_order_id=client_order_id,
    )


def _assert_place_order_error(
    exc_info: pytest.ExceptionInfo[ExchangeError],
    kind: ExchangeErrorKind,
) -> None:
    assert exc_info.value.detail.kind == kind


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    @pytest.mark.asyncio
    async def test_exchange_is_okx(self):
        client = OkxBrokerClient(FakeTrader())
        assert client.exchange == ExchangeName.OKX

    @pytest.mark.asyncio
    async def test_capabilities_exchange_is_okx(self):
        client = OkxBrokerClient(FakeTrader())
        assert client.capabilities.exchange == ExchangeName.OKX
        assert client.capabilities.supports_hedge_mode is True
        assert client.capabilities.supports_reduce_only is True


# ---------------------------------------------------------------------------
# fetch_instrument
# ---------------------------------------------------------------------------


class TestFetchInstrument:
    @pytest.mark.asyncio
    async def test_returns_broker_instrument(self):
        client = OkxBrokerClient(FakeTrader(symbol="ETH-USDT-SWAP"))
        inst = await client.fetch_instrument("ETH-USDT-SWAP")
        assert inst.symbol == "ETH-USDT-SWAP"
        assert inst.exchange == ExchangeName.OKX

    @pytest.mark.asyncio
    async def test_symbol_mismatch_raises_invalid_symbol(self):
        client = OkxBrokerClient(FakeTrader(symbol="ETH-USDT-SWAP"))
        with pytest.raises(ExchangeError) as exc_info:
            await client.fetch_instrument("BTC-USDT-SWAP")
        assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_SYMBOL


# ---------------------------------------------------------------------------
# fetch_balance
# ---------------------------------------------------------------------------


class TestFetchBalance:
    @pytest.mark.asyncio
    async def test_usdt_balance(self):
        client = OkxBrokerClient(FakeTrader(equity=123.45))
        bal = await client.fetch_balance("USDT")
        assert bal.exchange == ExchangeName.OKX
        assert bal.asset == "USDT"
        assert bal.total == Decimal("123.45")
        assert bal.available == Decimal("123.45")
        assert bal.equity == Decimal("123.45")

    @pytest.mark.asyncio
    async def test_btc_balance_unsupported(self):
        client = OkxBrokerClient(FakeTrader())
        with pytest.raises(ExchangeError) as exc_info:
            await client.fetch_balance("BTC")
        assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION


# ---------------------------------------------------------------------------
# fetch_position
# ---------------------------------------------------------------------------


class TestFetchPosition:
    @pytest.mark.asyncio
    async def test_maps_position_snapshot(self):
        snap = _FakePositionSnapshot(
            side="LONG",
            contracts=Decimal("1.5"),
            avg_entry_price=3400.0,
            eth_qty=0.15,
            raw_pos=Decimal("1.5"),
        )
        client = OkxBrokerClient(FakeTrader(position_snapshot=snap))
        pos = await client.fetch_position("ETH-USDT-SWAP")
        assert pos.side == BrokerPositionSide.LONG
        assert pos.contracts == Decimal("1.5")

    @pytest.mark.asyncio
    async def test_symbol_mismatch_raises_invalid_symbol(self):
        client = OkxBrokerClient(FakeTrader())
        with pytest.raises(ExchangeError) as exc_info:
            await client.fetch_position("BTC-USDT-SWAP")
        assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_SYMBOL


# ---------------------------------------------------------------------------
# fetch_open_orders
# ---------------------------------------------------------------------------


class TestFetchOpenOrders:
    @pytest.mark.asyncio
    async def test_maps_pending_orders(self):
        orders = [
            {
                "ordId": "100",
                "side": "buy",
                "posSide": "long",
                "ordType": "limit",
                "state": "live",
                "px": "3400.00",
                "sz": "1.0",
                "accFillSz": "0",
            }
        ]
        client = OkxBrokerClient(FakeTrader(pending_orders=orders))
        result = await client.fetch_open_orders("ETH-USDT-SWAP")
        assert len(result) == 1
        assert result[0].order_id == "100"
        assert result[0].side == BrokerOrderSide.BUY

    @pytest.mark.asyncio
    async def test_empty_orders(self):
        client = OkxBrokerClient(FakeTrader())
        result = await client.fetch_open_orders("ETH-USDT-SWAP")
        assert result == []

    @pytest.mark.asyncio
    async def test_symbol_mismatch(self):
        client = OkxBrokerClient(FakeTrader())
        with pytest.raises(ExchangeError) as exc_info:
            await client.fetch_open_orders("BTC-USDT-SWAP")
        assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_SYMBOL


# ---------------------------------------------------------------------------
# place_order — MARKET entry
# ---------------------------------------------------------------------------


class TestPlaceOrderMarketEntry:
    @pytest.mark.asyncio
    async def test_market_entry_long(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.MARKET,
            quantity=Decimal("1"),
        )
        result = await client.place_order(req)
        assert isinstance(result, BrokerOrderResult)
        assert result.order_id == "1"
        assert result.status == BrokerOrderStatus.NEW
        # Verify the request was made
        assert len(trader.requests) == 1
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/order"
        assert r["payload"]["side"] == "buy"
        assert r["payload"]["ordType"] == "market"
        assert r["payload"]["sz"] == "1"

    @pytest.mark.asyncio
    async def test_market_entry_short(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.SHORT,
            order_type=BrokerOrderType.MARKET,
            quantity=Decimal("2"),
        )
        result = await client.place_order(req)
        assert isinstance(result, BrokerOrderResult)
        assert result.order_id == "1"
        assert trader.requests[0]["payload"]["side"] == "sell"
        assert trader.requests[0]["payload"]["sz"] == "2"

    @pytest.mark.asyncio
    async def test_market_entry_with_ioc(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            order_type=BrokerOrderType.MARKET,
            quantity=Decimal("1"),
            time_in_force=BrokerTimeInForce.IOC,
        )
        result = await client.place_order(req)
        assert result.order_id == "1"

    @pytest.mark.asyncio
    async def test_market_entry_item_level_error_raises_exchange_error(self):
        response = {
            "code": "0",
            "data": [
                {"ordId": "", "sCode": "51008", "sMsg": "Insufficient balance"}
            ],
        }
        trader = FakeTrader(
            order_response=response
        )
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.MARKET,
            quantity=Decimal("1"),
        )
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        assert exc_info.value.detail.kind == ExchangeErrorKind.INSUFFICIENT_MARGIN
        assert exc_info.value.detail.code == "51008"
        assert "Failed to place market entry" in exc_info.value.detail.message
        assert "Insufficient balance" in exc_info.value.detail.message
        assert "OKX INSUFFICIENT_MARGIN" not in exc_info.value.detail.message
        assert exc_info.value.detail.raw == response
        assert len(trader.requests) == 1


# ---------------------------------------------------------------------------
# place_order — validation
# ---------------------------------------------------------------------------


class TestPlaceOrderValidation:
    @pytest.mark.asyncio
    async def test_place_order_symbol_mismatch_raises_invalid_symbol(self):
        trader = FakeTrader(symbol="ETH-USDT-SWAP")
        client = OkxBrokerClient(trader)
        req = _broker_order_request(symbol="BTC-USDT-SWAP")
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        _assert_place_order_error(exc_info, ExchangeErrorKind.INVALID_SYMBOL)
        assert trader.requests == []

    @pytest.mark.asyncio
    async def test_place_order_exchange_mismatch_raises_unsupported(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(exchange=ExchangeName.BINANCE)
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        _assert_place_order_error(exc_info, ExchangeErrorKind.UNSUPPORTED_OPERATION)
        assert trader.requests == []

    @pytest.mark.asyncio
    async def test_market_entry_long_requires_buy_side(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.LONG,
        )
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        _assert_place_order_error(exc_info, ExchangeErrorKind.UNSUPPORTED_OPERATION)
        assert trader.requests == []

    @pytest.mark.asyncio
    async def test_market_entry_short_requires_sell_side(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.SHORT,
        )
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        _assert_place_order_error(exc_info, ExchangeErrorKind.UNSUPPORTED_OPERATION)
        assert trader.requests == []

    @pytest.mark.asyncio
    async def test_reduce_only_tp_long_requires_sell_side(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.LIMIT,
            price=Decimal("3500.00"),
            reduce_only=True,
        )
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        _assert_place_order_error(exc_info, ExchangeErrorKind.UNSUPPORTED_OPERATION)
        assert trader.requests == []

    @pytest.mark.asyncio
    async def test_reduce_only_tp_short_requires_buy_side(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.SHORT,
            order_type=BrokerOrderType.LIMIT,
            price=Decimal("3000.00"),
            reduce_only=True,
        )
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        _assert_place_order_error(exc_info, ExchangeErrorKind.UNSUPPORTED_OPERATION)
        assert trader.requests == []


# ---------------------------------------------------------------------------
# place_order — LIMIT reduce-only TP
# ---------------------------------------------------------------------------


class TestPlaceOrderLimitReduceOnlyTP:
    @pytest.mark.asyncio
    async def test_limit_reduce_only_tp_long(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("3500.00"),
            reduce_only=True,
            client_order_id="tp-001",
        )
        result = await client.place_order(req)
        assert isinstance(result, BrokerOrderResult)
        assert result.order_id == "1"
        assert trader.requests[0]["payload"]["side"] == "sell"
        assert trader.requests[0]["payload"]["ordType"] == "limit"
        assert trader.requests[0]["payload"]["reduceOnly"] == "true"
        assert trader.requests[0]["payload"]["px"] == "3500.00"
        assert trader.requests[0]["payload"]["clOrdId"] == "tp-001"

    @pytest.mark.asyncio
    async def test_limit_reduce_only_tp_short(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.SHORT,
            order_type=BrokerOrderType.LIMIT,
            quantity=Decimal("2"),
            price=Decimal("3000.00"),
            reduce_only=True,
        )
        result = await client.place_order(req)
        assert result.order_id == "1"
        assert trader.requests[0]["payload"]["side"] == "buy"
        assert trader.requests[0]["payload"]["reduceOnly"] == "true"

    @pytest.mark.asyncio
    async def test_limit_reduce_only_without_price_raises_unsupported(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            order_type=BrokerOrderType.LIMIT,
            quantity=Decimal("1"),
            reduce_only=True,
            price=None,  # No price
        )
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION

    @pytest.mark.asyncio
    async def test_limit_reduce_only_tp_item_level_error_raises_exchange_error(self):
        trader = FakeTrader(
            order_response={
                "code": "0",
                "data": [
                    {"ordId": "", "sCode": "51008", "sMsg": "Insufficient balance"}
                ],
            }
        )
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("3500.00"),
            reduce_only=True,
        )
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        assert exc_info.value.detail.kind == ExchangeErrorKind.INSUFFICIENT_MARGIN
        assert exc_info.value.detail.code == "51008"
        assert len(trader.requests) == 1


# ---------------------------------------------------------------------------
# place_order — Unsupported
# ---------------------------------------------------------------------------


class TestPlaceOrderUnsupported:
    @pytest.mark.asyncio
    async def test_stop_market_places_protective_sl_algo(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            order_type=BrokerOrderType.STOP_MARKET,
            quantity=Decimal("1"),
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.LONG,
            reduce_only=True,
            close_position=True,
            trigger_price=Decimal("3100"),
        )
        result = await client.place_order(req)
        assert result.order_id == "algo-1"
        assert len(trader.requests) == 1
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/order-algo"
        assert r["payload"]["ordType"] == "conditional"
        assert r["payload"]["slTriggerPx"] == "3100.00"
        assert r["payload"]["reduceOnly"] == "true"

    @pytest.mark.asyncio
    async def test_take_profit_market_unsupported(self):
        client = OkxBrokerClient(FakeTrader())
        req = _broker_order_request(
            order_type=BrokerOrderType.TAKE_PROFIT_MARKET,
            quantity=Decimal("1"),
            reduce_only=True,
        )
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION

    @pytest.mark.asyncio
    async def test_market_reduce_only_close_supported(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            order_type=BrokerOrderType.MARKET,
            quantity=Decimal("1"),
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.LONG,
            reduce_only=True,
        )
        result = await client.place_order(req)
        assert result.order_id == "1"
        assert len(trader.requests) == 1
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/order"
        assert r["payload"]["ordType"] == "market"
        assert r["payload"]["side"] == "sell"
        assert r["payload"]["reduceOnly"] == "true"

    @pytest.mark.asyncio
    async def test_market_close_position_supported(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        req = _broker_order_request(
            order_type=BrokerOrderType.MARKET,
            quantity=Decimal("1"),
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.LONG,
            reduce_only=True,
            close_position=True,
        )
        result = await client.place_order(req)
        assert result.order_id == "1"

    @pytest.mark.asyncio
    async def test_market_with_net_position_side_unsupported(self):
        client = OkxBrokerClient(FakeTrader())
        req = _broker_order_request(
            order_type=BrokerOrderType.MARKET,
            quantity=Decimal("1"),
            position_side=BrokerPositionSide.NET,
        )
        with pytest.raises(ExchangeError) as exc_info:
            await client.place_order(req)
        assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_order_calls_correct_endpoint(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        await client.cancel_order("ETH-USDT-SWAP", "12345")
        assert len(trader.requests) == 1
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/cancel-order"
        assert r["payload"]["instId"] == "ETH-USDT-SWAP"
        assert r["payload"]["ordId"] == "12345"

    @pytest.mark.asyncio
    async def test_cancel_order_symbol_mismatch(self):
        client = OkxBrokerClient(FakeTrader(symbol="ETH-USDT-SWAP"))
        with pytest.raises(ExchangeError) as exc_info:
            await client.cancel_order("BTC-USDT-SWAP", "12345")
        assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_SYMBOL

    @pytest.mark.asyncio
    async def test_cancel_order_item_level_order_not_found_raises_exchange_error(self):
        response = {
            "code": "0",
            "data": [
                {
                    "ordId": "12345",
                    "sCode": "51603",
                    "sMsg": "Order does not exist",
                }
            ],
        }
        trader = FakeTrader(
            cancel_response=response
        )
        client = OkxBrokerClient(trader)
        with pytest.raises(ExchangeError) as exc_info:
            await client.cancel_order("ETH-USDT-SWAP", "12345")
        assert exc_info.value.detail.kind == ExchangeErrorKind.ORDER_NOT_FOUND
        assert exc_info.value.detail.code == "51603"
        assert exc_info.value.detail.raw == response
        assert len(trader.requests) == 1


# ---------------------------------------------------------------------------
# cancel_all_open_orders
# ---------------------------------------------------------------------------


class TestCancelAllOpenOrders:
    @pytest.mark.asyncio
    async def test_cancels_all_ordinary_pending_orders(self):
        orders = [
            {
                "ordId": "ord-1",
                "side": "buy",
                "posSide": "long",
                "ordType": "limit",
                "state": "live",
                "px": "3400.00",
                "sz": "1.0",
                "accFillSz": "0",
            },
            {
                "ordId": "ord-2",
                "side": "sell",
                "posSide": "long",
                "ordType": "limit",
                "state": "live",
                "px": "3600.00",
                "sz": "0.5",
                "accFillSz": "0",
            },
        ]
        trader = FakeTrader(pending_orders=orders)
        client = OkxBrokerClient(trader)
        await client.cancel_all_open_orders("ETH-USDT-SWAP")
        # Should have 2 cancel requests (one per order) + 1 fetch pending orders request
        # Actually fetch_open_orders calls trader.fetch_pending_orders which doesn't go through request()
        # Then cancel_order calls trader.request for each order
        cancel_requests = [r for r in trader.requests if r["endpoint"] == "/api/v5/trade/cancel-order"]
        assert len(cancel_requests) == 2

    @pytest.mark.asyncio
    async def test_no_orders_nothing_to_cancel(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        await client.cancel_all_open_orders("ETH-USDT-SWAP")
        assert len(trader.requests) == 0

    @pytest.mark.asyncio
    async def test_skips_orders_with_empty_order_id(self):
        orders = [
            {
                "side": "buy",  # No ordId
                "ordType": "limit",
                "state": "live",
                "sz": "1.0",
                "accFillSz": "0",
            },
        ]
        trader = FakeTrader(pending_orders=orders)
        client = OkxBrokerClient(trader)
        await client.cancel_all_open_orders("ETH-USDT-SWAP")
        # fetch_open_orders will return an order with empty order_id → skipped in cancel loop
        cancel_requests = [r for r in trader.requests if r["endpoint"] == "/api/v5/trade/cancel-order"]
        assert len(cancel_requests) == 0


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_delegates_to_trader(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        await client.close()
        assert trader.closed is True


# ---------------------------------------------------------------------------
# fetch_algo_orders
# ---------------------------------------------------------------------------


class TestFetchAlgoOrders:
    @pytest.mark.asyncio
    async def test_maps_pending_algo_orders(self):
        algo_orders = [
            {
                "algoId": "algo-100",
                "algoClOrdId": "algo-client-1",
                "side": "sell",
                "posSide": "long",
                "ordType": "conditional",
                "state": "live",
                "slTriggerPx": "3100.00",
                "sz": "1.0",
                "accFillSz": "0",
            }
        ]
        client = OkxBrokerClient(FakeTrader(pending_algo_orders=algo_orders))
        result = await client.fetch_algo_orders("ETH-USDT-SWAP")
        assert len(result) == 1
        assert result[0].order_id == "algo-100"
        assert result[0].client_order_id == "algo-client-1"
        assert result[0].order_type == BrokerOrderType.STOP_MARKET
        assert result[0].trigger_price == Decimal("3100.00")

    @pytest.mark.asyncio
    async def test_empty_algo_orders(self):
        client = OkxBrokerClient(FakeTrader())
        result = await client.fetch_algo_orders("ETH-USDT-SWAP")
        assert result == []

    @pytest.mark.asyncio
    async def test_symbol_mismatch(self):
        client = OkxBrokerClient(FakeTrader())
        with pytest.raises(ExchangeError) as exc_info:
            await client.fetch_algo_orders("BTC-USDT-SWAP")
        assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_SYMBOL


class TestCancelAlgoOrder:
    @pytest.mark.asyncio
    async def test_cancel_algo_order_calls_correct_endpoint(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        await client.cancel_algo_order("ETH-USDT-SWAP", "algo-123")
        assert len(trader.requests) == 1
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/cancel-algos"
        assert r["payload"] == [{"instId": "ETH-USDT-SWAP", "algoId": "algo-123"}]

    @pytest.mark.asyncio
    async def test_cancel_algo_order_symbol_mismatch(self):
        client = OkxBrokerClient(FakeTrader())
        with pytest.raises(ExchangeError) as exc_info:
            await client.cancel_algo_order("BTC-USDT-SWAP", "algo-123")
        assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_SYMBOL

    @pytest.mark.asyncio
    async def test_cancel_algo_order_non_zero_scode_raises_exchange_error(self):
        response = {
            "code": "0",
            "data": [
                {
                    "algoId": "algo-123",
                    "sCode": "51603",
                    "sMsg": "Order does not exist",
                }
            ],
        }
        client = OkxBrokerClient(FakeTrader(cancel_algo_response=response))
        with pytest.raises(ExchangeError) as exc_info:
            await client.cancel_algo_order("ETH-USDT-SWAP", "algo-123")
        assert exc_info.value.detail.kind == ExchangeErrorKind.ORDER_NOT_FOUND


# ---------------------------------------------------------------------------
# map_live_trade_result
# ---------------------------------------------------------------------------


class TestMapLiveTradeResult:
    @staticmethod
    def _fake_result(**overrides: object) -> object:
        """Build a minimal fake LiveTradeResult-like object."""

        class _FakeResult:
            ok: bool = True
            action: str = "OPEN_LONG"
            order_id: str | None = "order-123"
            tp_order_id: str | None = "tp-456"
            tp_order_ids: tuple = ()
            protective_sl_order_id: str | None = None
            contracts: str | None = "1.0"
            tp_price: str | None = "3500.00"
            message: str = "ok"
            entry_filled: bool = True
            tp_ok: bool | None = True
            protective_sl_price: str | None = ""
            protective_sl_ok: bool | None = None

            def __init__(self, **kw: object) -> None:
                for k, v in kw.items():
                    object.__setattr__(self, k, v)

        return _FakeResult(**overrides)

    def test_returns_broker_execution_result(self):
        trader = FakeTrader(symbol="ETH-USDT-SWAP")
        client = OkxBrokerClient(trader)
        fake = self._fake_result()
        br = client.map_live_trade_result(fake)
        assert isinstance(br, BrokerExecutionResult)
        assert br.exchange == ExchangeName.OKX
        assert br.symbol == "ETH-USDT-SWAP"
        assert br.action == BrokerExecutionAction.OPEN_LONG
        assert br.ok is True

    def test_symbol_comes_from_trader(self):
        trader = FakeTrader(symbol="BTC-USDT-SWAP")
        client = OkxBrokerClient(trader)
        fake = self._fake_result()
        br = client.map_live_trade_result(fake)
        assert br.symbol == "BTC-USDT-SWAP"
        assert br.exchange == ExchangeName.OKX
