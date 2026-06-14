#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_semantic_executor_algo_sl.py
@Description: Tests verifying that BinanceBrokerSemanticExecutor routes
              protective stops through the Algo Order API.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.binance.semantic_executor import BinanceBrokerSemanticExecutor
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
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
)

SYMBOL = "ETHUSDT"


# ======================================================================
# Helpers
# ======================================================================


class FakeBrokerClient:
    """Fake broker client that records calls."""

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
        return BrokerOrderResult(
            exchange=ExchangeName.BINANCE,
            symbol=request.symbol,
            ok=True,
            order_id="order-1",
            client_order_id=request.client_order_id,
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
        )

    async def fetch_open_orders(self, symbol: str) -> list[BrokerOrder]:
        return self.open_orders

    async def fetch_position(self, symbol: str) -> BrokerPosition | None:
        return self.position


class FakeAlgoClient:
    """Fake algo client that records calls."""

    def __init__(self) -> None:
        self.place_calls: list[dict] = []
        self.cancel_calls: list[dict] = []
        self.fetch_count: int = 0
        self._next_place_result: BrokerOrderResult | None = None
        self._next_cancel_result: BrokerCancelResult | None = None
        self._open_algo_orders: list[dict] = []
        self._place_raises: Exception | None = None

    async def place_stop_loss(self, **kwargs) -> BrokerOrderResult:
        self.place_calls.append(kwargs)
        if self._place_raises:
            raise self._place_raises
        if self._next_place_result is not None:
            return self._next_place_result
        return BrokerOrderResult(
            exchange=ExchangeName.BINANCE,
            symbol=kwargs.get("symbol", SYMBOL),
            ok=True,
            order_id=str(kwargs.get("client_algo_id", "algo-1")),
            client_order_id=kwargs.get("client_algo_id", ""),
        )

    async def cancel_algo_order(self, **kwargs) -> BrokerCancelResult:
        self.cancel_calls.append(kwargs)
        if self._next_cancel_result is not None:
            return self._next_cancel_result
        return BrokerCancelResult(
            exchange=ExchangeName.BINANCE,
            symbol=kwargs.get("symbol", SYMBOL),
            ok=True,
            order_id=kwargs.get("client_algo_id"),
            client_order_id=kwargs.get("client_algo_id"),
        )

    async def fetch_open_algo_orders(self, **kwargs) -> list[dict]:
        self.fetch_count += 1
        return self._open_algo_orders


def _make_executor(algo_client=None):
    broker = FakeBrokerClient()
    return BinanceBrokerSemanticExecutor(broker, algo_client=algo_client), broker


# ======================================================================
# Without Algo Client — PLACE_PROTECTIVE_STOP raises clear error
# ======================================================================


class TestProtectiveStopRequiresAlgoClient:
    """Without algo_client, PLACE_PROTECTIVE_STOP must raise clear error."""

    @pytest.mark.asyncio
    async def test_place_protective_stop_without_algo_raises(self) -> None:
        executor, broker = _make_executor(algo_client=None)
        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            trigger_price=Decimal("3000"),
            client_order_id="RE_MAIN_sl",
        )

        with pytest.raises(ExchangeError) as exc_info:
            await executor.execute(request)
        assert "requires BinanceAlgoOrderClient" in str(exc_info.value)
        # Ensure it did NOT call regular place_order
        assert len(broker.place_order_requests) == 0

    @pytest.mark.asyncio
    async def test_place_protective_stop_does_not_fallback_to_stop_market(self) -> None:
        """Without algo_client, should never fall back to STOP_MARKET."""
        executor, broker = _make_executor(algo_client=None)
        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerPositionSide.SHORT,
            quantity=Decimal("1"),
            trigger_price=Decimal("3500"),
            client_order_id="RE_MAIN_sl",
        )

        with pytest.raises(ExchangeError):
            await executor.execute(request)
        # No regular order calls
        assert broker.place_order_requests == []


# ======================================================================
# With Algo Client — PLACE_PROTECTIVE_STOP uses algo API
# ======================================================================


class TestProtectiveStopWithAlgoClient:
    """With algo_client, PLACE_PROTECTIVE_STOP routes to Algo Order API."""

    @pytest.mark.asyncio
    async def test_place_protective_stop_uses_algo_client(self) -> None:
        algo = FakeAlgoClient()
        executor, broker = _make_executor(algo_client=algo)

        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("0.1"),
            trigger_price=Decimal("3000"),
            client_order_id="RE_MAIN_sl",
        )

        result = await executor.execute(request)

        assert result.ok
        assert result.client_order_id == "RE_MAIN_sl"
        # Must have called algo client, NOT broker_client.place_order
        assert len(algo.place_calls) == 1
        assert len(broker.place_order_requests) == 0

        call = algo.place_calls[0]
        assert call["side"] == "SELL"  # LONG position → SELL to close
        assert call["client_algo_id"] == "RE_MAIN_sl"
        assert call["trigger_price"] == Decimal("3000")

    @pytest.mark.asyncio
    async def test_place_protective_stop_short_closes_with_buy(self) -> None:
        """SHORT position → BUY to close."""
        algo = FakeAlgoClient()
        executor, _ = _make_executor(algo_client=algo)

        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerPositionSide.SHORT,
            quantity=Decimal("0.1"),
            trigger_price=Decimal("3500"),
            client_order_id="RE_MAIN_sl_short",
        )

        await executor.execute(request)
        assert algo.place_calls[0]["side"] == "BUY"

    @pytest.mark.asyncio
    async def test_place_protective_stop_converts_contracts_to_base(self) -> None:
        """Quantity is converted from CONTRACTS to base-asset for algo API."""
        algo = FakeAlgoClient()
        executor, _ = _make_executor(algo_client=algo)

        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),  # 1 contract = 0.1 ETH
            trigger_price=Decimal("3000"),
            client_order_id="RE_MAIN_sl",
        )

        await executor.execute(request)
        # 1 contract * 0.1 = 0.1 ETH base-asset
        assert algo.place_calls[0]["quantity"] == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_place_protective_stop_no_position_side(self) -> None:
        """Algo SL must NEVER emit positionSide."""
        algo = FakeAlgoClient()
        executor, _ = _make_executor(algo_client=algo)

        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            trigger_price=Decimal("3000"),
            client_order_id="RE_MAIN_sl",
        )

        await executor.execute(request)
        assert "positionSide" not in algo.place_calls[0]


# ======================================================================
# CANCEL_PROTECTIVE_STOP with Algo Client
# ======================================================================


class TestCancelProtectiveStopWithAlgoClient:
    """CANCEL_PROTECTIVE_STOP uses algo cancel, not regular cancel."""

    @pytest.mark.asyncio
    async def test_cancel_protective_stop_uses_algo_cancel(self) -> None:
        algo = FakeAlgoClient()
        executor, broker = _make_executor(algo_client=algo)

        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            order_id="RE_MAIN_sl",
        )

        result = await executor.execute(request)

        assert result.ok
        # Must use algo cancel, not regular cancel_order
        assert len(algo.cancel_calls) == 1
        assert len(broker.cancel_order_calls) == 0
        assert algo.cancel_calls[0]["client_algo_id"] == "RE_MAIN_sl"

    @pytest.mark.asyncio
    async def test_cancel_protective_stop_without_algo_raises(self) -> None:
        """Without algo_client, CANCEL_PROTECTIVE_STOP must raise."""
        executor, broker = _make_executor(algo_client=None)
        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            order_id="RE_MAIN_sl",
        )

        with pytest.raises(ExchangeError):
            await executor.execute(request)
        # Does NOT fall back to regular cancel_order
        assert broker.cancel_order_calls == []


# ======================================================================
# FETCH_ALGO_ORDERS
# ======================================================================


class TestFetchAlgoOrders:
    """FETCH_ALGO_ORDERS returns algo orders or empty."""

    @pytest.mark.asyncio
    async def test_fetch_algo_orders_with_client(self) -> None:
        algo = FakeAlgoClient()
        algo._open_algo_orders = [
            {"algoId": 1, "clientAlgoId": "RE_MAIN_sl", "orderType": "STOP_MARKET",
             "side": "SELL", "quantity": "0.1", "triggerPrice": "3000"},
        ]
        executor, _ = _make_executor(algo_client=algo)

        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.FETCH_ALGO_ORDERS,
            role=BrokerSemanticOrderRole.UNKNOWN,
        )

        result = await executor.execute(request)
        assert result.ok
        assert len(result.orders) == 1
        assert result.orders[0].order_id == "1"

    @pytest.mark.asyncio
    async def test_fetch_algo_orders_without_client(self) -> None:
        """Without algo_client, returns empty orders tuple."""
        executor, _ = _make_executor(algo_client=None)
        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.FETCH_ALGO_ORDERS,
            role=BrokerSemanticOrderRole.UNKNOWN,
        )

        result = await executor.execute(request)
        assert result.ok
        assert len(result.orders) == 0


# ======================================================================
# RECOVER_OPEN_ORDERS includes algo orders
# ======================================================================


class TestRecoverOpenOrdersWithAlgo:
    """RECOVER_OPEN_ORDERS returns both normal and algo orders."""

    @pytest.mark.asyncio
    async def test_recovers_both_order_types(self) -> None:
        algo = FakeAlgoClient()
        algo._open_algo_orders = [
            {"algoId": 100, "clientAlgoId": "RE_MAIN_sl", "orderType": "STOP_MARKET",
             "side": "SELL", "quantity": "0.1", "triggerPrice": "3000"},
        ]
        executor, broker = _make_executor(algo_client=algo)
        broker.open_orders = [
            BrokerOrder(
                exchange=ExchangeName.BINANCE,
                symbol=SYMBOL,
                order_id="tp-1",
                client_order_id="RE_MAIN_tp",
                side=BrokerOrderSide.SELL,
                position_side=BrokerPositionSide.LONG,
                order_type=BrokerOrderType.LIMIT,
                status=BrokerOrderStatus.OPEN,
                price=Decimal("3100"),
                quantity=Decimal("0.1"),
                quantity_unit=BrokerQuantityUnit.BASE_ASSET,
                reduce_only=True,
            ),
        ]

        request = BrokerSemanticRequest(
            exchange=ExchangeName.BINANCE,
            symbol=SYMBOL,
            action=BrokerSemanticAction.RECOVER_OPEN_ORDERS,
            role=BrokerSemanticOrderRole.RECOVERY,
        )

        result = await executor.execute(request)
        assert result.ok
        assert len(result.orders) == 2  # 1 ordinary + 1 algo

        sources = {getattr(o, "metadata", {}).get("source") for o in result.orders}
        assert "ordinary" in sources
        assert "algo" in sources
