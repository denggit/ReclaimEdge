#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_live_trader.py
@Description: Tests for BinanceLiveTrader — construction, quantity conversion,
              position mapping, execute_intent routing.  All tests use fake
              clients — no network.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from src.execution.binance_live_trader import BinanceLiveTrader, CLIENT_ORDER_ID_PREFIX
from src.execution.trader import PositionSnapshot
from src.strategies.boll_cvd_reclaim_strategy import PositionSide, TradeIntent


# ======================================================================
# Helpers
# ======================================================================


class FakeBrokerClient:
    """Fake BinanceBrokerClient for testing."""

    def __init__(self) -> None:
        self.place_order_requests: list[Any] = []
        self.cancel_order_calls: list[tuple[str, str]] = []
        self._position: Any | None = None

    async def fetch_position(self, symbol: str) -> Any | None:
        return self._position


class FakeSemanticExecutor:
    """Fake BinanceBrokerSemanticExecutor for testing."""

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.next_result: Any = None
        self._exchange = None

    @property
    def exchange(self) -> Any:
        if self._exchange is None:
            from src.exchanges.models import ExchangeName
            self._exchange = ExchangeName.BINANCE
        return self._exchange

    async def execute(self, request: Any) -> Any:
        self.requests.append(request)
        if self.next_result is not None:
            return self.next_result
        from src.exchanges.semantic_models import BrokerSemanticResult
        return BrokerSemanticResult(
            exchange=self.exchange,
            symbol="ETHUSDT",
            action=request.action,
            role=request.role,
            ok=True,
            order_id="order-1",
            client_order_id=request.client_order_id,
        )


class FakeAlgoClient:
    """Fake BinanceAlgoOrderClient for testing."""

    def __init__(self) -> None:
        self.place_calls: list[dict] = []
        self.cancel_calls: list[dict] = []
        self._open_algo_orders: list[dict] = []

    async def place_stop_loss(self, **kwargs) -> Any:
        self.place_calls.append(kwargs)
        from src.exchanges.models import BrokerOrderResult, ExchangeName
        return BrokerOrderResult(
            exchange=ExchangeName.BINANCE,
            symbol=kwargs.get("symbol", "ETHUSDT"),
            ok=True,
            order_id=str(kwargs.get("client_algo_id", "algo-1")),
            client_order_id=kwargs.get("client_algo_id", ""),
        )

    async def cancel_algo_order(self, **kwargs) -> Any:
        self.cancel_calls.append(kwargs)
        from src.exchanges.models import BrokerCancelResult, ExchangeName
        return BrokerCancelResult(
            exchange=ExchangeName.BINANCE,
            symbol=kwargs.get("symbol", "ETHUSDT"),
            ok=True,
            order_id=kwargs.get("client_algo_id"),
            client_order_id=kwargs.get("client_algo_id"),
        )

    async def fetch_open_algo_orders(self, **kwargs) -> list[dict]:
        return self._open_algo_orders


def _make_trader(
    broker_client=None,
    semantic_executor=None,
    algo_client=None,
    env=None,
) -> BinanceLiveTrader:
    return BinanceLiveTrader(
        broker_client=broker_client or FakeBrokerClient(),
        semantic_executor=semantic_executor or FakeSemanticExecutor(),
        algo_client=algo_client or FakeAlgoClient(),
        env=env or {},
    )


def _make_intent(
    intent_type: str = "OPEN_LONG",
    side: PositionSide = "LONG",
    eth_qty: float = 0.1,
    price: float = 3000.0,
    tp_price: float = 3100.0,
) -> TradeIntent:
    from src.risk.simple_position_sizer import PositionSize
    size = PositionSize(
        margin_usdt=0.0,
        notional_usdt=eth_qty * price,
        eth_qty=eth_qty,
        layer_index=0,
        layer_multiplier=1.0,
    )
    return TradeIntent(
        intent_type=intent_type,
        side=side,
        price=price,
        layer_index=0,
        tp_price=tp_price,
        reason="test",
        size=size,
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.5,
        sell_ratio=0.5,
        boll_upper=3100.0,
        boll_middle=3000.0,
        boll_lower=2900.0,
        ts_ms=0,
        avg_entry_price=price,
        breakeven_price=price,
        tp_mode="SINGLE",
    )


# ======================================================================
# Construction — no network
# ======================================================================


class TestConstruction:
    """BinanceLiveTrader constructs without network."""

    def test_construct_with_fake_clients(self) -> None:
        t = _make_trader()
        assert t.symbol == "ETHUSDT"
        assert t.contract_multiplier == Decimal("0.1")
        assert t.contract_precision == Decimal("0.01")
        assert t.min_contracts == Decimal("0.01")
        assert t.broker_exchange_name == "binance"

    def test_start_and_close_call_transport(self) -> None:
        t = _make_trader()

        async def run():
            await t.start()
            await t.close()

        asyncio.get_event_loop().run_until_complete(run())

    def test_protocol_attributes(self) -> None:
        t = _make_trader()
        assert isinstance(t.account_equity_usdt, float)
        assert isinstance(t.position_contracts, Decimal)
        assert isinstance(t.leverage, int)


# ======================================================================
# Quantity conversion
# ======================================================================


class TestQuantityConversion:
    """eth_qty_to_contracts and contracts_to_eth_qty."""

    def test_eth_qty_to_contracts(self) -> None:
        t = _make_trader()
        assert t.eth_qty_to_contracts(Decimal("0.1")) == Decimal("1")
        assert t.eth_qty_to_contracts(Decimal("0.05")) == Decimal("0.5")
        assert t.eth_qty_to_contracts(Decimal("1")) == Decimal("10")

    def test_contracts_to_eth_qty(self) -> None:
        t = _make_trader()
        assert t.contracts_to_eth_qty(Decimal("1")) == Decimal("0.1")
        assert t.contracts_to_eth_qty(Decimal("0.5")) == Decimal("0.05")

    def test_round_trip(self) -> None:
        t = _make_trader()
        eth = Decimal("0.15")
        contracts = t.eth_qty_to_contracts(eth)
        back = t.contracts_to_eth_qty(contracts)
        assert abs(back - eth) <= t.contract_precision * t.contract_multiplier

    def test_decimal_to_str(self) -> None:
        assert BinanceLiveTrader.decimal_to_str(Decimal("1.5")) == "1.5"
        assert BinanceLiveTrader.decimal_to_str(Decimal("0.1")) == "0.1"
        assert BinanceLiveTrader.decimal_to_str(Decimal("100")) == "100"


# ======================================================================
# PositionSnapshot mapping
# ======================================================================


class TestFetchPositionSnapshot:
    """fetch_position_snapshot via fake broker client."""

    def test_no_position(self) -> None:
        broker = FakeBrokerClient()
        broker._position = None
        t = _make_trader(broker_client=broker)

        async def run():
            return await t.fetch_position_snapshot()

        snap = asyncio.get_event_loop().run_until_complete(run())
        assert snap.side is None
        assert snap.contracts == Decimal("0")
        assert snap.eth_qty == 0.0

    def test_long_position(self) -> None:
        from src.exchanges.models import (
            BrokerPosition,
            BrokerPositionSide,
            BrokerQuantityUnit,
            ExchangeName,
        )

        broker = FakeBrokerClient()
        broker._position = BrokerPosition(
            exchange=ExchangeName.BINANCE,
            symbol="ETHUSDT",
            position_side=BrokerPositionSide.LONG,
            quantity=Decimal("0.1"),
            quantity_unit=BrokerQuantityUnit.BASE_ASSET,
            average_entry_price=Decimal("3000"),
        )
        t = _make_trader(broker_client=broker)

        async def run():
            return await t.fetch_position_snapshot()

        snap = asyncio.get_event_loop().run_until_complete(run())
        assert snap.side == "LONG"
        assert snap.contracts == Decimal("1")  # 0.1 ETH / 0.1
        assert snap.eth_qty == 0.1
        assert snap.avg_entry_price == 3000.0

    def test_short_position(self) -> None:
        from src.exchanges.models import (
            BrokerPosition,
            BrokerPositionSide,
            BrokerQuantityUnit,
            ExchangeName,
        )

        broker = FakeBrokerClient()
        broker._position = BrokerPosition(
            exchange=ExchangeName.BINANCE,
            symbol="ETHUSDT",
            position_side=BrokerPositionSide.SHORT,
            quantity=Decimal("0.2"),
            quantity_unit=BrokerQuantityUnit.BASE_ASSET,
            average_entry_price=Decimal("3200"),
        )
        t = _make_trader(broker_client=broker)

        async def run():
            return await t.fetch_position_snapshot()

        snap = asyncio.get_event_loop().run_until_complete(run())
        assert snap.side == "SHORT"
        assert snap.contracts == Decimal("2")
        assert snap.eth_qty == 0.2


# ======================================================================
# execute_intent — OPEN_LONG
# ======================================================================


class TestExecuteOpenLong:
    """OPEN_LONG builds correct semantic request."""

    @pytest.mark.asyncio
    async def test_open_long_builds_semantic_request(self) -> None:
        exec = FakeSemanticExecutor()
        t = _make_trader(semantic_executor=exec)
        t.position_contracts = Decimal("0")

        broker = FakeBrokerClient()
        broker._position = None
        t._broker_client = broker

        intent = _make_intent("OPEN_LONG", "LONG", eth_qty=0.1, price=3000, tp_price=3100)

        t._max_order_notional = Decimal("10000")
        t._max_position_notional = Decimal("10000")

        result = await t.execute_intent(intent)

        assert result.entry_filled
        assert result.ok
        reqs = exec.requests
        assert len(reqs) >= 1
        entry_req = reqs[0]
        assert entry_req.symbol == "ETHUSDT"
        assert entry_req.client_order_id.startswith(CLIENT_ORDER_ID_PREFIX)

    @pytest.mark.asyncio
    async def test_open_long_order_cap_exceeded(self) -> None:
        """When order notional > cap, no semantic call is made."""
        exec = FakeSemanticExecutor()
        t = _make_trader(semantic_executor=exec)
        t._max_order_notional = Decimal("10")

        intent = _make_intent("OPEN_LONG", "LONG", eth_qty=0.1, price=3000, tp_price=3100)

        result = await t.execute_intent(intent)
        assert not result.ok
        assert "live_max_order_notional_exceeded" in result.message
        assert len(exec.requests) == 0

    @pytest.mark.asyncio
    async def test_open_long_position_cap_exceeded(self) -> None:
        """When projected position notional > cap, no semantic call."""
        exec = FakeSemanticExecutor()
        t = _make_trader(semantic_executor=exec)
        t.position_contracts = Decimal("10")
        t._max_order_notional = Decimal("10000")
        t._max_position_notional = Decimal("100")

        intent = _make_intent("ADD_LONG", "LONG", eth_qty=0.1, price=3000, tp_price=3100)

        result = await t.execute_intent(intent)
        assert not result.ok
        assert "live_max_position_notional_exceeded" in result.message
        assert len(exec.requests) == 0


# ======================================================================
# execute_intent — unsupported intent
# ======================================================================


class TestUnsupportedIntent:
    """Unsupported intents return ok=False."""

    @pytest.mark.asyncio
    async def test_unsupported_intent(self) -> None:
        t = _make_trader()
        base_intent = _make_intent("OPEN_LONG", "LONG", eth_qty=0.1, price=3000, tp_price=3100)
        intent = TradeIntent(
            intent_type="UNSUPPORTED_FOO",
            side="LONG",
            price=3000,
            layer_index=0,
            tp_price=3100,
            reason="test",
            size=base_intent.size,
            fast_cvd=0.0,
            previous_fast_cvd=0.0,
            buy_ratio=0.5,
            sell_ratio=0.5,
            boll_upper=3100,
            boll_middle=3000,
            boll_lower=2900,
            ts_ms=0,
            avg_entry_price=3000,
            breakeven_price=3000,
            tp_mode="SINGLE",
        )
        result = await t.execute_intent(intent)
        assert not result.ok
        assert "unsupported_binance_intent" in result.message


# ======================================================================
# execute_intent — MARKET_EXIT
# ======================================================================


class TestMarketExit:
    """MARKET_EXIT and MARKET_EXIT_RUNNER."""

    @pytest.mark.asyncio
    async def test_market_exit_reduce_only(self) -> None:
        exec = FakeSemanticExecutor()
        broker = FakeBrokerClient()
        from src.exchanges.models import (
            BrokerPosition,
            BrokerPositionSide,
            BrokerQuantityUnit,
            ExchangeName,
        )
        broker._position = BrokerPosition(
            exchange=ExchangeName.BINANCE,
            symbol="ETHUSDT",
            position_side=BrokerPositionSide.LONG,
            quantity=Decimal("0.1"),
            quantity_unit=BrokerQuantityUnit.BASE_ASSET,
        )
        t = _make_trader(semantic_executor=exec, broker_client=broker)
        t.position_contracts = Decimal("1")
        t._max_order_notional = Decimal("10000")
        t._max_position_notional = Decimal("10000")

        intent = _make_intent("MARKET_EXIT", "LONG")

        result = await t.execute_intent(intent)
        assert result.ok
        reqs = exec.requests
        assert len(reqs) >= 1
        exit_req = reqs[0]
        assert exit_req.reduce_only is True


# ======================================================================
# UPDATE_TP
# ======================================================================


class TestUpdateTp:
    """UPDATE_TP cancels old TP and places new one."""

    @pytest.mark.asyncio
    async def test_update_tp_places_new_tp(self) -> None:
        exec = FakeSemanticExecutor()
        broker = FakeBrokerClient()
        from src.exchanges.models import (
            BrokerPosition,
            BrokerPositionSide,
            BrokerQuantityUnit,
            ExchangeName,
        )
        broker._position = BrokerPosition(
            exchange=ExchangeName.BINANCE,
            symbol="ETHUSDT",
            position_side=BrokerPositionSide.LONG,
            quantity=Decimal("0.1"),
            quantity_unit=BrokerQuantityUnit.BASE_ASSET,
        )
        t = _make_trader(semantic_executor=exec, broker_client=broker)
        t.position_contracts = Decimal("1")

        intent = _make_intent("UPDATE_TP", "LONG", tp_price=3150)

        result = await t.execute_intent(intent)
        assert result.ok
        tp_requests = [r for r in exec.requests
                       if hasattr(r, 'action') and 'TP' in str(r.action)]
        assert len(tp_requests) >= 1


# ======================================================================
# Boundary checks
# ======================================================================


class TestBoundaries:
    """BinanceLiveTrader boundary checks."""

    def test_no_btc(self) -> None:
        t = _make_trader()
        assert t.symbol == "ETHUSDT"
        assert "BTC" not in t.symbol

    def test_no_spot(self) -> None:
        t = _make_trader()
        assert t.symbol == "ETHUSDT"

    def test_client_order_id_prefix(self) -> None:
        assert CLIENT_ORDER_ID_PREFIX == "RE_MAIN_"
        assert not CLIENT_ORDER_ID_PREFIX.startswith("RE_SMOKE_")

    def test_fetch_broker_open_orders_returns_tuple(self) -> None:
        t = _make_trader()

        async def run():
            return await t.fetch_broker_open_orders()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert isinstance(result, tuple)

    def test_fetch_broker_algo_orders_returns_tuple(self) -> None:
        t = _make_trader()

        async def run():
            return await t.fetch_broker_algo_orders()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert isinstance(result, tuple)

    def test_recover_broker_open_orders_returns_tuple(self) -> None:
        t = _make_trader()

        async def run():
            return await t.recover_broker_open_orders()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert isinstance(result, tuple)

    def test_mark_flat_clears_state(self) -> None:
        t = _make_trader()
        t.position_contracts = Decimal("5")
        t.tp_order_id = "tp-123"
        t._managed_order_ids.add("tp-123")
        t.mark_flat()
        assert t.position_contracts == Decimal("0")
        assert t.tp_order_id is None
        assert len(t._managed_order_ids) == 0
