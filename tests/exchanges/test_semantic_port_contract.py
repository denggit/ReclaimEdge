#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : test_semantic_port_contract.py
@Description: Contract tests for BrokerSemanticExecutor convenience methods.

Uses a RecordingSemanticExecutor fake to verify every convenience method
constructs the correct BrokerSemanticRequest and delegates to execute().
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.models import (
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
    BrokerSemanticResult,
)
from src.exchanges.semantics import BrokerSemanticExecutor


# ---------------------------------------------------------------------------
# Fake / Recording executor
# ---------------------------------------------------------------------------


class RecordingSemanticExecutor(BrokerSemanticExecutor):
    """A test double that records every request and returns a trivial OK."""

    def __init__(self) -> None:
        self.requests: list[BrokerSemanticRequest] = []

    @property
    def exchange(self) -> ExchangeName:
        return ExchangeName.OKX

    async def execute(self, request: BrokerSemanticRequest) -> BrokerSemanticResult:
        self.requests.append(request)
        return BrokerSemanticResult(
            exchange=request.exchange,
            symbol=request.symbol,
            action=request.action,
            role=request.role,
            ok=True,
            message="ok",
        )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def executor() -> RecordingSemanticExecutor:
    return RecordingSemanticExecutor()


# ---------------------------------------------------------------------------
# Convenience method tests
# ---------------------------------------------------------------------------


class TestConvenienceMethodsDelegateToExecute:
    """Every convenience method must record exactly one request."""

    @pytest.mark.asyncio
    async def test_place_reduce_only_tp_action_default_role_is_core_tp(
        self, executor: RecordingSemanticExecutor,
    ) -> None:
        await executor.place_reduce_only_tp(
            symbol="BTC-USDT-SWAP",
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            trigger_price=Decimal("52000"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.PLACE_REDUCE_ONLY_TP
        assert req.role == BrokerSemanticOrderRole.CORE_TP
        assert req.reduce_only is True
        assert req.price == Decimal("52000")
        assert req.trigger_price is None

    @pytest.mark.asyncio
    async def test_place_reduce_only_tp_action_explicit_role_tp1(
        self, executor: RecordingSemanticExecutor,
    ) -> None:
        await executor.place_reduce_only_tp(
            symbol="BTC-USDT-SWAP",
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            trigger_price=Decimal("52000"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            role=BrokerSemanticOrderRole.TP1,
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.PLACE_REDUCE_ONLY_TP
        assert req.role == BrokerSemanticOrderRole.TP1
        assert req.reduce_only is True
        assert req.price == Decimal("52000")
        assert req.trigger_price is None

    @pytest.mark.asyncio
    async def test_place_reduce_only_tp_order_price_takes_precedence(
        self, executor: RecordingSemanticExecutor,
    ) -> None:
        await executor.place_reduce_only_tp(
            symbol="ETH-USDT-SWAP",
            side=BrokerPositionSide.LONG,
            quantity=Decimal("10"),
            trigger_price=Decimal("3500"),
            order_price=Decimal("3501"),
        )

        req = executor.requests[-1]
        assert req.action == BrokerSemanticAction.PLACE_REDUCE_ONLY_TP
        assert req.price == Decimal("3501")
        assert req.trigger_price is None

    @pytest.mark.asyncio
    async def test_place_protective_stop_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.place_protective_stop(
            symbol="ETH-USDT-SWAP",
            side=BrokerPositionSide.SHORT,
            quantity=Decimal("2"),
            trigger_price=Decimal("2800"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.PLACE_PROTECTIVE_STOP
        assert req.role == BrokerSemanticOrderRole.PROTECTIVE_SL
        assert req.reduce_only is True

    @pytest.mark.asyncio
    async def test_market_exit_runner_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.market_exit_runner(
            symbol="BTC-USDT-SWAP",
            side=BrokerPositionSide.LONG,
            quantity=Decimal("0.5"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.MARKET_EXIT_RUNNER
        assert req.role == BrokerSemanticOrderRole.MARKET_EXIT
        assert req.reduce_only is True

    @pytest.mark.asyncio
    async def test_fetch_open_orders_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.fetch_open_orders(symbol="BTC-USDT-SWAP")
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.FETCH_OPEN_ORDERS

    @pytest.mark.asyncio
    async def test_recover_open_orders_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.recover_open_orders(symbol="BTC-USDT-SWAP")
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.RECOVER_OPEN_ORDERS
        assert req.role == BrokerSemanticOrderRole.RECOVERY

    @pytest.mark.asyncio
    async def test_open_position_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.open_position(
            symbol="BTC-USDT-SWAP",
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            price=Decimal("50000"),
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.OPEN_POSITION
        assert req.role == BrokerSemanticOrderRole.ENTRY
        assert req.price == Decimal("50000")

    @pytest.mark.asyncio
    async def test_add_position_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.add_position(
            symbol="BTC-USDT-SWAP",
            side=BrokerPositionSide.LONG,
            quantity=Decimal("0.5"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.ADD_POSITION
        assert req.role == BrokerSemanticOrderRole.ADD

    @pytest.mark.asyncio
    async def test_cancel_order_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.cancel_order(symbol="BTC-USDT-SWAP", order_id="okx-123")
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.CANCEL_ORDER
        assert req.order_id == "okx-123"

    @pytest.mark.asyncio
    async def test_cancel_reduce_only_tp_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.cancel_reduce_only_tp(
            symbol="BTC-USDT-SWAP",
            order_id="okx-456",
            role=BrokerSemanticOrderRole.CORE_TP,
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP
        assert req.order_id == "okx-456"

    @pytest.mark.asyncio
    async def test_cancel_protective_stop_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.cancel_protective_stop(
            symbol="BTC-USDT-SWAP",
            order_id="okx-789",
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.CANCEL_PROTECTIVE_STOP
        assert req.role == BrokerSemanticOrderRole.PROTECTIVE_SL

    @pytest.mark.asyncio
    async def test_market_exit_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.market_exit(
            symbol="BTC-USDT-SWAP",
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.MARKET_EXIT
        assert req.reduce_only is True

    @pytest.mark.asyncio
    async def test_sidecar_entry_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.sidecar_entry(
            symbol="BTC-USDT-SWAP",
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            price=Decimal("50000"),
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.SIDECAR_ENTRY
        assert req.role == BrokerSemanticOrderRole.SIDECAR_ENTRY

    @pytest.mark.asyncio
    async def test_sidecar_tp_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.sidecar_tp(
            symbol="BTC-USDT-SWAP",
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            trigger_price=Decimal("52000"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.SIDECAR_TP
        assert req.role == BrokerSemanticOrderRole.SIDECAR_TP
        assert req.reduce_only is True
        assert req.price == Decimal("52000")
        assert req.trigger_price is None

    @pytest.mark.asyncio
    async def test_sidecar_tp_order_price_takes_precedence(
        self, executor: RecordingSemanticExecutor,
    ) -> None:
        await executor.sidecar_tp(
            symbol="ETH-USDT-SWAP",
            side=BrokerPositionSide.LONG,
            quantity=Decimal("10"),
            trigger_price=Decimal("3500"),
            order_price=Decimal("3501"),
        )

        req = executor.requests[-1]
        assert req.action == BrokerSemanticAction.SIDECAR_TP
        assert req.price == Decimal("3501")
        assert req.trigger_price is None

    @pytest.mark.asyncio
    async def test_fetch_position_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.fetch_position(symbol="BTC-USDT-SWAP")
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.FETCH_POSITION

    @pytest.mark.asyncio
    async def test_fetch_algo_orders_action(self, executor: RecordingSemanticExecutor) -> None:
        await executor.fetch_algo_orders(symbol="BTC-USDT-SWAP")
        assert len(executor.requests) == 1
        req = executor.requests[0]
        assert req.action == BrokerSemanticAction.FETCH_ALGO_ORDERS


class TestAllConvenienceMethodsCallExecuteOnly:
    """Smoke test – every convenience method returns a BrokerSemanticResult."""

    @pytest.mark.asyncio
    async def test_all_methods_return_result(self) -> None:
        executor = RecordingSemanticExecutor()

        methods_and_kwargs = [
            (
                executor.open_position,
                {
                    "symbol": "X-USDT-SWAP",
                    "side": BrokerPositionSide.LONG,
                    "quantity": Decimal("1"),
                },
            ),
            (
                executor.add_position,
                {
                    "symbol": "X-USDT-SWAP",
                    "side": BrokerPositionSide.LONG,
                    "quantity": Decimal("1"),
                },
            ),
            (
                executor.place_reduce_only_tp,
                {
                    "symbol": "X-USDT-SWAP",
                    "side": BrokerPositionSide.LONG,
                    "quantity": Decimal("1"),
                    "trigger_price": Decimal("100"),
                },
            ),
            (
                executor.place_protective_stop,
                {
                    "symbol": "X-USDT-SWAP",
                    "side": BrokerPositionSide.LONG,
                    "quantity": Decimal("1"),
                    "trigger_price": Decimal("90"),
                },
            ),
            (
                executor.cancel_order,
                {"symbol": "X-USDT-SWAP", "order_id": "1"},
            ),
            (
                executor.cancel_reduce_only_tp,
                {"symbol": "X-USDT-SWAP", "order_id": "1"},
            ),
            (
                executor.cancel_protective_stop,
                {"symbol": "X-USDT-SWAP", "order_id": "1"},
            ),
            (
                executor.market_exit,
                {
                    "symbol": "X-USDT-SWAP",
                    "side": BrokerPositionSide.LONG,
                    "quantity": Decimal("1"),
                },
            ),
            (
                executor.market_exit_runner,
                {
                    "symbol": "X-USDT-SWAP",
                    "side": BrokerPositionSide.LONG,
                    "quantity": Decimal("1"),
                },
            ),
            (
                executor.sidecar_entry,
                {
                    "symbol": "X-USDT-SWAP",
                    "side": BrokerPositionSide.LONG,
                    "quantity": Decimal("1"),
                },
            ),
            (
                executor.sidecar_tp,
                {
                    "symbol": "X-USDT-SWAP",
                    "side": BrokerPositionSide.LONG,
                    "quantity": Decimal("1"),
                    "trigger_price": Decimal("100"),
                },
            ),
            (executor.fetch_position, {"symbol": "X-USDT-SWAP"}),
            (executor.fetch_open_orders, {"symbol": "X-USDT-SWAP"}),
            (executor.fetch_algo_orders, {"symbol": "X-USDT-SWAP"}),
            (executor.recover_open_orders, {"symbol": "X-USDT-SWAP"}),
        ]

        for method, kwargs in methods_and_kwargs:
            result = await method(**kwargs)
            assert isinstance(result, BrokerSemanticResult)
            assert result.ok is True

        # Every call should have recorded exactly one request
        assert len(executor.requests) == len(methods_and_kwargs)
