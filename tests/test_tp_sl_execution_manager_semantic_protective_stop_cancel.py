#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_tp_sl_execution_manager_semantic_protective_stop_cancel.py
@Description: Tests for the optional semantic protective SL cancel path.
"""

from __future__ import annotations

import pytest

from src.exchanges.models import ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticResult,
)
from src.execution.okx_trading_client import OkxTradingClient
from src.execution.tp_sl_execution_manager import TpSlExecutionManager


class FakeSemanticExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.ok = True
        self.message = ""

    async def cancel_protective_stop(
        self,
        *,
        symbol: str,
        order_id: str,
    ) -> BrokerSemanticResult:
        self.calls.append((symbol, order_id))
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            ok=self.ok,
            order_id=order_id,
            message=self.message,
        )


class FakeTrader:
    symbol = "ETH-USDT-SWAP"

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, object | None]] = []
        self.semantic = FakeSemanticExecutor()
        self.near_tp_protective_sl_order_id = "algo-1"
        self.middle_runner_protective_sl_order_id = "algo-1"
        self.trend_runner_sl_order_id = "algo-1"
        self.three_stage_post_tp1_protective_sl_order_id = "algo-1"
        self.middle_bucket_fast_sl_order_id = "algo-1"
        self._manager: TpSlExecutionManager | None = None
        self.pos_side_mode = "net"
        self.contract_multiplier = __import__("decimal").Decimal("0.1")

        # Create a fake trading client that records cancel_algo_order calls
        from src.execution.trading_client_port import CancelResult
        class _FakeTC:
            def __init__(self, outer):
                self._outer = outer
            async def fetch_open_orders(self):
                return []
            async def fetch_open_algo_orders(self):
                return ()
            async def cancel_order(self, *, order_id=None, client_order_id=None):
                return CancelResult(ok=True, order_id=order_id)
            async def cancel_algo_order(self, *, order_id=None, client_order_id=None):
                if order_id:
                    body = [{"instId": self._outer.symbol, "algoId": order_id}]
                else:
                    body = {"instId": self._outer.symbol, "algoClOrdId": client_order_id}
                self._outer.requests.append(("POST", "/api/v5/trade/cancel-algos", body))
                return CancelResult(ok=True, order_id=order_id)
            async def place_market_order(self, **kwargs):
                from src.execution.trading_client_port import OrderResult
                return OrderResult(ok=True, order_id="fake-order")
            async def place_limit_order(self, **kwargs):
                from src.execution.trading_client_port import OrderResult
                return OrderResult(ok=True, order_id="fake-order")
            async def place_stop_market_order(self, **kwargs):
                from src.execution.trading_client_port import OrderResult
                return OrderResult(ok=True, order_id="fake-order")
            async def fetch_balance(self):
                from src.execution.trading_client_port import BalanceSnapshot
                return BalanceSnapshot(asset="USDT", total=__import__("decimal").Decimal("100"))
            async def fetch_position(self):
                from src.execution.trading_client_port import PositionSnapshot
                return PositionSnapshot(side=None, qty=__import__("decimal").Decimal("0"))
            async def fetch_order_status(self, **kwargs):
                from src.execution.trading_client_port import OrderStatusSnapshot
                return OrderStatusSnapshot(order_id=None, status="UNKNOWN")
            async def configure_instrument(self):
                pass
        self.trading_client = _FakeTC(self)

    @property
    def broker_semantic_executor(self) -> FakeSemanticExecutor:
        return self.semantic

    async def request(self, method: str, endpoint: str, payload=None) -> dict:
        self.requests.append((method, endpoint, payload))
        return {"data": [{"algoId": payload[0]["algoId"], "sCode": "0"}]}

    async def cancel_near_tp_protective_stop(self, order_id: str | None) -> bool:
        assert self._manager is not None
        return await self._manager.cancel_near_tp_protective_stop(order_id)


def make_manager() -> tuple[FakeTrader, TpSlExecutionManager]:
    trader = FakeTrader()
    manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
    trader._tp_sl_manager = manager  # type: ignore[assignment]
    trader._manager = manager
    return trader, manager


@pytest.mark.asyncio
async def test_default_off_uses_legacy_cancel_algos(monkeypatch) -> None:
    monkeypatch.delenv("BROKER_SEMANTIC_PROTECTIVE_SL_CANCEL_ENABLED", raising=False)
    trader, manager = make_manager()

    ok = await manager.cancel_near_tp_protective_stop("algo-1")

    assert ok is True
    assert len(trader.requests) == 1
    method, endpoint, payload = trader.requests[0]
    assert method == "POST"
    assert endpoint == "/api/v5/trade/cancel-algos"
    assert payload == [{"instId": "ETH-USDT-SWAP", "algoId": "algo-1"}]
    assert trader.semantic.calls == []
    assert trader.near_tp_protective_sl_order_id is None


@pytest.mark.asyncio
async def test_enabled_uses_semantic_cancel_without_legacy_request(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_CANCEL_ENABLED", "true")
    trader, manager = make_manager()

    ok = await manager.cancel_near_tp_protective_stop("algo-1")

    assert ok is True
    assert trader.semantic.calls == [("ETH-USDT-SWAP", "algo-1")]
    assert trader.requests == []
    assert trader.near_tp_protective_sl_order_id is None


@pytest.mark.asyncio
async def test_semantic_failure_returns_false_without_legacy_fallback(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_CANCEL_ENABLED", "true")
    trader, manager = make_manager()
    trader.semantic.ok = False
    trader.semantic.message = "boom"

    ok = await manager.cancel_near_tp_protective_stop("algo-1")

    assert ok is False
    assert trader.semantic.calls == [("ETH-USDT-SWAP", "algo-1")]
    assert trader.requests == []
    assert trader.near_tp_protective_sl_order_id == "algo-1"


@pytest.mark.asyncio
async def test_semantic_already_absent_returns_true_without_legacy_request(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_CANCEL_ENABLED", "true")
    trader, manager = make_manager()
    trader.semantic.ok = False
    trader.semantic.message = "order not found"

    ok = await manager.cancel_near_tp_protective_stop("algo-1")

    assert ok is True
    assert trader.semantic.calls == [("ETH-USDT-SWAP", "algo-1")]
    assert trader.requests == []


@pytest.mark.asyncio
async def test_none_order_id_returns_true_without_any_cancel(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_CANCEL_ENABLED", "true")
    trader, manager = make_manager()

    ok = await manager.cancel_near_tp_protective_stop(None)

    assert ok is True
    assert trader.requests == []
    assert trader.semantic.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "attr_name"),
    [
        ("cancel_middle_runner_protective_stop", "middle_runner_protective_sl_order_id"),
        ("cancel_trend_runner_protective_stop", "trend_runner_sl_order_id"),
        (
            "cancel_three_stage_post_tp1_protective_stop",
            "three_stage_post_tp1_protective_sl_order_id",
        ),
        ("cancel_middle_bucket_fast_protective_stop", "middle_bucket_fast_sl_order_id"),
    ],
)
async def test_wrappers_reuse_center_cancel_with_semantic_enabled(
    monkeypatch,
    method_name: str,
    attr_name: str,
) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_CANCEL_ENABLED", "true")
    trader, manager = make_manager()
    method = getattr(manager, method_name)

    ok = await method("algo-1")

    assert ok is True
    assert getattr(trader, attr_name) is None
    assert trader.semantic.calls == [("ETH-USDT-SWAP", "algo-1")]
    assert trader.requests == []
