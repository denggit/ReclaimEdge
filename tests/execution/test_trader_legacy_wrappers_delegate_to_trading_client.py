#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_trader_legacy_wrappers_delegate_to_trading_client.py
@Description: Boundary tests verifying Trader legacy wrappers delegate to
              TradingClientPort and do NOT contain direct /api/v5 calls.
"""

from __future__ import annotations

from pathlib import Path


TRADER_SOURCE = Path(__file__).resolve().parents[2] / "src" / "execution" / "trader.py"


def _read_source() -> str:
    return TRADER_SOURCE.read_text(encoding="utf-8")


def _method_source(method_name: str) -> str | None:
    """Extract the source of a specific async def from trader.py."""
    import ast
    tree = ast.parse(_read_source())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return ast.get_source_segment(_read_source(), node)
    return None


class TestFetchUsdtEquityDelegates:
    """fetch_usdt_equity must delegate to TradingClientPort.fetch_balance()."""

    def test_does_not_contain_api_v5(self) -> None:
        src = _method_source("fetch_usdt_equity")
        assert src is not None, "fetch_usdt_equity method not found"
        assert "/api/v5" not in src, "fetch_usdt_equity must NOT contain /api/v5"

    def test_calls_fetch_balance(self) -> None:
        src = _method_source("fetch_usdt_equity")
        assert src is not None
        assert "fetch_balance()" in src, "fetch_usdt_equity must call trading_client.fetch_balance()"


class TestFetchPositionSnapshotDelegates:
    """fetch_position_snapshot must delegate to TradingClientPort.fetch_position()."""

    def test_does_not_contain_api_v5(self) -> None:
        src = _method_source("fetch_position_snapshot")
        assert src is not None, "fetch_position_snapshot method not found"
        assert "/api/v5" not in src, "fetch_position_snapshot must NOT contain /api/v5"

    def test_calls_fetch_position(self) -> None:
        src = _method_source("fetch_position_snapshot")
        assert src is not None
        assert "fetch_position()" in src, "fetch_position_snapshot must call trading_client.fetch_position()"


class TestFetchPendingOrdersDelegates:
    """fetch_pending_orders must delegate to TradingClientPort.fetch_open_orders()."""

    def test_does_not_contain_api_v5_as_primary_path(self) -> None:
        src = _method_source("fetch_pending_orders")
        assert src is not None, "fetch_pending_orders method not found"
        # The /api/v5 reference must not be in this method
        assert "/api/v5/trade/orders-pending" not in src, (
            "fetch_pending_orders must NOT contain /api/v5/trade/orders-pending"
        )

    def test_calls_fetch_open_orders(self) -> None:
        src = _method_source("fetch_pending_orders")
        assert src is not None
        assert "fetch_open_orders()" in src, "fetch_pending_orders must call trading_client.fetch_open_orders()"


class TestFetchPendingAlgoOrdersDelegates:
    """fetch_pending_algo_orders must delegate to TradingClientPort.fetch_open_algo_orders()."""

    def test_does_not_contain_api_v5_as_primary_path(self) -> None:
        src = _method_source("fetch_pending_algo_orders")
        assert src is not None, "fetch_pending_algo_orders method not found"
        assert "/api/v5/trade/orders-algo-pending" not in src, (
            "fetch_pending_algo_orders must NOT contain /api/v5/trade/orders-algo-pending"
        )

    def test_calls_fetch_open_algo_orders(self) -> None:
        src = _method_source("fetch_pending_algo_orders")
        assert src is not None
        assert "fetch_open_algo_orders()" in src, (
            "fetch_pending_algo_orders must call trading_client.fetch_open_algo_orders()"
        )


class TestSetLeverageDelegates:
    """set_leverage must delegate to TradingClientPort.configure_instrument()."""

    def test_does_not_contain_api_v5(self) -> None:
        src = _method_source("set_leverage")
        assert src is not None, "set_leverage method not found"
        assert "/api/v5" not in src, "set_leverage must NOT contain /api/v5"

    def test_calls_configure_instrument(self) -> None:
        src = _method_source("set_leverage")
        assert src is not None
        assert "configure_instrument()" in src, (
            "set_leverage must call trading_client.configure_instrument()"
        )
