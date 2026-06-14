#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_trading_client_port_boundaries.py
@Description: Boundary tests for TradingClientPort — the port must NOT
              import concrete exchanges, Trader, or specific symbols.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

# ======================================================================
# Paths
# ======================================================================

_PORT_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "trading_client_port.py"


def _read_port_text() -> str:
    return _PORT_PATH.read_text(encoding="utf-8")


# ======================================================================
# File existence / compilation
# ======================================================================


def test_port_file_exists() -> None:
    assert _PORT_PATH.exists(), f"TradingClientPort file not found at {_PORT_PATH}"
    assert _PORT_PATH.is_file()


def test_port_file_compiles() -> None:
    text = _read_port_text()
    compile(text, str(_PORT_PATH), "exec")


# ======================================================================
# Import-ability
# ======================================================================


def test_port_can_be_imported() -> None:
    from src.execution.trading_client_port import TradingClientPort  # noqa: F401


def test_dtos_can_be_imported() -> None:
    from src.execution.trading_client_port import (
        BalanceSnapshot,
        CancelResult,
        OrderResult,
        OrderSnapshot,
        PositionSnapshot,
    )  # noqa: F401


# ======================================================================
# Forbidden tokens — no concrete exchange references
# ======================================================================


class TestNoConcreteExchangeReferences:
    """The port source must NOT reference any concrete exchange."""

    def test_no_okx_reference(self) -> None:
        text = _read_port_text()
        assert "OKX" not in text
        assert "okx" not in text

    def test_no_binance_reference(self) -> None:
        text = _read_port_text()
        assert "Binance" not in text
        assert "binance" not in text

    def test_no_ethusdt_symbol(self) -> None:
        text = _read_port_text()
        assert "ETH-USDT-SWAP" not in text
        assert "ETHUSDT" not in text

    def test_no_api_paths(self) -> None:
        text = _read_port_text()
        assert "/api/v5" not in text
        assert "/fapi" not in text

    def test_no_trader_import(self) -> None:
        text = _read_port_text()
        assert "src.execution.trader" not in text
        assert "from src.execution.trader" not in text
        assert "import Trader" not in text

    def test_no_okx_private_client(self) -> None:
        text = _read_port_text()
        assert "okx_private_client" not in text

    def test_no_binance_client_import(self) -> None:
        text = _read_port_text()
        assert "src.exchanges" not in text


# ======================================================================
# DTO behaviour
# ======================================================================


class TestDtoBehaviour:
    """DTOs must behave as expected."""

    def test_balance_snapshot_defaults(self) -> None:
        from src.execution.trading_client_port import BalanceSnapshot

        b = BalanceSnapshot(asset="USDT", total=Decimal("100.0"))
        assert b.asset == "USDT"
        assert b.total == Decimal("100.0")
        assert b.available is None
        assert b.raw == {}

    def test_position_snapshot_has_position_true(self) -> None:
        from src.execution.trading_client_port import PositionSnapshot

        p = PositionSnapshot(side="long", qty=Decimal("1.5"))
        assert p.has_position is True

    def test_position_snapshot_has_position_false_no_side(self) -> None:
        from src.execution.trading_client_port import PositionSnapshot

        p = PositionSnapshot(side=None, qty=Decimal("0"))
        assert p.has_position is False

    def test_position_snapshot_has_position_false_zero_qty(self) -> None:
        from src.execution.trading_client_port import PositionSnapshot

        p = PositionSnapshot(side="long", qty=Decimal("0"))
        assert p.has_position is False

    def test_order_result_defaults(self) -> None:
        from src.execution.trading_client_port import OrderResult

        r = OrderResult(ok=True)
        assert r.ok is True
        assert r.order_id is None
        assert r.message == ""

    def test_cancel_result_defaults(self) -> None:
        from src.execution.trading_client_port import CancelResult

        r = CancelResult(ok=False, message="not found")
        assert r.ok is False
        assert r.message == "not found"

    def test_order_snapshot_defaults(self) -> None:
        from src.execution.trading_client_port import OrderSnapshot

        o = OrderSnapshot(
            order_id="123",
            client_order_id="abc",
            side="buy",
            qty=Decimal("1.0"),
        )
        assert o.reduce_only is False
        assert o.price is None
        assert o.trigger_price is None

    def test_dtos_are_frozen(self) -> None:
        from src.execution.trading_client_port import OrderResult

        r = OrderResult(ok=True, order_id="123")
        with pytest.raises(Exception):
            r.order_id = "456"  # type: ignore[misc]

    def test_balance_snapshot_is_frozen(self) -> None:
        from src.execution.trading_client_port import BalanceSnapshot

        b = BalanceSnapshot(asset="USDT", total=Decimal("100"))
        with pytest.raises(Exception):
            b.total = Decimal("200")  # type: ignore[misc]


# Local import for mypy's sake
import pytest  # noqa: E402
