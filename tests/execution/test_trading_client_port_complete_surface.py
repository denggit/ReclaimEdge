#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_trading_client_port_complete_surface.py
@Description: Verify the complete TradingClientPort surface includes all
              strategy-required methods and DTOs.
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
# Protocol method presence
# ======================================================================


def test_port_has_configure_instrument() -> None:
    text = _read_port_text()
    assert "async def configure_instrument" in text, (
        "TradingClientPort must declare configure_instrument"
    )


def test_port_has_fetch_order_status() -> None:
    text = _read_port_text()
    assert "async def fetch_order_status" in text, (
        "TradingClientPort must declare fetch_order_status"
    )


def test_port_has_fetch_open_algo_orders() -> None:
    text = _read_port_text()
    assert "async def fetch_open_algo_orders" in text, (
        "TradingClientPort must declare fetch_open_algo_orders"
    )


# ======================================================================
# DTO field presence
# ======================================================================


def test_order_status_snapshot_fields() -> None:
    from src.execution.trading_client_port import OrderStatusSnapshot

    snap = OrderStatusSnapshot(
        order_id="ord-1",
        client_order_id="cid-1",
        status="OPEN",
        filled_qty=Decimal("0.5"),
        avg_fill_price=Decimal("3100.00"),
        raw={"state": "live"},
    )
    assert snap.order_id == "ord-1"
    assert snap.client_order_id == "cid-1"
    assert snap.status == "OPEN"
    assert snap.filled_qty == Decimal("0.5")
    assert snap.avg_fill_price == Decimal("3100.00")
    assert snap.raw == {"state": "live"}


def test_order_status_snapshot_defaults() -> None:
    from src.execution.trading_client_port import OrderStatusSnapshot

    snap = OrderStatusSnapshot(order_id="ord-1", client_order_id=None, status="UNKNOWN")
    assert snap.order_id == "ord-1"
    assert snap.client_order_id is None
    assert snap.status == "UNKNOWN"
    assert snap.filled_qty is None
    assert snap.avg_fill_price is None
    assert snap.raw == {}


def test_algo_order_snapshot_fields() -> None:
    from src.execution.trading_client_port import AlgoOrderSnapshot

    snap = AlgoOrderSnapshot(
        order_id="algo-1",
        client_order_id="cid-algo",
        side="sell",
        qty=Decimal("1.5"),
        trigger_price=Decimal("2900.00"),
        status="OPEN",
        raw={"algoId": "algo-1"},
    )
    assert snap.order_id == "algo-1"
    assert snap.client_order_id == "cid-algo"
    assert snap.side == "sell"
    assert snap.qty == Decimal("1.5")
    assert snap.trigger_price == Decimal("2900.00")
    assert snap.status == "OPEN"
    assert snap.raw == {"algoId": "algo-1"}


def test_algo_order_snapshot_defaults() -> None:
    from src.execution.trading_client_port import AlgoOrderSnapshot

    snap = AlgoOrderSnapshot(order_id="algo-1", client_order_id=None)
    assert snap.side is None
    assert snap.qty is None
    assert snap.trigger_price is None
    assert snap.status == "OPEN"
    assert snap.raw == {}


def test_dtos_are_frozen() -> None:
    from src.execution.trading_client_port import OrderStatusSnapshot, AlgoOrderSnapshot

    s = OrderStatusSnapshot(order_id="x", client_order_id=None, status="OPEN")
    with pytest.raises(Exception):
        s.status = "FILLED"  # type: ignore[misc]

    a = AlgoOrderSnapshot(order_id="x", client_order_id=None)
    with pytest.raises(Exception):
        a.order_id = "y"  # type: ignore[misc]


# ======================================================================
# No sidecar-specific names in port
# ======================================================================


def test_port_has_no_sidecar_specific_names() -> None:
    text = _read_port_text()
    assert "sidecar" not in text.lower(), (
        "TradingClientPort must not contain sidecar-specific names"
    )


def test_port_has_no_okx_specific_names() -> None:
    text = _read_port_text()
    assert "okx" not in text.lower(), (
        "TradingClientPort must not contain OKX-specific names"
    )


# ======================================================================
# Protocol can be imported
# ======================================================================


def test_new_dtos_importable() -> None:
    from src.execution.trading_client_port import (
        OrderStatusSnapshot,
        AlgoOrderSnapshot,
    )  # noqa: F401


import pytest  # noqa: E402
