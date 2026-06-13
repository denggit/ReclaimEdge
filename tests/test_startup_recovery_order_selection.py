"""Tests for startup recovery reduce-only order selection helpers.

These tests exercise the private helpers in
``src/live/startup_recovery/order_recovery`` without touching a live
exchange, confirming compatibility with both OKX raw dicts and
BrokerOrder-like objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.live.startup_recovery.order_recovery import (
    _order_id,
    _order_reduce_only,
    _order_symbol,
    _order_value,
    _select_recoverable_reduce_only_orders,
)


# ---------------------------------------------------------------------------
# Helper-specific unit tests
# ---------------------------------------------------------------------------


def test_order_value_from_dict() -> None:
    order = {"ordId": "abc-123", "instId": "ETH-USDT-SWAP"}
    assert _order_value(order, "ordId", "order_id") == "abc-123"
    assert _order_value(order, "order_id", "ordId") == "abc-123"
    assert _order_value(order, "order_id", "id") is None


def test_order_value_from_object() -> None:
    @dataclass
    class BrokerOrder:
        order_id: str
        symbol: str

    order = BrokerOrder(order_id="abc-123", symbol="ETH-USDT-SWAP")
    assert _order_value(order, "order_id", "ordId") == "abc-123"
    assert _order_value(order, "ordId", "order_id") == "abc-123"
    assert _order_value(order, "reduceOnly", "reduce_only") is None


def test_order_value_missing_fields() -> None:
    assert _order_value({}, "ordId", "order_id") is None
    assert _order_value({"instId": "X"}, "ordId", "order_id") is None


def test_order_id_okx_dict() -> None:
    assert _order_id({"ordId": "tp-1"}) == "tp-1"
    assert _order_id({"order_id": "tp-2"}) == "tp-2"
    assert _order_id({"id": "tp-3"}) == "tp-3"
    assert _order_id({}) == ""


def test_order_id_broker_order() -> None:
    @dataclass
    class BrokerOrder:
        order_id: str

    assert _order_id(BrokerOrder(order_id="tp-1")) == "tp-1"


def test_order_symbol_okx_dict() -> None:
    assert _order_symbol({"instId": "ETH-USDT-SWAP"}) == "ETH-USDT-SWAP"
    assert _order_symbol({"symbol": "BTC-USDT-SWAP"}) == "BTC-USDT-SWAP"
    assert _order_symbol({}) == ""


def test_order_symbol_broker_order() -> None:
    @dataclass
    class BrokerOrder:
        symbol: str

    assert _order_symbol(BrokerOrder(symbol="ETH-USDT-SWAP")) == "ETH-USDT-SWAP"


# ---------------------------------------------------------------------------
# reduce_only variants
# ---------------------------------------------------------------------------


def test_order_reduce_only_variants() -> None:
    assert _order_reduce_only({"reduceOnly": True}) is True
    assert _order_reduce_only({"reduceOnly": "true"}) is True
    assert _order_reduce_only({"reduceOnly": "TRUE"}) is True
    assert _order_reduce_only({"reduceOnly": "1"}) is True
    assert _order_reduce_only({"reduceOnly": "yes"}) is True
    assert _order_reduce_only({"reduceOnly": "y"}) is True
    assert _order_reduce_only({"reduceOnly": "on"}) is True
    assert _order_reduce_only({"reduceOnly": "YES"}) is True
    assert _order_reduce_only({"reduceOnly": "ON"}) is True
    assert _order_reduce_only({"reduceOnly": False}) is False
    assert _order_reduce_only({"reduceOnly": "false"}) is False
    assert _order_reduce_only({"reduceOnly": ""}) is False
    assert _order_reduce_only({"reduceOnly": "no"}) is False
    assert _order_reduce_only({"reduceOnly": "off"}) is False
    assert _order_reduce_only({"reduceOnly": 0}) is False
    assert _order_reduce_only({"reduceOnly": None}) is False
    assert _order_reduce_only({}) is False


def test_order_reduce_only_broker_order() -> None:
    @dataclass
    class BrokerOrder:
        reduce_only: bool

    assert _order_reduce_only(BrokerOrder(reduce_only=True)) is True
    assert _order_reduce_only(BrokerOrder(reduce_only=False)) is False


# ---------------------------------------------------------------------------
# _select_recoverable_reduce_only_orders
# ---------------------------------------------------------------------------


def test_select_recoverable_reduce_only_orders_from_okx_raw_dicts() -> None:
    pending = [
        {"instId": "ETH-USDT-SWAP", "reduceOnly": "true", "ordId": "tp-1"},
        {"instId": "ETH-USDT-SWAP", "reduceOnly": True, "ordId": "tp-2"},
        {"instId": "BTC-USDT-SWAP", "reduceOnly": "true", "ordId": "btc-tp"},
        {"instId": "ETH-USDT-SWAP", "reduceOnly": "false", "ordId": "entry-1"},
        {"instId": "ETH-USDT-SWAP", "reduceOnly": "true", "ordId": "sidecar-protected"},
    ]

    selected = _select_recoverable_reduce_only_orders(
        pending,
        symbol="ETH-USDT-SWAP",
        protected_order_ids={"sidecar-protected"},
    )

    assert [_order_id(item) for item in selected] == ["tp-1", "tp-2"]


def test_select_recoverable_reduce_only_orders_from_broker_order_like_objects() -> None:
    @dataclass
    class FakeBrokerOrder:
        symbol: str
        reduce_only: bool
        order_id: str

    pending = [
        FakeBrokerOrder(symbol="ETH-USDT-SWAP", reduce_only=True, order_id="tp-1"),
        FakeBrokerOrder(symbol="ETH-USDT-SWAP", reduce_only=False, order_id="entry-1"),
        FakeBrokerOrder(symbol="BTC-USDT-SWAP", reduce_only=True, order_id="btc-tp"),
        FakeBrokerOrder(symbol="ETH-USDT-SWAP", reduce_only=True, order_id="sidecar-protected"),
    ]

    selected = _select_recoverable_reduce_only_orders(
        pending,
        symbol="ETH-USDT-SWAP",
        protected_order_ids={"sidecar-protected"},
    )

    assert [_order_id(item) for item in selected] == ["tp-1"]


def test_select_recoverable_reduce_only_orders_empty() -> None:
    assert _select_recoverable_reduce_only_orders(
        [],
        symbol="ETH-USDT-SWAP",
        protected_order_ids=set(),
    ) == []


def test_select_recoverable_reduce_only_orders_all_protected() -> None:
    pending = [
        {"instId": "ETH-USDT-SWAP", "reduceOnly": "true", "ordId": "tp-1"},
        {"instId": "ETH-USDT-SWAP", "reduceOnly": "true", "ordId": "tp-2"},
    ]
    selected = _select_recoverable_reduce_only_orders(
        pending,
        symbol="ETH-USDT-SWAP",
        protected_order_ids={"tp-1", "tp-2"},
    )
    assert selected == []


def test_select_recoverable_reduce_only_orders_ignores_missing_fields() -> None:
    pending = [
        {},
        {"instId": "ETH-USDT-SWAP"},
        {"reduceOnly": "true"},
        {"instId": "ETH-USDT-SWAP", "reduceOnly": "true"},
    ]

    selected = _select_recoverable_reduce_only_orders(
        pending,
        symbol="ETH-USDT-SWAP",
        protected_order_ids=set(),
    )

    # All items lack a usable order id — none should be selected.
    assert selected == []


def test_select_recoverable_reduce_only_orders_requires_order_id() -> None:
    pending = [
        {"instId": "ETH-USDT-SWAP", "reduceOnly": "true"},
        {"symbol": "ETH-USDT-SWAP", "reduce_only": True},
        {"instId": "ETH-USDT-SWAP", "reduceOnly": "true", "ordId": "tp-1"},
    ]

    selected = _select_recoverable_reduce_only_orders(
        pending,
        symbol="ETH-USDT-SWAP",
        protected_order_ids=set(),
    )

    assert [_order_id(item) for item in selected] == ["tp-1"]


# ---------------------------------------------------------------------------
# Source-level verification — no inline OKX raw comprehension remains
# ---------------------------------------------------------------------------


def test_apply_main_tp_startup_recovery_uses_order_selection_helper() -> None:
    from pathlib import Path

    text = Path("src/live/startup_recovery/order_recovery.py").read_text(encoding="utf-8")

    assert "_select_recoverable_reduce_only_orders(" in text
    assert 'item.get("instId") == trader.symbol' not in text
    assert 'item.get("reduceOnly"' not in text
