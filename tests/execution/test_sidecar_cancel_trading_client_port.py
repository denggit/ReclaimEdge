#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_sidecar_cancel_trading_client_port.py
@Description: Tests — sidecar TP cancel routed through TradingClientPort
              (20C-CLEAN-PORTS-11A).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from decimal import Decimal

import pytest

if importlib.util.find_spec("aiohttp") is None:
    aiohttp = types.ModuleType("aiohttp")
    sys.modules.setdefault("aiohttp", aiohttp)

from src.execution.trading_client_port import CancelResult, OrderResult  # noqa: E402


# ======================================================================
# FakeTradingClient
# ======================================================================


class FakeTradingClient:
    """Trading client that records cancel calls."""

    def __init__(self):
        self.cancel_calls: list[dict] = []
        self.next_ok = True
        self.next_raw: dict = {"fake": True}
        self.raise_exc: Exception | None = None

    async def cancel_order(self, *, order_id=None, client_order_id=None):
        self.cancel_calls.append({
            "order_id": order_id,
            "client_order_id": client_order_id,
        })
        if self.raise_exc is not None:
            raise self.raise_exc
        return CancelResult(
            ok=self.next_ok,
            order_id=order_id,
            client_order_id=client_order_id,
            raw=self.next_raw,
        )

    async def place_limit_order(self, *, side, qty, price, reduce_only, client_order_id):
        return OrderResult(
            ok=True,
            order_id="tp-1",
            client_order_id=client_order_id or None,
            raw={"fake": True},
        )

    async def place_market_order(self, *, side, qty, reduce_only, client_order_id):
        return OrderResult(
            ok=True,
            order_id="entry-1",
            client_order_id=None,
            raw={"fake": True},
        )


# ======================================================================
# FakeTrader — minimal trader with trader.request blocked
# ======================================================================


class FakeTrader:
    """Minimal fake trader for sidecar cancel tests.

    ``trader.request()`` must never be called from sidecar cancel.
    """

    symbol = "ETH-USDT-SWAP"

    def __init__(self):
        self.requests: list = []

    async def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))
        raise AssertionError("must not call trader.request for sidecar cancel")

    @staticmethod
    def decimal_to_str(value):
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price):
        return f"{float(price):.2f}"

    @property
    def broker_semantic_executor(self):
        raise AssertionError("semantic executor must not be accessed when semantic is disabled")


# ======================================================================
# Helpers
# ======================================================================


def _make_manager(trader, trading_client):
    """Create a SidecarTpManager with a given trader and trading client."""
    from src.execution.tp_sl_sidecar_manager import SidecarTpManager
    return SidecarTpManager(trader, trading_client)


# ======================================================================
# Tests: legacy cancel routed through TradingClientPort
# ======================================================================


@pytest.mark.asyncio
async def test_legacy_sidecar_cancel_uses_trading_client_port(monkeypatch):
    """Legacy cancel_sidecar_take_profit calls trading_client.cancel_order()."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", raising=False)
    fake = FakeTradingClient()
    trader = FakeTrader()
    manager = _make_manager(trader, fake)

    ok = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    assert ok is True
    assert fake.cancel_calls == [{"order_id": "sidecar-tp-1", "client_order_id": None}]
    assert trader.requests == []


@pytest.mark.asyncio
async def test_legacy_sidecar_cancel_no_order_id_returns_true(monkeypatch):
    """cancel_sidecar_take_profit(None) returns True without any call."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", raising=False)
    fake = FakeTradingClient()
    trader = FakeTrader()
    manager = _make_manager(trader, fake)

    ok = await manager.cancel_sidecar_take_profit(None)

    assert ok is True
    assert fake.cancel_calls == []
    assert trader.requests == []


@pytest.mark.asyncio
async def test_legacy_sidecar_cancel_not_found_returns_true(monkeypatch):
    """When cancel_order raises 'order not found', still return True."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", raising=False)
    fake = FakeTradingClient()
    fake.raise_exc = RuntimeError("order not found")
    trader = FakeTrader()
    manager = _make_manager(trader, fake)

    ok = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    assert ok is True
    assert len(fake.cancel_calls) == 1


@pytest.mark.asyncio
async def test_legacy_sidecar_cancel_already_absent_returns_true(monkeypatch):
    """When cancel_order raises 'does not exist', still return True."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", raising=False)
    fake = FakeTradingClient()
    fake.raise_exc = RuntimeError("order does not exist")
    trader = FakeTrader()
    manager = _make_manager(trader, fake)

    ok = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    assert ok is True
    assert len(fake.cancel_calls) == 1


@pytest.mark.asyncio
async def test_legacy_sidecar_cancel_already_canceled_returns_true(monkeypatch):
    """When cancel_order raises 'already canceled', still return True."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", raising=False)
    fake = FakeTradingClient()
    fake.raise_exc = RuntimeError("already canceled")
    trader = FakeTrader()
    manager = _make_manager(trader, fake)

    ok = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    assert ok is True
    assert len(fake.cancel_calls) == 1


@pytest.mark.asyncio
async def test_legacy_sidecar_cancel_network_error_returns_false(monkeypatch):
    """When cancel_order raises a non-absent error, return False."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", raising=False)
    fake = FakeTradingClient()
    fake.raise_exc = RuntimeError("network broken")
    trader = FakeTrader()
    manager = _make_manager(trader, fake)

    ok = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    assert ok is False
    assert len(fake.cancel_calls) == 1


@pytest.mark.asyncio
async def test_legacy_sidecar_cancel_result_not_ok_returns_false(monkeypatch):
    """When CancelResult.ok is False, return False."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", raising=False)
    fake = FakeTradingClient()
    fake.next_ok = False
    fake.next_raw = {"sCode": "1", "sMsg": "cancel failed"}
    trader = FakeTrader()
    manager = _make_manager(trader, fake)

    ok = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    assert ok is False
    assert len(fake.cancel_calls) == 1


@pytest.mark.asyncio
async def test_legacy_sidecar_cancel_does_not_call_trader_request(monkeypatch):
    """trader.request must NOT be called from the cancel path."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", raising=False)
    fake = FakeTradingClient()
    trader = FakeTrader()
    manager = _make_manager(trader, fake)

    await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    # trader.request should never have been called
    assert len(trader.requests) == 0
