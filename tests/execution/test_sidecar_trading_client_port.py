#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_sidecar_trading_client_port.py
@Description: Tests — sidecar market entry and fixed TP routed through
              TradingClientPort (20C-CLEAN-PORTS-08).
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

from tests.conftest import FakeOkxClient
import src.execution.trader as trader_module  # noqa: E402
from src.execution.trader import Trader  # noqa: E402
from src.execution.trading_client_port import OrderResult  # noqa: E402


# ======================================================================
# FakeTradingClient
# ======================================================================


class FakeTradingClient:
    def __init__(self):
        self.market_calls: list[dict] = []
        self.limit_calls: list[dict] = []
        self.next_market_order_id = "sidecar-entry-1"
        self.next_limit_order_id = "sidecar-tp-1"
        self._fail_market = False
        self._fail_limit = False

    async def place_market_order(self, *, side, qty, reduce_only, client_order_id):
        self.market_calls.append({
            "side": side,
            "qty": qty,
            "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        })
        order_id = None if self._fail_market else self.next_market_order_id
        return OrderResult(
            ok=True,
            order_id=order_id,
            client_order_id=None,
            raw={"fake": True},
        )

    async def place_limit_order(self, *, side, qty, price, reduce_only, client_order_id):
        self.limit_calls.append({
            "side": side,
            "qty": qty,
            "price": price,
            "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        })
        order_id = None if self._fail_limit else self.next_limit_order_id
        return OrderResult(
            ok=True,
            order_id=order_id,
            client_order_id=client_order_id or None,
            raw={"fake": True},
        )


# ======================================================================
# Helpers
# ======================================================================


def _make_trader(**overrides) -> Trader:
    """Create a bare Trader without calling __init__."""
    t = Trader.__new__(Trader)
    t.base_url = "https://www.okx.test"
    t.api_key = "key"
    t.secret_key = "secret"
    t.passphrase = "pass"
    t._timeout_seconds = 7.0
    t.symbol = "ETH-USDT-SWAP"
    t.td_mode = "isolated"
    t.leverage = "50"
    t.pos_side_mode = "net"
    t.live_trading = True
    t.max_live_equity_usdt = 30.0
    t.contract_multiplier = Decimal("0.1")
    t.contract_precision = Decimal("0.01")
    t.min_contracts = Decimal("0.01")
    t.tp_order_id = None
    t.middle_runner_protective_sl_order_id = None
    t.three_stage_post_tp1_protective_sl_order_id = None
    t.trend_runner_sl_order_id = None
    t.position_contracts = Decimal("0")
    t.account_equity_usdt = 0.0
    t._protected_reduce_only_order_ids = set()
    t._managed_reduce_only_order_ids = set()
    t._allow_cancel_unmanaged_reduce_only = True
    t._client = FakeOkxClient(t)
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


# ======================================================================
# sidecar market entry tests
# ======================================================================


@pytest.mark.asyncio
async def test_sidecar_market_entry_uses_trading_client_port():
    """place_sidecar_market_order calls trading_client.place_market_order(reduce_only=False)."""
    fake = FakeTradingClient()
    trader = _make_trader(trading_client=fake)

    result = await trader.place_sidecar_market_order(side="LONG", eth_qty=0.5)

    # One market call
    assert len(fake.market_calls) == 1
    call = fake.market_calls[0]
    assert call["reduce_only"] is False
    assert call["side"] == "LONG"
    assert call["qty"] == Decimal("5")  # 0.5 / 0.1 = 5 contracts
    assert call["client_order_id"] == ""

    # No limit calls
    assert len(fake.limit_calls) == 0

    # Result shape preserved
    assert result["order_id"] == "sidecar-entry-1"
    assert result["contracts"] == "5"
    assert result["qty"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_sidecar_market_entry_short_side():
    """place_sidecar_market_order passes SHORT side correctly."""
    fake = FakeTradingClient()
    trader = _make_trader(trading_client=fake)

    await trader.place_sidecar_market_order(side="SHORT", eth_qty=0.3)

    assert fake.market_calls[0]["side"] == "SHORT"
    assert fake.market_calls[0]["qty"] == Decimal("3")  # 0.3 / 0.1


@pytest.mark.asyncio
async def test_sidecar_market_entry_missing_order_id_raises():
    """When trading_client returns order_id=None, raise RuntimeError."""
    fake = FakeTradingClient()
    fake._fail_market = True
    trader = _make_trader(trading_client=fake)

    with pytest.raises(RuntimeError, match="sidecar_market_entry_missing_order_id"):
        await trader.place_sidecar_market_order(side="LONG", eth_qty=0.1)

    assert len(fake.market_calls) == 1


# ======================================================================
# sidecar fixed TP tests
# ======================================================================


def _make_sidecar_manager(trader, trading_client):
    """Create a SidecarTpManager with a given trader and trading client."""
    from src.execution.tp_sl_sidecar_manager import SidecarTpManager
    return SidecarTpManager(trader, trading_client)


@pytest.mark.asyncio
async def test_sidecar_fixed_tp_uses_trading_client_port(monkeypatch):
    """place_sidecar_fixed_take_profit calls trading_client.place_limit_order(reduce_only=True)."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", raising=False)
    fake = FakeTradingClient()
    trader = _make_trader()
    manager = _make_sidecar_manager(trader, fake)

    order_id = await manager.place_sidecar_fixed_take_profit(
        side="LONG",
        contracts=Decimal("3"),
        tp_price=3500.0,
    )

    assert order_id == "sidecar-tp-1"
    assert len(fake.limit_calls) == 1
    call = fake.limit_calls[0]
    assert call["reduce_only"] is True
    assert call["side"] == "LONG"
    assert call["qty"] == Decimal("3")
    assert call["price"] == Decimal("3500.0")
    assert call["client_order_id"] == ""  # no client_order_id → ""

    # No market calls
    assert len(fake.market_calls) == 0


@pytest.mark.asyncio
async def test_sidecar_fixed_tp_short_side(monkeypatch):
    """place_sidecar_fixed_take_profit passes SHORT side correctly."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", raising=False)
    fake = FakeTradingClient()
    trader = _make_trader()
    manager = _make_sidecar_manager(trader, fake)

    await manager.place_sidecar_fixed_take_profit(
        side="SHORT",
        contracts=Decimal("2"),
        tp_price=2900.0,
    )

    call = fake.limit_calls[0]
    assert call["side"] == "SHORT"
    assert call["price"] == Decimal("2900.0")


@pytest.mark.asyncio
async def test_sidecar_fixed_tp_with_client_order_id(monkeypatch):
    """place_sidecar_fixed_take_profit forwards client_order_id after sanitization."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", raising=False)
    fake = FakeTradingClient()
    trader = _make_trader()
    manager = _make_sidecar_manager(trader, fake)

    await manager.place_sidecar_fixed_take_profit(
        side="LONG",
        contracts=Decimal("3"),
        tp_price=3500.0,
        client_order_id="SC-97644895de-L1-47229",
    )

    call = fake.limit_calls[0]
    # Should be sanitized (dashes stripped)
    assert call["client_order_id"] == "SC97644895deL147229"


@pytest.mark.asyncio
async def test_sidecar_fixed_tp_missing_order_id_raises(monkeypatch):
    """When trading_client returns order_id=None, raise RuntimeError."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", raising=False)
    fake = FakeTradingClient()
    fake._fail_limit = True
    trader = _make_trader()
    manager = _make_sidecar_manager(trader, fake)

    with pytest.raises(RuntimeError, match="sidecar_fixed_tp_missing_order_id"):
        await manager.place_sidecar_fixed_take_profit(
            side="LONG",
            contracts=Decimal("3"),
            tp_price=3500.0,
        )

    assert len(fake.limit_calls) == 1


@pytest.mark.asyncio
async def test_sidecar_fixed_tp_contracts_from_string(monkeypatch):
    """contracts can be passed as a string — converted to Decimal."""
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", raising=False)
    fake = FakeTradingClient()
    trader = _make_trader()
    manager = _make_sidecar_manager(trader, fake)

    await manager.place_sidecar_fixed_take_profit(
        side="LONG",
        contracts="0.69",
        tp_price=3012.0,
    )

    call = fake.limit_calls[0]
    assert call["qty"] == Decimal("0.69")
