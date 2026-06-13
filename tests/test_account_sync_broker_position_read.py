"""Tests for optional broker position read in account sync position snapshot."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.models import (
    BrokerPosition,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.execution.trader import PositionSnapshot
from src.live.account_sync.pre_core_position import (
    _fetch_account_sync_position_snapshot,
    _position_snapshot_from_broker_position,
)


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    contract_multiplier = Decimal("0.1")

    def __init__(self):
        self.legacy_called = 0
        self.broker_called = 0
        self.broker_position = None
        self.broker_error = None
        self.legacy_position = PositionSnapshot(
            side="LONG",
            contracts=Decimal("5"),
            avg_entry_price=3300.0,
            eth_qty=0.5,
            raw_pos=Decimal("5"),
        )

    async def fetch_position_snapshot(self):
        self.legacy_called += 1
        return self.legacy_position

    async def fetch_broker_position(self):
        self.broker_called += 1
        if self.broker_error is not None:
            raise self.broker_error
        return self.broker_position


# ── Default: legacy path ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_account_sync_position_default_uses_legacy(monkeypatch) -> None:
    monkeypatch.delenv("BROKER_SEMANTIC_ACCOUNT_SYNC_POSITION_ENABLED", raising=False)
    trader = FakeTrader()

    position = await _fetch_account_sync_position_snapshot(trader)

    assert position == trader.legacy_position
    assert trader.legacy_called == 1
    assert trader.broker_called == 0


# ── Enabled: broker position read ─────────────────────────────────


@pytest.mark.asyncio
async def test_account_sync_position_enabled_uses_broker_position(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_ACCOUNT_SYNC_POSITION_ENABLED", "true")
    trader = FakeTrader()
    trader.broker_position = BrokerPosition(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        position_side=BrokerPositionSide.LONG,
        quantity=Decimal("10"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        average_entry_price=Decimal("3300"),
    )

    position = await _fetch_account_sync_position_snapshot(trader)

    assert position.side == "LONG"
    assert position.contracts == Decimal("10")
    assert position.avg_entry_price == 3300.0
    assert position.eth_qty == 1.0
    assert position.raw_pos == Decimal("10")
    assert trader.broker_called == 1
    assert trader.legacy_called == 0


# ── SHORT raw_pos negative ────────────────────────────────────────


@pytest.mark.asyncio
async def test_account_sync_position_enabled_maps_short_raw_pos_negative(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_ACCOUNT_SYNC_POSITION_ENABLED", "true")
    trader = FakeTrader()
    trader.broker_position = BrokerPosition(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        position_side=BrokerPositionSide.SHORT,
        quantity=Decimal("7"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        average_entry_price=Decimal("3200"),
    )

    position = await _fetch_account_sync_position_snapshot(trader)

    assert position.side == "SHORT"
    assert position.contracts == Decimal("7")
    assert position.raw_pos == Decimal("-7")
    assert position.eth_qty == 0.7


# ── Broker None → flat ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_account_sync_position_enabled_broker_none_returns_flat(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_ACCOUNT_SYNC_POSITION_ENABLED", "true")
    trader = FakeTrader()
    trader.broker_position = None

    position = await _fetch_account_sync_position_snapshot(trader)

    assert position.has_position is False
    assert position.contracts == Decimal("0")
    assert trader.broker_called == 1
    assert trader.legacy_called == 0


# ── Broker failure → fallback legacy ──────────────────────────────


@pytest.mark.asyncio
async def test_account_sync_position_enabled_broker_failure_fallbacks_legacy(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_ACCOUNT_SYNC_POSITION_ENABLED", "true")
    trader = FakeTrader()
    trader.broker_error = RuntimeError("boom")

    position = await _fetch_account_sync_position_snapshot(trader)

    assert position == trader.legacy_position
    assert trader.broker_called == 1
    assert trader.legacy_called == 1


# ── source-level: no direct broker_semantic_executor call ─────────


def test_account_sync_position_does_not_directly_call_broker_semantic_executor() -> None:
    from pathlib import Path

    text = Path("src/live/account_sync/pre_core_position.py").read_text(encoding="utf-8")

    assert "broker_semantic_executor" not in text
    assert "fetch_broker_position()" in text
