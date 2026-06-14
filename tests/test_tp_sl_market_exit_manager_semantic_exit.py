from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from src.execution.tp_sl_market_exit_manager import MarketExitManager
from src.execution.trading_client_port import OrderResult
from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit, ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticResult,
)


from src.execution.trading_client_port import PositionSnapshot as PortPositionSnapshot


class FakeTradingClient:
    """A fake trading client that records market order calls."""

    def __init__(self):
        self.market_calls: list[dict] = []
        self.next_order_id: str | None = "fake-market-exit-1"
        self.position_sequence: list[PortPositionSnapshot] = []

    async def fetch_position(self) -> PortPositionSnapshot:
        if self.position_sequence:
            return self.position_sequence.pop(0)
        return PortPositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={})

    async def place_market_order(self, *, side, qty, reduce_only, client_order_id):
        self.market_calls.append({
            "side": side,
            "qty": qty,
            "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        })
        return OrderResult(
            ok=True,
            order_id=self.next_order_id,
            client_order_id=None,
            raw={"fake": True},
        )


class FakePositionSnapshot:
    def __init__(self, *, has_position: bool, side: str | None, contracts: Decimal):
        self.has_position = has_position
        self.side = side
        self.contracts = contracts
        self.qty = contracts
        self.avg_entry_price = None


class FakeSemanticExecutor:
    def __init__(self):
        self.calls = []
        self.ok = True
        self.order_id = "semantic-exit-1"
        self.message = ""

    async def market_exit(
        self,
        *,
        symbol,
        side,
        quantity,
        quantity_unit,
        label=None,
    ):
        self.calls.append(
            {
                "method": "market_exit",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "quantity_unit": quantity_unit,
                "label": label,
            }
        )
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.MARKET_EXIT,
            role=BrokerSemanticOrderRole.MARKET_EXIT,
            ok=self.ok,
            order_id=self.order_id if self.ok else None,
            message=self.message,
        )

    async def market_exit_runner(
        self,
        *,
        symbol,
        side,
        quantity,
        quantity_unit,
        label=None,
    ):
        self.calls.append(
            {
                "method": "market_exit_runner",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "quantity_unit": quantity_unit,
                "label": label,
            }
        )
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.MARKET_EXIT_RUNNER,
            role=BrokerSemanticOrderRole.MARKET_EXIT,
            ok=self.ok,
            order_id=self.order_id if self.ok else None,
            message=self.message,
        )


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    min_contracts = Decimal("0.01")
    position_contracts = Decimal("10")

    def __init__(self):
        self.requests = []
        self.semantic = FakeSemanticExecutor()
        self.snapshots = []
        self.cleanup_called = 0
        self.trading_client = FakeTradingClient()

    @property
    def broker_semantic_executor(self):
        return self.semantic

    async def fetch_position_snapshot(self):
        if not self.snapshots:
            raise AssertionError("no more snapshots")
        return self.snapshots.pop(0)

    def _reduce_only_market_order_body(self, side, contracts):
        return {"legacy_market_exit": True, "side": side, "contracts": str(contracts)}

    async def request(self, method, endpoint, payload=None):
        self.requests.append((method, endpoint, payload))
        return {"data": [{"ordId": "legacy-exit-1"}]}

    @staticmethod
    def extract_order_id(res):
        return str(res["data"][0]["ordId"])

    async def _cleanup_after_market_exit(self):
        self.cleanup_called += 1

    @staticmethod
    def decimal_to_str(value):
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")


@pytest.mark.asyncio
async def test_market_exit_defaults_to_legacy_market_order(monkeypatch) -> None:
    monkeypatch.delenv("BROKER_SEMANTIC_MARKET_EXIT_ENABLED", raising=False)
    trader = FakeTrader()
    trader.trading_client.position_sequence = [
        PortPositionSnapshot(side="LONG", qty=Decimal("10"), avg_entry_price=None, raw={}),
        PortPositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={}),
    ]
    manager = MarketExitManager(trader, trader.trading_client)

    ok, message = await manager.market_exit_remaining_position_with_retries(
        "LONG",
        1,
        context="legacy-default",
    )

    assert ok is True
    assert "fake-market-exit-1" in message
    # After migration, the legacy path routes through trading_client
    assert len(trader.trading_client.market_calls) == 1
    call = trader.trading_client.market_calls[0]
    assert call["reduce_only"] is True
    assert call["side"] == "LONG"
    assert call["client_order_id"] == ""
    assert trader.requests == []
    assert trader.semantic.calls == []
    assert trader.cleanup_called == 1


@pytest.mark.asyncio
async def test_market_exit_uses_semantic_executor_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_MARKET_EXIT_ENABLED", "true")
    trader = FakeTrader()
    trader.trading_client.position_sequence = [
        PortPositionSnapshot(side="LONG", qty=Decimal("10"), avg_entry_price=None, raw={}),
        PortPositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={}),
    ]
    manager = MarketExitManager(trader, trader.trading_client)
    context = "semantic-enabled"

    ok, message = await manager.market_exit_remaining_position_with_retries(
        "LONG",
        1,
        context=context,
    )

    assert ok is True
    assert "semantic-exit-1" in message
    assert trader.requests == []
    assert len(trader.semantic.calls) == 1
    call = trader.semantic.calls[0]
    assert call["method"] == "market_exit"
    assert call["symbol"] == "ETH-USDT-SWAP"
    assert call["side"] == BrokerPositionSide.LONG
    assert call["quantity"] == Decimal("10")
    assert call["quantity_unit"] == BrokerQuantityUnit.CONTRACTS
    assert call["label"] == context
    assert trader.cleanup_called == 1


@pytest.mark.asyncio
async def test_market_exit_runner_context_uses_runner_semantic_action(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_MARKET_EXIT_ENABLED", "true")
    trader = FakeTrader()
    trader.trading_client.position_sequence = [
        PortPositionSnapshot(side="LONG", qty=Decimal("10"), avg_entry_price=None, raw={}),
        PortPositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={}),
    ]
    manager = MarketExitManager(trader, trader.trading_client)

    ok, message = await manager.market_exit_remaining_position_with_retries(
        "LONG",
        1,
        context="near_tp_market_exit_runner",
    )

    assert ok is True
    assert "semantic-exit-1" in message
    assert trader.requests == []
    assert len(trader.semantic.calls) == 1
    call = trader.semantic.calls[0]
    assert call["method"] == "market_exit_runner"
    assert call["label"] == "near_tp_market_exit_runner"
    assert call["side"] == BrokerPositionSide.LONG
    assert call["quantity"] == Decimal("10")
    assert call["quantity_unit"] == BrokerQuantityUnit.CONTRACTS


@pytest.mark.asyncio
async def test_market_exit_semantic_maps_short_side(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_MARKET_EXIT_ENABLED", "true")
    trader = FakeTrader()
    trader.trading_client.position_sequence = [
        PortPositionSnapshot(side="SHORT", qty=Decimal("10"), avg_entry_price=None, raw={}),
        PortPositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={}),
    ]
    manager = MarketExitManager(trader, trader.trading_client)

    ok, _message = await manager.market_exit_remaining_position_with_retries(
        "SHORT",
        1,
        context="semantic-short",
    )

    assert ok is True
    assert trader.semantic.calls[0]["side"] == BrokerPositionSide.SHORT


@pytest.mark.asyncio
async def test_market_exit_already_flat_does_not_place_order(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_MARKET_EXIT_ENABLED", "true")
    trader = FakeTrader()
    trader.trading_client.position_sequence = [
        PortPositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={}),
    ]
    manager = MarketExitManager(trader, trader.trading_client)

    ok, message = await manager.market_exit_remaining_position_with_retries(
        "LONG",
        1,
        context="already-flat",
    )

    assert ok is True
    assert message == "already_flat"
    assert trader.requests == []
    assert trader.semantic.calls == []
    assert trader.cleanup_called == 1


@pytest.mark.asyncio
async def test_market_exit_target_side_absent_does_not_place_order(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_MARKET_EXIT_ENABLED", "true")
    trader = FakeTrader()
    trader.trading_client.position_sequence = [
        PortPositionSnapshot(side="SHORT", qty=Decimal("10"), avg_entry_price=None, raw={}),
    ]
    manager = MarketExitManager(trader, trader.trading_client)

    ok, message = await manager.market_exit_remaining_position_with_retries(
        "LONG",
        1,
        context="wrong-side",
    )

    assert ok is True
    assert message == "target_side_absent"
    assert trader.requests == []
    assert trader.semantic.calls == []
    assert trader.cleanup_called == 1


@pytest.mark.asyncio
async def test_market_exit_semantic_failure_does_not_fallback_legacy(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_MARKET_EXIT_ENABLED", "true")
    trader = FakeTrader()
    trader.semantic.ok = False
    trader.semantic.message = "boom"
    trader.trading_client.position_sequence = [
        PortPositionSnapshot(side="LONG", qty=Decimal("10"), avg_entry_price=None, raw={}),
    ]
    manager = MarketExitManager(trader, trader.trading_client)

    ok, message = await manager.market_exit_remaining_position_with_retries(
        "LONG",
        1,
        context="semantic-failure",
    )

    assert ok is False
    assert "semantic_market_exit_order_failed" in message or "boom" in message
    assert trader.requests == []
    assert len(trader.semantic.calls) == 1
    assert trader.cleanup_called == 0


def test_semantic_market_exit_uses_explicit_broker_semantic_executor_access() -> None:
    text = Path("src/execution/tp_sl_market_exit_manager.py").read_text(encoding="utf-8")

    assert '"broker_semantic" "_executor"' not in text
    assert "getattr(t, \"broker_semantic\"" not in text
    assert "t.broker_semantic_executor" in text


def test_is_runner_market_exit_context() -> None:
    assert MarketExitManager._is_runner_market_exit_context("near_tp_market_exit_runner") is True
    assert MarketExitManager._is_runner_market_exit_context("trend_runner_exit") is True
    assert MarketExitManager._is_runner_market_exit_context("generic") is False
    assert MarketExitManager._is_runner_market_exit_context("") is False
