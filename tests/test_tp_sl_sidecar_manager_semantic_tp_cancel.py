from __future__ import annotations

from pathlib import Path

import pytest

from src.execution.tp_sl_sidecar_manager import SidecarTpManager
from src.execution.trading_client_port import CancelResult
from src.exchanges.models import ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticResult,
)


class FakeTradingClient:
    """Trading client for cancel tests."""

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


class FakeSemanticExecutor:
    def __init__(self):
        self.calls = []
        self.ok = True
        self.message = ""

    async def cancel_reduce_only_tp(
        self,
        *,
        symbol,
        order_id,
        role=None,
        label=None,
    ):
        self.calls.append(
            {
                "symbol": symbol,
                "order_id": order_id,
                "role": role,
                "label": label,
            }
        )
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
            role=role or BrokerSemanticOrderRole.SIDECAR_TP,
            ok=self.ok,
            order_id=order_id,
            message=self.message,
        )


class FakeTrader:
    symbol = "ETH-USDT-SWAP"

    def __init__(self):
        self.requests = []
        self.semantic = FakeSemanticExecutor()

    @property
    def broker_semantic_executor(self):
        return self.semantic

    async def request(self, method, endpoint, payload=None):
        self.requests.append((method, endpoint, payload))
        return {"data": [{"ordId": payload["ordId"], "sCode": "0"}]}


@pytest.mark.asyncio
async def test_sidecar_tp_cancel_default_disabled_uses_trading_client_port(monkeypatch) -> None:
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", raising=False)
    trader = FakeTrader()
    trading_client = FakeTradingClient()
    manager = SidecarTpManager(trader, trading_client)  # type: ignore[arg-type]

    ok = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    assert ok is True
    # Must route through TradingClientPort, NOT trader.request
    assert trader.requests == []
    assert trading_client.cancel_calls == [{"order_id": "sidecar-tp-1", "client_order_id": None}]
    assert trader.semantic.calls == []


@pytest.mark.asyncio
async def test_sidecar_tp_cancel_semantic_enabled_uses_semantic_without_legacy_request(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", "true")
    trader = FakeTrader()
    manager = SidecarTpManager(trader, None)  # type: ignore[arg-type]

    ok = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    assert ok is True
    assert trader.requests == []
    assert len(trader.semantic.calls) == 1
    call = trader.semantic.calls[0]
    assert call["symbol"] == "ETH-USDT-SWAP"
    assert call["order_id"] == "sidecar-tp-1"
    assert call["role"] == BrokerSemanticOrderRole.SIDECAR_TP
    assert call["label"] == "sidecar_tp"


@pytest.mark.asyncio
async def test_sidecar_tp_cancel_semantic_failure_does_not_fallback_legacy(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", "true")
    trader = FakeTrader()
    trader.semantic.ok = False
    trader.semantic.message = "boom"
    manager = SidecarTpManager(trader, None)  # type: ignore[arg-type]

    ok = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    assert ok is False
    assert trader.requests == []
    assert len(trader.semantic.calls) == 1


@pytest.mark.asyncio
async def test_sidecar_tp_cancel_semantic_already_absent_returns_true(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", "true")
    trader = FakeTrader()
    trader.semantic.ok = False
    trader.semantic.message = "order not found"
    manager = SidecarTpManager(trader, None)  # type: ignore[arg-type]

    ok = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

    assert ok is True
    assert trader.requests == []
    assert len(trader.semantic.calls) == 1


@pytest.mark.asyncio
async def test_sidecar_tp_cancel_order_id_none_returns_true(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED", "true")
    trader = FakeTrader()
    manager = SidecarTpManager(trader, None)  # type: ignore[arg-type]

    ok = await manager.cancel_sidecar_take_profit(None)

    assert ok is True
    assert trader.requests == []
    assert trader.semantic.calls == []


def test_sidecar_tp_cancel_uses_explicit_broker_semantic_executor_access() -> None:
    from pathlib import Path

    text = Path("src/execution/tp_sl_sidecar_manager.py").read_text(encoding="utf-8")

    assert '"broker_semantic" "_executor"' not in text
    assert "getattr(t, \"broker_semantic\"" not in text
    assert "t.broker_semantic_executor" in text
