from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from src.execution.okx_trading_client import OkxTradingClient
from src.execution.tp_sl_sidecar_manager import SidecarTpManager
from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit, ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticResult,
)


class FakeSemanticExecutor:
    def __init__(self):
        self.calls = []
        self.ok = True
        self.order_id = "semantic-sidecar-tp-1"
        self.message = ""

    async def sidecar_tp(
        self,
        *,
        symbol,
        side,
        quantity,
        trigger_price,
        quantity_unit,
        order_price=None,
        client_order_id=None,
        label=None,
    ):
        self.calls.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "trigger_price": trigger_price,
                "quantity_unit": quantity_unit,
                "order_price": order_price,
                "client_order_id": client_order_id,
                "label": label,
            }
        )
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.SIDECAR_TP,
            role=BrokerSemanticOrderRole.SIDECAR_TP,
            ok=self.ok,
            order_id=self.order_id if self.ok else None,
            message=self.message,
        )


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    td_mode = "isolated"
    pos_side_mode = "long_short"

    def __init__(self):
        self.requests = []
        self.semantic = FakeSemanticExecutor()
        self._client = self  # backward compat: serve as own private_client

    @property
    def broker_semantic_executor(self):
        return self.semantic

    async def request(self, method, endpoint, payload=None):
        self.requests.append((method, endpoint, payload))
        return {"data": [{"ordId": "legacy-sidecar-tp-1"}]}

    @staticmethod
    def extract_order_id(res):
        return str(res["data"][0]["ordId"])

    @staticmethod
    def decimal_to_str(value):
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price):
        return f"{float(price):.2f}"


@pytest.mark.asyncio
async def test_sidecar_tp_default_disabled_uses_legacy_request(monkeypatch) -> None:
    monkeypatch.delenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", raising=False)
    trader = FakeTrader()
    manager = SidecarTpManager(trader, OkxTradingClient(trader, private_client=trader._client))
    raw_client_order_id = "sidecar client id too long maybe"

    order_id = await manager.place_sidecar_fixed_take_profit(
        side="LONG",
        contracts=Decimal("3"),
        tp_price=3500.0,
        client_order_id=raw_client_order_id,
    )

    assert order_id == "legacy-sidecar-tp-1"
    assert len(trader.requests) == 1
    method, endpoint, payload = trader.requests[0]
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert trader.semantic.calls == []
    assert payload["clOrdId"]
    assert payload["clOrdId"] != raw_client_order_id


@pytest.mark.asyncio
async def test_sidecar_tp_semantic_enabled_uses_semantic_without_legacy_request(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", "true")
    trader = FakeTrader()
    manager = SidecarTpManager(trader, OkxTradingClient(trader, private_client=trader._client))
    raw_client_order_id = "sidecar client id too long maybe"

    order_id = await manager.place_sidecar_fixed_take_profit(
        side="LONG",
        contracts=Decimal("3"),
        tp_price=3500.0,
        client_order_id=raw_client_order_id,
    )

    assert order_id == "semantic-sidecar-tp-1"
    assert trader.requests == []
    assert len(trader.semantic.calls) == 1
    call = trader.semantic.calls[0]
    assert call["symbol"] == "ETH-USDT-SWAP"
    assert call["side"] == BrokerPositionSide.LONG
    assert call["quantity"] == Decimal("3")
    assert call["trigger_price"] == Decimal("3500.0")
    assert call["quantity_unit"] == BrokerQuantityUnit.CONTRACTS
    assert call["label"] == "sidecar_tp"
    assert call["client_order_id"]
    assert call["client_order_id"] != raw_client_order_id


@pytest.mark.asyncio
async def test_sidecar_tp_semantic_maps_short_side(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", "true")
    trader = FakeTrader()
    manager = SidecarTpManager(trader, OkxTradingClient(trader, private_client=trader._client))

    await manager.place_sidecar_fixed_take_profit(
        side="SHORT",
        contracts=Decimal("3"),
        tp_price=3500.0,
    )

    assert trader.semantic.calls[0]["side"] == BrokerPositionSide.SHORT


@pytest.mark.asyncio
async def test_sidecar_tp_semantic_failure_does_not_fallback_legacy(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", "true")
    trader = FakeTrader()
    trader.semantic.ok = False
    trader.semantic.message = "boom"
    manager = SidecarTpManager(trader, OkxTradingClient(trader, private_client=trader._client))

    with pytest.raises(RuntimeError, match="semantic_sidecar_tp_order_failed"):
        await manager.place_sidecar_fixed_take_profit(
            side="LONG",
            contracts=Decimal("3"),
            tp_price=3500.0,
        )

    assert trader.requests == []
    assert len(trader.semantic.calls) == 1


@pytest.mark.asyncio
async def test_sidecar_tp_semantic_without_client_order_id_passes_none(monkeypatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED", "true")
    trader = FakeTrader()
    manager = SidecarTpManager(trader, OkxTradingClient(trader, private_client=trader._client))

    await manager.place_sidecar_fixed_take_profit(
        side="LONG",
        contracts=Decimal("3"),
        tp_price=3500.0,
        client_order_id=None,
    )

    assert trader.semantic.calls[0]["client_order_id"] is None


def test_sidecar_tp_uses_explicit_broker_semantic_executor_access() -> None:
    text = Path("src/execution/tp_sl_sidecar_manager.py").read_text(encoding="utf-8")

    assert '"broker_semantic" "_executor"' not in text
    assert "getattr(t, \"broker_semantic\"" not in text
    assert "t.broker_semantic_executor" in text
