from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from src.execution.okx_trading_client import OkxTradingClient
from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager
from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit, ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticResult,
)


class FakeSemanticExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result_ok = True
        self.order_id = "semantic-tp-1"
        self.order_ids: list[str] | None = None
        self.message = ""

    async def place_reduce_only_tp(
        self,
        *,
        symbol,
        side,
        quantity,
        trigger_price,
        quantity_unit,
        role,
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
                "role": role,
                "client_order_id": client_order_id,
                "label": label,
            }
        )
        order_id = self.order_id
        if self.order_ids is not None:
            order_id = self.order_ids[len(self.calls) - 1]
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            role=role,
            ok=self.result_ok,
            order_id=order_id if self.result_ok else None,
            message=self.message,
        )


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    position_contracts = Decimal("10")
    min_contracts = Decimal("0.01")
    contract_precision = Decimal("0.01")
    contract_multiplier = Decimal("0.1")
    td_mode = "isolated"
    pos_side_mode = "long_short"

    def __init__(self):
        self.requests = []
        self.semantic = FakeSemanticExecutor()

    @property
    def broker_semantic_executor(self):
        return self.semantic

    def _reduce_only_tp_order_body(self, side, contracts, price):
        return {
            "legacy": True,
            "side": side,
            "contracts": str(contracts),
            "price": str(price),
        }

    async def request(self, method, endpoint, payload=None):
        self.requests.append((method, endpoint, payload))
        return {"data": [{"ordId": "legacy-tp-1"}]}

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


def _intent(side: str = "LONG") -> SimpleNamespace:
    return SimpleNamespace(side=side)


@pytest.mark.asyncio
async def test_default_disabled_uses_legacy_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BROKER_SEMANTIC_TP_PLACEMENT_ENABLED", raising=False)
    fake_trader = FakeTrader()
    trading_client = OkxTradingClient(fake_trader)
    manager = CoreTakeProfitManager(fake_trader, protective_stops=None,
                                    trading_client=trading_client)

    order_ids = await manager._place_reduce_only_take_profit_orders(
        intent=_intent("LONG"),
        specs=[("final", Decimal("10"), 3500.0)],
    )

    assert order_ids == ["legacy-tp-1"]
    assert len(fake_trader.requests) == 1
    method, endpoint, body = fake_trader.requests[0]
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert body["sz"] == "10"
    assert body["px"] == "3500.00"
    assert body["reduceOnly"] == "true"
    assert body["ordType"] == "limit"
    assert fake_trader.semantic.calls == []


@pytest.mark.asyncio
async def test_enabled_uses_semantic_executor_not_legacy_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_TP_PLACEMENT_ENABLED", "true")
    fake_trader = FakeTrader()
    manager = CoreTakeProfitManager(fake_trader, protective_stops=None,
                                    trading_client=OkxTradingClient(fake_trader))

    order_ids = await manager._place_reduce_only_take_profit_orders(
        intent=_intent("LONG"),
        specs=[("final", Decimal("10"), 3500.0)],
    )

    assert order_ids == ["semantic-tp-1"]
    assert fake_trader.requests == []
    assert len(fake_trader.semantic.calls) == 1
    call = fake_trader.semantic.calls[0]
    assert call["symbol"] == "ETH-USDT-SWAP"
    assert call["side"] == BrokerPositionSide.LONG
    assert call["quantity"] == Decimal("10")
    assert call["trigger_price"] == Decimal("3500.0")
    assert call["quantity_unit"] == BrokerQuantityUnit.CONTRACTS
    assert call["client_order_id"] is None
    assert call["label"] is None


@pytest.mark.asyncio
async def test_enabled_maps_short_side(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_TP_PLACEMENT_ENABLED", "true")
    fake_trader = FakeTrader()
    manager = CoreTakeProfitManager(fake_trader, protective_stops=None,
                                    trading_client=OkxTradingClient(fake_trader))

    await manager._place_reduce_only_take_profit_orders(
        intent=_intent("SHORT"),
        specs=[("final", Decimal("10"), 3500.0)],
    )

    assert fake_trader.semantic.calls[0]["side"] == BrokerPositionSide.SHORT


def test_broker_tp_role_for_label_mapping() -> None:
    assert CoreTakeProfitManager._broker_tp_role_for_label("tp1_middle_fast") == BrokerSemanticOrderRole.TP1
    assert CoreTakeProfitManager._broker_tp_role_for_label("tp2_outer") == BrokerSemanticOrderRole.TP2
    assert CoreTakeProfitManager._broker_tp_role_for_label("runner") == BrokerSemanticOrderRole.RUNNER_TP
    assert CoreTakeProfitManager._broker_tp_role_for_label("final") == BrokerSemanticOrderRole.CORE_TP


@pytest.mark.asyncio
async def test_semantic_failure_does_not_fallback_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_TP_PLACEMENT_ENABLED", "true")
    fake_trader = FakeTrader()
    fake_trader.semantic.result_ok = False
    fake_trader.semantic.message = "boom"
    manager = CoreTakeProfitManager(fake_trader, protective_stops=None,
                                    trading_client=OkxTradingClient(fake_trader))

    with pytest.raises(RuntimeError, match="semantic_tp_order_failed"):
        await manager._place_reduce_only_take_profit_orders(
            intent=_intent("LONG"),
            specs=[("final", Decimal("10"), 3500.0)],
        )

    assert len(fake_trader.semantic.calls) == 1
    assert fake_trader.requests == []


@pytest.mark.asyncio
async def test_multiple_semantic_specs_return_order_ids_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_TP_PLACEMENT_ENABLED", "true")
    fake_trader = FakeTrader()
    fake_trader.semantic.order_ids = ["semantic-tp-1", "semantic-tp-2"]
    manager = CoreTakeProfitManager(fake_trader, protective_stops=None,
                                    trading_client=OkxTradingClient(fake_trader))

    order_ids = await manager._place_reduce_only_take_profit_orders(
        intent=_intent("LONG"),
        specs=[
            ("tp1_middle_fast", Decimal("4"), 3400.0),
            ("tp2_outer", Decimal("6"), 3500.0),
        ],
    )

    assert order_ids == ["semantic-tp-1", "semantic-tp-2"]
    assert [call["quantity"] for call in fake_trader.semantic.calls] == [Decimal("4"), Decimal("6")]
    assert [call["trigger_price"] for call in fake_trader.semantic.calls] == [
        Decimal("3400.0"),
        Decimal("3500.0"),
    ]
    assert [call["role"] for call in fake_trader.semantic.calls] == [
        BrokerSemanticOrderRole.TP1,
        BrokerSemanticOrderRole.TP2,
    ]
    assert fake_trader.requests == []


def test_semantic_tp_placement_uses_explicit_broker_semantic_executor_access() -> None:
    """Prove the source file uses explicit t.broker_semantic_executor, not hidden-string getattr."""
    from pathlib import Path

    text = Path("src/execution/tp_sl_core_tp_manager.py").read_text(encoding="utf-8")

    assert '"broker_" "semantic_executor"' not in text
    assert "getattr(t, \"broker_\"" not in text
    assert "t.broker_semantic_executor" in text
