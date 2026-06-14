#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_tp_sl_protective_stop_manager_semantic_placement.py
@Description: Tests for the optional semantic protective SL placement path
              in ProtectiveStopManager.place_near_tp_protective_stop_with_retries().
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit, ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticResult,
)
from src.execution.okx_trading_client import OkxTradingClient
from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager


# ---------------------------------------------------------------------------
# FakeSemanticExecutor
# ---------------------------------------------------------------------------


class FakeSemanticExecutor:
    """A test double that records place_protective_stop calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.ok: bool = True
        self.order_id: str = "semantic-sl-1"
        self.message: str = ""

    async def place_protective_stop(
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
    ) -> BrokerSemanticResult:
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
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=role,
            ok=self.ok,
            order_id=self.order_id if self.ok else None,
            message=self.message,
        )


# ---------------------------------------------------------------------------
# FakeTrader
# ---------------------------------------------------------------------------


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    contract_precision = Decimal("0.01")
    td_mode = "isolated"
    pos_side_mode = "long_short"

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, Any | None]] = []
        self.semantic = FakeSemanticExecutor()
        self.verify_result: bool = True
        self.fallback_verify_result: bool = True
        self._verify_calls: list[str] = []
        self.cancelled_unverified: list[tuple[str, str]] = []

    @property
    def broker_semantic_executor(self) -> FakeSemanticExecutor:
        return self.semantic

    @staticmethod
    def _to_decimal(value: Any) -> Decimal:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _near_tp_protective_sl_algo_body(
        self, side: str, contracts: Decimal, stop_price: float
    ) -> dict[str, Any]:
        return {
            "legacy_primary": True,
            "side": side,
            "contracts": str(contracts),
            "stop_price": str(stop_price),
        }

    def _near_tp_fallback_conditional_close_body(
        self, side: str, contracts: Decimal, stop_price: float
    ) -> dict[str, Any]:
        return {
            "legacy_fallback": True,
            "side": side,
            "contracts": str(contracts),
            "stop_price": str(stop_price),
        }

    async def request(
        self, method: str, endpoint: str, payload: Any | None = None
    ) -> dict[str, Any]:
        self.requests.append((method, endpoint, payload))
        if payload and payload.get("legacy_fallback"):
            return {"data": [{"algoId": "fallback-algo-1"}]}
        return {"data": [{"algoId": "legacy-algo-1"}]}

    @staticmethod
    def extract_algo_id(res: dict[str, Any]) -> str:
        item = res["data"][0]
        return str(item.get("algoId") or item.get("ordId"))

    async def verify_near_tp_protective_stop(
        self, algo_id: str, side: str, contracts: Decimal, stop_price: float
    ) -> bool:
        self._verify_calls.append(algo_id)
        # Use fallback_verify_result for fallback algo IDs
        if algo_id.startswith("fallback-"):
            return self.fallback_verify_result
        return self.verify_result

    async def _cancel_unverified_near_tp_algo(self, algo_id: str, *, phase: str) -> None:
        self.cancelled_unverified.append((algo_id, phase))

    @staticmethod
    def decimal_to_str(value: Any) -> str:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: Any) -> str:
        return f"{float(price):.2f}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manager(trader: FakeTrader, trading_client=None) -> ProtectiveStopManager:
    if trading_client is None:
        trading_client = OkxTradingClient(trader)
    return ProtectiveStopManager(trader, trading_client=trading_client)  # type: ignore[arg-type]


# ===================================================================
# Test: default off → legacy primary
# ===================================================================


@pytest.mark.asyncio
async def test_default_off_uses_legacy_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the env var is not set, legacy algo body path is used."""
    monkeypatch.delenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", raising=False)

    trader = FakeTrader()
    manager = make_manager(trader)

    ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
        side="LONG",
        contracts=Decimal("10"),
        stop_price=3400.0,
        retry_count=1,
        retry_interval_seconds=0,
    )

    assert ok is True
    assert algo_id == "legacy-algo-1"
    assert message == "protective_sl_placed"
    assert len(trader.requests) == 1
    method, endpoint, body = trader.requests[0]
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order-algo"
    assert body["sz"] == "10"
    assert body["slTriggerPx"] == "3400.00"
    assert body["reduceOnly"] == "true"
    assert body["ordType"] == "conditional"
    assert trader.semantic.calls == []


# ===================================================================
# Test: enabled → semantic primary, no legacy
# ===================================================================


@pytest.mark.asyncio
async def test_enabled_uses_semantic_primary_no_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When enabled, primary placement goes through semantic executor, not legacy request."""
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", "true")

    trader = FakeTrader()
    manager = make_manager(trader)

    ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
        side="LONG",
        contracts=Decimal("10"),
        stop_price=3400.0,
        retry_count=1,
        retry_interval_seconds=0,
    )

    assert ok is True
    assert algo_id == "semantic-sl-1"
    assert message == "protective_sl_placed"
    assert trader.requests == [], "Legacy request must not be made when semantic is enabled"
    assert len(trader.semantic.calls) == 1
    call = trader.semantic.calls[0]
    assert call["side"] == BrokerPositionSide.LONG
    assert call["quantity"] == Decimal("10")
    assert call["trigger_price"] == Decimal("3400.0")
    assert call["quantity_unit"] == BrokerQuantityUnit.CONTRACTS
    assert call["role"] == BrokerSemanticOrderRole.PROTECTIVE_SL


# ===================================================================
# Test: SHORT side mapping
# ===================================================================


@pytest.mark.asyncio
async def test_enabled_maps_short_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """SHORT side maps to BrokerPositionSide.SHORT."""
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", "true")

    trader = FakeTrader()
    manager = make_manager(trader)

    ok, algo_id, _message = await manager.place_near_tp_protective_stop_with_retries(
        side="SHORT",
        contracts=Decimal("5"),
        stop_price=3700.0,
        retry_count=1,
        retry_interval_seconds=0,
    )

    assert ok is True
    assert algo_id == "semantic-sl-1"
    assert len(trader.semantic.calls) == 1
    call = trader.semantic.calls[0]
    assert call["side"] == BrokerPositionSide.SHORT
    assert call["quantity"] == Decimal("5")
    assert call["trigger_price"] == Decimal("3700.0")


# ===================================================================
# Test: verify failure → cancel unverified
# ===================================================================


@pytest.mark.asyncio
async def test_verify_failure_cancels_unverified_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """When verify fails, cancel_unverified is called for the semantic algo_id."""
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", "true")

    trader = FakeTrader()
    trader.verify_result = False
    manager = make_manager(trader)

    ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
        side="LONG",
        contracts=Decimal("10"),
        stop_price=3400.0,
        retry_count=1,
        retry_interval_seconds=0,
    )

    assert ("semantic-sl-1", "primary") in trader.cancelled_unverified

    # After primary verify fail → cancel, fallback conditional close kicked in
    assert ok is True
    assert algo_id == "fallback-algo-1"
    assert message == "fallback_conditional_close_placed"

    # Semantic was called once for primary
    assert len(trader.semantic.calls) == 1

    # Fallback was via legacy request
    fallback_requests = [r for r in trader.requests if r[2] and r[2].get("legacy_fallback")]
    assert len(fallback_requests) == 1


# ===================================================================
# Test: semantic failure → retry/fallback
# ===================================================================


@pytest.mark.asyncio
async def test_semantic_failure_enters_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When semantic primary fails, the exception is caught and fallback conditional close is used."""
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", "true")

    trader = FakeTrader()
    trader.semantic.ok = False
    trader.semantic.message = "boom"
    manager = make_manager(trader)

    ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
        side="LONG",
        contracts=Decimal("10"),
        stop_price=3400.0,
        retry_count=1,
        retry_interval_seconds=0,
    )

    # Semantic was attempted once
    assert len(trader.semantic.calls) == 1

    # No legacy primary request
    primary_requests = [r for r in trader.requests if r[2] and r[2].get("legacy_primary")]
    assert primary_requests == [], "Legacy primary must not be used when semantic is enabled"

    # Fallback conditional close was used
    fallback_requests = [r for r in trader.requests if r[2] and r[2].get("legacy_fallback")]
    assert len(fallback_requests) == 1

    assert ok is True
    assert algo_id == "fallback-algo-1"
    assert message == "fallback_conditional_close_placed"


# ===================================================================
# Test: env var value variants
# ===================================================================


@pytest.mark.asyncio
class TestEnvVarVariants:
    @pytest.mark.parametrize("value", ["1", "true", "yes", "y", "on", "True", "YES", "ON"])
    async def test_env_var_enabled_variants(self, monkeypatch, value: str) -> None:
        """All truthy variants enable the semantic path."""
        monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", value)

        trader = FakeTrader()
        manager = make_manager(trader)

        ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
            side="LONG",
            contracts=Decimal("10"),
            stop_price=3400.0,
            retry_count=1,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "semantic-sl-1"
        assert message == "protective_sl_placed"
        assert len(trader.semantic.calls) == 1
        assert trader.requests == []

    @pytest.mark.parametrize("value", ["0", "false", "no", "n", "off", "", "   ", "maybe"])
    async def test_env_var_disabled_variants(self, monkeypatch, value: str) -> None:
        """All falsy variants keep legacy path."""
        monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", value)

        trader = FakeTrader()
        manager = make_manager(trader)

        ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
            side="LONG",
            contracts=Decimal("10"),
            stop_price=3400.0,
            retry_count=1,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "legacy-algo-1"
        assert message == "protective_sl_placed"
        assert trader.semantic.calls == []
        assert len(trader.requests) == 1
        method, endpoint, body = trader.requests[0]
        assert method == "POST"
        assert endpoint == "/api/v5/trade/order-algo"
        assert body["ordType"] == "conditional"
        assert body["reduceOnly"] == "true"


# ===================================================================
# Test: semantic protective SL placement helper methods
# ===================================================================


def test_broker_semantic_protective_sl_placement_enabled_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default returns False when the env var is absent."""
    monkeypatch.delenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", raising=False)

    assert ProtectiveStopManager._broker_semantic_protective_sl_placement_enabled() is False


def test_broker_position_side_long() -> None:
    assert ProtectiveStopManager._broker_position_side("LONG") == BrokerPositionSide.LONG


def test_broker_position_side_short() -> None:
    assert ProtectiveStopManager._broker_position_side("SHORT") == BrokerPositionSide.SHORT


def test_broker_position_side_unknown_raises() -> None:
    with pytest.raises(RuntimeError, match="unsupported_position_side_for_semantic_protective_sl"):
        ProtectiveStopManager._broker_position_side("UNKNOWN")


def test_fallback_conditional_close_remains_legacy_boundary() -> None:
    from pathlib import Path

    text = Path("src/execution/tp_sl_protective_stop_manager.py").read_text(encoding="utf-8")

    assert "_near_tp_fallback_conditional_close_body" in text
    assert "cross-exchange conditional-close" in text


def test_no_fallback_conditional_close_semantic_action_exists() -> None:
    from pathlib import Path

    for file_name in [
        "src/exchanges/semantic_models.py",
        "src/exchanges/semantics.py",
        "src/exchanges/okx/semantic_executor.py",
    ]:
        text = Path(file_name).read_text(encoding="utf-8")
        forbidden = [
            "FALLBACK_CONDITIONAL_CLOSE",
            "CONDITIONAL_CLOSE",
            "PLACE_FALLBACK",
            "FALLBACK_CLOSE",
        ]
        for token in forbidden:
            assert token not in text, f"{token} should not be introduced before 08C design approval"
