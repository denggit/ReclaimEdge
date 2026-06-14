#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_tp_sl_protective_stop_manager_semantic_placement.py
@Description: Tests for the optional semantic protective SL placement path
              in ProtectiveStopManager.place_near_tp_protective_stop_with_retries().
              When semantic is disabled, placement routes through TradingClientPort.
              When semantic is enabled, placement routes through semantic executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit, ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticResult,
)
from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager


# ---------------------------------------------------------------------------
# FakeTradingClient — records place_stop_market_order calls
# ---------------------------------------------------------------------------


@dataclass
class FakeOrderResult:
    ok: bool = True
    order_id: str | None = "sl-port-1"
    client_order_id: str | None = None
    message: str = ""


class FakeTradingClient:
    """Records every TradingClientPort call so tests can assert routing."""

    def __init__(self) -> None:
        self.place_stop_calls: list[dict[str, Any]] = []
        self._next_stop_result = FakeOrderResult(order_id="sl-port-1")

    async def place_stop_market_order(
        self, *, side, qty, trigger_price, reduce_only, client_order_id
    ) -> FakeOrderResult:
        self.place_stop_calls.append(
            {
                "side": side,
                "qty": qty,
                "trigger_price": trigger_price,
                "reduce_only": reduce_only,
                "client_order_id": client_order_id,
            }
        )
        return self._next_stop_result


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
        # Fallback algo IDs from fake trading client don't start with "fallback-"
        # since they come from FakeTradingClient, not from FakeTrader.request().
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


def make_manager(
    trader: FakeTrader, trading_client: FakeTradingClient | None = None
) -> ProtectiveStopManager:
    if trading_client is None:
        trading_client = FakeTradingClient()
    return ProtectiveStopManager(trader, trading_client=trading_client)  # type: ignore[arg-type]


# ===================================================================
# Test: default off → TradingClientPort primary
# ===================================================================


@pytest.mark.asyncio
async def test_default_off_uses_trading_client_port_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the env var is not set, primary placement routes through
    TradingClientPort.place_stop_market_order()."""
    monkeypatch.delenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", raising=False)

    trader = FakeTrader()
    fake_tc = FakeTradingClient()
    fake_tc._next_stop_result = FakeOrderResult(order_id="sl-tc-1")
    manager = make_manager(trader, trading_client=fake_tc)

    ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
        side="LONG",
        contracts=Decimal("10"),
        stop_price=3400.0,
        retry_count=1,
        retry_interval_seconds=0,
    )

    assert ok is True
    assert algo_id == "sl-tc-1"
    assert message == "protective_sl_placed"
    assert len(fake_tc.place_stop_calls) == 1
    call = fake_tc.place_stop_calls[0]
    assert call["side"] == "LONG"
    assert call["qty"] == Decimal("10")
    assert call["trigger_price"] == Decimal("3400.0")
    assert call["reduce_only"] is True
    assert call["client_order_id"] == ""
    assert trader.requests == [], "No direct REST request when routing through port"
    assert trader.semantic.calls == [], "Semantic not used when disabled"


# ===================================================================
# Test: enabled → semantic primary, no trading client
# ===================================================================


@pytest.mark.asyncio
async def test_enabled_uses_semantic_primary_no_trading_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When enabled, primary placement goes through semantic executor,
    not trading_client."""
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", "true")

    trader = FakeTrader()
    fake_tc = FakeTradingClient()
    manager = make_manager(trader, trading_client=fake_tc)

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
    assert fake_tc.place_stop_calls == [], (
        "Trading client must not be used when semantic primary succeeds"
    )
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
# Test: verify failure → cancel unverified, fallback through port
# ===================================================================


@pytest.mark.asyncio
async def test_verify_failure_cancels_unverified_semantic_fallback_via_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When verify fails, cancel_unverified is called for the semantic algo_id,
    and the fallback path routes through TradingClientPort."""
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", "true")

    trader = FakeTrader()
    fake_tc = FakeTradingClient()
    fake_tc._next_stop_result = FakeOrderResult(order_id="sl-fallback-tc")
    manager = make_manager(trader, trading_client=fake_tc)

    # Use a counting verify: primary (semantic) fails, fallback (port) succeeds
    verify_count = [0]

    async def counting_verify(_algo_id, _side, _contracts, _stop_price):
        verify_count[0] += 1
        return verify_count[0] > 1  # First call fails, subsequent succeed

    trader.verify_near_tp_protective_stop = counting_verify  # type: ignore[method-assign]

    ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
        side="LONG",
        contracts=Decimal("10"),
        stop_price=3400.0,
        retry_count=1,
        retry_interval_seconds=0,
    )

    # Primary semantic was attempted, verify failed → cancelled
    assert ("semantic-sl-1", "primary") in trader.cancelled_unverified

    # Fallback routes through trading_client (not legacy request)
    assert ok is True
    assert algo_id == "sl-fallback-tc"
    assert message == "fallback_conditional_close_placed"

    # Semantic was called once for primary
    assert len(trader.semantic.calls) == 1

    # Fallback via trading client, not legacy request
    assert len(fake_tc.place_stop_calls) == 1
    call = fake_tc.place_stop_calls[0]
    assert call["side"] == "LONG"
    assert call["qty"] == Decimal("10")
    assert call["trigger_price"] == Decimal("3400.0")
    assert call["reduce_only"] is True
    assert trader.requests == [], "No direct REST request in fallback path"


# ===================================================================
# Test: semantic failure → fallback through port
# ===================================================================


@pytest.mark.asyncio
async def test_semantic_failure_enters_fallback_via_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When semantic primary fails, the exception is caught and fallback
    uses TradingClientPort.place_stop_market_order()."""
    monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", "true")

    trader = FakeTrader()
    trader.semantic.ok = False
    trader.semantic.message = "boom"
    fake_tc = FakeTradingClient()
    fake_tc._next_stop_result = FakeOrderResult(order_id="sl-fallback-tc")
    manager = make_manager(trader, trading_client=fake_tc)

    ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
        side="LONG",
        contracts=Decimal("10"),
        stop_price=3400.0,
        retry_count=1,
        retry_interval_seconds=0,
    )

    # Semantic was attempted once
    assert len(trader.semantic.calls) == 1

    # No legacy request
    assert trader.requests == [], "Legacy request must not be used"

    # Fallback routes through trading client
    assert len(fake_tc.place_stop_calls) == 1
    call = fake_tc.place_stop_calls[0]
    assert call["side"] == "LONG"
    assert call["reduce_only"] is True

    assert ok is True
    assert algo_id == "sl-fallback-tc"
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
        assert manager.trading_client.place_stop_calls == []

    @pytest.mark.parametrize("value", ["0", "false", "no", "n", "off", "", "   ", "maybe"])
    async def test_env_var_disabled_variants(self, monkeypatch, value: str) -> None:
        """All falsy variants route through TradingClientPort."""
        monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", value)

        trader = FakeTrader()
        fake_tc = FakeTradingClient()
        fake_tc._next_stop_result = FakeOrderResult(order_id="sl-disabled-tc")
        manager = make_manager(trader, trading_client=fake_tc)

        ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
            side="LONG",
            contracts=Decimal("10"),
            stop_price=3400.0,
            retry_count=1,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "sl-disabled-tc"
        assert message == "protective_sl_placed"
        assert trader.semantic.calls == []
        assert trader.requests == [], "No direct REST when disabled — must route through port"
        assert len(fake_tc.place_stop_calls) == 1
        call = fake_tc.place_stop_calls[0]
        assert call["side"] == "LONG"
        assert call["qty"] == Decimal("10")
        assert call["trigger_price"] == Decimal("3400.0")
        assert call["reduce_only"] is True
        assert call["client_order_id"] == ""


# ===================================================================
# Test: semantic protective SL placement helper methods
# ===================================================================


def test_broker_semantic_protective_sl_placement_enabled_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_fallback_routes_through_trading_client_port() -> None:
    """The fallback path no longer uses _near_tp_fallback_conditional_close_body;
    it now routes through TradingClientPort."""
    from pathlib import Path

    text = Path("src/execution/tp_sl_protective_stop_manager.py").read_text(encoding="utf-8")

    # Fallback placement no longer uses the legacy conditional close body
    assert "_near_tp_fallback_conditional_close_body" not in text, (
        "fallback must not use legacy conditional close body — route through TradingClientPort"
    )
    # Both primary and fallback use TradingClientPort
    assert "trading_client.place_stop_market_order" in text


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
