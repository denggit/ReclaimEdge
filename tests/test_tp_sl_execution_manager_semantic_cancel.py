#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_tp_sl_execution_manager_semantic_cancel.py
@Description: Tests for the optional semantic reduce-only TP cancel path
              in TpSlExecutionManager.cancel_existing_reduce_only_orders().
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticResult,
)
from src.execution.okx_trading_client import OkxTradingClient
from src.execution.tp_sl_execution_manager import TpSlExecutionManager


# ---------------------------------------------------------------------------
# FakeSemanticExecutor
# ---------------------------------------------------------------------------


class FakeSemanticExecutor:
    """A test double that records cancel_reduce_only_tp calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.ok: bool = True
        self.message: str = ""

    async def cancel_reduce_only_tp(self, *, symbol: str, order_id: str) -> BrokerSemanticResult:
        self.calls.append((symbol, order_id))
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.TP1,
            ok=self.ok,
            order_id=order_id,
            message=self.message,
        )


# ---------------------------------------------------------------------------
# FakeTrader
# ---------------------------------------------------------------------------


class FakeTradingClientForCancel:
    """Minimal mock TradingClientPort for testing cancel operations."""

    def __init__(self, trader: FakeTrader) -> None:
        self._trader = trader

    async def fetch_open_orders(self):
        # Return fake OrderSnapshots matching trader.pending_orders
        # Filter by symbol (like real OkxTradingClient does)
        from src.execution.trading_client_port import OrderSnapshot
        from decimal import Decimal
        result = []
        for item in self._trader.pending_orders:
            if item.get("instId") != self._trader.symbol:
                continue
            result.append(OrderSnapshot(
                order_id=item.get("ordId"),
                client_order_id=item.get("clOrdId"),
                side=str(item.get("side", "sell")),
                qty=Decimal(str(item.get("sz", "1"))),
                price=Decimal(str(item.get("px", "0"))) if item.get("px") else None,
                reduce_only=str(item.get("reduceOnly", "")).lower() == "true",
                raw=item,
            ))
        return result

    async def cancel_order(self, *, order_id=None, client_order_id=None):
        from src.execution.trading_client_port import CancelResult
        self._trader.requests.append(("POST", "/api/v5/trade/cancel-order", {"ordId": order_id}))
        return CancelResult(ok=True, order_id=order_id, raw={"code": "0", "data": [{"ordId": order_id}]})

    async def cancel_algo_order(self, *, order_id=None, client_order_id=None):
        from src.execution.trading_client_port import CancelResult
        self._trader.requests.append(("POST", "/api/v5/trade/cancel-algos", {"algoId": order_id}))
        return CancelResult(ok=True, order_id=order_id, raw={"code": "0", "data": [{"algoId": order_id}]})


class FakeTrader:
    symbol = "ETH-USDT-SWAP"

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict]] = []
        self.pending_orders: list[dict] = []
        self._protected_reduce_only_order_ids: set[str] = set()
        self._managed_reduce_only_order_ids: set[str] = set()
        self._allow_cancel_unmanaged_reduce_only: bool = True
        self.semantic = FakeSemanticExecutor()
        self.trading_client = FakeTradingClientForCancel(self)

    @property
    def broker_semantic_executor(self) -> FakeSemanticExecutor:
        return self.semantic

    async def fetch_pending_orders(self) -> list[dict]:
        return list(self.pending_orders)

    async def fetch_broker_open_orders(self):
        """Return BrokerOrder objects built from the raw pending_orders dicts."""
        orders: list[BrokerOrder] = []
        for item in self.pending_orders:
            # Production fetch_broker_open_orders is symbol-filtered
            if item.get("instId") != self.symbol:
                continue
            orders.append(
                BrokerOrder(
                    exchange=ExchangeName.OKX,
                    symbol=item.get("instId", self.symbol),
                    order_id=item.get("ordId"),
                    client_order_id=item.get("clOrdId"),
                    side=(
                        BrokerOrderSide.BUY
                        if str(item.get("side", "")).lower() == "buy"
                        else BrokerOrderSide.SELL
                    ),
                    position_side=BrokerPositionSide.LONG,
                    order_type=BrokerOrderType.LIMIT,
                    status=BrokerOrderStatus.OPEN,
                    price=Decimal(item["px"]) if item.get("px") else None,
                    quantity=Decimal(item["sz"]) if item.get("sz") else Decimal("1"),
                    quantity_unit=BrokerQuantityUnit.CONTRACTS,
                    reduce_only=str(item.get("reduceOnly", "")).lower() == "true",
                    raw=item,
                )
            )
        return tuple(orders)

    async def request(self, method: str, endpoint: str, payload: dict | None = None) -> dict:
        self.requests.append((method, endpoint, payload or {}))
        return {"data": [{"ordId": payload.get("ordId", ""), "sCode": "0"}]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fake_trader(**overrides) -> FakeTrader:
    t = FakeTrader()
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


# ===================================================================
# Test: default off → legacy cancel-order
# ===================================================================


@pytest.mark.asyncio
class TestDefaultOffLegacyCancel:
    async def test_default_off_uses_legacy_cancel(self, monkeypatch) -> None:
        """When the env var is not set, legacy /api/v5/trade/cancel-order is used."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        assert len(trader.requests) == 1, "Expected 1 legacy request"
        _method, endpoint, payload = trader.requests[0]
        assert endpoint == "/api/v5/trade/cancel-order"
        assert payload["ordId"] == "tp-1"
        assert trader.semantic.calls == [], "Semantic path must not be called when disabled"

    async def test_default_off_uses_legacy_cancel_multiple_orders(self, monkeypatch) -> None:
        """With multiple eligible orders, all are cancelled via legacy path."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-2", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        assert len(trader.requests) == 2
        cancelled_ids = [r[2]["ordId"] for r in trader.requests]
        assert cancelled_ids == ["tp-1", "tp-2"]
        assert trader.semantic.calls == []


# ===================================================================
# Test: enabled → semantic cancel, no legacy
# ===================================================================


@pytest.mark.asyncio
class TestEnabledSemanticCancel:
    async def test_enabled_uses_semantic_cancel(self, monkeypatch) -> None:
        """When enabled, cancel goes through semantic executor, not legacy request."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        assert trader.semantic.calls == [("ETH-USDT-SWAP", "tp-1")], (
            "Semantic cancel must be called with correct symbol and ordId"
        )
        assert trader.requests == [], "Legacy request must not be made when semantic is enabled"

    async def test_enabled_multiple_orders_all_semantic(self, monkeypatch) -> None:
        """All eligible orders are cancelled via semantic path."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-2", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        assert trader.semantic.calls == [
            ("ETH-USDT-SWAP", "tp-1"),
            ("ETH-USDT-SWAP", "tp-2"),
        ]
        assert trader.requests == []


# ===================================================================
# Test: semantic failure → no fallback legacy
# ===================================================================


@pytest.mark.asyncio
class TestSemanticFailureNoFallback:
    async def test_semantic_failure_raises_no_fallback(self, monkeypatch) -> None:
        """When semantic cancel fails, RuntimeError is raised and legacy is NOT attempted."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
        ]
        trader.semantic.ok = False
        trader.semantic.message = "boom"

        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]

        with pytest.raises(RuntimeError) as exc_info:
            await manager.cancel_existing_reduce_only_orders()
        assert "semantic_reduce_only_cancel_failed" in str(exc_info.value)
        assert "boom" in str(exc_info.value)
        # No legacy fallback
        assert trader.requests == [], (
            "Legacy cancel must NOT be attempted when semantic cancel fails"
        )

    async def test_semantic_failure_no_fallback_even_with_mixed_orders(self, monkeypatch) -> None:
        """First order fails → exception is raised, second order never attempted."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-2", "reduceOnly": "true"},
        ]
        trader.semantic.ok = False
        trader.semantic.message = "boom"

        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]

        with pytest.raises(RuntimeError) as exc_info:
            await manager.cancel_existing_reduce_only_orders()
        assert "semantic_reduce_only_cancel_failed" in str(exc_info.value)
        # Only the first order was attempted
        assert len(trader.semantic.calls) == 1
        assert trader.requests == []


# ===================================================================
# Test: protected orders still skipped
# ===================================================================


@pytest.mark.asyncio
class TestProtectedOrderSkipped:
    async def test_protected_order_skipped_with_semantic_enabled(self, monkeypatch) -> None:
        """Protected orders are skipped regardless of semantic being enabled."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader._protected_reduce_only_order_ids = {"tp-1"}
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-2", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        # tp-1 is protected → skipped; only tp-2 is cancelled
        assert trader.semantic.calls == [("ETH-USDT-SWAP", "tp-2")]
        assert trader.requests == []

    async def test_protected_order_skipped_with_legacy(self, monkeypatch) -> None:
        """Protected orders are skipped in legacy path too."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        trader = make_fake_trader()
        trader._protected_reduce_only_order_ids = {"tp-1"}
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-2", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        cancelled_ids = [r[2]["ordId"] for r in trader.requests]
        assert "tp-1" not in cancelled_ids
        assert "tp-2" in cancelled_ids
        assert trader.semantic.calls == []


# ===================================================================
# Test: managed ids safety still enforced
# ===================================================================


@pytest.mark.asyncio
class TestManagedIdsSafety:
    async def test_unmanaged_order_with_managed_set_raises_semantic(self, monkeypatch) -> None:
        """When managed_order_ids is set and an order is not in it, raise identity error."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader._managed_reduce_only_order_ids = {"tp-1"}
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-2", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]

        with pytest.raises(RuntimeError) as exc_info:
            await manager.cancel_existing_reduce_only_orders()
        assert "reduce_only_order_identity_unknown" in str(exc_info.value)
        # tp-1 (managed) was cancelled before tp-2 (unmanaged) caused the error
        assert trader.semantic.calls == [("ETH-USDT-SWAP", "tp-1")]
        assert trader.requests == []

    async def test_unmanaged_order_with_managed_set_raises_legacy(self, monkeypatch) -> None:
        """Same safety check works in legacy path."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        trader = make_fake_trader()
        trader._managed_reduce_only_order_ids = {"tp-1"}
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-2", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]

        with pytest.raises(RuntimeError) as exc_info:
            await manager.cancel_existing_reduce_only_orders()
        assert "reduce_only_order_identity_unknown" in str(exc_info.value)
        # tp-1 was cancelled before the error
        cancelled_ids = [r[2]["ordId"] for r in trader.requests]
        assert "tp-1" in cancelled_ids
        assert trader.semantic.calls == []


# ===================================================================
# Test: allow_unmanaged=False still enforced
# ===================================================================


@pytest.mark.asyncio
class TestAllowUnmanagedFalse:
    async def test_allow_unmanaged_false_raises_semantic(self, monkeypatch) -> None:
        """When managed set is empty and allow_unmanaged=False, raise identity error."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = False
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]

        with pytest.raises(RuntimeError) as exc_info:
            await manager.cancel_existing_reduce_only_orders()
        assert "reduce_only_order_identity_unknown" in str(exc_info.value)
        assert trader.semantic.calls == []
        assert trader.requests == []

    async def test_allow_unmanaged_false_raises_legacy(self, monkeypatch) -> None:
        """Same safety check works in legacy path."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        trader = make_fake_trader()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = False
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]

        with pytest.raises(RuntimeError) as exc_info:
            await manager.cancel_existing_reduce_only_orders()
        assert "reduce_only_order_identity_unknown" in str(exc_info.value)
        assert trader.semantic.calls == []
        assert trader.requests == []


# ===================================================================
# Test: symbol / reduceOnly filtering still enforced
# ===================================================================


@pytest.mark.asyncio
class TestSymbolReduceOnlyFiltering:
    async def test_wrong_symbol_not_cancelled_semantic(self, monkeypatch) -> None:
        """Orders for other symbols are skipped."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "BTC-USDT-SWAP", "ordId": "btc-1", "reduceOnly": "true"},
            {"instId": "ETH-USDT-SWAP", "ordId": "eth-1", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        # Only ETH-USDT-SWAP orders are cancelled
        assert trader.semantic.calls == [("ETH-USDT-SWAP", "eth-1")]
        assert trader.requests == []

    async def test_not_reduce_only_skipped_semantic(self, monkeypatch) -> None:
        """Orders that are not reduceOnly are skipped."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "eth-1", "reduceOnly": "false"},
            {"instId": "ETH-USDT-SWAP", "ordId": "eth-2", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        # Only reduceOnly=true orders are cancelled
        assert trader.semantic.calls == [("ETH-USDT-SWAP", "eth-2")]
        assert trader.requests == []

    async def test_mixed_symbol_and_reduce_only_filtering_semantic(self, monkeypatch) -> None:
        """Combined symbol + reduceOnly filtering works correctly."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "BTC-USDT-SWAP", "ordId": "btc-1", "reduceOnly": "true"},
            {"instId": "ETH-USDT-SWAP", "ordId": "eth-1", "reduceOnly": "false"},
            {"instId": "ETH-USDT-SWAP", "ordId": "eth-2", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        # Only eth-2 matches: right symbol AND reduceOnly=true
        assert trader.semantic.calls == [("ETH-USDT-SWAP", "eth-2")]
        assert trader.requests == []

    async def test_mixed_symbol_and_reduce_only_filtering_legacy(self, monkeypatch) -> None:
        """Same filtering works in legacy path."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "BTC-USDT-SWAP", "ordId": "btc-1", "reduceOnly": "true"},
            {"instId": "ETH-USDT-SWAP", "ordId": "eth-1", "reduceOnly": "false"},
            {"instId": "ETH-USDT-SWAP", "ordId": "eth-2", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        cancelled_ids = [r[2]["ordId"] for r in trader.requests]
        assert cancelled_ids == ["eth-2"]
        assert trader.semantic.calls == []


# ===================================================================
# Test: ordId missing check still enforced
# ===================================================================


@pytest.mark.asyncio
class TestOrdIdMissingCheck:
    async def test_missing_ord_id_raises_semantic(self, monkeypatch) -> None:
        """Missing ordId raises identity error in semantic path."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", "true")

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]

        with pytest.raises(RuntimeError) as exc_info:
            await manager.cancel_existing_reduce_only_orders()
        assert "reduce_only_order_identity_unknown" in str(exc_info.value)
        assert trader.semantic.calls == []
        assert trader.requests == []

    async def test_missing_ord_id_raises_legacy(self, monkeypatch) -> None:
        """Missing ordId raises identity error in legacy path."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]

        with pytest.raises(RuntimeError) as exc_info:
            await manager.cancel_existing_reduce_only_orders()
        assert "reduce_only_order_identity_unknown" in str(exc_info.value)
        assert trader.semantic.calls == []
        assert trader.requests == []


# ===================================================================
# Test: env var value variants
# ===================================================================


@pytest.mark.asyncio
class TestEnvVarVariants:
    @pytest.mark.parametrize("value", ["1", "true", "yes", "y", "on", "True", "YES", "ON"])
    async def test_env_var_enabled_variants(self, monkeypatch, value: str) -> None:
        """All truthy variants enable the semantic path."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", value)

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        assert trader.semantic.calls == [("ETH-USDT-SWAP", "tp-1")]
        assert trader.requests == []

    @pytest.mark.parametrize("value", ["0", "false", "no", "n", "off", "", "   ", "maybe"])
    async def test_env_var_disabled_variants(self, monkeypatch, value: str) -> None:
        """All falsy variants keep legacy path."""
        monkeypatch.setenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", value)

        trader = make_fake_trader()
        trader.pending_orders = [
            {"instId": "ETH-USDT-SWAP", "ordId": "tp-1", "reduceOnly": "true"},
        ]
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]  # type: ignore[arg-type]
        await manager.cancel_existing_reduce_only_orders()

        assert len(trader.requests) == 1
        assert trader.requests[0][1] == "/api/v5/trade/cancel-order"
        assert trader.semantic.calls == []
