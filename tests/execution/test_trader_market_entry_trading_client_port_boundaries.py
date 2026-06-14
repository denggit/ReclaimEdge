#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_trader_market_entry_trading_client_port_boundaries.py
@Description: Boundary tests — verify that Trader source code meets the
              20C-CLEAN-PORTS-06 contract:
              - Trader.__init__ creates self.trading_client via OkxTradingClient(self)
              - execute_intent() contains .place_market_order(
              - execute_intent() does NOT contain legacy direct-request or
                build_market_entry_order_body patterns in the market-entry branch
              - other methods (sidecar, market exit, etc.) are untouched
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "trader.py"


def _read_source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def _extract_method(source: str, method_name: str) -> str:
    """Extract a single async def method body from source text.

    Returns everything from ``async def <method_name>`` to the next
    class-level ``async def`` or ``def`` (indented with exactly 4 spaces).
    """
    marker = f"async def {method_name}"
    idx = source.find(marker)
    if idx == -1:
        raise AssertionError(f"Method {method_name!r} not found in source")
    remaining = source[idx:]

    # Split at the next class-level function definition (4-space indent)
    for delim in ("\n    async def ", "\n    def "):
        parts = remaining.split(delim, 1)
        if len(parts) > 1:
            return parts[0]
    return remaining


def _extract_init(source: str) -> str:
    """Extract the __init__ method body (def, not async def)."""
    marker = "def __init__(self) -> None:"
    idx = source.find(marker)
    if idx == -1:
        raise AssertionError("__init__ not found in source")
    remaining = source[idx:]

    for delim in ("\n    async def ", "\n    def "):
        parts = remaining.split(delim, 1)
        if len(parts) > 1:
            return parts[0]
    return remaining


# ======================================================================
# Positive checks: trader.py __init__ must create trading_client
# ======================================================================


class TestTraderInitCreatesTradingClient:
    def test_init_has_trading_client_attribute(self):
        init_source = _extract_init(_read_source())
        assert "self.trading_client" in init_source, (
            "Trader.__init__ must assign self.trading_client"
        )

    def test_init_creates_okx_trading_client_with_self(self):
        init_source = _extract_init(_read_source())
        assert "OkxTradingClient(self)" in init_source, (
            "Trader.__init__ must create OkxTradingClient(self)"
        )


# ======================================================================
# Positive checks: execute_intent() must contain .place_market_order(
# ======================================================================


class TestExecuteIntentUsesPlaceMarketOrder:
    def test_execute_intent_contains_place_market_order(self):
        execute_source = _extract_method(_read_source(), "execute_intent")
        assert ".place_market_order(" in execute_source, (
            "execute_intent() must call .place_market_order("
        )


# ======================================================================
# Negative checks: execute_intent() must NOT contain legacy patterns
# ======================================================================


class TestExecuteIntentNoLegacyDirectRequest:
    """The market-entry branch of execute_intent() must NOT contain
    self.request("POST", "/api/v5/trade/order") or
    build_market_entry_order_body."""

    def test_execute_intent_no_direct_order_request(self):
        execute_source = _extract_method(_read_source(), "execute_intent")
        assert 'self.request("POST", "/api/v5/trade/order"' not in execute_source, (
            "execute_intent() must not call self.request('POST', '/api/v5/trade/order') directly"
        )

    def test_execute_intent_no_build_market_entry_order_body(self):
        execute_source = _extract_method(_read_source(), "execute_intent")
        assert "build_market_entry_order_body" not in execute_source, (
            "execute_intent() must not call build_market_entry_order_body directly"
        )


# ======================================================================
# Negative checks: no forbidden types / modules introduced
# ======================================================================


class TestNoForbiddenNewImports:
    def test_no_binance_import(self):
        text = _read_source()
        assert "binance" not in text.lower(), "trader.py must not import binance"

    def test_no_exchange_runtime_bundle_import(self):
        text = _read_source()
        assert "ExchangeRuntimeBundle" not in text

    def test_no_three_stage_adapter_import(self):
        text = _read_source()
        assert "ThreeStageAdapter" not in text

    def test_no_middle_runner_adapter_import(self):
        text = _read_source()
        assert "MiddleRunnerAdapter" not in text

    def test_no_sidecar_adapter_import(self):
        text = _read_source()
        assert "SidecarAdapter" not in text

    def test_no_broker_semantic_executor_in_execute_intent(self):
        execute_source = _extract_method(_read_source(), "execute_intent")
        assert "BrokerSemanticExecutor" not in execute_source


# ======================================================================
# Negative checks: other legacy functions are untouched
# ======================================================================


class TestSidecarMarketEntryUsesTradingClientPort:
    """Verify that place_sidecar_market_order now uses TradingClientPort
    instead of direct OKX request (20C-CLEAN-PORTS-08)."""

    def test_place_sidecar_market_order_uses_place_market_order(self):
        text = _read_source()
        sidecar_source = _extract_method(text, "place_sidecar_market_order")
        assert ".place_market_order(" in sidecar_source, (
            "place_sidecar_market_order must use .place_market_order("
        )

    def test_place_sidecar_market_order_no_direct_request(self):
        text = _read_source()
        sidecar_source = _extract_method(text, "place_sidecar_market_order")
        assert 'self.request("POST", "/api/v5/trade/order"' not in sidecar_source, (
            "place_sidecar_market_order must NOT use direct request (migrated)"
        )

    def test_place_sidecar_market_order_no_build_market_entry_order_body(self):
        """place_sidecar_market_order no longer calls build_market_entry_order_body."""
        sidecar_source = _extract_method(_read_source(), "place_sidecar_market_order")
        assert "build_market_entry_order_body" not in sidecar_source, (
            "place_sidecar_market_order must NOT use build_market_entry_order_body (migrated)"
        )

    def test_place_sidecar_market_order_checks_missing_order_id(self):
        sidecar_source = _extract_method(_read_source(), "place_sidecar_market_order")
        assert "sidecar_market_entry_missing_order_id" in sidecar_source, (
            "place_sidecar_market_order must raise on missing order_id"
        )

    def test_reduce_only_market_order_body_still_present(self):
        """The _reduce_only_market_order_body helper must still exist."""
        text = _read_source()
        assert "def _reduce_only_market_order_body" in text


# ======================================================================
# Positive import checks
# ======================================================================


class TestRequiredImportsPresent:
    def test_okx_trading_client_imported(self):
        text = _read_source()
        assert "from src.execution.okx_trading_client import OkxTradingClient" in text

    def test_trading_client_port_imported(self):
        text = _read_source()
        assert "from src.execution.trading_client_port import TradingClientPort" in text


# ======================================================================
# Compilation check
# ======================================================================


class TestFileCompiles:
    def test_trader_py_compiles(self):
        text = _read_source()
        compile(text, str(_SOURCE_PATH), "exec")
