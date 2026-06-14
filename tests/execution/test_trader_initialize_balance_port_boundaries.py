#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_trader_initialize_balance_port_boundaries.py
@Description: Boundary tests — verify that Trader.initialize() uses
              TradingClientPort.fetch_balance() and does NOT call the
              legacy fetch_usdt_equity() internally.
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


# ======================================================================
# Positive checks: initialize() must contain .fetch_balance(
# ======================================================================


class TestInitializeUsesTradingClientFetchBalance:
    def test_initialize_contains_fetch_balance(self):
        init_source = _extract_method(_read_source(), "initialize")
        assert "self.trading_client.fetch_balance(" in init_source, (
            "initialize() must call self.trading_client.fetch_balance("
        )


# ======================================================================
# Negative checks: initialize() must NOT contain fetch_usdt_equity
# ======================================================================


class TestInitializeDoesNotCallFetchUsdtEquity:
    def test_initialize_no_fetch_usdt_equity(self):
        init_source = _extract_method(_read_source(), "initialize")
        assert "fetch_usdt_equity(" not in init_source, (
            "initialize() must not call fetch_usdt_equity("
        )

    def test_initialize_no_self_fetch_usdt_equity(self):
        init_source = _extract_method(_read_source(), "initialize")
        assert "self.fetch_usdt_equity(" not in init_source, (
            "initialize() must not call self.fetch_usdt_equity("
        )


# ======================================================================
# Positive check: fetch_usdt_equity() method preserved in file
# ======================================================================


class TestFetchUsdtEquityMethodPreserved:
    def test_fetch_usdt_equity_method_exists(self):
        """The fetch_usdt_equity method must be preserved in trader.py
        (it is still used by OkxTradingClient.fetch_balance as the legacy
        bridge)."""
        source = _read_source()
        assert "async def fetch_usdt_equity" in source, (
            "fetch_usdt_equity() method must remain in trader.py"
        )


# ======================================================================
# Negative checks: no forbidden strings in initialize() method
# ======================================================================


class TestInitializeNoForbiddenPatterns:
    def test_initialize_no_direct_balance_request(self):
        """initialize() must not call the REST API directly for balance."""
        init_source = _extract_method(_read_source(), "initialize")
        assert '/api/v5/account/balance' not in init_source, (
            "initialize() must not call /api/v5/account/balance directly"
        )


# ======================================================================
# Negative checks: no forbidden types / modules introduced at file level
# ======================================================================


class TestNoForbiddenNewImportsOrInstantiations:
    def test_no_binance_import(self):
        text = _read_source()
        assert "binance" not in text.lower(), "trader.py must not import binance"

    def test_no_exchange_runtime_bundle_import(self):
        text = _read_source()
        assert "ExchangeRuntimeBundle" not in text

    def test_no_broker_semantic_executor_at_file_level(self):
        """Verify no new plain BrokerSemanticExecutor import (OkxBrokerSemanticExecutor
        is pre-existing and acceptable)."""
        text = _read_source()
        # OkxBrokerSemanticExecutor is pre-existing; check plain BrokerSemanticExecutor
        # is not independently imported
        assert "import BrokerSemanticExecutor" not in text
        assert "from BrokerSemanticExecutor" not in text

    def test_no_three_stage_adapter_import(self):
        text = _read_source()
        assert "ThreeStageAdapter" not in text

    def test_no_middle_runner_adapter_import(self):
        text = _read_source()
        assert "MiddleRunnerAdapter" not in text

    def test_no_sidecar_adapter_import(self):
        text = _read_source()
        assert "SidecarAdapter" not in text

    def test_no_trader_instantiation(self):
        """Trader() must not appear as a top-level instantiation."""
        text = _read_source()
        assert "Trader()" not in text

    def test_no_okx_private_client_instantiation(self):
        """OkxPrivateClient() must not appear as a top-level instantiation."""
        text = _read_source()
        assert "OkxPrivateClient()" not in text

    def test_no_load_dotenv(self):
        text = _read_source()
        assert "load_dotenv" not in text


# ======================================================================
# Compilation check
# ======================================================================


class TestFileCompiles:
    def test_trader_py_compiles(self):
        text = _read_source()
        compile(text, str(_SOURCE_PATH), "exec")
