#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_sidecar_trading_client_port_boundaries.py
@Description: Boundary tests — verify that sidecar methods in trader.py and
              tp_sl_sidecar_manager.py meet the 20C-CLEAN-PORTS-08 contract.
              Method-level scans only.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ======================================================================
# Source file paths
# ======================================================================

_TRADER_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "trader.py"
_SIDECAR_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "tp_sl_sidecar_manager.py"


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_method(source: str, method_name: str) -> str:
    """Extract a single method body from source text."""
    for marker in (f"async def {method_name}", f"def {method_name}"):
        idx = source.find(marker)
        if idx != -1:
            break
    else:
        raise AssertionError(f"Method {method_name!r} not found in source")
    remaining = source[idx:]

    for delim in ("\n    async def ", "\n    def "):
        parts = remaining.split(delim, 1)
        if len(parts) > 1:
            return parts[0]
    return remaining


# ======================================================================
# 1. trader.py — place_sidecar_market_order
# ======================================================================


class TestPlaceSidecarMarketOrderMethod:
    """Method-level scan of trader.place_sidecar_market_order."""

    FORBIDDEN = [
        "build_market_entry_order_body",
        '"/api/v5/trade/order"',
        "'/api/v5/trade/order'",
        "extract_order_id(",
    ]

    REQUIRED = [
        ".place_market_order(",
        "reduce_only=False",
        "sidecar_market_entry_missing_order_id",
    ]

    def test_no_forbidden_patterns(self):
        text = _read_source(_TRADER_PATH)
        method_text = _extract_method(text, "place_sidecar_market_order")

        for forbidden in self.FORBIDDEN:
            lines = method_text.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if forbidden in line:
                    pytest.fail(
                        f"place_sidecar_market_order:{i} must not contain {forbidden}"
                    )

    def test_has_required_patterns(self):
        text = _read_source(_TRADER_PATH)
        method_text = _extract_method(text, "place_sidecar_market_order")

        for required in self.REQUIRED:
            assert required in method_text, (
                f"place_sidecar_market_order must contain {required}"
            )


# ======================================================================
# 2. tp_sl_sidecar_manager.py — place_sidecar_fixed_take_profit
# ======================================================================


class TestPlaceSidecarFixedTpMethod:
    """Method-level scan of SidecarTpManager.place_sidecar_fixed_take_profit."""

    FORBIDDEN = [
        "build_reduce_only_tp_order_body",
        '"/api/v5/trade/order"',
        "'/api/v5/trade/order'",
        "extract_order_id(",
    ]

    REQUIRED = [
        ".place_limit_order(",
        "reduce_only=True",
        "sidecar_fixed_tp_missing_order_id",
    ]

    def test_no_forbidden_patterns(self):
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "place_sidecar_fixed_take_profit")

        for forbidden in self.FORBIDDEN:
            lines = method_text.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if forbidden in line:
                    pytest.fail(
                        f"place_sidecar_fixed_take_profit:{i} must not contain {forbidden}"
                    )

    def test_has_required_patterns(self):
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "place_sidecar_fixed_take_profit")

        for required in self.REQUIRED:
            assert required in method_text, (
                f"place_sidecar_fixed_take_profit must contain {required}"
            )


# ======================================================================
# 3. SidecarTpManager.__init__ accepts trading_client
# ======================================================================


class TestSidecarTpManagerInit:
    """SidecarTpManager.__init__ must accept trading_client."""

    def test_init_accepts_trading_client(self):
        text = _read_source(_SIDECAR_PATH)
        init_text = _extract_method(text, "__init__")
        assert "trading_client" in init_text, (
            "SidecarTpManager.__init__ must accept trading_client"
        )

    def test_init_assigns_trading_client(self):
        text = _read_source(_SIDECAR_PATH)
        init_text = _extract_method(text, "__init__")
        assert "self.trading_client = trading_client" in init_text, (
            "SidecarTpManager.__init__ must assign self.trading_client"
        )


# ======================================================================
# 4. TpSlExecutionManager wires trading_client to SidecarTpManager
# ======================================================================


class TestTpSlExecutionManagerWiresSidecar:
    """TpSlExecutionManager.__init__ passes trading_client to SidecarTpManager."""

    def test_sidecar_gets_trading_client(self):
        from pathlib import Path
        exec_mgr_path = Path(__file__).resolve().parents[2] / "src" / "execution" / "tp_sl_execution_manager.py"
        text = exec_mgr_path.read_text(encoding="utf-8")

        # Find __init__
        idx = text.find("def __init__(self, trader: Trader, *, trading_client")
        if idx == -1:
            raise AssertionError("__init__ not found")
        remaining = text[idx:]
        for delim in ("\n    async def ", "\n    def "):
            parts = remaining.split(delim, 1)
            if len(parts) > 1:
                init_text = parts[0]
                break
        else:
            init_text = remaining

        assert "SidecarTpManager(trader, self.trading_client)" in init_text, (
            "TpSlExecutionManager must pass self.trading_client to SidecarTpManager"
        )


# ======================================================================
# 5. No forbidden abstractions in sidecar manager
# ======================================================================


class TestNoForbiddenInSidecarManager:
    """SidecarTpManager must not introduce forbidden abstractions."""

    FORBIDDEN = [
        "Binance",
        "binance",
        "ExchangeRuntimeBundle",
        "BrokerSemanticExecutor",
        "ThreeStageAdapter",
        "MiddleRunnerAdapter",
        "SidecarAdapter",
    ]

    def test_no_forbidden_tokens(self):
        text = _read_source(_SIDECAR_PATH)
        for token in self.FORBIDDEN:
            assert token not in text, (
                f"tp_sl_sidecar_manager.py must not reference {token}"
            )

    def test_no_new_trader_instantiation(self):
        text = _read_source(_SIDECAR_PATH)
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue
            if "= Trader(" in stripped:
                pytest.fail(f"tp_sl_sidecar_manager.py:{i} creates new Trader()")
            if "= OkxPrivateClient(" in stripped:
                pytest.fail(f"tp_sl_sidecar_manager.py:{i} creates new OkxPrivateClient()")
            if "= OkxTradingClient(" in stripped:
                pytest.fail(
                    f"tp_sl_sidecar_manager.py:{i} must not instantiate OkxTradingClient — "
                    f"receive from TpSlExecutionManager"
                )

    def test_no_new_env_reads(self):
        text = _read_source(_SIDECAR_PATH)
        # Only allow existing semantic env reads — no load_dotenv
        assert "load_dotenv" not in text, (
            "tp_sl_sidecar_manager.py must not call load_dotenv"
        )


# ======================================================================
# 6. Compilation check
# ======================================================================


class TestFilesCompile:
    @pytest.mark.parametrize("file_path", [
        _TRADER_PATH,
        _SIDECAR_PATH,
    ])
    def test_file_compiles(self, file_path: Path) -> None:
        text = _read_source(file_path)
        compile(text, str(file_path), "exec")
