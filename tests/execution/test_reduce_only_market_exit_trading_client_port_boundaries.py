#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_reduce_only_market_exit_trading_client_port_boundaries.py
@Description: Boundary tests — verify that the reduce-only market exit source
              code meets the 20C-CLEAN-PORTS-07 contract:
              - migrated methods do NOT contain legacy direct-request patterns
              - migrated methods DO contain .place_market_order(reduce_only=True)
              - unmigrated methods (sidecar entry/fixed TP) are untouched
              - no forbidden abstractions introduced
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ======================================================================
# Source file paths
# ======================================================================

_MARKET_EXIT_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "tp_sl_market_exit_manager.py"
_NEAR_TP_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "tp_sl_near_tp_manager.py"
_SIDECAR_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "tp_sl_sidecar_manager.py"
_EXECUTION_MGR_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "tp_sl_execution_manager.py"
_TRADER_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "trader.py"


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_method(source: str, method_name: str) -> str:
    """Extract a single async def method body from source text."""
    marker = f"async def {method_name}"
    idx = source.find(marker)
    if idx == -1:
        # Try def (non-async)
        marker = f"def {method_name}"
        idx = source.find(marker)
    if idx == -1:
        raise AssertionError(f"Method {method_name!r} not found in source")
    remaining = source[idx:]

    # Split at the next class-level function definition
    for delim in ("\n    async def ", "\n    def "):
        parts = remaining.split(delim, 1)
        if len(parts) > 1:
            return parts[0]
    return remaining


# ======================================================================
# 1. MarketExitManager — migrated method checks
# ======================================================================


class TestMarketExitManagerMigratedMethods:
    """market_exit_remaining_position_with_retries is the migrated method."""

    FORBIDDEN_IN_MIGRATED = [
        "build_reduce_only_market_order_body",
        "_reduce_only_market_order_body",
        '"/api/v5/trade/order"',
        '"/api/v5/trade/order\'',
        "extract_order_id(",
    ]

    REQUIRED_IN_MIGRATED = [
        ".place_market_order(",
        "reduce_only=True",
    ]

    def test_migrated_method_no_forbidden_patterns(self):
        """market_exit_remaining_position_with_retries must NOT contain
        legacy direct-request patterns."""
        text = _read_source(_MARKET_EXIT_PATH)
        method_text = _extract_method(text, "market_exit_remaining_position_with_retries")

        for forbidden in self.FORBIDDEN_IN_MIGRATED:
            # Allow in comments only
            lines = method_text.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if forbidden in line:
                    pytest.fail(
                        f"market_exit_remaining_position_with_retries:{i} "
                        f"must not contain {forbidden}"
                    )

    def test_migrated_method_has_required_patterns(self):
        """market_exit_remaining_position_with_retries must contain
        .place_market_order( and reduce_only=True."""
        text = _read_source(_MARKET_EXIT_PATH)
        method_text = _extract_method(text, "market_exit_remaining_position_with_retries")

        for required in self.REQUIRED_IN_MIGRATED:
            assert required in method_text, (
                f"market_exit_remaining_position_with_retries must contain {required}"
            )

    def test_migrated_method_checks_missing_order_id(self):
        """The migrated method must raise on missing order_id."""
        text = _read_source(_MARKET_EXIT_PATH)
        method_text = _extract_method(text, "market_exit_remaining_position_with_retries")

        assert "reduce_only_market_exit_missing_order_id" in method_text, (
            "Missing order_id must raise RuntimeError with descriptive message"
        )


# ======================================================================
# 2. NearTpExecutionManager — migrated method checks
# ======================================================================


class TestNearTpExecutionManagerMigratedMethods:
    """execute_near_tp_reduce is the migrated method."""

    FORBIDDEN_IN_MIGRATED = [
        "build_reduce_only_market_order_body",
        "_reduce_only_market_order_body",
        '"/api/v5/trade/order"',
        '"/api/v5/trade/order\'',
        "extract_order_id(",
    ]

    REQUIRED_IN_MIGRATED = [
        ".place_market_order(",
        "reduce_only=True",
    ]

    def test_migrated_method_no_forbidden_patterns(self):
        """execute_near_tp_reduce must NOT contain legacy patterns."""
        text = _read_source(_NEAR_TP_PATH)
        method_text = _extract_method(text, "execute_near_tp_reduce")

        for forbidden in self.FORBIDDEN_IN_MIGRATED:
            lines = method_text.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if forbidden in line:
                    pytest.fail(
                        f"execute_near_tp_reduce:{i} must not contain {forbidden}"
                    )

    def test_migrated_method_has_required_patterns(self):
        """execute_near_tp_reduce must contain .place_market_order( and reduce_only=True."""
        text = _read_source(_NEAR_TP_PATH)
        method_text = _extract_method(text, "execute_near_tp_reduce")

        for required in self.REQUIRED_IN_MIGRATED:
            assert required in method_text, (
                f"execute_near_tp_reduce must contain {required}"
            )

    def test_migrated_method_checks_missing_order_id(self):
        """The migrated method must raise on missing order_id."""
        text = _read_source(_NEAR_TP_PATH)
        method_text = _extract_method(text, "execute_near_tp_reduce")

        assert "near_tp_reduce_only_market_order_missing_order_id" in method_text, (
            "Missing order_id must raise RuntimeError with descriptive message"
        )


# ======================================================================
# 3. MarketExitManager __init__ accepts trading_client
# ======================================================================


class TestMarketExitManagerInit:
    """MarketExitManager.__init__ must accept trading_client."""

    def test_init_accepts_trading_client(self):
        text = _read_source(_MARKET_EXIT_PATH)
        init_text = _extract_method(text, "__init__")
        assert "trading_client" in init_text, (
            "MarketExitManager.__init__ must accept trading_client"
        )

    def test_init_assigns_trading_client(self):
        text = _read_source(_MARKET_EXIT_PATH)
        init_text = _extract_method(text, "__init__")
        assert "self.trading_client = trading_client" in init_text, (
            "MarketExitManager.__init__ must assign self.trading_client"
        )


# ======================================================================
# 4. NearTpExecutionManager __init__ accepts trading_client
# ======================================================================


class TestNearTpExecutionManagerInit:
    """NearTpExecutionManager.__init__ must accept trading_client."""

    def test_init_accepts_trading_client(self):
        text = _read_source(_NEAR_TP_PATH)
        init_text = _extract_method(text, "__init__")
        assert "trading_client" in init_text, (
            "NearTpExecutionManager.__init__ must accept trading_client"
        )

    def test_init_assigns_trading_client(self):
        text = _read_source(_NEAR_TP_PATH)
        init_text = _extract_method(text, "__init__")
        assert "self.trading_client = trading_client" in init_text, (
            "NearTpExecutionManager.__init__ must assign self.trading_client"
        )


# ======================================================================
# 5. TpSlExecutionManager wires trading_client
# ======================================================================


class TestTpSlExecutionManagerWiring:
    """TpSlExecutionManager.__init__ must pass trading_client to sub-managers."""

    def test_market_exit_gets_trading_client(self):
        text = _read_source(_EXECUTION_MGR_PATH)
        init_text = _extract_method(text, "__init__")
        assert "MarketExitManager(trader, self.trading_client)" in init_text, (
            "TpSlExecutionManager must pass self.trading_client to MarketExitManager"
        )

    def test_near_tp_gets_trading_client(self):
        text = _read_source(_EXECUTION_MGR_PATH)
        init_text = _extract_method(text, "__init__")
        assert "trading_client=self.trading_client" in init_text, (
            "TpSlExecutionManager must pass self.trading_client to NearTpExecutionManager"
        )


# ======================================================================
# 6. Sidecar is NOT migrated (untouched)
# ======================================================================


class TestSidecarMigratedToTradingClientPort:
    """Sidecar fixed TP is now migrated to TradingClientPort.place_limit_order."""

    def test_sidecar_tp_placement_uses_place_limit_order(self):
        """place_sidecar_fixed_take_profit now uses .place_limit_order(
        (migrated in 20C-CLEAN-PORTS-08)."""
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "place_sidecar_fixed_take_profit")

        assert ".place_limit_order(" in method_text, (
            "Sidecar fixed TP must use .place_limit_order( (migrated)"
        )
        # Must NOT contain place_market_order
        assert ".place_market_order(" not in method_text, (
            "Sidecar fixed TP is a limit order — must NOT call place_market_order"
        )

    def test_sidecar_tp_placement_no_direct_request(self):
        """place_sidecar_fixed_take_profit must NOT contain direct
        /api/v5/trade/order request."""
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "place_sidecar_fixed_take_profit")

        for forbidden in ('"/api/v5/trade/order"', "'/api/v5/trade/order'",
                          "build_reduce_only_tp_order_body", "extract_order_id("):
            lines = method_text.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if forbidden in line:
                    pytest.fail(
                        f"place_sidecar_fixed_take_profit:{i} must not contain {forbidden}"
                    )

    def test_sidecar_tp_placement_checks_missing_order_id(self):
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "place_sidecar_fixed_take_profit")
        assert "sidecar_fixed_tp_missing_order_id" in method_text, (
            "Sidecar fixed TP must raise on missing order_id"
        )

    def test_sidecar_has_no_reduce_only_market_order(self):
        """SidecarTpManager has no reduce-only market order to migrate."""
        text = _read_source(_SIDECAR_PATH)
        # The sidecar manager should not contain any market order placement
        assert ".place_market_order(" not in text, (
            "SidecarTpManager has no reduce-only market order"
        )


# ======================================================================
# 7. Trader._reduce_only_market_order_body still exists
# ======================================================================


class TestTraderReduceOnlyMarketOrderBodyExists:
    """The _reduce_only_market_order_body helper on Trader must still exist."""

    def test_method_still_present(self):
        text = _read_source(_TRADER_PATH)
        assert "def _reduce_only_market_order_body" in text, (
            "Trader._reduce_only_market_order_body must still exist"
        )


# ======================================================================
# 8. No Binance or forbidden abstractions
# ======================================================================


class TestNoForbiddenImports:
    """Migrated files must not reference forbidden abstractions."""

    MIGRATED_FILES = [
        _MARKET_EXIT_PATH,
        _NEAR_TP_PATH,
        _EXECUTION_MGR_PATH,
        _SIDECAR_PATH,
    ]

    FORBIDDEN_TOKENS = [
        "Binance",
        "binance",
        "ExchangeRuntimeBundle",
        "ThreeStageAdapter",
        "MiddleRunnerAdapter",
        "SidecarAdapter",
    ]

    @pytest.mark.parametrize("file_path", MIGRATED_FILES)
    def test_no_forbidden_tokens(self, file_path: Path) -> None:
        text = _read_source(file_path)
        for token in self.FORBIDDEN_TOKENS:
            assert token not in text, (
                f"{file_path.name} must not reference {token}"
            )


# ======================================================================
# 9. No new Trader() / OkxPrivateClient() instantiation
# ======================================================================


class TestNoNewClientInstantiation_OLD:
    """Migrated files must not create new Trader or OkxPrivateClient.

    Note: TpSlExecutionManager is allowed to create OkxTradingClient —
    it is the canonical creation point that distributes the instance
    to sub-managers."""

    MIGRATED_FILES = [
        _MARKET_EXIT_PATH,
        _NEAR_TP_PATH,
        _SIDECAR_PATH,
    ]

    # TpSlExecutionManager is checked separately below
    EXECUTION_MGR_PATH = _EXECUTION_MGR_PATH

    @pytest.mark.parametrize("file_path", MIGRATED_FILES)
    def test_no_trader_instantiation(self, file_path: Path) -> None:
        text = _read_source(file_path)
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue
            if "= Trader(" in stripped:
                pytest.fail(f"{file_path.name}:{i} creates new Trader()")
            if "= OkxPrivateClient(" in stripped:
                pytest.fail(f"{file_path.name}:{i} creates new OkxPrivateClient()")

    @pytest.mark.parametrize("file_path", MIGRATED_FILES)
    def test_no_okx_trading_client_instantiation(self, file_path: Path) -> None:
        """Migrated sub-managers must NOT create OkxTradingClient directly —
        they receive it from TpSlExecutionManager."""
        text = _read_source(file_path)
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "OkxTradingClient(" in stripped:
                pytest.fail(
                    f"{file_path.name}:{i} must not instantiate OkxTradingClient — "
                    f"receive from TpSlExecutionManager"
                )

    def test_execution_manager_accepts_trading_client(self):
        """TpSlExecutionManager receives trading_client via injection, no longer creates it."""
        text = _read_source(_EXECUTION_MGR_PATH)
        assert "OkxTradingClient(trader)" not in text, (
            "TpSlExecutionManager must NOT create OkxTradingClient — it is injected"
        )
        assert "trading_client: TradingClientPort" in text, (
            "TpSlExecutionManager must accept trading_client: TradingClientPort"
        )


# ======================================================================
# 10. No env reads in new code paths
# ======================================================================


class TestNoNewEnvReads:
    """The trading_client wiring in __init__ must not read env vars."""

    MIGRATED_FILES = [_MARKET_EXIT_PATH, _NEAR_TP_PATH, _SIDECAR_PATH]

    @pytest.mark.parametrize("file_path", MIGRATED_FILES)
    def test_no_load_dotenv(self, file_path: Path) -> None:
        text = _read_source(file_path)
        assert "load_dotenv" not in text, (
            f"{file_path.name} must not call load_dotenv"
        )


# ======================================================================
# 11. Unmigrated methods still allowed to have direct request
# ======================================================================


class TestUnmigratedMethodsAllowedDirectRequest:
    """Methods that are NOT migrated may still use direct request."""

    def test_execute_market_exit_runner_still_delegates(self):
        """execute_market_exit_runner delegates to market_exit which is migrated,
        but the method itself is just a delegate — no direct request."""
        text = _read_source(_NEAR_TP_PATH)
        method_text = _extract_method(text, "execute_market_exit_runner")
        # This method calls market_exit_remaining_position_with_retries (delegated)
        # and should NOT contain direct request itself
        assert '"/api/v5/trade/order"' not in method_text, (
            "execute_market_exit_runner delegates to market_exit, no direct request needed"
        )

    def test_sidecar_tp_placement_uses_trading_client_port(self):
        """Sidecar TP placement now uses TradingClientPort.place_limit_order
        (migrated in 20C-CLEAN-PORTS-08)."""
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "place_sidecar_fixed_take_profit")
        assert ".place_limit_order(" in method_text, (
            "Sidecar fixed TP must use .place_limit_order( (migrated)"
        )


# ======================================================================
# 12. Compilation check
# ======================================================================


class TestFilesCompile:
    @pytest.mark.parametrize("file_path", [
        _MARKET_EXIT_PATH,
        _NEAR_TP_PATH,
        _EXECUTION_MGR_PATH,
        _SIDECAR_PATH,
    ])
    def test_file_compiles(self, file_path: Path) -> None:
        text = _read_source(file_path)
        compile(text, str(file_path), "exec")
