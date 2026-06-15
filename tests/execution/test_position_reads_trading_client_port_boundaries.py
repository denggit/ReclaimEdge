#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_position_reads_trading_client_port_boundaries.py
@Description: Boundary tests — verify that MarketExitManager and
              NearTpExecutionManager position reads route through
              TradingClientPort.fetch_position() at the source-code level.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ======================================================================
# Source file paths
# ======================================================================

_MARKET_EXIT_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "tp_sl_market_exit_manager.py"


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_method(source: str, method_name: str) -> str:
    """Extract a single async def / def method body from source text."""
    for prefix in (f"async def {method_name}", f"def {method_name}"):
        idx = source.find(prefix)
        if idx != -1:
            break
    else:
        raise AssertionError(f"Method {method_name!r} not found in source")
    remaining = source[idx:]

    # Split at the next class-level or top-level function definition
    for delim in ("\n    async def ", "\n    def "):
        parts = remaining.split(delim, 1)
        if len(parts) > 1:
            return parts[0]
    return remaining


# ======================================================================
# 1. MarketExitManager — migrated method checks
# ======================================================================


class TestMarketExitManagerPositionReadMigrated:
    """market_exit_remaining_position_with_retries must use
    self.trading_client.fetch_position() and must NOT use
    trader.fetch_position_snapshot()."""

    METHOD = "market_exit_remaining_position_with_retries"

    REQUIRED = [
        "self.trading_client.fetch_position(",
    ]

    FORBIDDEN = [
        "t.fetch_position_snapshot(",
        "self.trader.fetch_position_snapshot(",
        "trader.fetch_position_snapshot(",
    ]

    def test_method_contains_fetch_position(self):
        text = _read_source(_MARKET_EXIT_PATH)
        method_text = _extract_method(text, self.METHOD)

        for required in self.REQUIRED:
            assert required in method_text, (
                f"{self.METHOD} must contain {required}"
            )

    def test_method_no_legacy_position_snapshot(self):
        text = _read_source(_MARKET_EXIT_PATH)
        method_text = _extract_method(text, self.METHOD)

        for forbidden in self.FORBIDDEN:
            lines = method_text.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if forbidden in line:
                    pytest.fail(
                        f"{self.METHOD}:{i} must not contain {forbidden}"
                    )

    def test_field_mapping_contracts_to_qty(self):
        """position.contracts and refreshed.contracts must NOT appear;
        only position.qty and refreshed.qty are used."""
        text = _read_source(_MARKET_EXIT_PATH)
        method_text = _extract_method(text, self.METHOD)

        # position.contracts should not appear
        assert "position.contracts" not in method_text, (
            f"{self.METHOD} must use position.qty, not position.contracts"
        )
        # refreshed.contracts should not appear
        assert "refreshed.contracts" not in method_text, (
            f"{self.METHOD} must use refreshed.qty, not refreshed.contracts"
        )


# ======================================================================
# 3. File-level forbidden tokens
# ======================================================================


class TestNoForbiddenTokensInMigratedFiles:
    """Migrated files must not contain forbidden abstractions or patterns."""

    MIGRATED_FILES = [_MARKET_EXIT_PATH]

    FORBIDDEN_TOKENS = [
        "Binance",
        "ExchangeRuntimeBundle",
        "BrokerSemanticExecutor",
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

    @pytest.mark.parametrize("file_path", MIGRATED_FILES)
    def test_no_new_trader_instantiation(self, file_path: Path) -> None:
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
    def test_no_load_dotenv(self, file_path: Path) -> None:
        text = _read_source(file_path)
        assert "load_dotenv" not in text, (
            f"{file_path.name} must not call load_dotenv"
        )


# ======================================================================
# 4. Compilation check
# ======================================================================


class TestFilesCompile:
    @pytest.mark.parametrize("file_path", [_MARKET_EXIT_PATH])
    def test_file_compiles(self, file_path: Path) -> None:
        text = _read_source(file_path)
        compile(text, str(file_path), "exec")
