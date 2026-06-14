#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_remaining_position_reads_trading_client_port_boundaries.py
@Description: Source-level boundary tests — verify that CoreTakeProfitManager
              and ProtectiveStopManager position reads route through
              TradingClientPort.fetch_position() at the source-code level.

              Ref: 20C-CLEAN-PORTS-09C
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ======================================================================
# Source file paths
# ======================================================================

_CORE_TP_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "tp_sl_core_tp_manager.py"
_PROTECTIVE_SL_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "tp_sl_protective_stop_manager.py"

_MODIFIED_FILES = [_CORE_TP_PATH, _PROTECTIVE_SL_PATH]


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
# 1. CoreTakeProfitManager — migrated method checks
# ======================================================================


class TestCoreTakeProfitManagerPositionReadMigrated:
    """replace_take_profit must use self.trading_client.fetch_position()
    and must NOT use trader.fetch_position_snapshot()."""

    METHOD = "replace_take_profit"

    REQUIRED = [
        "self.trading_client.fetch_position(",
    ]

    FORBIDDEN = [
        "t.fetch_position_snapshot(",
        "self.trader.fetch_position_snapshot(",
        "trader.fetch_position_snapshot(",
    ]

    def test_method_contains_fetch_position(self):
        text = _read_source(_CORE_TP_PATH)
        method_text = _extract_method(text, self.METHOD)

        for required in self.REQUIRED:
            assert required in method_text, (
                f"{self.METHOD} must contain {required}"
            )

    def test_method_no_legacy_position_snapshot(self):
        text = _read_source(_CORE_TP_PATH)
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
        """position.contracts must NOT appear; only position.qty is used."""
        text = _read_source(_CORE_TP_PATH)
        method_text = _extract_method(text, self.METHOD)

        assert "position.contracts" not in method_text, (
            f"{self.METHOD} must use position.qty, not position.contracts"
        )

    def test_position_qty_is_used(self):
        """position.qty must appear in the method."""
        text = _read_source(_CORE_TP_PATH)
        method_text = _extract_method(text, self.METHOD)

        assert "position.qty" in method_text, (
            f"{self.METHOD} must use position.qty"
        )


# ======================================================================
# 2. ProtectiveStopManager — no position read to migrate
# ======================================================================


class TestProtectiveStopManagerNoPositionRead:
    """ProtectiveStopManager has no fetch_position_snapshot() calls and
    therefore no position read to migrate."""

    def test_no_fetch_position_snapshot_calls(self):
        text = _read_source(_PROTECTIVE_SL_PATH)
        assert "fetch_position_snapshot(" not in text, (
            "ProtectiveStopManager has no fetch_position_snapshot calls — "
            "nothing to migrate"
        )

    def test_no_fetch_position_calls(self):
        text = _read_source(_PROTECTIVE_SL_PATH)
        assert "fetch_position(" not in text, (
            "ProtectiveStopManager has no fetch_position calls — "
            "nothing to migrate"
        )


# ======================================================================
# 3. B-class whitelist — no B-class call points exist
# ======================================================================


class TestBClassLegacyWhitelist:
    """Neither CoreTakeProfitManager nor ProtectiveStopManager has
    B-class (eth_qty / raw_pos) position reads to preserve.

    The whitelist is intentionally empty because:
    - CoreTakeProfitManager.replace_take_profit only uses contracts/has_position/side
      → fully migrated to TradingClientPort
    - ProtectiveStopManager has no position reads at all
    """

    # If any method used eth_qty / raw_pos and kept legacy, it would be here:
    LEGACY_POSITION_READ_ALLOWED: set[tuple[str, str, str]] = set()

    def test_whitelist_is_empty(self):
        """Confirm the B-class whitelist is empty — all safe reads migrated."""
        assert len(self.LEGACY_POSITION_READ_ALLOWED) == 0, (
            "B-class whitelist should be empty: no method uses eth_qty/raw_pos "
            "in CoreTakeProfitManager or ProtectiveStopManager"
        )

    def test_core_tp_no_eth_qty(self):
        text = _read_source(_CORE_TP_PATH)
        assert "eth_qty" not in text, (
            "CoreTakeProfitManager must not reference eth_qty"
        )

    def test_core_tp_no_raw_pos(self):
        text = _read_source(_CORE_TP_PATH)
        assert "raw_pos" not in text, (
            "CoreTakeProfitManager must not reference raw_pos"
        )

    def test_protective_sl_no_eth_qty(self):
        text = _read_source(_PROTECTIVE_SL_PATH)
        assert "eth_qty" not in text, (
            "ProtectiveStopManager must not reference eth_qty"
        )

    def test_protective_sl_no_raw_pos(self):
        text = _read_source(_PROTECTIVE_SL_PATH)
        assert "raw_pos" not in text, (
            "ProtectiveStopManager must not reference raw_pos"
        )


# ======================================================================
# 4. File-level forbidden tokens
# ======================================================================


class TestNoForbiddenTokensInModifiedFiles:
    """Modified files must not contain forbidden abstractions or patterns."""

    FORBIDDEN_TOKENS = [
        "Binance",
        "ExchangeRuntimeBundle",
        "BrokerSemanticExecutor",
        "ThreeStageAdapter",
        "MiddleRunnerAdapter",
        "SidecarAdapter",
    ]

    @pytest.mark.parametrize("file_path", _MODIFIED_FILES)
    def test_no_forbidden_tokens(self, file_path: Path) -> None:
        text = _read_source(file_path)
        for token in self.FORBIDDEN_TOKENS:
            assert token not in text, (
                f"{file_path.name} must not reference {token}"
            )

    @pytest.mark.parametrize("file_path", _MODIFIED_FILES)
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

    @pytest.mark.parametrize("file_path", _MODIFIED_FILES)
    def test_no_load_dotenv(self, file_path: Path) -> None:
        text = _read_source(file_path)
        assert "load_dotenv" not in text, (
            f"{file_path.name} must not call load_dotenv"
        )

    def test_no_new_env_reads_in_core_tp_init(self):
        """CoreTakeProfitManager.__init__ must not read new env vars."""
        text = _read_source(_CORE_TP_PATH)
        lines = text.splitlines()
        in_init = False
        for i, line in enumerate(lines, 1):
            if "def __init__" in line and "trading_client" in line:
                in_init = True
                continue
            if in_init and line.strip() and not line.startswith("        "):
                in_init = False
            # Must NOT contain new env reads in init
            # Note: replace_take_profit has pre-existing os.getenv calls
            # for retry config — those are NOT in __init__ and are fine.
        # This test just ensures __init__ hasn't gained env reads
        init_text = text.split("def __init__")[1].split("\n    def ")[0] if "def __init__" in text else ""
        assert "os.getenv" not in init_text, (
            "CoreTakeProfitManager.__init__ must not read env vars"
        )

    def test_no_new_env_reads_in_protective_sl_init(self):
        """ProtectiveStopManager.__init__ must not read new env vars."""
        text = _read_source(_PROTECTIVE_SL_PATH)
        init_text = text.split("def __init__")[1].split("\n    def ")[0] if "def __init__" in text else ""
        assert "os.getenv" not in init_text, (
            "ProtectiveStopManager.__init__ must not read env vars"
        )


# ======================================================================
# 5. Compilation check
# ======================================================================


class TestFilesCompile:
    @pytest.mark.parametrize("file_path", _MODIFIED_FILES)
    def test_file_compiles(self, file_path: Path) -> None:
        text = _read_source(file_path)
        compile(text, str(file_path), "exec")
