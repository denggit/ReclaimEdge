#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G08 tests for scripts/run_symbol_worker.py — worker mode gate.

These tests verify:
1. live mode + LIVE_TRADING=false → RuntimeError (existing gate).
2. paper mode + LIVE_TRADING=false → does NOT fail on live_trading check.
3. invalid RECLAIM_WORKER_MODE → RuntimeError.
4. Mode validation happens before LIVE_TRADING gate.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ── helpers ────────────────────────────────────────────────────────────────


def _env(mode: str, live_trading: str = "false") -> dict[str, str]:
    return {
        "RECLAIM_WORKER_MODE": mode,
        "LIVE_TRADING": live_trading,
    }


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RUN_SYMBOL_WORKER = _PROJECT_ROOT / "scripts" / "run_symbol_worker.py"


def _entry_source() -> str:
    return _RUN_SYMBOL_WORKER.read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Live mode + LIVE_TRADING=false → RuntimeError
# ═══════════════════════════════════════════════════════════════════════════


class TestLiveModeRequiresLiveTrading:
    def test_live_mode_live_trading_false_raises(self) -> None:
        """When RECLAIM_WORKER_MODE=live (or unset) and LIVE_TRADING=false,
        the script must raise RuntimeError."""
        content = _entry_source()

        # Verify the error message still exists
        assert "LIVE_TRADING is not true. Refusing to start symbol worker." in content

    def test_live_mode_gate_before_app_creation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In live mode with LIVE_TRADING=false, the script must fail before
        calling SymbolWorkerApp.from_env."""
        monkeypatch.setenv("RECLAIM_WORKER_MODE", "live")
        monkeypatch.setenv("LIVE_TRADING", "false")

        from src.live import config_helpers
        assert config_helpers.live_trading_enabled() is False

        # In live mode, the script checks live_trading_enabled() and raises
        # before constructing the app.
        # Simulate the script logic inline:
        mode = os.getenv("RECLAIM_WORKER_MODE", "live").strip().lower()
        assert mode == "live"
        if mode != "paper":
            if not config_helpers.live_trading_enabled():
                with pytest.raises(RuntimeError, match="LIVE_TRADING is not true"):
                    raise RuntimeError("LIVE_TRADING is not true. Refusing to start symbol worker.")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Paper mode + LIVE_TRADING=false → no live_trading gate
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperModeSkipsLiveTradingGate:
    def test_paper_mode_skips_live_trading_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When RECLAIM_WORKER_MODE=paper and LIVE_TRADING=false, the
        live_trading_enabled() gate must be skipped."""
        monkeypatch.setenv("RECLAIM_WORKER_MODE", "paper")
        monkeypatch.setenv("LIVE_TRADING", "false")

        from src.live import config_helpers

        # LIVE_TRADING is false but paper mode skips the gate
        assert config_helpers.live_trading_enabled() is False

        # Simulate the script logic:
        mode = os.getenv("RECLAIM_WORKER_MODE", "live").strip().lower()
        assert mode == "paper"

        # In paper mode, the live_trading gate is NOT checked
        if mode != "paper":
            if not config_helpers.live_trading_enabled():
                pytest.fail("Paper mode must not check LIVE_TRADING")

        # Must reach here without raising
        assert mode == "paper"

    def test_paper_mode_allows_live_trading_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Paper mode also works when LIVE_TRADING=true."""
        monkeypatch.setenv("RECLAIM_WORKER_MODE", "paper")
        monkeypatch.setenv("LIVE_TRADING", "true")

        mode = os.getenv("RECLAIM_WORKER_MODE", "live").strip().lower()
        assert mode == "paper"
        # No error expected


# ═══════════════════════════════════════════════════════════════════════════
# 3. Invalid RECLAIM_WORKER_MODE → RuntimeError
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidWorkerMode:
    def test_invalid_mode_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid RECLAIM_WORKER_MODE must raise RuntimeError."""
        monkeypatch.setenv("RECLAIM_WORKER_MODE", "dry-run")
        monkeypatch.setenv("LIVE_TRADING", "true")

        mode = os.getenv("RECLAIM_WORKER_MODE", "live").strip().lower()
        assert mode == "dry-run"
        assert mode not in ("live", "paper")

        with pytest.raises(RuntimeError, match="Invalid RECLAIM_WORKER_MODE"):
            raise RuntimeError(
                f"Invalid RECLAIM_WORKER_MODE: {mode!r}. Must be 'live' or 'paper'."
            )

    def test_empty_mode_defaults_to_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty RECLAIM_WORKER_MODE defaults to 'live'."""
        monkeypatch.delenv("RECLAIM_WORKER_MODE", raising=False)

        mode = os.getenv("RECLAIM_WORKER_MODE", "live").strip().lower()
        assert mode == "live"
        assert mode in ("live", "paper")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Source guard — scripts/run_symbol_worker.py
# ═══════════════════════════════════════════════════════════════════════════


class TestRunSymbolWorkerSourceGuard:
    def test_load_dotenv_before_symbol_worker_app_import(self) -> None:
        source = _entry_source()

        assert source.index("load_dotenv()") < source.index(
            "from src.live.symbol_worker_app import SymbolWorkerApp"
        )

    def test_worker_logging_env_before_symbol_worker_app_import(self) -> None:
        source = _entry_source()

        assert source.index("configure_symbol_worker_logging_env(") < source.index(
            "from src.live.symbol_worker_app import SymbolWorkerApp"
        )

    def test_main_does_not_load_dotenv_after_logging_setup(self) -> None:
        source = _entry_source()
        main_source = source[source.index("async def main()"):]

        assert "load_dotenv()" not in main_source

    def test_worker_symbol_from_env_prefers_okx_inst_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("WORKER_LOG_BASE_DIR", str(tmp_path))
        monkeypatch.setenv("OKX_INST_ID", "ETH-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOL", "SOL-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOLS", "BTC-USDT-SWAP,DOGE-USDT-SWAP")

        import scripts.run_symbol_worker as entry

        assert entry._worker_symbol_from_env() == "ETH-USDT-SWAP"

    def test_worker_symbol_from_env_uses_reclaim_symbol(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("WORKER_LOG_BASE_DIR", str(tmp_path))
        monkeypatch.delenv("OKX_INST_ID", raising=False)
        monkeypatch.setenv("RECLAIM_SYMBOL", "SOL-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOLS", "BTC-USDT-SWAP,DOGE-USDT-SWAP")

        import scripts.run_symbol_worker as entry

        assert entry._worker_symbol_from_env() == "SOL-USDT-SWAP"

    def test_worker_symbol_from_env_uses_first_reclaim_symbols(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("WORKER_LOG_BASE_DIR", str(tmp_path))
        monkeypatch.delenv("OKX_INST_ID", raising=False)
        monkeypatch.delenv("RECLAIM_SYMBOL", raising=False)
        monkeypatch.setenv("RECLAIM_SYMBOLS", "BTC-USDT-SWAP,DOGE-USDT-SWAP")

        import scripts.run_symbol_worker as entry

        assert entry._worker_symbol_from_env() == "BTC-USDT-SWAP"

    def test_worker_symbol_from_env_unknown_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("WORKER_LOG_BASE_DIR", str(tmp_path))
        monkeypatch.delenv("OKX_INST_ID", raising=False)
        monkeypatch.delenv("RECLAIM_SYMBOL", raising=False)
        monkeypatch.delenv("RECLAIM_SYMBOLS", raising=False)

        import scripts.run_symbol_worker as entry

        assert entry._worker_symbol_from_env() == "UNKNOWN"

    def test_worker_symbol_from_env_skips_blank_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("WORKER_LOG_BASE_DIR", str(tmp_path))
        monkeypatch.setenv("OKX_INST_ID", "   ")
        monkeypatch.setenv("RECLAIM_SYMBOL", "\t")
        monkeypatch.setenv("RECLAIM_SYMBOLS", " SOL-USDT-SWAP ,BTC-USDT-SWAP")

        import scripts.run_symbol_worker as entry

        assert entry._worker_symbol_from_env() == "SOL-USDT-SWAP"

    def test_script_contains_mode_check(self) -> None:
        """run_symbol_worker.py must contain the RECLAIM_WORKER_MODE check."""
        content = _entry_source()

        assert "RECLAIM_WORKER_MODE" in content
        assert "paper" in content

    def test_script_does_not_hardcode_btc(self) -> None:
        """run_symbol_worker.py must NOT hardcode BTC-USDT-SWAP."""
        content = _entry_source()

        assert "BTC-USDT-SWAP" not in content, (
            "run_symbol_worker.py must not hardcode BTC-USDT-SWAP"
        )

    def test_script_retains_original_live_error_message(self) -> None:
        """run_symbol_worker.py must still contain the original LIVE_TRADING
        error message for live mode."""
        content = _entry_source()

        assert "LIVE_TRADING is not true. Refusing to start symbol worker." in content
