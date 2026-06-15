#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_binance_trading_client_boundaries.py
@Description: Source-level boundary scan — BinanceTradingClient and
              BinancePrivateClient must NOT leak into strategies, monitors,
              live runtime, trader, or scripts.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_PATHS: list[Path] = [
    _PROJECT_ROOT / "src" / "strategies",
    _PROJECT_ROOT / "src" / "monitors",
    _PROJECT_ROOT / "src" / "live",
    _PROJECT_ROOT / "src" / "execution" / "trader.py",
    _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py",
]

_FORBIDDEN_TOKENS = [
    "BinanceTradingClient",
    "src.exchanges.binance.trading_client",
    "src.exchanges.binance.private_client",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoBinanceTradingClientLeakage:
    """BinanceTradingClient must not appear in strategies, monitors, live, trader, or scripts."""

    @pytest.mark.parametrize("directory", [
        d for d in _FORBIDDEN_PATHS if d.is_dir()
    ])
    def test_no_binance_trading_client_in_directory(self, directory: Path) -> None:
        for py_file in directory.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            for token in _FORBIDDEN_TOKENS:
                assert token not in text, (
                    f"Forbidden token {token!r} found in {py_file}"
                )

    @pytest.mark.parametrize("file_path", [
        p for p in _FORBIDDEN_PATHS if p.is_file()
    ])
    def test_no_binance_trading_client_in_file(self, file_path: Path) -> None:
        text = file_path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_TOKENS:
            assert token not in text, (
                f"Forbidden token {token!r} found in {file_path}"
            )


class TestTradingClientPortRemainsGeneric:
    """TradingClientPort itself must not reference Binance."""

    def test_port_no_binance_reference(self) -> None:
        port_path = _PROJECT_ROOT / "src" / "execution" / "trading_client_port.py"
        text = port_path.read_text(encoding="utf-8")
        assert "Binance" not in text
        assert "binance" not in text
        assert "fapi" not in text


class TestNewFilesCompile:
    """Quick compilation check for all new source files."""

    @pytest.mark.parametrize("file_path", [
        "src/exchanges/binance/private_client.py",
        "src/exchanges/binance/trading_mappers.py",
        "src/exchanges/binance/trading_client.py",
        "src/exchanges/binance/credentials.py",
    ])
    def test_file_compiles(self, file_path: str) -> None:
        path = _PROJECT_ROOT / file_path
        text = path.read_text(encoding="utf-8")
        compile(text, str(path), "exec")


class TestNoLiveImportsInNewFiles:
    """New Binance files must not import src.live or src.strategies."""

    _NEW_FILES = [
        "src/exchanges/binance/private_client.py",
        "src/exchanges/binance/trading_mappers.py",
        "src/exchanges/binance/trading_client.py",
        "src/exchanges/binance/credentials.py",
    ]

    @pytest.mark.parametrize("rel_path", _NEW_FILES)
    def test_no_live_import(self, rel_path: str) -> None:
        text = (_PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
        assert "src.live" not in text
        assert "src.strategies" not in text
        assert "src.monitors" not in text
        assert "src.execution.trader" not in text

    @pytest.mark.parametrize("rel_path", _NEW_FILES)
    def test_no_scripts_import(self, rel_path: str) -> None:
        text = (_PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
        assert "scripts." not in text


class TestSigningFileBoundaryIntact:
    """signing.py boundary tests should still pass with new constants."""

    def test_signing_file_compiles(self) -> None:
        path = _PROJECT_ROOT / "src" / "exchanges" / "binance" / "signing.py"
        text = path.read_text(encoding="utf-8")
        compile(text, str(path), "exec")
