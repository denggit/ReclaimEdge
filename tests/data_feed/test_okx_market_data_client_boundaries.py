#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_okx_market_data_client_boundaries.py
@Description: Boundary tests for OkxMarketDataClient — the source must NOT
              contain forbidden imports, patterns, or references.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PATH = Path(__file__).resolve().parents[2] / "src" / "data_feed" / "okx_market_data_client.py"


def _read_source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


# ======================================================================
# File existence / compilation
# ======================================================================


def test_file_exists() -> None:
    assert _SOURCE_PATH.exists(), f"OkxMarketDataClient file not found at {_SOURCE_PATH}"
    assert _SOURCE_PATH.is_file()


def test_file_compiles() -> None:
    text = _read_source()
    compile(text, str(_SOURCE_PATH), "exec")


# ======================================================================
# Import-ability
# ======================================================================


def test_can_be_imported() -> None:
    from src.data_feed.okx_market_data_client import OkxMarketDataClient  # noqa: F401


# ======================================================================
# Implements MarketDataClientPort
# ======================================================================


def test_has_all_port_methods() -> None:
    from src.data_feed.okx_market_data_client import OkxMarketDataClient

    required = {"fetch_recent_klines", "stream_market_events", "close"}
    actual = {
        name
        for name in dir(OkxMarketDataClient)
        if not name.startswith("_") and callable(getattr(OkxMarketDataClient, name, None))
    }
    missing = required - actual
    assert not missing, f"OkxMarketDataClient is missing methods: {missing}"


# ======================================================================
# Forbidden tokens — no Binance references
# ======================================================================


class TestNoBinanceReferences:
    def test_no_binance_word(self) -> None:
        text = _read_source()
        assert "binance" not in text
        assert "Binance" not in text

    def test_no_ethusdt_symbol(self) -> None:
        text = _read_source()
        assert "ETHUSDT" not in text

    def test_no_fapi(self) -> None:
        text = _read_source()
        assert "/fapi" not in text


# ======================================================================
# Forbidden imports
# ======================================================================


class TestNoForbiddenImports:
    def test_no_binance_import(self) -> None:
        text = _read_source()
        assert "src.exchanges.binance" not in text
        assert "src.data_feed.binance" not in text

    def test_no_scripts_import(self) -> None:
        text = _read_source()
        assert "scripts." not in text

    def test_no_env_import(self) -> None:
        text = _read_source()
        assert "os.getenv" not in text
        assert "load_dotenv" not in text

    def test_no_aiohttp_import(self) -> None:
        text = _read_source()
        assert "import aiohttp" not in text

    def test_no_websockets_import(self) -> None:
        text = _read_source()
        assert "import websockets" not in text

    def test_no_execution_import(self) -> None:
        text = _read_source()
        assert "src.execution" not in text

    def test_no_strategies_import(self) -> None:
        text = _read_source()
        assert "src.strategies" not in text

    def test_no_risk_import(self) -> None:
        text = _read_source()
        assert "src.risk" not in text

    def test_no_reporting_import(self) -> None:
        text = _read_source()
        assert "src.reporting" not in text


# ======================================================================
# Forbidden patterns — no monitor construction
# ======================================================================


class TestNoMonitorConstruction:
    def test_no_monitor_instantiation(self) -> None:
        """The source must NOT create a BollBandBreakoutMonitor — only accept one via __init__."""
        text = _read_source()
        assert "BollBandBreakoutMonitor(" not in text

    def test_no_config_from_env(self) -> None:
        text = _read_source()
        assert "BollBandBreakoutMonitorConfig" not in text
        assert ".from_env()" not in text

    def test_monitor_import_only_in_type_checking(self) -> None:
        """BollBandBreakoutMonitor must only be imported under TYPE_CHECKING."""
        text = _read_source()
        lines = text.split("\n")
        in_type_checking = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("if TYPE_CHECKING:"):
                in_type_checking = True
                continue
            if in_type_checking:
                if stripped == "" or stripped.startswith("#"):
                    continue
                # After TYPE_CHECKING block ends (unindented non-empty), we're out
                if stripped and not stripped.startswith(" ") and not stripped.startswith("\t"):
                    in_type_checking = False
                    continue
            if not in_type_checking:
                if "BollBandBreakoutMonitor" in stripped and ("import" in stripped or "from" in stripped):
                    raise AssertionError(
                        f"Runtime import of BollBandBreakoutMonitor found: {stripped!r}"
                    )


# ======================================================================
# Forbidden patterns — no adapters
# ======================================================================


class TestNoAdapterPatterns:
    def test_no_three_stage_adapter(self) -> None:
        text = _read_source()
        assert "ThreeStageAdapter" not in text

    def test_no_middle_runner_adapter(self) -> None:
        text = _read_source()
        assert "MiddleRunnerAdapter" not in text

    def test_no_sidecar_adapter(self) -> None:
        text = _read_source()
        assert "SidecarAdapter" not in text

    def test_no_near_tp_adapter(self) -> None:
        text = _read_source()
        assert "NearTpAdapter" not in text

    def test_no_exchange_runtime_bundle(self) -> None:
        text = _read_source()
        assert "ExchangeRuntimeBundle" not in text

    def test_no_broker_semantic_executor(self) -> None:
        text = _read_source()
        assert "BrokerSemanticExecutor" not in text
