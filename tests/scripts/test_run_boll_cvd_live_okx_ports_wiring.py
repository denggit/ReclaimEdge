#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_run_boll_cvd_live_okx_ports_wiring.py
@Description: Tests verifying that run_boll_cvd_live.py uses
              create_runtime_bundle for adapter wiring.

All tests are source-level — no network, no API keys, no live runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_BOLL_SOURCE = ROOT / "scripts" / "run_boll_cvd_live.py"

_SOURCE_TEXT = RUN_BOLL_SOURCE.read_text(encoding="utf-8")


# ======================================================================
# Import tests
# ======================================================================


class TestRuntimeBundleImports:
    """Verify that create_runtime_bundle is imported."""

    def test_imports_create_runtime_bundle(self) -> None:
        assert "from src.live.runtime_factory import create_runtime_bundle" in _SOURCE_TEXT

    def test_no_direct_okx_trading_client_import(self) -> None:
        """OkxTradingClient should NOT be directly imported — it comes from the bundle."""
        assert "from src.execution.okx_trading_client import OkxTradingClient" not in _SOURCE_TEXT

    def test_no_direct_okx_market_data_client_import(self) -> None:
        """OkxMarketDataClient should NOT be directly imported — it comes from the bundle."""
        assert "from src.data_feed.okx_market_data_client import OkxMarketDataClient" not in _SOURCE_TEXT

    def test_no_legacy_live_trader_factory(self) -> None:
        """Legacy live_trader_factory is deleted — only create_runtime_bundle is supported."""
        assert "live_trader_factory" not in _SOURCE_TEXT
        assert "create_live_trader" not in _SOURCE_TEXT


# ======================================================================
# Bundle usage tests
# ======================================================================


class TestBundleUsage:
    """Verify that the runtime bundle is used correctly."""

    def test_calls_create_runtime_bundle(self) -> None:
        assert "create_runtime_bundle(os.environ)" in _SOURCE_TEXT

    def test_uses_bundle_trader(self) -> None:
        assert "bundle.trader" in _SOURCE_TEXT

    def test_uses_bundle_trading_client(self) -> None:
        assert "bundle.trading_client" in _SOURCE_TEXT

    def test_uses_bundle_market_data_client(self) -> None:
        assert "bundle.market_data_client" in _SOURCE_TEXT

    def test_uses_bundle_runtime_config(self) -> None:
        assert "bundle.runtime_config" in _SOURCE_TEXT

    def test_ports_ready_log_message(self) -> None:
        assert "OKX_RUNTIME_PORTS_READY" in _SOURCE_TEXT


# ======================================================================
# No legacy OKX env reads for monitor config
# ======================================================================


class TestNoLegacyOkxEnvForMonitor:
    """Monitor config must NOT use OKX_INST_ID / OKX_BAR from env."""

    def test_no_okx_inst_id_in_monitor_config(self) -> None:
        """monitor_config should use rt_config.okx_inst_id, not OKX_INST_ID env."""
        # Check that BollBandBreakoutMonitorConfig is NOT created via from_env()
        assert "BollBandBreakoutMonitorConfig.from_env()" not in _SOURCE_TEXT

    def test_monitor_config_uses_runtime_config(self) -> None:
        """monitor_config inst_id should come from rt_config."""
        assert "rt_config.okx_inst_id" in _SOURCE_TEXT

    def test_monitor_config_bar_uses_runtime_config(self) -> None:
        """monitor_config bar should come from rt_config."""
        assert "rt_config.kline_interval" in _SOURCE_TEXT


# ======================================================================
# No port method calls
# ======================================================================


class TestNoPortMethodCalls:
    """Verify that NO port methods are called on the new objects."""

    def test_no_fetch_balance(self) -> None:
        assert "trading_client_port.fetch_balance(" not in _SOURCE_TEXT

    def test_no_fetch_position(self) -> None:
        assert "trading_client_port.fetch_position(" not in _SOURCE_TEXT

    def test_no_place_market_order(self) -> None:
        assert "trading_client_port.place_market_order(" not in _SOURCE_TEXT

    def test_no_place_limit_order(self) -> None:
        assert "trading_client_port.place_limit_order(" not in _SOURCE_TEXT

    def test_no_place_stop_market_order(self) -> None:
        assert "trading_client_port.place_stop_market_order(" not in _SOURCE_TEXT

    def test_no_cancel_order(self) -> None:
        assert "trading_client_port.cancel_order(" not in _SOURCE_TEXT

    def test_no_fetch_recent_klines(self) -> None:
        assert "market_data_client_port.fetch_recent_klines(" not in _SOURCE_TEXT

    def test_no_stream_market_events(self) -> None:
        assert "market_data_client_port.stream_market_events(" not in _SOURCE_TEXT


# ======================================================================
# No forbidden abstractions
# ======================================================================


class TestNoForbiddenAbstractions:
    """Verify that no premature abstractions leak into the source."""

    def test_no_exchange_runtime_bundle(self) -> None:
        assert "ExchangeRuntimeBundle" not in _SOURCE_TEXT

    def test_no_broker_semantic_executor(self) -> None:
        assert "BrokerSemanticExecutor" not in _SOURCE_TEXT

    def test_no_three_stage_adapter(self) -> None:
        assert "ThreeStageAdapter" not in _SOURCE_TEXT

    def test_no_sidecar_adapter(self) -> None:
        assert "SidecarAdapter" not in _SOURCE_TEXT

    def test_no_middle_runner_adapter(self) -> None:
        assert "MiddleRunnerAdapter" not in _SOURCE_TEXT


# ======================================================================
# No OKX legacy path
# ======================================================================


class TestNoOkxLegacyPath:
    """Verify the old OKX legacy path comment is gone."""

    def test_no_okx_legacy_path_comment(self) -> None:
        assert "# ── OKX legacy path continues below" not in _SOURCE_TEXT

    def test_no_okx_legacy_path(self) -> None:
        assert "# ── OKX legacy path" not in _SOURCE_TEXT

    def test_no_direct_trader_instantiation_in_okx_path(self) -> None:
        """Trader() should come from the bundle, not instantiated directly."""
        # The only Trader() instantiation happens inside create_runtime_bundle
        assert "from src.execution.trader import Trader" in _SOURCE_TEXT
        assert "Trader()" not in _SOURCE_TEXT
