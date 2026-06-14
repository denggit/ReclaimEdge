#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_run_boll_live_runtime_adapter_wiring.py
@Description: Tests verifying run_boll_cvd_live.py adapter wiring.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_BOLL_SOURCE = ROOT / "scripts" / "run_boll_cvd_live.py"

_SOURCE_TEXT = RUN_BOLL_SOURCE.read_text(encoding="utf-8")


class TestAdapterWiring:
    """run_boll_cvd_live.py uses the adapter runtime bundle."""

    def test_no_okx_legacy_path(self) -> None:
        """The OKX legacy path comment/pattern must be gone."""
        assert "# ── OKX legacy path continues below" not in _SOURCE_TEXT

    def test_uses_create_runtime_bundle(self) -> None:
        """The script must use create_runtime_bundle instead of manual setup."""
        assert "create_runtime_bundle(os.environ)" in _SOURCE_TEXT

    def test_no_okx_trading_client_direct_import(self) -> None:
        """OkxTradingClient must NOT be directly imported — comes from bundle."""
        assert "from src.execution.okx_trading_client import OkxTradingClient" not in _SOURCE_TEXT

    def test_no_okx_market_data_client_direct_import(self) -> None:
        """OkxMarketDataClient must NOT be directly imported — comes from bundle."""
        assert "from src.data_feed.okx_market_data_client import OkxMarketDataClient" not in _SOURCE_TEXT

    def test_bundle_ports_are_used(self) -> None:
        """The bundle's ports are used for the runtime ports log."""
        assert "bundle.trading_client" in _SOURCE_TEXT
        assert "bundle.market_data_client" in _SOURCE_TEXT

    def test_monitor_config_uses_runtime_config(self) -> None:
        """Monitor config uses bundle's runtime_config, not OKX env vars."""
        assert "rt_config.okx_inst_id" in _SOURCE_TEXT
        assert "rt_config.kline_interval" in _SOURCE_TEXT

    def test_no_BollBandBreakoutMonitorConfig_from_env(self) -> None:
        """Must NOT call from_env() on the monitor config."""
        assert "BollBandBreakoutMonitorConfig.from_env()" not in _SOURCE_TEXT
