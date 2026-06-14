#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_run_boll_cvd_live_okx_ports_wiring.py
@Description: Tests verifying that run_boll_cvd_live.py instantiates
              OkxTradingClient / OkxMarketDataClient ports in the OKX
              legacy path without calling any port methods.

All tests are source-level — no network, no API keys, no live runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_BOLL_SOURCE = ROOT / "scripts" / "run_boll_cvd_live.py"

_SOURCE_TEXT = RUN_BOLL_SOURCE.read_text(encoding="utf-8")


# ======================================================================
# Helpers
# ======================================================================


def _okx_path_lines() -> list[str]:
    """Return source lines from the OKX legacy path marker onwards."""
    lines = _SOURCE_TEXT.split("\n")
    for i, line in enumerate(lines):
        if "# ── OKX legacy path continues below" in line:
            return lines[i:]
    return []


# ======================================================================
# Import tests
# ======================================================================


class TestOkxPortsImports:
    """Verify that Okx client classes are imported."""

    def test_imports_okx_trading_client(self) -> None:
        assert "from src.execution.okx_trading_client import OkxTradingClient" in _SOURCE_TEXT

    def test_imports_okx_market_data_client(self) -> None:
        assert "from src.data_feed.okx_market_data_client import OkxMarketDataClient" in _SOURCE_TEXT


# ======================================================================
# Instantiation tests
# ======================================================================


class TestOkxPortsInstantiation:
    """Verify that Okx ports are instantiated with existing objects."""

    def test_creates_okx_trading_client_with_trader(self) -> None:
        assert "OkxTradingClient(trader)" in _SOURCE_TEXT

    def test_creates_okx_market_data_client_with_monitor(self) -> None:
        assert "OkxMarketDataClient(monitor)" in _SOURCE_TEXT

    def test_ports_ready_log_message(self) -> None:
        assert "OKX_RUNTIME_PORTS_READY" in _SOURCE_TEXT


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
# Binance blocked branch safety
# ======================================================================


class TestBinanceBlockedBranchNoPorts:
    """Verify that ports are NOT created in the Binance blocked branch."""

    def test_okx_ports_not_in_source_before_okx_path_comment(self) -> None:
        """Ports creation must appear after the OKX legacy path comment."""
        okx_lines = _okx_path_lines()
        okx_text = "\n".join(okx_lines)
        assert "OkxTradingClient(trader)" in okx_text
        assert "OkxMarketDataClient(monitor)" in okx_text

    def test_binance_blocked_branch_does_not_instantiate_ports(self) -> None:
        """The Binance blocked branch (before OKX path comment) must not contain port instantiation."""
        lines = _SOURCE_TEXT.split("\n")
        okx_path_start: int | None = None
        for i, line in enumerate(lines):
            if "# ── OKX legacy path continues below" in line:
                okx_path_start = i
                break

        assert okx_path_start is not None, "OKX legacy path comment not found"

        pre_okx_text = "\n".join(lines[:okx_path_start])
        # Module-level imports are fine — they don't create objects.
        # Only instantiation must be after the OKX path comment.
        assert "OkxTradingClient(trader)" not in pre_okx_text
        assert "OkxMarketDataClient(monitor)" not in pre_okx_text
