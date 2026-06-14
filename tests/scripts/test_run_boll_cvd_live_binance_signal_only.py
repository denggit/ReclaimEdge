#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_run_boll_cvd_live_binance_signal_only.py
@Description: Tests for the Binance signal-only branch in run_boll_cvd_live.py.

All tests monkeypatch the runtime entry point — no network, no API keys.
"""

from __future__ import annotations

import asyncio
import os
from unittest import mock

import pytest


# ======================================================================
# Helpers
# ======================================================================


def _base_binance_env() -> dict[str, str]:
    return {
        "EXCHANGE": "binance",
        "TRADE_ASSET": "ETH",
        "QUOTE_ASSET": "USDT",
        "MARKET_TYPE": "PERPETUAL",
        "KLINE_INTERVAL": "15m",
        "SIGNAL_ONLY": "true",
    }


# ======================================================================
# Binance signal-only branch tests
# ======================================================================


class TestBinanceSignalOnlyBranch:
    """Tests for the Binance signal-only branch in main()."""

    def test_binance_signal_only_true_calls_runtime(self) -> None:
        """EXCHANGE=binance + SIGNAL_ONLY=true calls run_binance_signal_only."""
        env = _base_binance_env()

        called_with = {}

        async def fake_run_binance_signal_only(env_arg=None):
            called_with["called"] = True
            called_with["env"] = env_arg

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            # Patch the import target BEFORE importing main
            with mock.patch(
                "src.live.binance_signal_only_runtime.run_binance_signal_only",
                fake_run_binance_signal_only,
            ):
                from scripts.run_boll_cvd_live import main
                asyncio.run(main())

        assert called_with.get("called") is True

    def test_binance_signal_only_false_calls_factory(self) -> None:
        """EXCHANGE=binance + SIGNAL_ONLY=false now enters main path via factory."""
        env = {**_base_binance_env(), "SIGNAL_ONLY": "false", "LIVE_TRADING": "true"}

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False), \
             mock.patch("scripts.run_boll_cvd_live.EmailSender"), \
             mock.patch("scripts.run_boll_cvd_live.LiveTradeJournal"), \
             mock.patch("scripts.run_boll_cvd_live.RollingLossGuard"), \
             mock.patch("scripts.run_boll_cvd_live.LiveStateStore"), \
             mock.patch("scripts.run_boll_cvd_live.DailyTradeReporter"):
            with mock.patch(
                "scripts.run_boll_cvd_live.create_live_trader"
            ) as mock_factory:
                mock_factory.side_effect = RuntimeError("STOP_AFTER_FACTORY")
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError, match="STOP_AFTER_FACTORY"):
                    asyncio.run(main())
                mock_factory.assert_called_once()

    def test_binance_signal_only_missing_enters_main_path(self) -> None:
        """EXCHANGE=binance without SIGNAL_ONLY enters main path via factory."""
        env = _base_binance_env()
        del env["SIGNAL_ONLY"]
        env["LIVE_TRADING"] = "true"

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False), \
             mock.patch("scripts.run_boll_cvd_live.EmailSender"), \
             mock.patch("scripts.run_boll_cvd_live.LiveTradeJournal"), \
             mock.patch("scripts.run_boll_cvd_live.RollingLossGuard"), \
             mock.patch("scripts.run_boll_cvd_live.LiveStateStore"), \
             mock.patch("scripts.run_boll_cvd_live.DailyTradeReporter"):
            with mock.patch(
                "scripts.run_boll_cvd_live.create_live_trader"
            ) as mock_factory:
                mock_factory.side_effect = RuntimeError("STOP_AFTER_FACTORY")
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError, match="STOP_AFTER_FACTORY"):
                    asyncio.run(main())
                mock_factory.assert_called_once()

    def test_binance_signal_only_no_trader_instantiation(self) -> None:
        """Binance signal-only path must NOT instantiate Trader."""
        env = _base_binance_env()

        async def fake_run_binance_signal_only(env_arg=None):
            return

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "src.live.binance_signal_only_runtime.run_binance_signal_only",
                fake_run_binance_signal_only,
            ):
                # The main import chain would import Trader at module level,
                # but the Trader() call must not happen in the Binance branch.
                # We verify by mocking Trader and checking it's not called.
                with mock.patch("scripts.run_boll_cvd_live.Trader") as mock_trader:
                    from scripts.run_boll_cvd_live import main
                    asyncio.run(main())
                    # Trader should NOT have been instantiated
                    mock_trader.assert_not_called()

    def test_binance_signal_only_no_execution_worker(self) -> None:
        """Binance signal-only path must NOT call execution_worker."""
        env = _base_binance_env()

        async def fake_run_binance_signal_only(env_arg=None):
            return

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "src.live.binance_signal_only_runtime.run_binance_signal_only",
                fake_run_binance_signal_only,
            ):
                with mock.patch(
                    "scripts.run_boll_cvd_live.execution_worker_module"
                ) as mock_ew:
                    from scripts.run_boll_cvd_live import main
                    asyncio.run(main())
                    # execution_worker must NOT have been called
                    mock_ew.execution_worker.assert_not_called()


# ======================================================================
# OKX default path unchanged
# ======================================================================


class TestOkxDefaultPathUnchanged:
    """Verify OKX default path behavior is unchanged."""

    def test_okx_default_requires_live_trading(self) -> None:
        """With default env (no EXCHANGE set), LIVE_TRADING guard still applies."""
        env = {
            "EXCHANGE": "okx",
            "LIVE_TRADING": "false",
        }
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            from scripts.run_boll_cvd_live import main
            with pytest.raises(RuntimeError, match="LIVE_TRADING is not true"):
                asyncio.run(main())

    def test_okx_exchange_still_uses_live_trading_guard(self) -> None:
        """EXCHANGE=okx still requires LIVE_TRADING=true."""
        env = {
            "EXCHANGE": "okx",
            "LIVE_TRADING": "false",
            "SIGNAL_ONLY": "true",  # irrelevant for OKX
        }
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            from scripts.run_boll_cvd_live import main
            with pytest.raises(RuntimeError, match="LIVE_TRADING is not true"):
                asyncio.run(main())

    def test_default_exchange_goes_to_okx_path(self) -> None:
        """When EXCHANGE is not set, default to OKX path (LIVE_TRADING guard)."""
        env = {
            "LIVE_TRADING": "false",
        }
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            from scripts.run_boll_cvd_live import main
            with pytest.raises(RuntimeError, match="LIVE_TRADING is not true"):
                asyncio.run(main())


# ======================================================================
# Binance signal-only path isolation
# ======================================================================


class TestBinanceSignalOnlyIsolation:
    """Verify Binance signal-only path doesn't touch OKX execution machinery."""

    def test_binance_branch_does_not_create_strategy_tick_queue(self) -> None:
        """Binance signal-only path does not create the strategy tick queue."""
        env = _base_binance_env()

        async def fake_run_binance_signal_only(env_arg=None):
            return

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "src.live.binance_signal_only_runtime.run_binance_signal_only",
                fake_run_binance_signal_only,
            ):
                with mock.patch("asyncio.Queue") as mock_queue:
                    from scripts.run_boll_cvd_live import main
                    asyncio.run(main())
                    # asyncio.Queue should not be called (it's used for
                    # strategy_tick_queue and execution_queue in OKX path)
                    mock_queue.assert_not_called()

    def test_binance_branch_returns_early(self) -> None:
        """Binance branch returns before reaching monitor setup."""
        env = _base_binance_env()

        async def fake_run_binance_signal_only(env_arg=None):
            return

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "src.live.binance_signal_only_runtime.run_binance_signal_only",
                fake_run_binance_signal_only,
            ):
                with mock.patch(
                    "scripts.run_boll_cvd_live.BollBandBreakoutMonitor"
                ) as mock_monitor:
                    from scripts.run_boll_cvd_live import main
                    asyncio.run(main())
                    # BollBandBreakoutMonitor should NOT be instantiated
                    mock_monitor.assert_not_called()


# ======================================================================
# Unsupported exchange
# ======================================================================


class TestUnsupportedExchange:
    """Unsupported EXCHANGE values raise ValueError."""

    def test_bybit_raises_valueerror(self) -> None:
        """EXCHANGE=bybit raises ValueError from selector."""
        env = {"EXCHANGE": "bybit"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            from scripts.run_boll_cvd_live import main
            with pytest.raises(ValueError, match="Unsupported exchange"):
                asyncio.run(main())

    def test_unknown_exchange_raises_valueerror(self) -> None:
        """EXCHANGE=ftx raises ValueError from selector."""
        env = {"EXCHANGE": "ftx"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            from scripts.run_boll_cvd_live import main
            with pytest.raises(ValueError, match="Unsupported exchange"):
                asyncio.run(main())
