#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_run_boll_cvd_live_trader_factory.py
@Description: Tests verifying that run_boll_cvd_live.py uses the trader factory
              for OKX path and keeps Binance paths unchanged.
"""

from __future__ import annotations

import asyncio
import os
from unittest import mock

import pytest


# ======================================================================
# OKX path uses create_live_trader
# ======================================================================


class TestOkxPathUsesFactory:
    """The OKX default path calls create_live_trader instead of Trader()."""

    def test_okx_path_calls_create_live_trader(self) -> None:
        """EXCHANGE=okx path should call create_live_trader."""
        env = {
            "EXCHANGE": "okx",
            "LIVE_TRADING": "true",
        }

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
                # The factory is called after initialisation guards.
                # Raise here to stop execution before real Trader construction.
                mock_factory.side_effect = RuntimeError("STOP_AFTER_FACTORY")
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError, match="STOP_AFTER_FACTORY"):
                    asyncio.run(main())
                mock_factory.assert_called_once()

    def test_okx_path_does_not_directly_instantiate_trader(self) -> None:
        """The OKX path must not call Trader() directly."""
        env = {
            "EXCHANGE": "okx",
            "LIVE_TRADING": "false",
        }

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch("scripts.run_boll_cvd_live.Trader") as mock_trader:
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError):
                    asyncio.run(main())
                # Trader class should never be called directly
                mock_trader.assert_not_called()

    def test_okx_default_no_exchange_uses_factory(self) -> None:
        """Default (no EXCHANGE) should also use create_live_trader."""
        env = {
            "LIVE_TRADING": "true",
        }

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


# ======================================================================
# Binance signal-only does not call factory
# ======================================================================


class TestBinanceSignalOnlySkipsFactory:
    """The Binance signal-only path must not call create_live_trader."""

    def test_signal_only_does_not_call_factory(self) -> None:
        """EXCHANGE=binance + SIGNAL_ONLY=true skips the factory."""
        env = {
            "EXCHANGE": "binance",
            "SIGNAL_ONLY": "true",
        }

        async def fake_run_signal_only(env_arg=None):
            return

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "src.live.binance_signal_only_runtime.run_binance_signal_only",
                fake_run_signal_only,
            ):
                with mock.patch(
                    "scripts.run_boll_cvd_live.create_live_trader"
                ) as mock_factory:
                    from scripts.run_boll_cvd_live import main
                    asyncio.run(main())
                    mock_factory.assert_not_called()

    def test_signal_only_does_not_call_trader_directly(self) -> None:
        """Signal-only path must not call Trader() directly."""
        env = {
            "EXCHANGE": "binance",
            "SIGNAL_ONLY": "true",
        }

        async def fake_run_signal_only(env_arg=None):
            return

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "src.live.binance_signal_only_runtime.run_binance_signal_only",
                fake_run_signal_only,
            ):
                with mock.patch("scripts.run_boll_cvd_live.Trader") as mock_trader:
                    from scripts.run_boll_cvd_live import main
                    asyncio.run(main())
                    mock_trader.assert_not_called()


# ======================================================================
# Binance live blocked does not create trader
# ======================================================================


class TestBinanceLiveBlockedNoTrader:
    """Binance live blocked path now falls through to factory.
    It no longer raises from the blocked branch.  Factory creates BinanceLiveTrader."""

    def test_blocked_now_calls_runtime(self) -> None:
        """Binance blocked path now calls run_binance_main_live."""
        env = {
            "EXCHANGE": "binance",
            "SIGNAL_ONLY": "false",
            "LIVE_TRADING": "true",
        }

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "src.live.binance_main_live_runtime.run_binance_main_live"
            ) as mock_runtime:
                async def fake_main_live(env=None):
                    raise RuntimeError("STOP_AFTER_RUNTIME")
                mock_runtime.side_effect = fake_main_live
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError, match="STOP_AFTER_RUNTIME"):
                    asyncio.run(main())
                mock_runtime.assert_called_once()

    def test_blocked_does_not_call_trader_directly(self) -> None:
        """Binance path must not call OKX Trader() directly."""
        env = {
            "EXCHANGE": "binance",
            "SIGNAL_ONLY": "false",
            "LIVE_TRADING": "true",
        }

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch("scripts.run_boll_cvd_live.Trader") as mock_trader:
                with mock.patch(
                    "src.live.binance_main_live_runtime.run_binance_main_live"
                ) as mock_runtime:
                    async def fake_main_live(env=None):
                        raise RuntimeError("STOP_AFTER_RUNTIME")
                    mock_runtime.side_effect = fake_main_live
                    from scripts.run_boll_cvd_live import main
                    with pytest.raises(RuntimeError, match="STOP_AFTER_RUNTIME"):
                        asyncio.run(main())
                    mock_trader.assert_not_called()
