#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_run_boll_cvd_live_binance_live_blocked.py
@Description: Tests for the Binance live blocked branch in run_boll_cvd_live.py
              after BinanceLiveTrader wiring.

Binance main live path is now wired — BINANCE_LIVE_BLOCKED no longer raises.
It falls through to the main path where create_live_trader produces a
BinanceLiveTrader.  Preflight is done by BinanceLiveTrader.initialize().

All tests monkeypatch — no network, no API keys.
"""

from __future__ import annotations

import asyncio
import os
from unittest import mock

import pytest

from src.live.binance_live_preflight import BINANCE_LIVE_CONFIRMATION_PHRASE


# ======================================================================
# Helpers
# ======================================================================


def _binance_blocked_env() -> dict[str, str]:
    """Minimal env that triggers BINANCE_LIVE_BLOCKED."""
    return {
        "EXCHANGE": "binance",
        "SIGNAL_ONLY": "false",
    }


def _binance_all_confirmations_env() -> dict[str, str]:
    """Every preflight env set to valid values."""
    return {
        "EXCHANGE": "binance",
        "SIGNAL_ONLY": "false",
        "LIVE_ENABLED": "true",
        "LIVE_ALLOW_ORDERS": "true",
        "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
        "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
        "LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
        "LIVE_LEVERAGE": "20",
    }


# ======================================================================
# Binance live blocked now falls through to main path
# ======================================================================


class TestBinanceLiveBlockedFallsThrough:
    """BINANCE_LIVE_BLOCKED no longer raises — it passes through
    to the main path where create_live_trader handles it."""

    def test_blocked_no_longer_raises_from_preflight(self) -> None:
        """BINANCE_LIVE_BLOCKED now passes through to main path."""
        env = _binance_blocked_env()
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

    def test_blocked_with_confirmations_calls_factory(self) -> None:
        """All confirmations set → Binance live path calls factory."""
        env = _binance_all_confirmations_env()
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


# ======================================================================
# Binance signal-only does not call preflight or factory
# ======================================================================


class TestSignalOnlyDoesNotCallPreflight:
    """The signal-only path must NOT invoke preflight."""

    def test_signal_only_skips_preflight(self) -> None:
        """EXCHANGE=binance + SIGNAL_ONLY=true bypasses preflight."""
        env = {
            "EXCHANGE": "binance",
            "SIGNAL_ONLY": "true",
        }

        async def fake_run_binance_signal_only(env_arg=None):
            return

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "src.live.binance_signal_only_runtime.run_binance_signal_only",
                fake_run_binance_signal_only,
            ):
                with mock.patch(
                    "src.live.binance_live_preflight.build_binance_live_preflight_report"
                ) as mock_build:
                    from scripts.run_boll_cvd_live import main
                    asyncio.run(main())
                    # preflight must NOT be called in signal-only path
                    mock_build.assert_not_called()


# ======================================================================
# OKX path does not call preflight
# ======================================================================


class TestOkxPathDoesNotCallPreflight:
    """The OKX default path must NOT invoke preflight."""

    def test_okx_skips_preflight(self) -> None:
        """EXCHANGE=okx (default) bypasses Binance preflight."""
        env = {
            "EXCHANGE": "okx",
            "LIVE_TRADING": "false",
        }

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "src.live.binance_live_preflight.build_binance_live_preflight_report"
            ) as mock_build:
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError, match="LIVE_TRADING is not true"):
                    asyncio.run(main())
                # preflight must NOT be called in OKX path
                mock_build.assert_not_called()

    def test_okx_default_no_exchange_skips_preflight(self) -> None:
        """Default (no EXCHANGE set) bypasses Binance preflight."""
        env = {
            "LIVE_TRADING": "false",
        }

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "src.live.binance_live_preflight.build_binance_live_preflight_report"
            ) as mock_build:
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError, match="LIVE_TRADING is not true"):
                    asyncio.run(main())
                mock_build.assert_not_called()
