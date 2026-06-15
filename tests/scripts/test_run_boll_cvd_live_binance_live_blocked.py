#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_run_boll_cvd_live_binance_live_blocked.py
@Description: Tests for the Binance live blocked branch in run_boll_cvd_live.py
              after the preflight guard integration.

All tests monkeypatch — no network, no API keys.
"""

from __future__ import annotations

import asyncio
import os
from unittest import mock

import pytest

from src.exchanges.binance.live_preflight import BINANCE_LIVE_CONFIRMATION_PHRASE


# ======================================================================
# Helpers
# ======================================================================


def _binance_blocked_env() -> dict[str, str]:
    """Minimal env that triggers BINANCE_LIVE_BLOCKED."""
    return {
        "EXCHANGE": "binance",
        "LIVE_TRADING": "true",
        "SIGNAL_ONLY": "false",
    }


def _binance_all_confirmations_env() -> dict[str, str]:
    """Every preflight env set to valid values."""
    return {
        "EXCHANGE": "binance",
        "LIVE_TRADING": "true",
        "SIGNAL_ONLY": "false",
        "LIVE_ENABLED": "true",
        "LIVE_ALLOW_ORDERS": "true",
        "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
        "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
        "LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
        "LIVE_LEVERAGE": "20",
    }


# ======================================================================
# Binance live blocked uses preflight message
# ======================================================================


class TestBinanceLiveBlockedUsesPreflight:
    """The BINANCE_LIVE_BLOCKED branch now uses the preflight guard."""

    def test_blocked_raises_with_preflight_message(self) -> None:
        """Basic blocked env raises RuntimeError with preflight details."""
        env = _binance_blocked_env()

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            from scripts.run_boll_cvd_live import main
            with pytest.raises(RuntimeError) as exc_info:
                asyncio.run(main())

            msg = str(exc_info.value)
            assert "Binance live trading runtime is not wired yet" in msg
            assert "blocking_reasons=" in msg
            assert "binance_live_orders_disabled_by_build" in msg

    def test_blocked_does_not_instantiate_trader(self) -> None:
        """BINANCE_LIVE_BLOCKED must NOT instantiate Trader."""
        env = _binance_blocked_env()

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch("scripts.run_boll_cvd_live.Trader") as mock_trader:
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError):
                    asyncio.run(main())
                mock_trader.assert_not_called()

    def test_blocked_does_not_call_execution_worker(self) -> None:
        """BINANCE_LIVE_BLOCKED must NOT call execution_worker."""
        env = _binance_blocked_env()

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "scripts.run_boll_cvd_live.execution_worker_module"
            ) as mock_ew:
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError):
                    asyncio.run(main())
                mock_ew.execution_worker.assert_not_called()


# ======================================================================
# All confirmations set still blocked
# ======================================================================


class TestAllConfirmationsStillBlocked:
    """Even with all env confirmations, Binance live remains blocked."""

    def test_all_confirmations_still_raises(self) -> None:
        """All env set correctly but orders_globally_enabled=False still blocks."""
        env = _binance_all_confirmations_env()

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            from scripts.run_boll_cvd_live import main
            with pytest.raises(RuntimeError) as exc_info:
                asyncio.run(main())

            msg = str(exc_info.value)
            assert "binance_live_orders_disabled_by_build" in msg

    def test_all_confirmations_no_trader(self) -> None:
        """All env confirmations still do NOT instantiate Trader."""
        env = _binance_all_confirmations_env()

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch("scripts.run_boll_cvd_live.Trader") as mock_trader:
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError):
                    asyncio.run(main())
                mock_trader.assert_not_called()

    def test_all_confirmations_no_execution_worker(self) -> None:
        """All env confirmations still do NOT call execution_worker."""
        env = _binance_all_confirmations_env()

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "scripts.run_boll_cvd_live.execution_worker_module"
            ) as mock_ew:
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError):
                    asyncio.run(main())
                mock_ew.execution_worker.assert_not_called()

    def test_all_confirmations_does_not_enter_okx_path(self) -> None:
        """All confirmations + EXCHANGE=binance does NOT enter OKX path.

        live_trading_enabled() is called before create_runtime_bundle(),
        which then handles the Binance block internally.
        """
        env = _binance_all_confirmations_env()

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("scripts.run_boll_cvd_live.load_dotenv", return_value=False):
            with mock.patch(
                "scripts.run_boll_cvd_live.live_config_helpers"
            ) as mock_lch:
                mock_lch.live_trading_enabled.return_value = True
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError):
                    asyncio.run(main())
                # live_trading_enabled() IS called (it gates before bundle creation)
                mock_lch.live_trading_enabled.assert_called()

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
                "src.exchanges.binance.live_preflight.build_binance_live_preflight_report"
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
                "src.exchanges.binance.live_preflight.build_binance_live_preflight_report"
            ) as mock_build:
                from scripts.run_boll_cvd_live import main
                with pytest.raises(RuntimeError, match="LIVE_TRADING is not true"):
                    asyncio.run(main())
                mock_build.assert_not_called()
