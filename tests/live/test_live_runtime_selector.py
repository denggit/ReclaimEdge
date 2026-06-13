#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_live_runtime_selector.py
@Description: Unit tests for the live runtime selector.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from src.live.live_runtime_selector import (
    LiveRuntimeKind,
    LiveRuntimeSelection,
    select_live_runtime,
)


# ======================================================================
# OKX default / explicit
# ======================================================================


class TestOkxDefault:
    """EXCHANGE not set or empty → OKX_LEGACY."""

    def test_no_exchange_defaults_to_okx(self) -> None:
        env: dict[str, str] = {}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.OKX_LEGACY
        assert result.exchange == "okx"
        assert result.signal_only is False
        assert "okx" in result.reason

    def test_empty_exchange_defaults_to_okx(self) -> None:
        env = {"EXCHANGE": ""}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.OKX_LEGACY
        assert result.exchange == "okx"

    def test_explicit_okx_lowercase(self) -> None:
        env = {"EXCHANGE": "okx"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.OKX_LEGACY
        assert result.exchange == "okx"
        assert result.signal_only is False

    def test_explicit_okx_uppercase(self) -> None:
        env = {"EXCHANGE": "OKX"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.OKX_LEGACY
        assert result.exchange == "okx"


# ======================================================================
# Binance signal-only (truthy values)
# ======================================================================


class TestBinanceSignalOnlyTruthy:
    """EXCHANGE=binance + BINANCE_SIGNAL_ONLY=<truthy> → BINANCE_SIGNAL_ONLY."""

    def test_true_lowercase(self) -> None:
        env = {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "true"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_SIGNAL_ONLY
        assert result.exchange == "binance"
        assert result.signal_only is True
        assert result.reason == "binance_signal_only"

    def test_one(self) -> None:
        env = {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "1"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_SIGNAL_ONLY

    def test_yes(self) -> None:
        env = {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "yes"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_SIGNAL_ONLY

    def test_y(self) -> None:
        env = {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "y"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_SIGNAL_ONLY

    def test_on(self) -> None:
        env = {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "on"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_SIGNAL_ONLY


# ======================================================================
# Binance live blocked (falsy / missing)
# ======================================================================


class TestBinanceLiveBlocked:
    """EXCHANGE=binance without truthy signal-only → BINANCE_LIVE_BLOCKED."""

    def test_signal_only_false_explicit(self) -> None:
        env = {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "false"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_LIVE_BLOCKED
        assert result.exchange == "binance"
        assert result.signal_only is False
        assert result.reason == "binance_live_not_wired"

    def test_signal_only_zero(self) -> None:
        env = {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "0"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_LIVE_BLOCKED

    def test_signal_only_missing(self) -> None:
        env = {"EXCHANGE": "binance"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_LIVE_BLOCKED
        assert result.signal_only is False


# ======================================================================
# Unsupported exchange
# ======================================================================


class TestUnsupportedExchange:
    """Unknown exchanges raise ValueError."""

    def test_bybit_raises_valueerror(self) -> None:
        env = {"EXCHANGE": "bybit"}
        with pytest.raises(ValueError, match="Unsupported exchange"):
            select_live_runtime(env)

    def test_unknown_exchange_raises_valueerror(self) -> None:
        env = {"EXCHANGE": "kraken"}
        with pytest.raises(ValueError, match="Unsupported exchange"):
            select_live_runtime(env)


# ======================================================================
# Dataclass / immutability
# ======================================================================


class TestSelectionDataclass:
    """LiveRuntimeSelection is frozen and well-formed."""

    def test_frozen_prevents_mutation(self) -> None:
        sel = LiveRuntimeSelection(
            kind=LiveRuntimeKind.OKX_LEGACY,
            exchange="okx",
            signal_only=False,
            reason="test",
        )
        with pytest.raises(Exception):
            sel.exchange = "binance"  # type: ignore[misc]

    def test_reason_is_populated(self) -> None:
        for kind in LiveRuntimeKind:
            # Construct minimal example for each kind
            sel = LiveRuntimeSelection(
                kind=kind,
                exchange="okx" if kind == LiveRuntimeKind.OKX_LEGACY else "binance",
                signal_only=(kind == LiveRuntimeKind.BINANCE_SIGNAL_ONLY),
                reason=f"dummy_{kind.value}",
            )
            assert isinstance(sel.reason, str)
            assert len(sel.reason) > 0


# ======================================================================
# Immutability of env
# ======================================================================


class TestEnvImmutability:
    """select_live_runtime must not mutate the env dict."""

    def test_env_not_mutated(self) -> None:
        env = {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "true"}
        snapshot = dict(env)
        select_live_runtime(env)
        assert env == snapshot

    def test_empty_env_not_mutated(self) -> None:
        env: dict[str, str] = {}
        snapshot = dict(env)
        select_live_runtime(env)
        assert env == snapshot


# ======================================================================
# env=None uses os.environ
# ======================================================================


class TestEnvNoneUsesOsEnviron:
    """When env is None, the selector reads os.environ."""

    def test_env_none_uses_os_environ(self) -> None:
        env = {"EXCHANGE": "okx"}
        with mock.patch.dict(os.environ, env, clear=True):
            result = select_live_runtime(None)
            assert result.kind == LiveRuntimeKind.OKX_LEGACY

    def test_env_none_binance_signal_only(self) -> None:
        env = {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "true"}
        with mock.patch.dict(os.environ, env, clear=True):
            result = select_live_runtime(None)
            assert result.kind == LiveRuntimeKind.BINANCE_SIGNAL_ONLY

    def test_env_none_binance_blocked(self) -> None:
        env = {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "false"}
        with mock.patch.dict(os.environ, env, clear=True):
            result = select_live_runtime(None)
            assert result.kind == LiveRuntimeKind.BINANCE_LIVE_BLOCKED

    def test_env_none_unsupported(self) -> None:
        env = {"EXCHANGE": "bybit"}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Unsupported exchange"):
                select_live_runtime(None)
