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

    def test_explicit_okx_uppercase(self) -> None:
        env = {"EXCHANGE": "OKX"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.OKX_LEGACY
        assert result.exchange == "okx"


# ======================================================================
# Binance — always blocked
# ======================================================================


class TestBinanceAlwaysBlocked:
    """EXCHANGE=binance always returns BINANCE_LIVE_BLOCKED."""

    def test_binance_blocked(self) -> None:
        env = {"EXCHANGE": "binance"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_LIVE_BLOCKED
        assert result.exchange == "binance"
        assert result.reason == "binance_live_not_wired"

    def test_binance_uppercase_blocked(self) -> None:
        env = {"EXCHANGE": "BINANCE"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_LIVE_BLOCKED

    def test_binance_with_arbitrary_env_blocked(self) -> None:
        env = {"EXCHANGE": "binance", "SIGNAL_ONLY": "true"}
        result = select_live_runtime(env)
        assert result.kind == LiveRuntimeKind.BINANCE_LIVE_BLOCKED


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
            reason="test",
        )
        with pytest.raises(Exception):
            sel.exchange = "binance"  # type: ignore[misc]

    def test_reason_is_populated(self) -> None:
        for kind in LiveRuntimeKind:
            sel = LiveRuntimeSelection(
                kind=kind,
                exchange="okx" if kind == LiveRuntimeKind.OKX_LEGACY else "binance",
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
        env = {"EXCHANGE": "binance", "SIGNAL_ONLY": "true"}
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

    def test_env_none_binance_blocked(self) -> None:
        env = {"EXCHANGE": "binance"}
        with mock.patch.dict(os.environ, env, clear=True):
            result = select_live_runtime(None)
            assert result.kind == LiveRuntimeKind.BINANCE_LIVE_BLOCKED

    def test_env_none_unsupported(self) -> None:
        env = {"EXCHANGE": "bybit"}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Unsupported exchange"):
                select_live_runtime(None)
