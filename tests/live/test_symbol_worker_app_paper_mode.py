#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G08b tests for BTC paper mode runtime config bootstrap.

These tests verify:
1. _runtime_config_env_for_worker_mode forces legacy env for paper mode.
2. Live mode returns None (no env override).
3. build_live_symbol_runtime_configs works for BTC paper with legacy path.
4. paper mode env DOES NOT mutate os.environ.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.live.symbol_worker_app import _runtime_config_env_for_worker_mode


# ═══════════════════════════════════════════════════════════════════════════
# 1. _runtime_config_env_for_worker_mode helper
# ═══════════════════════════════════════════════════════════════════════════


class TestRuntimeConfigEnvForWorkerMode:
    def test_paper_mode_forces_legacy_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RECLAIM_USE_SYMBOL_TOML", "true")
        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP")

        env = _runtime_config_env_for_worker_mode(
            mode="paper",
            trader_symbol="BTC-USDT-SWAP",
        )

        assert env is not None
        assert env["RECLAIM_USE_SYMBOL_TOML"] == "false"
        assert env["RECLAIM_SYMBOLS"] == "BTC-USDT-SWAP"

    def test_paper_mode_does_not_mutate_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RECLAIM_USE_SYMBOL_TOML", "true")
        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP")

        _runtime_config_env_for_worker_mode(
            mode="paper",
            trader_symbol="BTC-USDT-SWAP",
        )

        # os.environ must NOT be mutated
        assert os.environ["RECLAIM_USE_SYMBOL_TOML"] == "true"
        assert os.environ["RECLAIM_SYMBOLS"] == "ETH-USDT-SWAP"

    def test_live_mode_returns_none(self) -> None:
        env = _runtime_config_env_for_worker_mode(
            mode="live",
            trader_symbol="ETH-USDT-SWAP",
        )
        assert env is None

    def test_default_mode_returns_none(self) -> None:
        """Any mode other than 'paper' must return None."""
        env = _runtime_config_env_for_worker_mode(
            mode="something-else",
            trader_symbol="ETH-USDT-SWAP",
        )
        assert env is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. BTC paper runtime config smoke test
# ═══════════════════════════════════════════════════════════════════════════


class TestBtcPaperRuntimeConfigs:
    def test_btc_paper_does_not_load_toml(self) -> None:
        """BTC paper mode with legacy path must produce symbol_config=None."""
        from config.live_symbol_config_bootstrap import build_live_symbol_runtime_configs

        paper_env = _runtime_config_env_for_worker_mode(
            mode="paper",
            trader_symbol="BTC-USDT-SWAP",
        )
        assert paper_env is not None

        configs = build_live_symbol_runtime_configs(
            env=paper_env,
            account_equity_usdt=1000.0,
        )

        # Legacy path → symbol_config is None
        assert configs.symbol_config is None
        # env_runtime.symbols should be set to BTC-USDT-SWAP
        assert configs.env_runtime.symbols == ("BTC-USDT-SWAP",)
        # Strategy / monitor / CVD / sizer should be from legacy env
        assert configs.strategy is not None
        assert configs.monitor is not None
        assert configs.cvd is not None
        assert configs.position_sizer is not None

    def test_eth_live_with_default_env_unchanged(self) -> None:
        """Live mode with default env still loads TOML (existing behaviour)."""
        # Without env override, the TOML path is used by default
        # But this test only verifies the helper returns None for live mode,
        # which means the existing TOML behaviour is fully preserved.
        env = _runtime_config_env_for_worker_mode(
            mode="live",
            trader_symbol="ETH-USDT-SWAP",
        )
        assert env is None
