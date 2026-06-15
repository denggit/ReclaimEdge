#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests confirming that OKX credentials are resolved correctly through
the OKX adapter credential resolver (src/exchanges/okx/credentials.py).

Legacy OKX credential fallback has been moved out of config/env_loader.py
and into the OKX adapter layer.  The unified EXCHANGE_API_* vars are read
by load_unified_runtime_config(); legacy OKX_* fallback is handled only by
resolve_okx_credentials() in the OKX adapter.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.exchanges.okx.credentials import resolve_okx_credentials
from src.exchanges.runtime_config import ExchangeRuntimeConfig, ExchangeName


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FACTORY_PATH = ROOT / "src" / "live" / "runtime_factory.py"
OKX_CREDENTIALS_PATH = ROOT / "src" / "exchanges" / "okx" / "credentials.py"
RUNTIME_CONFIG_PATH = ROOT / "src" / "exchanges" / "runtime_config.py"
ENV_LOADER_PATH = ROOT / "config" / "env_loader.py"

RUNTIME_FACTORY_SOURCE = RUNTIME_FACTORY_PATH.read_text(encoding="utf-8")
OKX_CREDENTIALS_SOURCE = OKX_CREDENTIALS_PATH.read_text(encoding="utf-8")
RUNTIME_CONFIG_SOURCE = RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")
ENV_LOADER_SOURCE = ENV_LOADER_PATH.read_text(encoding="utf-8")


def _make_config(api_key="", api_secret="", api_passphrase=""):
    """Build a minimal ExchangeRuntimeConfig for testing."""
    return ExchangeRuntimeConfig(
        exchange=ExchangeName.OKX,
        trade_asset="ETH",
        quote_asset="USDT",
        market_type="PERPETUAL",
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
    )


# ══════════════════════════════════════════════════════════════════════════════
# resolve_okx_credentials() functional tests
# ══════════════════════════════════════════════════════════════════════════════


class TestResolveOkxCredentials:
    """resolve_okx_credentials() correctly applies unified > legacy priority."""

    def test_unified_credentials_are_used(self) -> None:
        config = _make_config(
            api_key="u-key", api_secret="u-secret", api_passphrase="u-pass"
        )
        api_key, api_secret, api_passphrase = resolve_okx_credentials(config, {})
        assert api_key == "u-key"
        assert api_secret == "u-secret"
        assert api_passphrase == "u-pass"

    def test_unified_over_legacy_priority(self) -> None:
        config = _make_config(
            api_key="u-key", api_secret="u-secret", api_passphrase="u-pass"
        )
        env = {
            "OKX_API_KEY": "l-key",
            "OKX_SECRET_KEY": "l-secret",
            "OKX_PASSPHASE": "l-pass",
        }
        api_key, api_secret, api_passphrase = resolve_okx_credentials(config, env)
        assert api_key == "u-key"
        assert api_secret == "u-secret"
        assert api_passphrase == "u-pass"

    def test_legacy_api_key_fallback(self) -> None:
        config = _make_config()
        env = {"OKX_API_KEY": "legacy-key"}
        api_key, _, _ = resolve_okx_credentials(config, env)
        assert api_key == "legacy-key"

    def test_legacy_secret_key_fallback(self) -> None:
        config = _make_config()
        env = {"OKX_SECRET_KEY": "legacy-secret"}
        _, api_secret, _ = resolve_okx_credentials(config, env)
        assert api_secret == "legacy-secret"

    def test_legacy_api_secret_fallback(self) -> None:
        """OKX_API_SECRET is a fallback for secret_key."""
        config = _make_config()
        env = {
            "OKX_API_KEY": "k",
            "OKX_API_SECRET": "legacy-api-secret",
        }
        _, api_secret, _ = resolve_okx_credentials(config, env)
        assert api_secret == "legacy-api-secret"

    def test_okx_secret_key_over_api_secret(self) -> None:
        """OKX_SECRET_KEY > OKX_API_SECRET in the fallback chain."""
        config = _make_config()
        env = {
            "OKX_API_KEY": "k",
            "OKX_SECRET_KEY": "primary-legacy-secret",
            "OKX_API_SECRET": "secondary-legacy-secret",
        }
        _, api_secret, _ = resolve_okx_credentials(config, env)
        assert api_secret == "primary-legacy-secret"

    def test_legacy_passphrase_fallback(self) -> None:
        config = _make_config()
        env = {
            "OKX_API_KEY": "k",
            "OKX_SECRET_KEY": "s",
            "OKX_PASSPHASE": "legacy-pass",
        }
        _, _, api_passphrase = resolve_okx_credentials(config, env)
        assert api_passphrase == "legacy-pass"

    def test_passphrase_correct_spelling_fallback(self) -> None:
        """OKX_PASSPHRASE (correct spelling) is a fallback for passphrase."""
        config = _make_config()
        env = {
            "OKX_API_KEY": "k",
            "OKX_SECRET_KEY": "s",
            "OKX_PASSPHRASE": "legacy-pass-correct",
        }
        _, _, api_passphrase = resolve_okx_credentials(config, env)
        assert api_passphrase == "legacy-pass-correct"

    def test_passphase_misspelling_over_correct(self) -> None:
        """OKX_PASSPHASE (misspelling) checked before OKX_PASSPHRASE (correct)."""
        config = _make_config()
        env = {
            "OKX_API_KEY": "k",
            "OKX_SECRET_KEY": "s",
            "OKX_PASSPHASE": "misspelled",
            "OKX_PASSPHRASE": "correct-spelling",
        }
        _, _, api_passphrase = resolve_okx_credentials(config, env)
        assert api_passphrase == "misspelled"

    def test_empty_credentials_when_nothing_set(self) -> None:
        config = _make_config()
        api_key, api_secret, api_passphrase = resolve_okx_credentials(config, {})
        assert api_key == ""
        assert api_secret == ""
        assert api_passphrase == ""


# ══════════════════════════════════════════════════════════════════════════════
# Source-level boundary checks — credentials live in OKX adapter only
# ══════════════════════════════════════════════════════════════════════════════


class TestCredentialsOnlyInOkxAdapter:
    """Legacy OKX credential fallback must ONLY exist in OKX adapter layer."""

    def test_okx_credentials_has_legacy_fallback(self) -> None:
        """resolve_okx_credentials reads legacy OKX_* env vars."""
        assert "OKX_API_KEY" in OKX_CREDENTIALS_SOURCE, (
            "credentials.py must handle legacy OKX_API_KEY fallback"
        )
        assert "OKX_SECRET_KEY" in OKX_CREDENTIALS_SOURCE, (
            "credentials.py must handle legacy OKX_SECRET_KEY fallback"
        )

    def test_runtime_config_no_legacy_credential_reading(self) -> None:
        """runtime_config.py must NOT read legacy OKX credential vars."""
        for var in ("OKX_API_KEY", "OKX_SECRET_KEY", "OKX_API_SECRET",
                     "OKX_PASSPHASE", "OKX_PASSPHRASE"):
            assert f'values.get("{var}"' not in RUNTIME_CONFIG_SOURCE, (
                f"runtime_config.py must NOT read {var}"
            )

    def test_env_loader_no_legacy_credential_reading(self) -> None:
        for var in ("OKX_API_KEY", "OKX_SECRET_KEY", "OKX_API_SECRET",
                     "OKX_PASSPHASE", "OKX_PASSPHRASE"):
            assert f'.get("{var}"' not in ENV_LOADER_SOURCE, (
                f"env_loader.py must NOT read {var}"
            )


class TestRuntimeFactoryBoundaries:
    """runtime_factory.py must NOT import exchange-specific config."""

    def test_no_okx_config_import(self) -> None:
        assert "from config.env_loader import OKX_CONFIG" not in RUNTIME_FACTORY_SOURCE, (
            "runtime_factory.py must NOT import OKX_CONFIG"
        )

    def test_no_okx_private_client_import(self) -> None:
        assert "from src.execution.okx_private_client import" not in RUNTIME_FACTORY_SOURCE

    def test_no_okx_trading_client_import(self) -> None:
        assert "from src.execution.okx_trading_client import" not in RUNTIME_FACTORY_SOURCE

    def test_no_okx_market_data_client_import(self) -> None:
        assert "from src.data_feed.okx_market_data_client import" not in RUNTIME_FACTORY_SOURCE

    def test_no_okx_broker_client_import(self) -> None:
        assert "from src.exchanges.okx.client import OkxBrokerClient" not in RUNTIME_FACTORY_SOURCE

    def test_no_okx_broker_semantic_executor_import(self) -> None:
        assert "from src.exchanges.okx.semantic_executor import" not in RUNTIME_FACTORY_SOURCE
