#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified exchange credential loading tests for config/env_loader.py.

Covers:
    Case A — EXCHANGE_API_* takes priority over OKX_* when both are present.
    Case B — Legacy OKX_* fallback when EXCHANGE_API_* is absent.
    Case C — OKX_PASSPHRASE (correct spelling) fallback when OKX_PASSPHASE is absent.
    Case D — os.environ overrides .env file values for the same key.
    Case E — OKX_API_SECRET as additional secret_key fallback.
"""

from __future__ import annotations

import os
from unittest.mock import mock_open, patch

import pytest

import config.env_loader


# ══════════════════════════════════════════════════════════════════════════════
# Case A — Unified EXCHANGE_API_* takes priority
# ══════════════════════════════════════════════════════════════════════════════


class TestUnifiedCredentialsPriority:
    """When both unified and legacy vars are set, unified wins."""

    UNIFIED_CONFIG = {
        "EXCHANGE_API_KEY": "unified-key",
        "EXCHANGE_API_SECRET": "unified-secret",
        "EXCHANGE_API_PASSPHRASE": "unified-pass",
        "OKX_API_KEY": "legacy-key",
        "OKX_SECRET_KEY": "legacy-secret",
        "OKX_PASSPHASE": "legacy-pass",
    }

    def test_api_key_returns_unified(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(self.UNIFIED_CONFIG),
        )
        result = config.env_loader.get_okx_config()
        assert result["api_key"] == "unified-key"

    def test_secret_key_returns_unified(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(self.UNIFIED_CONFIG),
        )
        result = config.env_loader.get_okx_config()
        assert result["secret_key"] == "unified-secret"

    def test_passphrase_returns_unified(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(self.UNIFIED_CONFIG),
        )
        result = config.env_loader.get_okx_config()
        assert result["passphrase"] == "unified-pass"

    def test_unified_wins_all_three_fields(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(self.UNIFIED_CONFIG),
        )
        result = config.env_loader.get_okx_config()
        assert result == {
            "api_key": "unified-key",
            "secret_key": "unified-secret",
            "passphrase": "unified-pass",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Case B — Legacy fallback when EXCHANGE_API_* is absent
# ══════════════════════════════════════════════════════════════════════════════


class TestLegacyFallback:
    """When only legacy OKX_* vars are set, they are used."""

    LEGACY_CONFIG = {
        "OKX_API_KEY": "legacy-key",
        "OKX_SECRET_KEY": "legacy-secret",
        "OKX_PASSPHASE": "legacy-pass",
    }

    def test_api_key_falls_back_to_okx_api_key(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(self.LEGACY_CONFIG),
        )
        result = config.env_loader.get_okx_config()
        assert result["api_key"] == "legacy-key"

    def test_secret_key_falls_back_to_okx_secret_key(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(self.LEGACY_CONFIG),
        )
        result = config.env_loader.get_okx_config()
        assert result["secret_key"] == "legacy-secret"

    def test_passphrase_falls_back_to_okx_passphase(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(self.LEGACY_CONFIG),
        )
        result = config.env_loader.get_okx_config()
        assert result["passphrase"] == "legacy-pass"

    def test_empty_when_no_vars_set(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config.env_loader, "load_env_config", lambda: {},
        )
        result = config.env_loader.get_okx_config()
        assert result["api_key"] == ""
        assert result["secret_key"] == ""
        assert result["passphrase"] == ""


# ══════════════════════════════════════════════════════════════════════════════
# Case C — OKX_PASSPHRASE (correct spelling) fallback
# ══════════════════════════════════════════════════════════════════════════════


class TestCorrectSpellingFallback:
    """OKX_PASSPHRASE (correct spelling) is a fallback for passphrase."""

    CORRECT_SPELLING_CONFIG = {
        "OKX_API_KEY": "legacy-key",
        "OKX_SECRET_KEY": "legacy-secret",
        "OKX_PASSPHRASE": "legacy-pass-correct",
    }

    def test_passphrase_reads_okx_passphrase_correct_spelling(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(self.CORRECT_SPELLING_CONFIG),
        )
        result = config.env_loader.get_okx_config()
        assert result["passphrase"] == "legacy-pass-correct"

    def test_okx_passphase_misspelling_has_higher_priority_than_correct(self, monkeypatch) -> None:
        """When both misspelling and correct spelling are set, misspelling wins
        (it is checked first in the fallback chain)."""
        both_config = {
            "OKX_API_KEY": "k",
            "OKX_SECRET_KEY": "s",
            "OKX_PASSPHASE": "misspelled",
            "OKX_PASSPHRASE": "correct-spelling",
        }
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(both_config),
        )
        result = config.env_loader.get_okx_config()
        assert result["passphrase"] == "misspelled"


# ══════════════════════════════════════════════════════════════════════════════
# Case D — os.environ overrides .env file values
# ══════════════════════════════════════════════════════════════════════════════


class TestOsEnvironOverrideDotEnv:
    """os.environ values override .env file values for the same key."""

    def test_os_environ_overrides_dotenv(self, monkeypatch) -> None:
        """Simulate .env file containing KEY_A=from-dotenv, with os.environ
        also setting KEY_A=from-environ. The result should be from-environ."""
        dotenv_content = "KEY_A=from-dotenv\nKEY_B=from-dotenv-b\n"

        # Test load_env_config in isolation: monkeypatch os.path.exists and open,
        # then verify os.environ overrides .env.
        monkeypatch.setattr(os.path, "exists", lambda _path: True)
        m_open = mock_open(read_data=dotenv_content)
        monkeypatch.setattr("builtins.open", m_open)
        monkeypatch.setitem(os.environ, "KEY_A", "from-environ")

        # load_env_config reads the (mocked) .env then merges os.environ
        result = config.env_loader.load_env_config()
        assert result["KEY_A"] == "from-environ"
        assert result["KEY_B"] == "from-dotenv-b"

    def test_exchange_api_key_env_overrides_dotenv(self, monkeypatch) -> None:
        """Unified credential set via os.environ should override .env file."""
        dotenv_content = (
            "EXCHANGE_API_KEY=from-dotenv\n"
            "EXCHANGE_API_SECRET=from-dotenv\n"
            "EXCHANGE_API_PASSPHRASE=from-dotenv\n"
        )
        monkeypatch.setattr(os.path, "exists", lambda _path: True)
        m_open = mock_open(read_data=dotenv_content)
        monkeypatch.setattr("builtins.open", m_open)

        monkeypatch.setitem(os.environ, "EXCHANGE_API_KEY", "from-shell-env")
        monkeypatch.setitem(os.environ, "EXCHANGE_API_SECRET", "from-shell-env")
        monkeypatch.setitem(os.environ, "EXCHANGE_API_PASSPHRASE", "from-shell-env")

        result = config.env_loader.load_env_config()
        assert result["EXCHANGE_API_KEY"] == "from-shell-env"
        assert result["EXCHANGE_API_SECRET"] == "from-shell-env"
        assert result["EXCHANGE_API_PASSPHRASE"] == "from-shell-env"

    def test_os_environ_only_no_dotenv(self, monkeypatch) -> None:
        """When no .env file exists, os.environ values are still loaded."""
        monkeypatch.setattr(os.path, "exists", lambda _path: False)
        monkeypatch.setitem(os.environ, "EXCHANGE_API_KEY", "env-only-key")

        result = config.env_loader.load_env_config()
        assert result["EXCHANGE_API_KEY"] == "env-only-key"


# ══════════════════════════════════════════════════════════════════════════════
# Case E — OKX_API_SECRET fallback (OKX_SECRET_KEY > OKX_API_SECRET)
# ══════════════════════════════════════════════════════════════════════════════


class TestSecretKeyFallbackChain:
    """EXCHANGE_API_SECRET > OKX_SECRET_KEY > OKX_API_SECRET > ''"""

    def test_okx_api_secret_fallback(self, monkeypatch) -> None:
        """When only OKX_API_SECRET is set, it is used for secret_key."""
        config_data = {
            "OKX_API_KEY": "k",
            "OKX_API_SECRET": "legacy-api-secret",
        }
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(config_data),
        )
        result = config.env_loader.get_okx_config()
        assert result["secret_key"] == "legacy-api-secret"

    def test_okx_secret_key_priority_over_okx_api_secret(self, monkeypatch) -> None:
        """OKX_SECRET_KEY > OKX_API_SECRET in the fallback chain."""
        config_data = {
            "OKX_API_KEY": "k",
            "OKX_SECRET_KEY": "primary-legacy-secret",
            "OKX_API_SECRET": "secondary-legacy-secret",
        }
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: dict(config_data),
        )
        result = config.env_loader.get_okx_config()
        assert result["secret_key"] == "primary-legacy-secret"
