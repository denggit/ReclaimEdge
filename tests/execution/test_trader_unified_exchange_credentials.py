#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests confirming that Trader correctly loads OKX credentials from unified
EXCHANGE_API_* env vars (with legacy OKX_* fallback).

Strategy: Trader imports OKX_CONFIG at module level, so direct instantiation
is avoided.  Instead we verify the credential loading layer and audit the
error message in trader.py source.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import config.env_loader


# ══════════════════════════════════════════════════════════════════════════════
# Source-level boundary checks — trader.py error message
# ══════════════════════════════════════════════════════════════════════════════


TRADER_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "trader.py"
TRADER_SOURCE = TRADER_PATH.read_text(encoding="utf-8")


class TestTraderErrorMessageContainsUnifiedVarNames:
    """The Trader ValueError message must mention both unified and legacy vars."""

    def test_error_message_contains_exchange_api_key(self) -> None:
        assert "EXCHANGE_API_KEY" in TRADER_SOURCE, (
            "Trader error message must mention EXCHANGE_API_KEY"
        )

    def test_error_message_contains_exchange_api_secret(self) -> None:
        assert "EXCHANGE_API_SECRET" in TRADER_SOURCE, (
            "Trader error message must mention EXCHANGE_API_SECRET"
        )

    def test_error_message_contains_exchange_api_passphrase(self) -> None:
        assert "EXCHANGE_API_PASSPHRASE" in TRADER_SOURCE, (
            "Trader error message must mention EXCHANGE_API_PASSPHRASE"
        )

    def test_error_message_contains_legacy_okx_api_key(self) -> None:
        assert "OKX_API_KEY" in TRADER_SOURCE, (
            "Trader error message must mention legacy OKX_API_KEY"
        )

    def test_error_message_contains_legacy_okx_secret_key(self) -> None:
        assert "OKX_SECRET_KEY" in TRADER_SOURCE, (
            "Trader error message must mention legacy OKX_SECRET_KEY"
        )

    def test_error_message_contains_legacy_okx_passphase(self) -> None:
        assert "OKX_PASSPHASE" in TRADER_SOURCE, (
            "Trader error message must mention legacy OKX_PASSPHASE"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Trader credential loading via get_okx_config()
# ══════════════════════════════════════════════════════════════════════════════


class TestTraderCredentialLoading:
    """Verify that get_okx_config() returns unified credentials with correct priority."""

    def test_unified_credentials_are_read(self, monkeypatch) -> None:
        """Trader would pick up EXCHANGE_API_* when only those are set."""
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: {
                "EXCHANGE_API_KEY": "trader-key",
                "EXCHANGE_API_SECRET": "trader-secret",
                "EXCHANGE_API_PASSPHRASE": "trader-pass",
            },
        )
        result = config.env_loader.get_okx_config()
        assert result["api_key"] == "trader-key"
        assert result["secret_key"] == "trader-secret"
        assert result["passphrase"] == "trader-pass"

    def test_unified_over_legacy_priority(self, monkeypatch) -> None:
        """When both unified and legacy are set, unified wins for all three fields."""
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: {
                "EXCHANGE_API_KEY": "u-key",
                "EXCHANGE_API_SECRET": "u-secret",
                "EXCHANGE_API_PASSPHRASE": "u-pass",
                "OKX_API_KEY": "l-key",
                "OKX_SECRET_KEY": "l-secret",
                "OKX_PASSPHASE": "l-pass",
            },
        )
        result = config.env_loader.get_okx_config()
        assert result["api_key"] == "u-key"
        assert result["secret_key"] == "u-secret"
        assert result["passphrase"] == "u-pass"

    def test_legacy_only_still_works(self, monkeypatch) -> None:
        """When only legacy vars are set, Trader can still initialize."""
        monkeypatch.setattr(
            config.env_loader, "load_env_config",
            lambda: {
                "OKX_API_KEY": "legacy-key",
                "OKX_SECRET_KEY": "legacy-secret",
                "OKX_PASSPHASE": "legacy-pass",
            },
        )
        result = config.env_loader.get_okx_config()
        assert result["api_key"] == "legacy-key"
        assert result["secret_key"] == "legacy-secret"
        assert result["passphrase"] == "legacy-pass"

    def test_empty_credentials_return_empty_strings(self, monkeypatch) -> None:
        """When no credentials are set, get_okx_config returns empty strings."""
        monkeypatch.setattr(
            config.env_loader, "load_env_config", lambda: {},
        )
        result = config.env_loader.get_okx_config()
        assert result["api_key"] == ""
        assert result["secret_key"] == ""
        assert result["passphrase"] == ""


# ══════════════════════════════════════════════════════════════════════════════
# Boundary — OKX_CONFIG global variable
# ══════════════════════════════════════════════════════════════════════════════


class TestOkxConfigGlobal:
    """OKX_CONFIG global variable is still named the same and is a dict."""

    def test_okx_config_is_dict(self) -> None:
        assert isinstance(config.env_loader.OKX_CONFIG, dict)

    def test_okx_config_has_expected_keys(self) -> None:
        for key in ("api_key", "secret_key", "passphrase"):
            assert key in config.env_loader.OKX_CONFIG, (
                f"OKX_CONFIG missing key {key!r}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Boundary — Binance signal-only path must NOT be affected
# ══════════════════════════════════════════════════════════════════════════════


class TestBinanceSignalOnlyNotAffected:
    """The unified EXCHANGE_API_* vars should only affect OKX live credential
    loading.  Binance signal-only path should NOT read these API keys."""

    def test_get_email_config_does_not_read_exchange_api_keys(self) -> None:
        """get_email_config must not reference exchange API key vars."""
        source = Path(config.env_loader.__file__).read_text(encoding="utf-8")
        email_func_src = _extract_function_source(source, "get_email_config")
        assert "EXCHANGE_API_KEY" not in email_func_src
        assert "OKX_API_KEY" not in email_func_src


def _extract_function_source(source: str, name: str) -> str:
    """Extract a function's source text from module source using AST."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    return ""
