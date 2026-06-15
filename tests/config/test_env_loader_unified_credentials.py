#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for config/env_loader.py — environment variable loading only.

config/env_loader.py no longer reads any exchange-specific credential
variables (OKX_API_KEY, OKX_SECRET_KEY, etc.).  Legacy OKX credential
fallback now lives exclusively in src/exchanges/okx/credentials.py.

Covers:
    - load_env_config() reads .env and merges os.environ
    - get_email_config() returns expected keys
    - EMAIL_CONFIG global is a dict
    - No exchange credential vars in env_loader source
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

import config.env_loader


ROOT = Path(__file__).resolve().parents[2]
ENV_LOADER_PATH = ROOT / "config" / "env_loader.py"
ENV_LOADER_SOURCE = ENV_LOADER_PATH.read_text(encoding="utf-8")

# Legacy OKX credential vars that must NOT appear in env_loader.py
FORBIDDEN_EXCHANGE_CREDENTIAL_VARS = [
    "OKX_API_KEY",
    "OKX_SECRET_KEY",
    "OKX_API_SECRET",
    "OKX_PASSPHASE",
    "OKX_PASSPHRASE",
    "BINANCE_API_KEY",
    "BINANCE_SECRET",
]


class TestEnvLoaderNoExchangeCredentials:
    """config/env_loader.py must NOT read exchange-specific credential vars."""

    def test_no_okx_legacy_credential_vars_in_source(self) -> None:
        violations = []
        for var in FORBIDDEN_EXCHANGE_CREDENTIAL_VARS:
            for i, line in enumerate(ENV_LOADER_SOURCE.split("\n"), 1):
                if var in line and not line.strip().startswith("#"):
                    violations.append(f"config/env_loader.py:{i}: {line.strip()}")
        assert not violations, (
            "config/env_loader.py must NOT contain exchange credential vars:\n"
            + "\n".join(violations)
        )

    def test_no_get_okx_config_function(self) -> None:
        assert "def get_okx_config" not in ENV_LOADER_SOURCE, (
            "config/env_loader.py must NOT define get_okx_config()"
        )

    def test_no_okx_config_global(self) -> None:
        assert "OKX_CONFIG" not in ENV_LOADER_SOURCE, (
            "config/env_loader.py must NOT have OKX_CONFIG global"
        )


class TestLoadEnvConfig:
    """load_env_config() reads .env and merges os.environ."""

    def test_os_environ_overrides_dotenv(self, monkeypatch) -> None:
        dotenv_content = "KEY_A=from-dotenv\nKEY_B=from-dotenv-b\n"
        monkeypatch.setattr(os.path, "exists", lambda _path: True)
        m_open = mock_open(read_data=dotenv_content)
        monkeypatch.setattr("builtins.open", m_open)
        monkeypatch.setitem(os.environ, "KEY_A", "from-environ")

        result = config.env_loader.load_env_config()
        assert result["KEY_A"] == "from-environ"
        assert result["KEY_B"] == "from-dotenv-b"

    def test_exchange_api_key_env_overrides_dotenv(self, monkeypatch) -> None:
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
        monkeypatch.setattr(os.path, "exists", lambda _path: False)
        monkeypatch.setitem(os.environ, "EXCHANGE_API_KEY", "env-only-key")
        result = config.env_loader.load_env_config()
        assert result["EXCHANGE_API_KEY"] == "env-only-key"


class TestEmailConfig:
    """get_email_config() and EMAIL_CONFIG are unchanged."""

    def test_email_config_is_dict(self) -> None:
        assert isinstance(config.env_loader.EMAIL_CONFIG, dict)

    def test_get_email_config_has_expected_keys(self) -> None:
        result = config.env_loader.get_email_config()
        for key in ("sender", "password", "receiver"):
            assert key in result, f"get_email_config missing key {key!r}"

    def test_get_email_config_does_not_read_exchange_api_keys(self) -> None:
        """get_email_config must not reference exchange API key vars."""
        email_func_text = _extract_function_source(ENV_LOADER_SOURCE, "get_email_config")
        for var in ("EXCHANGE_API_KEY", "OKX_API_KEY"):
            assert var not in email_func_text, (
                f"get_email_config must NOT reference {var}"
            )


def _extract_function_source(source: str, name: str) -> str:
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    return ""
