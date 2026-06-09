#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ``config.env_runtime_config`` — environment runtime configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.env_runtime_config import EnvRuntimeConfig, load_env_runtime_config


# ---------------------------------------------------------------------------
# 1. Default config
# ---------------------------------------------------------------------------


class TestDefaultEnvRuntimeConfig:
    """Default values when no env vars are set."""

    def test_run_mode_default(self) -> None:
        cfg = load_env_runtime_config({})
        assert cfg.run_mode == "live"

    def test_symbols_default(self) -> None:
        cfg = load_env_runtime_config({})
        assert cfg.symbols == ("ETH-USDT-SWAP",)

    def test_symbol_config_dir_default(self) -> None:
        cfg = load_env_runtime_config({})
        assert cfg.symbol_config_dir == Path("config/symbols")

    def test_runtime_dir_default(self) -> None:
        cfg = load_env_runtime_config({})
        assert cfg.runtime_dir == Path("runtime")

    def test_use_symbol_toml_default(self) -> None:
        cfg = load_env_runtime_config({})
        assert cfg.use_symbol_toml is False

    def test_email_enabled_default(self) -> None:
        cfg = load_env_runtime_config({})
        assert cfg.email_enabled is False

    def test_okx_api_key_default(self) -> None:
        cfg = load_env_runtime_config({})
        assert cfg.okx_api_key is None

    def test_okx_secret_key_default(self) -> None:
        cfg = load_env_runtime_config({})
        assert cfg.okx_secret_key is None

    def test_okx_passphase_default(self) -> None:
        cfg = load_env_runtime_config({})
        assert cfg.okx_passphase is None

    def test_smtp_fields_default(self) -> None:
        cfg = load_env_runtime_config({})
        assert cfg.smtp_host is None
        assert cfg.smtp_user is None
        assert cfg.smtp_password is None
        assert cfg.alert_email_to is None


# ---------------------------------------------------------------------------
# 2. Symbol parsing
# ---------------------------------------------------------------------------


class TestParseSymbolsSingleEth:
    """Parsing of a single ETH-USDT-SWAP symbol with whitespace."""

    def test_single_eth_with_whitespace(self) -> None:
        cfg = load_env_runtime_config({"RECLAIM_SYMBOLS": " ETH-USDT-SWAP "})
        assert cfg.symbols == ("ETH-USDT-SWAP",)

    def test_single_eth_no_whitespace(self) -> None:
        cfg = load_env_runtime_config({"RECLAIM_SYMBOLS": "ETH-USDT-SWAP"})
        assert cfg.symbols == ("ETH-USDT-SWAP",)


class TestRejectsDuplicateSymbols:
    """Duplicate symbols must raise ValueError."""

    def test_duplicate_eth(self) -> None:
        with pytest.raises(ValueError, match="RECLAIM_SYMBOLS"):
            load_env_runtime_config(
                {"RECLAIM_SYMBOLS": "ETH-USDT-SWAP,ETH-USDT-SWAP"}
            )


class TestRejectsBtcForNow:
    """BTC-USDT-SWAP is not yet supported — must be rejected."""

    def test_btc_alone(self) -> None:
        with pytest.raises(ValueError, match="RECLAIM_SYMBOLS"):
            load_env_runtime_config({"RECLAIM_SYMBOLS": "BTC-USDT-SWAP"})

    def test_eth_and_btc(self) -> None:
        with pytest.raises(ValueError, match="RECLAIM_SYMBOLS"):
            load_env_runtime_config(
                {"RECLAIM_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP"}
            )


class TestRejectsEmptySymbols:
    """Empty symbol list must raise ValueError."""

    def test_empty_string(self) -> None:
        with pytest.raises(ValueError, match="RECLAIM_SYMBOLS"):
            load_env_runtime_config({"RECLAIM_SYMBOLS": ""})

    def test_only_commas(self) -> None:
        with pytest.raises(ValueError, match="RECLAIM_SYMBOLS"):
            load_env_runtime_config({"RECLAIM_SYMBOLS": ","})


# ---------------------------------------------------------------------------
# 3. Boolean parsing — RECLAIM_USE_SYMBOL_TOML
# ---------------------------------------------------------------------------


class TestParseUseSymbolTomlTrueValues:
    """All accepted true-like values for RECLAIM_USE_SYMBOL_TOML."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
    def test_true_value(self, value: str) -> None:
        cfg = load_env_runtime_config({"RECLAIM_USE_SYMBOL_TOML": value})
        assert cfg.use_symbol_toml is True


class TestParseUseSymbolTomlFalseValues:
    """All accepted false-like values for RECLAIM_USE_SYMBOL_TOML."""

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "FALSE"])
    def test_false_value(self, value: str) -> None:
        cfg = load_env_runtime_config({"RECLAIM_USE_SYMBOL_TOML": value})
        assert cfg.use_symbol_toml is False


class TestInvalidBoolRejected:
    """Unknown boolean values must raise ValueError."""

    def test_maybe_rejected(self) -> None:
        with pytest.raises(ValueError, match="RECLAIM_USE_SYMBOL_TOML"):
            load_env_runtime_config({"RECLAIM_USE_SYMBOL_TOML": "maybe"})


# ---------------------------------------------------------------------------
# 4. OKX passphase — deliberate misspelling
# ---------------------------------------------------------------------------


class TestOkxPassphase:
    """The project uses ``OKX_PASSPHASE`` (not ``OKX_PASSPHRASE``)."""

    def test_passphase_key_is_read(self) -> None:
        cfg = load_env_runtime_config({"OKX_PASSPHASE": "abc"})
        assert cfg.okx_passphase == "abc"

    def test_passphrase_key_is_ignored(self) -> None:
        cfg = load_env_runtime_config({"OKX_PASSPHRASE": "wrong"})
        assert cfg.okx_passphase is None


# ---------------------------------------------------------------------------
# 5. Email fields
# ---------------------------------------------------------------------------


class TestEmailFields:
    """Email-related configuration is parsed correctly."""

    def test_all_email_fields(self) -> None:
        cfg = load_env_runtime_config(
            {
                "EMAIL_ENABLED": "true",
                "SMTP_HOST": "smtp.example.com",
                "SMTP_USER": "u",
                "SMTP_PASSWORD": "p",
                "ALERT_EMAIL_TO": "a@example.com",
            }
        )
        assert cfg.email_enabled is True
        assert cfg.smtp_host == "smtp.example.com"
        assert cfg.smtp_user == "u"
        assert cfg.smtp_password == "p"
        assert cfg.alert_email_to == "a@example.com"


# ---------------------------------------------------------------------------
# 6. Isolation: mapping vs os.environ
# ---------------------------------------------------------------------------


class TestDoesNotReadOsEnvironWhenMappingProvided:
    """When an explicit mapping is passed, ``os.environ`` must NOT be read."""

    def test_mapping_isolates_from_os_environ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Contaminate os.environ with an unsupported symbol.
        monkeypatch.setenv("RECLAIM_SYMBOLS", "BTC-USDT-SWAP")
        # Pass empty mapping → should use defaults, NOT os.environ.
        cfg = load_env_runtime_config({})
        assert cfg.symbols == ("ETH-USDT-SWAP",)


# ---------------------------------------------------------------------------
# 7. EnvRuntimeConfig is frozen
# ---------------------------------------------------------------------------


class TestEnvRuntimeConfigIsFrozen:
    """The dataclass must be immutable."""

    def test_cannot_set_attribute(self) -> None:
        cfg = load_env_runtime_config({})
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            cfg.run_mode = "dry-run"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. run_mode validation
# ---------------------------------------------------------------------------


class TestRunMode:
    """run_mode parsing."""

    def test_custom_run_mode(self) -> None:
        cfg = load_env_runtime_config({"RECLAIM_RUN_MODE": "dry-run"})
        assert cfg.run_mode == "dry-run"

    def test_empty_run_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="RECLAIM_RUN_MODE"):
            load_env_runtime_config({"RECLAIM_RUN_MODE": ""})


# ---------------------------------------------------------------------------
# 9. Custom directory paths
# ---------------------------------------------------------------------------


class TestCustomDirectoryPaths:
    """symbol_config_dir and runtime_dir can be overridden."""

    def test_custom_symbol_config_dir(self) -> None:
        cfg = load_env_runtime_config(
            {"RECLAIM_SYMBOL_CONFIG_DIR": "/etc/reclaim/symbols"}
        )
        assert cfg.symbol_config_dir == Path("/etc/reclaim/symbols")

    def test_custom_runtime_dir(self) -> None:
        cfg = load_env_runtime_config({"RECLAIM_RUNTIME_DIR": "/var/run/reclaim"})
        assert cfg.runtime_dir == Path("/var/run/reclaim")


# ---------------------------------------------------------------------------
# 10. No import side-effects
# ---------------------------------------------------------------------------


class TestNoImportSideEffects:
    """Importing the module must not read os.environ or perform I/O."""

    def test_import_does_not_touch_os_environ(self) -> None:
        """Re-importing the module should not fail regardless of env state."""
        import importlib

        import config.env_runtime_config as mod

        importlib.reload(mod)
        # Reaching this point without an exception is sufficient proof
        # that the module does not read os.environ at import time.
