#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for A09 configuration samples and documentation."""

from __future__ import annotations

from pathlib import Path

from config.symbol_config_loader import load_symbol_config
from config.symbol_config_validator import validate_symbol_config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _read(path: str) -> str:
    return (_PROJECT_ROOT / path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------


def test_env_example_exists() -> None:
    assert (_PROJECT_ROOT / ".env.example").is_file()


def test_sample_toml_exists() -> None:
    assert (_PROJECT_ROOT / "config" / "symbols" / "sample.toml").is_file()


def test_configuration_doc_exists() -> None:
    assert (_PROJECT_ROOT / "docs" / "configuration.md").is_file()


# ---------------------------------------------------------------------------
# 2. sample.toml loads and passes validator
# ---------------------------------------------------------------------------


def test_sample_toml_loads_and_validates() -> None:
    config = load_symbol_config(
        _PROJECT_ROOT / "config" / "symbols" / "sample.toml"
    )
    validate_symbol_config(config)

    assert config.inst_id == "ETH-USDT-SWAP"
    assert config.symbol.live_trading is False
    assert config.tp.three_stage_tp2_use_structure_boll is True
    assert config.sidecar.enabled is False
    assert config.risk.order_failure_market_exit_delay_seconds >= 1800


# ---------------------------------------------------------------------------
# 3. .env.example documents A08 default
# ---------------------------------------------------------------------------


def test_env_example_documents_a08_default() -> None:
    text = _read(".env.example")

    # Default is TOML path.
    assert "RECLAIM_USE_SYMBOL_TOML=true" in text

    # Legacy section is clearly marked.
    assert "Legacy strategy parameters" in text

    # Legacy params are documented as ignored by default.
    assert "ignored by default after A08" in text

    # OKX_PASSPHASE is the correct spelling.
    assert "OKX_PASSPHASE" in text

    # LIVE_TRADING is documented.
    assert "LIVE_TRADING" in text


# ---------------------------------------------------------------------------
# 4. .env.example does not enable live trading by default
# ---------------------------------------------------------------------------


def test_env_example_does_not_enable_live_by_default() -> None:
    text = _read(".env.example")

    # The active LIVE_TRADING line must be false.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("LIVE_TRADING="):
            # Must not be set to true.
            assert "LIVE_TRADING=true" not in stripped, (
                f"LIVE_TRADING must default to false: {stripped!r}"
            )
            assert stripped == "LIVE_TRADING=false", (
                f"Expected LIVE_TRADING=false, got: {stripped!r}"
            )


# ---------------------------------------------------------------------------
# 5. .env.example explains legacy duplication
# ---------------------------------------------------------------------------


def test_env_example_explains_legacy_duplication() -> None:
    text = _read(".env.example")

    # Must mention intentional duplication.
    assert "intentionally duplicate" in text

    # Must reference the explicit opt-out flag.
    assert "RECLAIM_USE_SYMBOL_TOML=false" in text

    # Must warn about pending TOML fields.
    assert "Some TOML fields are pending wiring" in text


# ---------------------------------------------------------------------------
# 6. configuration.md documents boundaries
# ---------------------------------------------------------------------------


def test_configuration_doc_documents_boundaries() -> None:
    text = _read("docs/configuration.md")

    assert "RECLAIM_USE_SYMBOL_TOML=true" in text
    assert "config/symbols/ETH-USDT-SWAP.toml" in text
    assert "LIVE_TRADING" in text
    assert "OKX_PASSPHASE" in text
    # BTC is mentioned — must not be enabled yet.
    assert "BTC" in text


# ---------------------------------------------------------------------------
# 7. configuration.md documents TOML wiring status
# ---------------------------------------------------------------------------


def test_configuration_doc_documents_wiring_status() -> None:
    text = _read("docs/configuration.md")

    # Must have a wiring status section.
    assert "TOML Field Wiring Status" in text

    # Must mention pending fields.
    assert "Pending" in text

    # Must reference specific sections from the wiring table.
    assert "[runtime]" in text or "[runtime].*" in text

    # alert_freeze_seconds must be documented as not a trade entry gate.
    assert "alert_freeze_seconds" in text
    assert "not a live trade entry gate" in text.lower()


# ---------------------------------------------------------------------------
# 8. sample.toml documents wired and pending fields
# ---------------------------------------------------------------------------


def test_sample_toml_documents_wired_and_pending_fields() -> None:
    text = _read("config/symbols/sample.toml").lower()

    assert "wired" in text
    assert "pending" in text
    assert "alert cooldown" in text
    # The comment is split across two TOML comment lines; verify both halves.
    assert "not a live" in text
    assert "trade entry gate" in text


# ---------------------------------------------------------------------------
# 9. ETH TOML documents wired and pending fields
# ---------------------------------------------------------------------------


def test_eth_toml_documents_wired_and_pending_fields() -> None:
    text = _read("config/symbols/ETH-USDT-SWAP.toml").lower()

    assert "wired" in text
    assert "pending" in text
    assert "alert cooldown" in text


# ---------------------------------------------------------------------------
# 10. No BTC symbol TOML created
# ---------------------------------------------------------------------------


def test_no_btc_symbol_toml_created() -> None:
    assert not (_PROJECT_ROOT / "config" / "symbols" / "BTC-USDT-SWAP.toml").exists()
