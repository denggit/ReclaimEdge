#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for on-disk symbol TOML config files (A03)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from config.symbol_config import SymbolConfig
from config.symbol_config_loader import load_symbol_config_from_dir
from config.symbol_config_mapper import MappedSymbolConfigs, map_symbol_config
from config.symbol_config_validator import validate_symbol_config

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SYMBOLS_DIR = _PROJECT_ROOT / "config" / "symbols"
_ETH_TOML = _SYMBOLS_DIR / "ETH-USDT-SWAP.toml"
_BTC_TOML = _SYMBOLS_DIR / "BTC-USDT-SWAP.toml"


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------


def test_default_eth_toml_exists() -> None:
    """The canonical ETH-USDT-SWAP.toml must exist on disk."""
    assert _ETH_TOML.is_file(), (
        f"Expected {_ETH_TOML} to exist, but it does not."
    )


# ---------------------------------------------------------------------------
# 2. Loading
# ---------------------------------------------------------------------------


def test_default_eth_toml_loads() -> None:
    """The TOML must be loadable and produce a config with the correct inst_id."""
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")
    assert config.inst_id == "ETH-USDT-SWAP"


# ---------------------------------------------------------------------------
# 3. Schema-defaults match
# ---------------------------------------------------------------------------


def test_default_eth_toml_loads_validates_and_maps() -> None:
    """The ETH TOML must load, validate, and map without pinning tunable values."""
    loaded = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")
    validate_symbol_config(loaded)
    mapped = map_symbol_config(loaded)

    assert loaded.inst_id == "ETH-USDT-SWAP"
    assert isinstance(loaded.symbol.enabled, bool)
    assert isinstance(loaded.symbol.live_trading, bool)
    assert loaded.market.contract_value == Decimal("0.1")
    assert isinstance(mapped, MappedSymbolConfigs)
    assert mapped.trader_preview.inst_id == "ETH-USDT-SWAP"
    assert mapped.trader_preview.contract_value == Decimal("0.1")


# ---------------------------------------------------------------------------
# 4. Live switches
# ---------------------------------------------------------------------------


def test_default_eth_toml_entry_timing_fields() -> None:
    """ETH entry timing config uses explicit first/subsequent add gates."""
    loaded = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")

    assert isinstance(loaded.entry.first_add_block_seconds, int)
    assert isinstance(loaded.entry.add_min_interval_seconds, int)


# ---------------------------------------------------------------------------
# 5. Validator acceptance
# ---------------------------------------------------------------------------


def test_default_eth_toml_passes_validator() -> None:
    loaded = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")
    validate_symbol_config(loaded)


# ---------------------------------------------------------------------------
# 6. BTC TOML — exists and maps current live switches
# ---------------------------------------------------------------------------


def test_btc_toml_exists_loads_validates_and_maps() -> None:
    """BTC-USDT-SWAP.toml must exist and load without pinning tunable switches."""
    assert _BTC_TOML.is_file(), (
        f"Expected {_BTC_TOML} to exist (F03 adds disabled BTC config)."
    )
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "BTC-USDT-SWAP")
    validate_symbol_config(config)
    mapped = map_symbol_config(config)

    assert config.inst_id == "BTC-USDT-SWAP"
    assert isinstance(config.symbol.enabled, bool)
    assert isinstance(config.symbol.live_trading, bool)
    assert isinstance(mapped, MappedSymbolConfigs)
    assert mapped.trader_preview.inst_id == "BTC-USDT-SWAP"


def test_btc_toml_market_metadata() -> None:
    """BTC TOML must carry correct OKX instrument metadata."""
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "BTC-USDT-SWAP")
    assert config.market.contract_value == Decimal("0.01")
    assert config.market.min_contracts == Decimal("0.01")
    assert config.market.contract_precision == Decimal("0.01")
    assert config.market.price_precision == Decimal("0.1")


def test_btc_toml_entry_timing_fields() -> None:
    """BTC entry timing config uses explicit first/subsequent add gates."""
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "BTC-USDT-SWAP")

    assert isinstance(config.entry.first_add_block_seconds, int)
    assert isinstance(config.entry.add_min_interval_seconds, int)


def test_btc_toml_passes_validator() -> None:
    """Disabled BTC config must pass validator without error."""
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "BTC-USDT-SWAP")
    validate_symbol_config(config)


# ---------------------------------------------------------------------------
# 8. Source guards — run_symbol_worker.py unchanged
# ---------------------------------------------------------------------------


def test_run_symbol_worker_must_contain_live_trading_gate() -> None:
    """run_symbol_worker.py must still contain the LIVE_TRADING gate check."""
    source = (_PROJECT_ROOT / "scripts" / "run_symbol_worker.py").read_text(
        encoding="utf-8"
    )
    assert "LIVE_TRADING is not true. Refusing to start symbol worker." in source, (
        "run_symbol_worker.py must still contain the LIVE_TRADING gate"
    )


def test_run_symbol_worker_must_not_contain_btc_usdt_swap() -> None:
    """run_symbol_worker.py must NOT contain BTC-USDT-SWAP."""
    source = (_PROJECT_ROOT / "scripts" / "run_symbol_worker.py").read_text(
        encoding="utf-8"
    )
    assert "BTC-USDT-SWAP" not in source, (
        "run_symbol_worker.py must NOT contain BTC-USDT-SWAP"
    )


# ---------------------------------------------------------------------------
# 9. Source guards — trader.py unchanged (no default BTC metadata)
# ---------------------------------------------------------------------------


def test_trader_module_has_no_default_btc_metadata() -> None:
    """trader.py must NOT define default instrument metadata for BTC."""
    source = (_PROJECT_ROOT / "src" / "execution" / "trader.py").read_text(
        encoding="utf-8"
    )
    # BTC must not appear as a value for inst_id in any TraderInstrumentMetadata
    # or DEFAULT_* constant definition.  The existing test
    # test_default_metadata_for_btc_raises_value_error already covers the
    # runtime behaviour; this is a source-level backstop.
    assert 'inst_id="BTC-USDT-SWAP"' not in source, (
        "trader.py must NOT define BTC default instrument metadata"
    )
