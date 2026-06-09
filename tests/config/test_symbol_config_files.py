#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for on-disk symbol TOML config files (A03).

These tests verify that the canonical ``config/symbols/ETH-USDT-SWAP.toml``
exists, can be loaded by the A02 loader, matches the Python-side
``SymbolConfig.default_eth()`` defaults, and that safety switches are correct.
"""

from __future__ import annotations

from pathlib import Path

from config.symbol_config import SymbolConfig
from config.symbol_config_loader import load_symbol_config_from_dir
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


def test_default_eth_toml_matches_schema_defaults() -> None:
    """Every key field in the TOML must match SymbolConfig.default_eth()."""
    loaded = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")
    default = SymbolConfig.default_eth()

    # market
    assert loaded.market.bar == default.market.bar
    assert loaded.market.contract_value == default.market.contract_value

    # capital
    assert loaded.capital.layer_margin_pct == default.capital.layer_margin_pct
    assert loaded.capital.leverage == default.capital.leverage
    assert loaded.capital.max_layers == default.capital.max_layers

    # entry
    assert loaded.entry.add_gap_pct == default.entry.add_gap_pct

    # cvd
    assert loaded.cvd.fast_window_seconds == default.cvd.fast_window_seconds

    # tp
    assert loaded.tp.three_stage_tp1_ratio == default.tp.three_stage_tp1_ratio
    assert loaded.tp.three_stage_tp2_ratio == default.tp.three_stage_tp2_ratio
    assert loaded.tp.three_stage_runner_ratio == default.tp.three_stage_runner_ratio

    # middle bucket split
    assert loaded.middle_bucket_split.fast_ratio == default.middle_bucket_split.fast_ratio

    # sidecar
    assert loaded.sidecar.margin_pct == default.sidecar.margin_pct

    # risk
    assert (
        loaded.risk.order_failure_market_exit_delay_seconds
        == default.risk.order_failure_market_exit_delay_seconds
    )

    # execution
    assert (
        loaded.execution.private_write_min_interval_seconds
        == default.execution.private_write_min_interval_seconds
    )

    # runtime
    assert (
        loaded.runtime.strategy_tick_queue_maxsize
        == default.runtime.strategy_tick_queue_maxsize
    )


# ---------------------------------------------------------------------------
# 4. Safety switches
# ---------------------------------------------------------------------------


def test_default_eth_toml_safety_switches() -> None:
    """Safety-critical fields must be locked to their safe defaults."""
    loaded = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")

    # live_trading must be off
    assert loaded.symbol.enabled is True
    assert loaded.symbol.live_trading is False
    assert loaded.is_live_trading_enabled is False

    # three_stage_tp2 must use structure Boll
    assert loaded.tp.three_stage_tp2_use_structure_boll is True

    # order failure market exit delay must be 1800
    assert loaded.risk.order_failure_market_exit_delay_seconds == 1800

    # sidecar must be disabled
    assert loaded.sidecar.enabled is False


# ---------------------------------------------------------------------------
# 5. Validator acceptance
# ---------------------------------------------------------------------------


def test_default_eth_toml_passes_validator() -> None:
    loaded = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")
    validate_symbol_config(loaded)


# ---------------------------------------------------------------------------
# 6. No BTC TOML created yet
# ---------------------------------------------------------------------------


def test_no_btc_toml_created_yet() -> None:
    """A03 only adds ETH — BTC-USDT-SWAP.toml must NOT exist yet."""
    assert (
        not _BTC_TOML.exists()
    ), f"{_BTC_TOML} must not exist; A03 only creates ETH TOML."
