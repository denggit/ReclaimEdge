#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for on-disk symbol TOML config files (A03).

These tests verify that the canonical ``config/symbols/ETH-USDT-SWAP.toml``
exists, can be loaded by the A02 loader, matches the Python-side
``SymbolConfig.default_eth()`` defaults, and that safety switches are correct.
"""

from __future__ import annotations

from decimal import Decimal
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


def test_default_eth_toml_matches_live_config() -> None:
    """The ETH TOML must contain the current live production values."""
    loaded = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")

    # market
    assert loaded.market.bar == "15m"
    assert loaded.market.td_mode == "isolated"
    assert loaded.market.pos_side_mode == "net"
    assert loaded.market.min_outside_pct == Decimal("0.0005")

    # capital (live values)
    assert loaded.capital.layer_margin_pct == Decimal("0.06")
    assert loaded.capital.leverage == Decimal("15")
    assert loaded.capital.max_layers == 10
    assert loaded.capital.dry_run_equity_usdt == Decimal("1000")

    # entry (live values)
    assert loaded.entry.add_gap_pct == Decimal("0.003")
    assert loaded.entry.add_freeze_seconds == 3600
    assert loaded.entry.first_add_block_seconds == 3600
    assert loaded.entry.add_min_interval_seconds == 1800
    assert loaded.entry.alert_freeze_seconds == 3600

    # cvd
    assert loaded.cvd.fast_window_seconds == Decimal("5")

    # tp (live values)
    assert loaded.tp.tp_min_net_profit_pct == Decimal("0.004")
    assert loaded.tp.three_stage_tp1_ratio == Decimal("0.80")
    assert loaded.tp.three_stage_tp2_ratio == Decimal("0.10")
    assert loaded.tp.three_stage_runner_ratio == Decimal("0.10")
    assert loaded.tp.split_tp_enabled is False

    # middle bucket split (live values)
    assert loaded.middle_bucket_split.enabled is True
    assert loaded.middle_bucket_split.fast_ratio == Decimal("0.70")

    # sidecar (live values)
    assert loaded.sidecar.enabled is True
    assert loaded.sidecar.margin_pct == Decimal("0.02")
    assert loaded.sidecar.tp_pct == Decimal("0.0044")
    assert loaded.sidecar.max_legs == 12

    # risk
    assert loaded.risk.order_failure_market_exit_delay_seconds == 1800

    # execution
    assert loaded.execution.private_write_min_interval_seconds == Decimal("0.6")

    # runtime (live values)
    assert loaded.runtime.market_tick_heartbeat_seconds == Decimal("300")
    assert loaded.runtime.strategy_tick_queue_maxsize == 20000


# ---------------------------------------------------------------------------
# 4. Safety switches
# ---------------------------------------------------------------------------


def test_default_eth_toml_safety_switches() -> None:
    """Safety-critical fields must be locked to their safe defaults."""
    loaded = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")

    # live_trading must be off (real gate is LIVE_TRADING in .env)
    assert loaded.symbol.enabled is True
    assert loaded.symbol.live_trading is False
    assert loaded.is_live_trading_enabled is False

    # three_stage_tp2 must use structure Boll
    assert loaded.tp.three_stage_tp2_use_structure_boll is True

    # order failure market exit delay must be at least 1800
    assert loaded.risk.order_failure_market_exit_delay_seconds >= 1800

    # tp_rate_limit_fail_action must be HALT_ONLY
    assert loaded.sidecar.tp_rate_limit_fail_action == "HALT_ONLY"


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
