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
    assert loaded.capital.max_layers == 8
    assert loaded.capital.dry_run_equity_usdt == Decimal("1000")

    # entry (live values)
    assert loaded.entry.add_gap_mode == "linear"
    assert loaded.entry.add_gap_base_pct == Decimal("0.003")
    assert loaded.entry.add_gap_step_pct == Decimal("0.001")
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
# 6. BTC TOML — exists but disabled (F03)
# ---------------------------------------------------------------------------


def test_btc_toml_exists_but_disabled() -> None:
    """BTC-USDT-SWAP.toml must exist, but enabled and live_trading must be False."""
    assert _BTC_TOML.is_file(), (
        f"Expected {_BTC_TOML} to exist (F03 adds disabled BTC config)."
    )
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "BTC-USDT-SWAP")
    assert config.inst_id == "BTC-USDT-SWAP"
    assert config.symbol.enabled is False
    assert config.symbol.live_trading is False
    assert config.is_enabled is False
    assert config.is_live_trading_enabled is False


def test_btc_toml_market_metadata() -> None:
    """BTC TOML must carry correct OKX instrument metadata."""
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "BTC-USDT-SWAP")
    assert config.market.contract_value == Decimal("0.01")
    assert config.market.min_contracts == Decimal("0.01")
    assert config.market.contract_precision == Decimal("0.01")
    assert config.market.price_precision == Decimal("0.1")


def test_btc_toml_safety_switches() -> None:
    """BTC safety-critical fields must be locked down."""
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "BTC-USDT-SWAP")

    # middle_bucket_split and sidecar must be disabled for BTC.
    assert config.middle_bucket_split.enabled is False
    assert config.sidecar.enabled is False

    # three_stage_tp2 must use structure Boll.
    assert config.tp.three_stage_tp2_use_structure_boll is True

    # tp_rate_limit_fail_action must be HALT_ONLY.
    assert config.sidecar.tp_rate_limit_fail_action == "HALT_ONLY"

    # order_failure_market_exit_delay must be >= 1800.
    assert config.risk.order_failure_market_exit_delay_seconds >= 1800


def test_btc_toml_passes_validator() -> None:
    """Disabled BTC config must pass validator without error."""
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "BTC-USDT-SWAP")
    validate_symbol_config(config)


# ---------------------------------------------------------------------------
# 7. BTC TOML source guards — must NOT contain enabled=true / live_trading=true
# ---------------------------------------------------------------------------


def test_btc_toml_must_not_contain_enabled_true_line() -> None:
    """BTC TOML must NOT contain the exact line 'enabled = true'."""
    content = _BTC_TOML.read_text(encoding="utf-8")
    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        assert stripped != "enabled = true", (
            f"BTC-USDT-SWAP.toml must not contain exact line 'enabled = true'. "
            f"Found: {stripped!r}"
        )


def test_btc_toml_must_not_contain_live_trading_true_line() -> None:
    """BTC TOML must NOT contain the exact line 'live_trading = true'."""
    content = _BTC_TOML.read_text(encoding="utf-8")
    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        assert stripped != "live_trading = true", (
            f"BTC-USDT-SWAP.toml must not contain exact line 'live_trading = true'. "
            f"Found: {stripped!r}"
        )


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
