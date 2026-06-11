#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ``config.symbol_config_check`` — pure config-check / dry-run preview module (F05)."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from config.symbol_config import SymbolConfig, SymbolIdentityConfig
from config.symbol_config_check import (
    SymbolConfigCheckResult,
    check_symbol_config,
)
from config.symbol_config_mapper import MappedSymbolConfigs
from config.symbol_config_validator import SymbolConfigValidationError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SYMBOLS_DIR = _PROJECT_ROOT / "config" / "symbols"


# ---------------------------------------------------------------------------
# 1. BTC config check succeeds
# ---------------------------------------------------------------------------


def test_btc_config_check_succeeds() -> None:
    """BTC-USDT-SWAP load -> validate -> map passes without pinning tunable values."""
    result = check_symbol_config(
        symbol_config_dir=_SYMBOLS_DIR,
        inst_id="BTC-USDT-SWAP",
    )
    assert isinstance(result, SymbolConfigCheckResult)
    assert result.inst_id == "BTC-USDT-SWAP"
    assert isinstance(result.enabled, bool)
    assert isinstance(result.live_trading, bool)
    assert result.contract_value == Decimal("0.01")
    assert result.min_contracts == Decimal("0.01")
    assert result.contract_precision == Decimal("0.01")
    assert result.price_precision == Decimal("0.1")
    assert isinstance(result.mapped, MappedSymbolConfigs)
    assert result.mapped.trader_preview.inst_id == "BTC-USDT-SWAP"
    assert result.mapped.trader_preview.contract_value == Decimal("0.01")
    assert result.mapped.trader_preview.live_trading is result.live_trading


# ---------------------------------------------------------------------------
# 2. ETH config check succeeds
# ---------------------------------------------------------------------------


def test_eth_config_check_succeeds() -> None:
    """ETH-USDT-SWAP config check passes without pinning tunable values."""
    result = check_symbol_config(
        symbol_config_dir=_SYMBOLS_DIR,
        inst_id="ETH-USDT-SWAP",
    )
    assert isinstance(result, SymbolConfigCheckResult)
    assert result.inst_id == "ETH-USDT-SWAP"
    assert isinstance(result.enabled, bool)
    assert isinstance(result.live_trading, bool)
    assert result.contract_value == Decimal("0.1")
    assert result.mapped.trader_preview.live_trading is result.live_trading


# ---------------------------------------------------------------------------
# 3. Empty inst_id fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inst_id", ["", "   ", "\t\n"])
def test_empty_inst_id_fails(inst_id: str) -> None:
    """Empty or whitespace-only inst_id raises ValueError."""
    with pytest.raises(ValueError, match="inst_id"):
        check_symbol_config(
            symbol_config_dir=_SYMBOLS_DIR,
            inst_id=inst_id,
        )


# ---------------------------------------------------------------------------
# 4. Unsupported inst_id fails
# ---------------------------------------------------------------------------


def test_unsupported_inst_id_fails() -> None:
    """SOL-USDT-SWAP is not supported and must fail validation."""
    with pytest.raises((SymbolConfigValidationError, FileNotFoundError)):
        check_symbol_config(
            symbol_config_dir=_SYMBOLS_DIR,
            inst_id="SOL-USDT-SWAP",
        )


# ---------------------------------------------------------------------------
# 5. BTC enabled=true succeeds (via temp TOML)
# ---------------------------------------------------------------------------


def test_btc_enabled_true_temp_toml_succeeds() -> None:
    """A BTC TOML with enabled=true passes symbol config validation."""
    btc_toml_content = """\
[symbol]
inst_id = "BTC-USDT-SWAP"
enabled = true
live_trading = false

[market]
bar = "15m"
td_mode = "isolated"
pos_side_mode = "net"
contract_value = "0.01"
min_contracts = "0.01"
contract_precision = "0.01"
price_precision = "0.1"
boll_window = 20
boll_std_multiplier = "2.0"
boll_distance_threshold_pct = "0.005"
tp_boll_window = 15
min_outside_pct = "0.0005"

[capital]
dry_run_equity_usdt = "1000"
layer_margin_pct = "0.06"
leverage = "15"
max_layers = 10
layer_multiplier_step = "0.15"

[entry]
add_gap_mode = "linear"
add_gap_base_pct = "0.003"
add_gap_step_pct = "0.001"
first_add_block_seconds = 3600
add_min_interval_seconds = 1800
alert_freeze_seconds = 3600

[cvd]
fast_window_seconds = "5"
price_stall_seconds = "2"
price_stall_tolerance_pct = "0.0005"
burst_window_seconds = "3"
burst_baseline_seconds = "60"
burst_min_move_ratio = "2.5"
burst_min_volume_ratio = "2.0"
burst_min_abs_range_pct = "0.0015"

[tp]
tp_min_net_profit_pct = "0.004"
tp_boll_enabled = true
three_stage_runner_enabled = true
three_stage_tp1_ratio = "0.80"
three_stage_tp2_ratio = "0.10"
three_stage_runner_ratio = "0.10"
three_stage_tp2_use_structure_boll = true
middle_runner_enabled = false
split_tp_enabled = false

[middle_bucket_split]
enabled = false
fast_ratio = "0.70"
fast_sl_enabled = true
fast_sl_fee_buffer_pct = "0.001"

[sidecar]
enabled = false
margin_pct = "0.02"
tp_pct = "0.0044"
skip_first_layer = true
max_legs = 12
order_status_check_seconds = "5"
tp_place_retry_count = 3
tp_place_retry_interval_seconds = "0.8"
tp_place_retry_backoff_multiplier = "1.5"
tp_rate_limit_fail_action = "HALT_ONLY"

[risk]
rolling_loss_guard_enabled = true
rolling_loss_warn_pct = "0.50"
rolling_loss_soft_halt_pct = "0.10"
order_failure_market_exit_delay_seconds = 1800

[execution]
private_write_min_interval_seconds = "0.6"
max_order_retries = 3

[runtime]
strategy_tick_queue_maxsize = 20000
execution_queue_maxsize = 1000
position_sync_seconds = "5"
account_sync_seconds = "60"
market_tick_heartbeat_seconds = "300"
account_snapshot_stale_warn_seconds = "30"
strategy_tick_lag_warn_seconds = "2"
execution_backlog_log_seconds = "30"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        toml_path = config_dir / "BTC-USDT-SWAP.toml"
        toml_path.write_text(btc_toml_content, encoding="utf-8")
        result = check_symbol_config(
            symbol_config_dir=config_dir,
            inst_id="BTC-USDT-SWAP",
        )
        assert result.inst_id == "BTC-USDT-SWAP"
        assert result.enabled is True
        assert result.live_trading is False


# ---------------------------------------------------------------------------
# 6. BTC live_trading=true succeeds (via temp TOML)
# ---------------------------------------------------------------------------


def test_btc_live_trading_true_temp_toml_succeeds() -> None:
    """A BTC TOML with live_trading=true passes symbol config validation."""
    btc_toml_content = """\
[symbol]
inst_id = "BTC-USDT-SWAP"
enabled = false
live_trading = true

[market]
bar = "15m"
td_mode = "isolated"
pos_side_mode = "net"
contract_value = "0.01"
min_contracts = "0.01"
contract_precision = "0.01"
price_precision = "0.1"
boll_window = 20
boll_std_multiplier = "2.0"
boll_distance_threshold_pct = "0.005"
tp_boll_window = 15
min_outside_pct = "0.0005"

[capital]
dry_run_equity_usdt = "1000"
layer_margin_pct = "0.06"
leverage = "15"
max_layers = 10
layer_multiplier_step = "0.15"

[entry]
add_gap_mode = "linear"
add_gap_base_pct = "0.003"
add_gap_step_pct = "0.001"
first_add_block_seconds = 3600
add_min_interval_seconds = 1800
alert_freeze_seconds = 3600

[cvd]
fast_window_seconds = "5"
price_stall_seconds = "2"
price_stall_tolerance_pct = "0.0005"
burst_window_seconds = "3"
burst_baseline_seconds = "60"
burst_min_move_ratio = "2.5"
burst_min_volume_ratio = "2.0"
burst_min_abs_range_pct = "0.0015"

[tp]
tp_min_net_profit_pct = "0.004"
tp_boll_enabled = true
three_stage_runner_enabled = true
three_stage_tp1_ratio = "0.80"
three_stage_tp2_ratio = "0.10"
three_stage_runner_ratio = "0.10"
three_stage_tp2_use_structure_boll = true
middle_runner_enabled = false
split_tp_enabled = false

[middle_bucket_split]
enabled = false
fast_ratio = "0.70"
fast_sl_enabled = true
fast_sl_fee_buffer_pct = "0.001"

[sidecar]
enabled = false
margin_pct = "0.02"
tp_pct = "0.0044"
skip_first_layer = true
max_legs = 12
order_status_check_seconds = "5"
tp_place_retry_count = 3
tp_place_retry_interval_seconds = "0.8"
tp_place_retry_backoff_multiplier = "1.5"
tp_rate_limit_fail_action = "HALT_ONLY"

[risk]
rolling_loss_guard_enabled = true
rolling_loss_warn_pct = "0.50"
rolling_loss_soft_halt_pct = "0.10"
order_failure_market_exit_delay_seconds = 1800

[execution]
private_write_min_interval_seconds = "0.6"
max_order_retries = 3

[runtime]
strategy_tick_queue_maxsize = 20000
execution_queue_maxsize = 1000
position_sync_seconds = "5"
account_sync_seconds = "60"
market_tick_heartbeat_seconds = "300"
account_snapshot_stale_warn_seconds = "30"
strategy_tick_lag_warn_seconds = "2"
execution_backlog_log_seconds = "30"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        toml_path = config_dir / "BTC-USDT-SWAP.toml"
        toml_path.write_text(btc_toml_content, encoding="utf-8")
        result = check_symbol_config(
            symbol_config_dir=config_dir,
            inst_id="BTC-USDT-SWAP",
        )
        assert result.inst_id == "BTC-USDT-SWAP"
        assert result.enabled is False
        assert result.live_trading is True
        assert result.mapped.trader_preview.live_trading is True


# ---------------------------------------------------------------------------
# 7. to_summary_dict
# ---------------------------------------------------------------------------


def test_to_summary_dict() -> None:
    """to_summary_dict() returns a JSON-serialisable dict with stable fields."""
    result = check_symbol_config(
        symbol_config_dir=_SYMBOLS_DIR,
        inst_id="BTC-USDT-SWAP",
    )
    d = result.to_summary_dict()
    assert d["inst_id"] == "BTC-USDT-SWAP"
    assert isinstance(d["enabled"], bool)
    assert isinstance(d["live_trading"], bool)
    assert d["contract_value"] == "0.01"
    assert d["price_precision"] == "0.1"
    assert isinstance(d["safe_for_config_check_only"], bool)
    tp = d["trader_preview"]
    assert tp["inst_id"] == "BTC-USDT-SWAP"
    assert tp["contract_value"] == "0.01"
    assert tp["live_trading"] is d["live_trading"]
    # Ensure it's valid JSON
    import json
    json.dumps(d)


# ---------------------------------------------------------------------------
# 8. safe_for_config_check_only
# ---------------------------------------------------------------------------


def test_safe_for_config_check_only() -> None:
    """Config check accepts ETH live_trading=true as TOML self-consistency.

    Live worker safety is enforced by startup preflight and SymbolWorkerApp's
    per-symbol live gate, not by the pure config-check preview.
    """
    result = check_symbol_config(
        symbol_config_dir=_SYMBOLS_DIR,
        inst_id="ETH-USDT-SWAP",
    )
    assert result.inst_id == "ETH-USDT-SWAP"
    assert isinstance(result.enabled, bool)
    assert isinstance(result.live_trading, bool)


# ---------------------------------------------------------------------------
# 9. Import source guard — no live/trader/network imports
# ---------------------------------------------------------------------------


def test_config_check_module_source_guard() -> None:
    """config/symbol_config_check.py must NOT import live/trader/network modules."""
    import io
    import tokenize

    source_path = Path(__file__).parent.parent.parent / "config" / "symbol_config_check.py"
    source_bytes = source_path.read_bytes()

    forbidden = {
        "Trader",
        "SymbolWorkerApp",
        "ReclaimSupervisor",
        "OkxPrivateClient",
        "EmailSender",
        "load_dotenv",
        "os.environ",
        "requests",
        "httpx",
        "websocket",
        "asyncio",
    }

    # Extract all NAME tokens from the source; ignore comments and strings.
    names_in_code: set[str] = set()
    try:
        for tok in tokenize.tokenize(io.BytesIO(source_bytes).readline):
            if tok.type == tokenize.NAME:
                names_in_code.add(tok.string)
    except tokenize.TokenError:
        pass  # tolerate incomplete source

    overlap = forbidden & names_in_code
    assert not overlap, (
        f"config/symbol_config_check.py must NOT contain any of {sorted(overlap)} "
        f"as code identifiers — this is a config-check only module"
    )
