#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F04 tests for ``src.live.supervisor.symbol_selection``."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.symbol_config_loader import load_symbol_config_from_dir
from config.symbol_config_validator import SymbolConfigValidationError
from src.live.supervisor.symbol_selection import (
    SupervisorSymbolSelection,
    require_single_enabled_symbol,
    select_enabled_supervisor_symbols,
)


# ---------------------------------------------------------------------------
# Helpers — write minimal TOML files for testing
# ---------------------------------------------------------------------------


def _write_toml(dir_path: Path, inst_id: str, *, enabled: bool = True) -> Path:
    """Write a minimal, validator-compliant TOML file for *inst_id*."""
    toml_path = dir_path / f"{inst_id}.toml"
    # Use the real ETH-USDT-SWAP TOML as a base to pass full validation.
    # We copy-paste a minimal valid config that passes the validator.
    content = f"""\
[symbol]
inst_id = "{inst_id}"
enabled = {str(enabled).lower()}
live_trading = false

[market]
bar = "15m"
td_mode = "isolated"
pos_side_mode = "net"
contract_value = "0.1"
min_contracts = "0.01"
contract_precision = "0.01"
price_precision = "0.01"
boll_window = 20
boll_std_multiplier = "2.0"
boll_distance_threshold_pct = "0.005"
tp_boll_window = 15
min_outside_pct = "0.0005"

[capital]
dry_run_equity_usdt = "1000"
layer_margin_pct = "0.03"
leverage = "50"
max_layers = 3
layer_multiplier_step = "0.15"

[entry]
add_gap_mode = "linear"
add_gap_base_pct = "0.006"
add_gap_step_pct = "0.001"
add_freeze_seconds = 3600
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
tp_min_net_profit_pct = "0.002"
tp_boll_enabled = true
three_stage_runner_enabled = true
three_stage_tp1_ratio = "0.70"
three_stage_tp2_ratio = "0.20"
three_stage_runner_ratio = "0.10"
three_stage_tp2_use_structure_boll = true
middle_runner_enabled = false
split_tp_enabled = false

[middle_bucket_split]
enabled = false
fast_ratio = "0.60"
fast_sl_enabled = true
fast_sl_fee_buffer_pct = "0.001"

[sidecar]
enabled = false
margin_pct = "0.01"
tp_pct = "0.004"
skip_first_layer = true
max_legs = 10
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
market_tick_heartbeat_seconds = "60"
account_snapshot_stale_warn_seconds = "30"
strategy_tick_lag_warn_seconds = "2"
execution_backlog_log_seconds = "30"
"""
    toml_path.write_text(content, encoding="utf-8")
    return toml_path


# ---------------------------------------------------------------------------
# 1. ETH only → enabled=(ETH,), skipped=()
# ---------------------------------------------------------------------------


def test_select_eth_only_enabled(tmp_path: Path) -> None:
    """symbols=(ETH,) → enabled=(ETH,), skipped=()."""
    _write_toml(tmp_path, "ETH-USDT-SWAP", enabled=True)

    selection = select_enabled_supervisor_symbols(
        symbols=("ETH-USDT-SWAP",),
        symbol_config_dir=tmp_path,
    )
    assert selection.requested_symbols == ("ETH-USDT-SWAP",)
    assert selection.enabled_symbols == ("ETH-USDT-SWAP",)
    assert selection.skipped_disabled_symbols == ()


def test_require_single_eth_returns_eth(tmp_path: Path) -> None:
    _write_toml(tmp_path, "ETH-USDT-SWAP", enabled=True)
    selection = select_enabled_supervisor_symbols(
        symbols=("ETH-USDT-SWAP",),
        symbol_config_dir=tmp_path,
    )
    assert require_single_enabled_symbol(selection) == "ETH-USDT-SWAP"


# ---------------------------------------------------------------------------
# 2. ETH + BTC (BTC disabled) → enabled=(ETH,), skipped=(BTC,)
# ---------------------------------------------------------------------------


def test_select_eth_and_btc_skips_disabled_btc(tmp_path: Path) -> None:
    """symbols=(ETH,BTC) with BTC enabled=false → enabled=(ETH,), skipped=(BTC,)."""
    _write_toml(tmp_path, "ETH-USDT-SWAP", enabled=True)
    _write_toml(tmp_path, "BTC-USDT-SWAP", enabled=False)

    selection = select_enabled_supervisor_symbols(
        symbols=("ETH-USDT-SWAP", "BTC-USDT-SWAP"),
        symbol_config_dir=tmp_path,
    )
    assert selection.requested_symbols == ("ETH-USDT-SWAP", "BTC-USDT-SWAP")
    assert selection.enabled_symbols == ("ETH-USDT-SWAP",)
    assert selection.skipped_disabled_symbols == ("BTC-USDT-SWAP",)

    assert require_single_enabled_symbol(selection) == "ETH-USDT-SWAP"


# ---------------------------------------------------------------------------
# 3. BTC only → enabled=(), skipped=(BTC,), require_single raises
# ---------------------------------------------------------------------------


def test_select_btc_only_no_enabled_symbols(tmp_path: Path) -> None:
    """symbols=(BTC,) with BTC enabled=false → enabled=(), skipped=(BTC,)."""
    _write_toml(tmp_path, "BTC-USDT-SWAP", enabled=False)

    selection = select_enabled_supervisor_symbols(
        symbols=("BTC-USDT-SWAP",),
        symbol_config_dir=tmp_path,
    )
    assert selection.requested_symbols == ("BTC-USDT-SWAP",)
    assert selection.enabled_symbols == ()
    assert selection.skipped_disabled_symbols == ("BTC-USDT-SWAP",)

    with pytest.raises(RuntimeError, match="No enabled symbols"):
        require_single_enabled_symbol(selection)


# ---------------------------------------------------------------------------
# 4. Two enabled → require_single raises
# ---------------------------------------------------------------------------


def test_two_enabled_require_single_raises() -> None:
    """When two symbols are in enabled_symbols, require_single_enabled_symbol raises.
    Tested directly via SupervisorSymbolSelection because the validator
    currently hard-rejects BTC enabled=true — this code path will be
    exercised via the loader once BTC is allowed to be enabled.
    """
    selection = SupervisorSymbolSelection(
        requested_symbols=("ETH-USDT-SWAP", "BTC-USDT-SWAP"),
        enabled_symbols=("ETH-USDT-SWAP", "BTC-USDT-SWAP"),
        skipped_disabled_symbols=(),
    )

    with pytest.raises(RuntimeError, match="Multiple enabled symbols"):
        require_single_enabled_symbol(selection)


# ---------------------------------------------------------------------------
# 5. Error: empty symbols
# ---------------------------------------------------------------------------


def test_empty_symbols_raises() -> None:
    with pytest.raises(ValueError, match="symbols must not be empty"):
        select_enabled_supervisor_symbols(
            symbols=(),
            symbol_config_dir=Path("/nonexistent"),
        )


# ---------------------------------------------------------------------------
# 6. Error: missing TOML
# ---------------------------------------------------------------------------


def test_missing_toml_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        select_enabled_supervisor_symbols(
            symbols=("ETH-USDT-SWAP",),
            symbol_config_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# 7. Error: unsupported symbol via validator
# ---------------------------------------------------------------------------


def test_unsupported_symbol_rejected_by_validator(tmp_path: Path) -> None:
    """Validator rejects SOL-USDT-SWAP (not in SUPPORTED_SYMBOLS_AT_THIS_STAGE)."""
    _write_toml(tmp_path, "SOL-USDT-SWAP", enabled=True)

    with pytest.raises(SymbolConfigValidationError, match="unsupported symbol"):
        select_enabled_supervisor_symbols(
            symbols=("SOL-USDT-SWAP",),
            symbol_config_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# 8. SupervisorSymbolSelection is frozen
# ---------------------------------------------------------------------------


def test_supervisor_symbol_selection_is_frozen() -> None:
    sel = SupervisorSymbolSelection(
        requested_symbols=("ETH-USDT-SWAP",),
        enabled_symbols=("ETH-USDT-SWAP",),
        skipped_disabled_symbols=(),
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        sel.enabled_symbols = ("BTC-USDT-SWAP",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 9. Error messages contain diagnostic info
# ---------------------------------------------------------------------------


def test_no_enabled_error_contains_diagnostics(tmp_path: Path) -> None:
    _write_toml(tmp_path, "BTC-USDT-SWAP", enabled=False)

    selection = select_enabled_supervisor_symbols(
        symbols=("BTC-USDT-SWAP",),
        symbol_config_dir=tmp_path,
    )
    with pytest.raises(RuntimeError) as exc_info:
        require_single_enabled_symbol(selection)
    msg = str(exc_info.value)
    assert "requested_symbols" in msg
    assert "enabled_symbols" in msg
    assert "skipped_disabled_symbols" in msg
    assert "BTC-USDT-SWAP" in msg


def test_multiple_enabled_error_contains_diagnostics() -> None:
    """Error message for multiple enabled symbols contains diagnostic info."""
    selection = SupervisorSymbolSelection(
        requested_symbols=("ETH-USDT-SWAP", "BTC-USDT-SWAP"),
        enabled_symbols=("ETH-USDT-SWAP", "BTC-USDT-SWAP"),
        skipped_disabled_symbols=(),
    )
    with pytest.raises(RuntimeError) as exc_info:
        require_single_enabled_symbol(selection)
    msg = str(exc_info.value)
    assert "requested_symbols" in msg
    assert "enabled_symbols" in msg
    assert "skipped_disabled_symbols" in msg
    assert "Multiple enabled symbols" in msg


# ---------------------------------------------------------------------------
# 10. No forbidden imports in selection module
# ---------------------------------------------------------------------------


def test_symbol_selection_no_forbidden_imports() -> None:
    """symbol_selection.py must NOT import Trader, worker, strategy, or network modules."""
    source_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "src" / "live" / "supervisor" / "symbol_selection.py"
    )
    source = source_path.read_text(encoding="utf-8")

    forbidden = [
        "Trader",
        "SymbolWorkerApp",
        "SymbolWorkerFactory",
        "strategy",
        "okx",
        "requests",
        "httpx",
        "websocket",
        "multiprocessing",
        "subprocess",
        "send_email",
    ]
    for token in forbidden:
        assert token not in source, (
            f"symbol_selection.py must NOT contain {token!r}"
        )
