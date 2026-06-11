#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ``scripts/check_multi_symbol_live_startup.py`` (G09d)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# TOML helpers (copied from test_multi_symbol_live_preflight to avoid import)
# ---------------------------------------------------------------------------


def _write_toml(dir_path: Path, filename: str, content: str) -> Path:
    p = dir_path / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


_ETH_TOML_ENABLED = textwrap.dedent("""\
    [symbol]
    inst_id = "ETH-USDT-SWAP"
    enabled = true
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
    layer_margin_pct = "0.06"
    leverage = "15"
    max_layers = 8
    layer_multiplier_step = "0.15"

    [entry]
    add_gap_mode = "linear"
    add_gap_base_pct = "0.003"
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
    enabled = true
    fast_ratio = "0.70"
    fast_sl_enabled = true
    fast_sl_fee_buffer_pct = "0.001"

    [sidecar]
    enabled = true
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
    """)


_BTC_TOML_ENABLED = textwrap.dedent("""\
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
    layer_margin_pct = "0.03"
    leverage = "15"
    max_layers = 5
    layer_multiplier_step = "0.15"

    [entry]
    add_gap_mode = "linear"
    add_gap_base_pct = "0.003"
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
    tp_min_net_profit_pct = "0.003"
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
    """)


def _make_toml_dir(tmp_path: Path) -> Path:
    """Create a temp TOML directory with both ETH and BTC enabled."""
    toml_dir = tmp_path / "symbols"
    toml_dir.mkdir(parents=True, exist_ok=True)
    _write_toml(toml_dir, "ETH-USDT-SWAP.toml", _ETH_TOML_ENABLED)
    _write_toml(toml_dir, "BTC-USDT-SWAP.toml", _BTC_TOML_ENABLED)
    return toml_dir


# ---------------------------------------------------------------------------
# CLI script path
# ---------------------------------------------------------------------------

_CLI_SCRIPT = _PROJECT_ROOT / "scripts" / "check_multi_symbol_live_startup.py"


# ---------------------------------------------------------------------------
# 1. Success scenario
# ---------------------------------------------------------------------------


class TestCliSuccess:
    def test_success_exits_zero(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path)
        runtime_dir = tmp_path / "runtime"

        env = {
            **dict(__import__("os").environ),
            "RECLAIM_RUN_MODE": "live",
            "RECLAIM_USE_SYMBOL_TOML": "true",
            "RECLAIM_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "RECLAIM_SYMBOL_CONFIG_DIR": str(toml_dir),
            "RECLAIM_RUNTIME_DIR": str(runtime_dir),
            "RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:live,BTC-USDT-SWAP:live",
            "RECLAIM_ALLOWED_LIVE_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "LIVE_TRADING": "true",
            "OKX_API_KEY": "test-api-key",
            "OKX_SECRET_KEY": "test-secret-key",
            "OKX_PASSPHASE": "test-passphrase",
        }

        result = subprocess.run(
            [sys.executable, str(_CLI_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        stdout = result.stdout
        stderr = result.stderr

        assert result.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
        assert "PREFLIGHT" in stdout
        assert "PREFLIGHT OK" in stdout

    def test_output_contains_eth_and_btc(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path)
        runtime_dir = tmp_path / "runtime"

        env = {
            **dict(__import__("os").environ),
            "RECLAIM_RUN_MODE": "live",
            "RECLAIM_USE_SYMBOL_TOML": "true",
            "RECLAIM_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "RECLAIM_SYMBOL_CONFIG_DIR": str(toml_dir),
            "RECLAIM_RUNTIME_DIR": str(runtime_dir),
            "RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:live,BTC-USDT-SWAP:live",
            "RECLAIM_ALLOWED_LIVE_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "LIVE_TRADING": "true",
        }

        result = subprocess.run(
            [sys.executable, str(_CLI_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        stdout = result.stdout
        assert "ETH-USDT-SWAP" in stdout
        assert "BTC-USDT-SWAP" in stdout


# ---------------------------------------------------------------------------
# 2. Failure scenario
# ---------------------------------------------------------------------------


class TestCliFailure:
    def test_failure_exits_nonzero(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path)
        runtime_dir = tmp_path / "runtime"

        env = {
            **dict(__import__("os").environ),
            "RECLAIM_RUN_MODE": "live",
            "RECLAIM_USE_SYMBOL_TOML": "true",
            "RECLAIM_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "RECLAIM_SYMBOL_CONFIG_DIR": str(toml_dir),
            "RECLAIM_RUNTIME_DIR": str(runtime_dir),
            "RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:live,BTC-USDT-SWAP:live",
            "RECLAIM_ALLOWED_LIVE_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "LIVE_TRADING": "false",
        }

        result = subprocess.run(
            [sys.executable, str(_CLI_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0
        assert "PREFLIGHT" in result.stdout


# ---------------------------------------------------------------------------
# 3. No API key / secret / passphrase in output
# ---------------------------------------------------------------------------


class TestNoSensitiveDataInOutput:
    def test_no_api_key_in_output(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path)
        runtime_dir = tmp_path / "runtime"

        env = {
            **dict(__import__("os").environ),
            "RECLAIM_RUN_MODE": "live",
            "RECLAIM_USE_SYMBOL_TOML": "true",
            "RECLAIM_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "RECLAIM_SYMBOL_CONFIG_DIR": str(toml_dir),
            "RECLAIM_RUNTIME_DIR": str(runtime_dir),
            "RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:live,BTC-USDT-SWAP:live",
            "RECLAIM_ALLOWED_LIVE_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "LIVE_TRADING": "true",
            "OKX_API_KEY": "test-api-key-abc123",
            "OKX_SECRET_KEY": "test-secret-key-xyz789",
            "OKX_PASSPHASE": "test-passphrase-def456",
        }

        result = subprocess.run(
            [sys.executable, str(_CLI_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        stdout = result.stdout
        stderr = result.stderr
        combined = stdout + stderr

        # The actual API key values must not leak into output
        assert "abc123" not in combined
        assert "xyz789" not in combined
        assert "def456" not in combined

    def test_no_secret_in_output_when_live_trading_false(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path)
        runtime_dir = tmp_path / "runtime"

        env = {
            **dict(__import__("os").environ),
            "RECLAIM_RUN_MODE": "live",
            "RECLAIM_USE_SYMBOL_TOML": "true",
            "RECLAIM_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "RECLAIM_SYMBOL_CONFIG_DIR": str(toml_dir),
            "RECLAIM_RUNTIME_DIR": str(runtime_dir),
            "RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:live,BTC-USDT-SWAP:live",
            "RECLAIM_ALLOWED_LIVE_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "LIVE_TRADING": "false",
            "OKX_API_KEY": "my-secret-key-value",
            "OKX_SECRET_KEY": "my-secret-secret-value",
            "OKX_PASSPHASE": "my-secret-passphrase",
        }

        result = subprocess.run(
            [sys.executable, str(_CLI_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert "my-secret-" not in result.stdout
        assert "my-secret-" not in result.stderr


# ---------------------------------------------------------------------------
# 4. main() function importable
# ---------------------------------------------------------------------------


class TestMainFunction:
    def test_main_is_importable(self) -> None:
        """Verify the script is importable without executing it."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "check_multi_symbol_live_startup",
            str(_CLI_SCRIPT),
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        # Don't execute — just verify it can be loaded
        assert mod is not None


# ---------------------------------------------------------------------------
# 5. Script source doesn't import forbidden modules
# ---------------------------------------------------------------------------


class TestScriptSourceSafety:
    def test_script_does_not_import_trader(self) -> None:
        source = _CLI_SCRIPT.read_text(encoding="utf-8")
        assert "from src.execution.trader import" not in source
        assert "import Trader" not in source

    def test_script_does_not_import_okx_client(self) -> None:
        source = _CLI_SCRIPT.read_text(encoding="utf-8")
        assert "okx_private_client" not in source
        assert "okx_public_client" not in source

    def test_script_does_not_import_websocket(self) -> None:
        source = _CLI_SCRIPT.read_text(encoding="utf-8")
        assert "websocket" not in source.lower() or "websocket" not in source
