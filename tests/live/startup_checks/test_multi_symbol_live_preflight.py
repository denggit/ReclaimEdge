#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ``src.live.startup_checks.multi_symbol_live_preflight`` (G09d)."""

from __future__ import annotations

import os
import textwrap
from decimal import Decimal
from pathlib import Path

import pytest

from src.live.startup_checks.multi_symbol_live_preflight import (
    MultiSymbolLivePreflightResult,
    SymbolPreflightResult,
    _symbol_worker_log_dir,
    run_multi_symbol_live_preflight,
)
from src.live.symbol_worker_app import _decimal_equal


# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------


def _write_toml(dir_path: Path, filename: str, content: str) -> Path:
    p = dir_path / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


_ETH_TOML_TEMPLATE = textwrap.dedent("""\
    [symbol]
    inst_id = "ETH-USDT-SWAP"
    enabled = {eth_enabled}
    live_trading = {eth_live_trading}

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


_BTC_TOML_TEMPLATE = textwrap.dedent("""\
    [symbol]
    inst_id = "BTC-USDT-SWAP"
    enabled = {btc_enabled}
    live_trading = {btc_live_trading}

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


def _make_toml_dir(tmp_path: Path, eth_enabled: bool = True, btc_enabled: bool = True,
                   eth_live_trading: bool = False, btc_live_trading: bool = False) -> Path:
    """Create a temporary TOML config directory with ETH and BTC configs."""
    toml_dir = tmp_path / "symbols"
    toml_dir.mkdir(parents=True, exist_ok=True)

    _write_toml(
        toml_dir,
        "ETH-USDT-SWAP.toml",
        _ETH_TOML_TEMPLATE.format(
            eth_enabled=str(eth_enabled).lower(),
            eth_live_trading=str(eth_live_trading).lower(),
        ),
    )
    _write_toml(
        toml_dir,
        "BTC-USDT-SWAP.toml",
        _BTC_TOML_TEMPLATE.format(
            btc_enabled=str(btc_enabled).lower(),
            btc_live_trading=str(btc_live_trading).lower(),
        ),
    )

    return toml_dir


def _base_env(tmp_path: Path, toml_dir: Path, **overrides: str) -> dict[str, str]:
    """Build a base environment dict for preflight testing."""
    runtime_dir = tmp_path / "runtime"
    env: dict[str, str] = {
        "RECLAIM_RUN_MODE": "live",
        "RECLAIM_USE_SYMBOL_TOML": "true",
        "RECLAIM_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
        "RECLAIM_SYMBOL_CONFIG_DIR": str(toml_dir),
        "RECLAIM_RUNTIME_DIR": str(runtime_dir),
        "RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:live,BTC-USDT-SWAP:live",
        "RECLAIM_ALLOWED_LIVE_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
        "LIVE_TRADING": "true",
    }
    env.update(overrides)
    return env


# ---------------------------------------------------------------------------
# 1. _decimal_equal helper
# ---------------------------------------------------------------------------


class TestDecimalEqual:
    def test_same_value_different_representation(self) -> None:
        assert _decimal_equal("50", Decimal("50.0")) is True

    def test_same_value_both_strings(self) -> None:
        assert _decimal_equal("50", "50") is True

    def test_same_value_both_decimals(self) -> None:
        assert _decimal_equal(Decimal("50"), Decimal("50.0")) is True

    def test_different_values(self) -> None:
        assert _decimal_equal("20", Decimal("15")) is False

    def test_none_is_not_equal(self) -> None:
        assert _decimal_equal(None, Decimal("50")) is False

    def test_bool_is_not_equal(self) -> None:
        assert _decimal_equal(True, Decimal("1")) is False

    def test_non_numeric_string(self) -> None:
        assert _decimal_equal("abc", Decimal("50")) is False

    def test_int_and_decimal(self) -> None:
        assert _decimal_equal(50, Decimal("50")) is True


# ---------------------------------------------------------------------------
# 2. _symbol_worker_log_dir helper
# ---------------------------------------------------------------------------


class TestSymbolWorkerLogDir:
    def test_eth_log_dir(self) -> None:
        log_dir = _symbol_worker_log_dir("logs/workers", "ETH-USDT-SWAP")
        assert log_dir == Path("logs/workers/ETH-USDT-SWAP")

    def test_btc_log_dir(self) -> None:
        log_dir = _symbol_worker_log_dir("logs/workers", "BTC-USDT-SWAP")
        assert log_dir == Path("logs/workers/BTC-USDT-SWAP")

    def test_eth_and_btc_are_different(self) -> None:
        eth = _symbol_worker_log_dir("logs/workers", "ETH-USDT-SWAP")
        btc = _symbol_worker_log_dir("logs/workers", "BTC-USDT-SWAP")
        assert eth != btc


# ---------------------------------------------------------------------------
# 3. ETH+BTC live success
# ---------------------------------------------------------------------------


class TestEthBtcLiveSuccess:
    def test_both_enabled_live_success(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env, strict_requested_symbols=True)

        assert result.ok is True
        assert len(result.errors) == 0
        assert result.requested_symbols == ("ETH-USDT-SWAP", "BTC-USDT-SWAP")
        assert result.enabled_symbols == ("ETH-USDT-SWAP", "BTC-USDT-SWAP")
        assert len(result.skipped_disabled_symbols) == 0

        assert len(result.worker_results) == 2
        eth_result = result.worker_results[0]
        btc_result = result.worker_results[1]

        assert eth_result.symbol == "ETH-USDT-SWAP"
        assert eth_result.worker_mode == "live"
        assert eth_result.metadata_ok is True
        assert eth_result.market_settings_ok is True

        assert btc_result.symbol == "BTC-USDT-SWAP"
        assert btc_result.worker_mode == "live"
        assert btc_result.metadata_ok is True
        assert btc_result.market_settings_ok is True

    def test_eth_enabled_btc_enabled_with_allowlist(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(
            tmp_path, toml_dir,
            RECLAIM_ALLOWED_LIVE_SYMBOLS="ETH-USDT-SWAP,BTC-USDT-SWAP",
        )

        result = run_multi_symbol_live_preflight(env=env)

        assert result.ok is True
        assert len(result.worker_results) == 2

    def test_child_env_single_symbol(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env)

        for wr in result.worker_results:
            assert "," not in wr.child_env_symbol
            assert wr.child_env_symbol == wr.symbol
            assert wr.okx_inst_id == wr.symbol

    def test_child_names_different(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env)

        eth_result = result.worker_results[0]
        btc_result = result.worker_results[1]
        # child_name is not stored on SymbolPreflightResult, but if they were
        # the same we'd see a path collision error
        assert eth_result.heartbeat_path != btc_result.heartbeat_path
        assert eth_result.event_outbox_path != btc_result.event_outbox_path
        assert eth_result.log_dir != btc_result.log_dir

    def test_heartbeat_paths_different(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env)

        eth = result.worker_results[0]
        btc = result.worker_results[1]
        assert eth.heartbeat_path != btc.heartbeat_path

    def test_event_outbox_paths_different(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env)

        eth = result.worker_results[0]
        btc = result.worker_results[1]
        assert eth.event_outbox_path != btc.event_outbox_path

    def test_log_paths_different(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env)

        eth = result.worker_results[0]
        btc = result.worker_results[1]
        assert eth.log_dir != btc.log_dir


# ---------------------------------------------------------------------------
# 4. BTC requested but disabled
# ---------------------------------------------------------------------------


class TestBtcRequestedButDisabled:
    def test_strict_mode_errors_on_disabled_btc(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=False)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env, strict_requested_symbols=True)

        assert result.ok is False
        assert "BTC-USDT-SWAP" in result.skipped_disabled_symbols
        assert any("disabled" in e.lower() and "BTC-USDT-SWAP" in e
                   for e in result.errors)

    def test_non_strict_mode_warns_on_disabled_btc(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=False)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env, strict_requested_symbols=False)

        # In non-strict mode, disabled is a warning, not an error
        # But the preflight still won't have BTC worker results because
        # build_symbol_worker_plans only gets enabled symbols
        assert "BTC-USDT-SWAP" in result.skipped_disabled_symbols
        assert any("disabled" in w.lower() or "skipped" in w.lower()
                   for w in result.warnings)


# ---------------------------------------------------------------------------
# 5. LIVE_TRADING=false but mode=live
# ---------------------------------------------------------------------------


class TestLiveTradingFalse:
    def test_live_trading_false_with_live_mode_fails(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir, LIVE_TRADING="false")

        result = run_multi_symbol_live_preflight(env=env)

        assert result.ok is False
        assert any("LIVE_TRADING" in e for e in result.errors)

    def test_live_trading_not_set_with_live_mode_fails(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)
        del env["LIVE_TRADING"]

        result = run_multi_symbol_live_preflight(env=env)

        assert result.ok is False
        assert any("LIVE_TRADING" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 6. BTC live not in allowlist
# ---------------------------------------------------------------------------


class TestBtcNotInAllowlist:
    def test_btc_live_not_in_allowlist_fails(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(
            tmp_path, toml_dir,
            RECLAIM_ALLOWED_LIVE_SYMBOLS="ETH-USDT-SWAP",
        )

        result = run_multi_symbol_live_preflight(env=env)

        assert result.ok is False
        assert any(
            "BTC-USDT-SWAP" in e and "RECLAIM_ALLOWED_LIVE_SYMBOLS" in e
            for e in result.errors
        )


# ---------------------------------------------------------------------------
# 7. Unsupported allowlist symbol
# ---------------------------------------------------------------------------


class TestUnsupportedAllowlistSymbol:
    def test_unsupported_symbol_in_allowlist_errors(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(
            tmp_path, toml_dir,
            RECLAIM_ALLOWED_LIVE_SYMBOLS="ETH-USDT-SWAP,BTC-USDT-SWAP,SOL-USDT-SWAP",
        )

        result = run_multi_symbol_live_preflight(env=env)

        assert result.ok is False
        assert any(
            "SOL-USDT-SWAP" in e and "unsupported" in e.lower()
            for e in result.errors
        )

    def test_wildcard_in_allowlist_fails(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(
            tmp_path, toml_dir,
            RECLAIM_ALLOWED_LIVE_SYMBOLS="*",
        )

        result = run_multi_symbol_live_preflight(env=env)

        assert result.ok is False
        assert any(
            "RECLAIM_ALLOWED_LIVE_SYMBOLS" in e
            for e in result.errors
        )


# ---------------------------------------------------------------------------
# 8. BTC metadata / market settings can be built
# ---------------------------------------------------------------------------


class TestBtcMetadataAndSettings:
    def test_btc_metadata_contract_multiplier_from_toml(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        from config.symbol_config_loader import load_symbol_config_from_dir
        from src.live.symbol_trader_config import (
            build_trader_instrument_metadata,
        )

        btc_config = load_symbol_config_from_dir(toml_dir, "BTC-USDT-SWAP")
        metadata = build_trader_instrument_metadata(btc_config)

        # BTC TOML has contract_value = "0.01"
        assert metadata.contract_multiplier == Decimal("0.01")
        assert metadata.inst_id == "BTC-USDT-SWAP"
        assert metadata.contract_precision == Decimal("0.01")
        assert metadata.min_contracts == Decimal("0.01")

    def test_btc_market_settings_leverage_from_toml(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        from config.symbol_config_loader import load_symbol_config_from_dir
        from src.live.symbol_trader_config import (
            build_trader_market_settings,
        )

        btc_config = load_symbol_config_from_dir(toml_dir, "BTC-USDT-SWAP")
        market_settings = build_trader_market_settings(btc_config)

        # BTC TOML has leverage = "15"
        assert market_settings.leverage == Decimal("15")
        assert market_settings.inst_id == "BTC-USDT-SWAP"
        assert market_settings.td_mode == "isolated"
        assert market_settings.pos_side_mode == "net"

    def test_preflight_result_shows_btc_metadata_ok(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env)

        btc_result = [wr for wr in result.worker_results
                      if wr.symbol == "BTC-USDT-SWAP"][0]
        assert btc_result.metadata_ok is True
        assert btc_result.market_settings_ok is True


# ---------------------------------------------------------------------------
# 9. Path conflict detection
# ---------------------------------------------------------------------------


class TestPathConflicts:
    def test_different_symbols_have_different_state_files(self, tmp_path: Path) -> None:
        from src.live.runtime_paths import build_runtime_paths

        runtime_dir = tmp_path / "runtime"
        eth_rp = build_runtime_paths(runtime_dir, "ETH-USDT-SWAP")
        btc_rp = build_runtime_paths(runtime_dir, "BTC-USDT-SWAP")

        assert eth_rp.state_file != btc_rp.state_file
        assert eth_rp.journal_file != btc_rp.journal_file
        assert eth_rp.heartbeat_file != btc_rp.heartbeat_file
        assert eth_rp.worker_event_outbox_file != btc_rp.worker_event_outbox_file
        assert eth_rp.log_file != btc_rp.log_file

    def test_same_symbol_would_collide(self, tmp_path: Path) -> None:
        from src.live.runtime_paths import build_runtime_paths

        runtime_dir = tmp_path / "runtime"
        rp1 = build_runtime_paths(runtime_dir, "ETH-USDT-SWAP")
        rp2 = build_runtime_paths(runtime_dir, "ETH-USDT-SWAP")

        assert rp1.state_file == rp2.state_file
        assert rp1.journal_file == rp2.journal_file


# ---------------------------------------------------------------------------
# 10. No env loads correctly
# ---------------------------------------------------------------------------


class TestEnvLoadFailure:
    def test_empty_env_fails_gracefully(self) -> None:
        """An empty env should fail because build_symbol_worker_plans requires symbols."""
        # With empty env, the default RECLAIM_SYMBOLS will be "ETH-USDT-SWAP"
        # But the RECLAIM_SYMBOL_CONFIG_DIR default "config/symbols" will try to
        # read real TOML files, and ETH TOML is disabled in the repo.
        # This test only validates that the result is returned (no crash).
        result = run_multi_symbol_live_preflight(env={})
        assert isinstance(result, MultiSymbolLivePreflightResult)
        # Will likely have errors due to live_trading not being true, etc.
        assert result.ok is False


# ---------------------------------------------------------------------------
# 11. Only ETH enabled → preflight with BTC disabled
# ---------------------------------------------------------------------------


class TestOnlyEthEnabled:
    def test_eth_only_enabled_btc_disabled_strict(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=False)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env, strict_requested_symbols=True)

        assert result.ok is False
        assert len(result.skipped_disabled_symbols) >= 1
        assert "BTC-USDT-SWAP" in result.skipped_disabled_symbols

    def test_eth_only_enabled_produces_one_worker(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=False)
        env = _base_env(tmp_path, toml_dir)
        # Also allow non-strict to get worker results
        env_strict_false = dict(env)

        result = run_multi_symbol_live_preflight(
            env=env_strict_false, strict_requested_symbols=False,
        )

        # With non-strict mode, only ETH worker is built (because only ETH enabled)
        eth_workers = [wr for wr in result.worker_results
                       if wr.symbol == "ETH-USDT-SWAP"]
        assert len(eth_workers) == 1
        assert eth_workers[0].enabled is True


# ---------------------------------------------------------------------------
# 12. No enabled symbols
# ---------------------------------------------------------------------------


class TestNoEnabledSymbols:
    def test_all_disabled_produces_error(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=False, btc_enabled=False)
        env = _base_env(tmp_path, toml_dir)

        result = run_multi_symbol_live_preflight(env=env, strict_requested_symbols=False)

        assert result.ok is False
        assert any("No enabled symbols" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 13. Worker mode validation
# ---------------------------------------------------------------------------


class TestWorkerModeValidation:
    def test_paper_mode_does_not_require_live_trading(self, tmp_path: Path) -> None:
        """Paper mode workers shouldn't enforce LIVE_TRADING check."""
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(
            tmp_path, toml_dir,
            RECLAIM_WORKER_MODES="ETH-USDT-SWAP:paper,BTC-USDT-SWAP:paper",
            LIVE_TRADING="false",
        )

        result = run_multi_symbol_live_preflight(env=env)

        # Paper mode doesn't trigger LIVE_TRADING error
        assert not any("LIVE_TRADING" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 14. Side effect safety
# ---------------------------------------------------------------------------


class TestNoSideEffects:
    def test_preflight_does_not_mutate_env(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        env_copy = dict(env)
        run_multi_symbol_live_preflight(env=env)

        assert env == env_copy

    def test_preflight_does_not_create_runtime_files(self, tmp_path: Path) -> None:
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(tmp_path, toml_dir)

        run_multi_symbol_live_preflight(env=env)

        runtime_dir = tmp_path / "runtime"
        # Check that no files were created
        assert not runtime_dir.exists() or not any(runtime_dir.iterdir())


# ---------------------------------------------------------------------------
# 15. Invalid RECLAIM_WORKER_MODES must not traceback
# ---------------------------------------------------------------------------


class TestInvalidWorkerModes:
    def test_invalid_mode_returns_errors_not_traceback(self, tmp_path: Path) -> None:
        """G09d-fix: parse_worker_modes with an invalid mode must not raise —
        the error must land in result.errors instead.
        """
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(
            tmp_path, toml_dir,
            RECLAIM_WORKER_MODES="ETH-USDT-SWAP:not-a-mode",
        )

        result = run_multi_symbol_live_preflight(env=env)

        assert result.ok is False
        assert any("RECLAIM_WORKER_MODES" in e for e in result.errors), (
            f"Expected RECLAIM_WORKER_MODES in errors, got: {result.errors}"
        )

    def test_invalid_format_returns_errors_not_traceback(self, tmp_path: Path) -> None:
        """Missing colon in RECLAIM_WORKER_MODES entry should also be caught."""
        toml_dir = _make_toml_dir(tmp_path, eth_enabled=True, btc_enabled=True)
        env = _base_env(
            tmp_path, toml_dir,
            RECLAIM_WORKER_MODES="ETH-USDT-SWAP",
        )

        result = run_multi_symbol_live_preflight(env=env)

        assert result.ok is False
        assert any("RECLAIM_WORKER_MODES" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 15. Result type consistency
# ---------------------------------------------------------------------------


class TestResultTypes:
    def test_symbol_preflight_result_is_frozen(self) -> None:
        r = SymbolPreflightResult(
            symbol="ETH-USDT-SWAP",
            worker_mode="live",
            enabled=True,
            live_trading=True,
            child_env_symbol="ETH-USDT-SWAP",
            okx_inst_id="ETH-USDT-SWAP",
            runtime_dir=Path("runtime"),
            heartbeat_path=Path("heartbeat.json"),
            event_outbox_path=Path("outbox.jsonl"),
            log_dir=Path("logs/workers/ETH-USDT-SWAP"),
            metadata_ok=True,
            market_settings_ok=True,
            sidecar_enabled=True,
        )
        with pytest.raises(Exception):
            r.metadata_ok = False  # type: ignore[misc]

    def test_multi_symbol_preflight_result_is_frozen(self) -> None:
        r = MultiSymbolLivePreflightResult(
            ok=True,
            requested_symbols=("ETH-USDT-SWAP",),
            enabled_symbols=("ETH-USDT-SWAP",),
            skipped_disabled_symbols=(),
            worker_results=(),
            errors=(),
            warnings=(),
        )
        with pytest.raises(Exception):
            r.ok = False  # type: ignore[misc]
