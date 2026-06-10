#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for SymbolConfig mapper (A06)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from config.symbol_config import SymbolConfig, SymbolIdentityConfig
from config.symbol_config_loader import load_symbol_config_from_dir
from config.symbol_config_mapper import (
    MappedSymbolConfigs,
    SymbolTraderConfigPreview,
    decimal_to_float,
    map_symbol_config,
    to_boll_monitor_config,
    to_cvd_tracker_config,
    to_position_sizer_config,
    to_strategy_config,
    to_trader_config_preview,
)
from config.symbol_config_validator import SymbolConfigValidationError
from src.indicators.cvd_tracker import CvdTrackerConfig
from src.monitors.boll_band_breakout_monitor import BollBandBreakoutMonitorConfig
from src.risk.simple_position_sizer import SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SYMBOLS_DIR = _PROJECT_ROOT / "config" / "symbols"


# ---------------------------------------------------------------------------
# 1. decimal_to_float
# ---------------------------------------------------------------------------


def test_decimal_to_float() -> None:
    """Decimal("0.1") → 0.1, non-Decimal raises TypeError."""
    assert decimal_to_float(Decimal("0.1")) == 0.1
    with pytest.raises(TypeError):
        decimal_to_float("0.1")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. to_boll_monitor_config
# ---------------------------------------------------------------------------


def test_to_boll_monitor_config_default_eth() -> None:
    config = SymbolConfig.default_eth()
    mapped = to_boll_monitor_config(config)
    assert isinstance(mapped, BollBandBreakoutMonitorConfig)
    assert mapped.inst_id == "ETH-USDT-SWAP"
    assert mapped.bar == "15m"
    assert mapped.boll_window == 20
    assert mapped.boll_std_multiplier == 2.0
    assert mapped.band_distance_threshold_pct == 0.005
    assert mapped.alert_freeze_seconds == 3600
    assert mapped.tp_boll_enabled is True
    assert mapped.tp_boll_window == 15
    # Unmapped fields keep their defaults:
    assert mapped.use_live_candle is True
    assert mapped.boll_recalc_seconds == 1.0


# ---------------------------------------------------------------------------
# 3. to_cvd_tracker_config
# ---------------------------------------------------------------------------


def test_to_cvd_tracker_config_default_eth() -> None:
    config = SymbolConfig.default_eth()
    mapped = to_cvd_tracker_config(config)
    assert isinstance(mapped, CvdTrackerConfig)
    assert mapped.fast_window_seconds == 5.0
    assert mapped.price_stall_seconds == 2.0
    assert mapped.price_stall_tolerance_pct == 0.0005
    assert mapped.burst_window_seconds == 3.0
    assert mapped.burst_baseline_seconds == 60.0
    assert mapped.burst_min_move_ratio == 2.5
    assert mapped.burst_min_volume_ratio == 2.0
    assert mapped.burst_min_abs_range_pct == 0.0015


# ---------------------------------------------------------------------------
# 4. to_position_sizer_config
# ---------------------------------------------------------------------------


def test_to_position_sizer_config_default_eth() -> None:
    config = SymbolConfig.default_eth()
    mapped = to_position_sizer_config(config)
    assert isinstance(mapped, SimplePositionSizerConfig)
    assert mapped.dry_run_equity_usdt == 1000.0
    assert mapped.layer_margin_pct == 0.03
    assert mapped.leverage == 50.0
    assert mapped.layer_multiplier_step == 0.15
    assert mapped.sidecar_enabled is False
    assert mapped.sidecar_margin_pct == 0.01
    assert mapped.sidecar_tp_pct == 0.004
    assert mapped.sidecar_close_when_core_flat is True
    assert mapped.sidecar_order_status_check_seconds == 5.0
    assert mapped.sidecar_max_legs == 10
    assert mapped.sidecar_skip_first_layer is True


# ---------------------------------------------------------------------------
# 5. to_strategy_config
# ---------------------------------------------------------------------------


def test_to_strategy_config_default_eth() -> None:
    config = SymbolConfig.default_eth()
    mapped = to_strategy_config(config)
    assert isinstance(mapped, BollCvdReclaimStrategyConfig)
    assert mapped.add_layer_gap_pct == 0.006
    assert mapped.max_layers == 3
    assert mapped.tp_min_net_profit_pct == 0.002
    assert mapped.tp_boll_enabled is True
    assert mapped.tp_boll_window == 15
    assert mapped.three_stage_runner_enabled is True
    assert mapped.three_stage_tp1_ratio == 0.70
    assert mapped.three_stage_tp2_ratio == 0.20
    assert mapped.three_stage_runner_ratio == 0.10
    assert mapped.three_stage_tp2_use_structure_boll is True
    assert mapped.middle_runner_enabled is False
    assert mapped.middle_bucket_split_enabled is False
    assert mapped.middle_bucket_split_fast_ratio == 0.60
    assert mapped.middle_bucket_split_fast_sl_enabled is True
    assert mapped.middle_bucket_split_fast_sl_fee_buffer_pct == 0.001
    assert mapped.min_outside_pct == 0.0005
    assert mapped.first_add_block_seconds == 3600
    assert mapped.add_min_interval_seconds == 1800
    assert mapped.split_tp_enabled is False


# ---------------------------------------------------------------------------
# 6. to_trader_config_preview
# ---------------------------------------------------------------------------


def test_to_trader_config_preview_default_eth() -> None:
    config = SymbolConfig.default_eth()
    mapped = to_trader_config_preview(config)
    assert isinstance(mapped, SymbolTraderConfigPreview)
    assert mapped.inst_id == "ETH-USDT-SWAP"
    assert mapped.td_mode == "isolated"
    assert mapped.pos_side_mode == "net"
    assert mapped.leverage == "50"
    assert mapped.contract_value == Decimal("0.1")
    assert mapped.contract_precision == Decimal("0.01")
    assert mapped.min_contracts == Decimal("0.01")
    assert mapped.live_trading is False


# ---------------------------------------------------------------------------
# 7. map_symbol_config (aggregate)
# ---------------------------------------------------------------------------


def test_map_symbol_config_default_eth() -> None:
    mapped = map_symbol_config(SymbolConfig.default_eth())
    assert isinstance(mapped, MappedSymbolConfigs)
    assert mapped.monitor.inst_id == "ETH-USDT-SWAP"
    assert mapped.cvd.fast_window_seconds == 5.0
    assert mapped.position_sizer.layer_multiplier_step == 0.15
    assert mapped.strategy.three_stage_tp2_use_structure_boll is True
    assert mapped.trader_preview.live_trading is False


# ---------------------------------------------------------------------------
# 8. Loaded ETH TOML maps successfully
# ---------------------------------------------------------------------------


def test_loaded_eth_toml_maps_successfully() -> None:
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")
    mapped = map_symbol_config(config)
    assert mapped.monitor.inst_id == "ETH-USDT-SWAP"
    assert mapped.cvd.fast_window_seconds == 5.0
    assert mapped.position_sizer.layer_multiplier_step == 0.15
    assert mapped.strategy.three_stage_tp2_use_structure_boll is True
    assert mapped.trader_preview.live_trading is False


# ---------------------------------------------------------------------------
# 9. Mapper rejects invalid SymbolConfig
# ---------------------------------------------------------------------------


def test_mapper_rejects_invalid_symbol_config() -> None:
    """Any mapper must raise SymbolConfigValidationError on an invalid config."""
    # Build a config that is clearly invalid — SOL-USDT-SWAP is not supported.
    invalid = SymbolConfig(
        symbol=SymbolIdentityConfig(inst_id="SOL-USDT-SWAP"),
    )
    with pytest.raises(SymbolConfigValidationError):
        to_boll_monitor_config(invalid)
    with pytest.raises(SymbolConfigValidationError):
        to_cvd_tracker_config(invalid)
    with pytest.raises(SymbolConfigValidationError):
        to_position_sizer_config(invalid)
    with pytest.raises(SymbolConfigValidationError):
        to_strategy_config(invalid)
    with pytest.raises(SymbolConfigValidationError):
        to_trader_config_preview(invalid)
    with pytest.raises(SymbolConfigValidationError):
        map_symbol_config(invalid)


# ---------------------------------------------------------------------------
# 10. BTC mapper — preview only (F03)
# ---------------------------------------------------------------------------


def test_btc_mapper_preview_only() -> None:
    """BTC config can be mapped for preview only; no Trader wiring."""
    config = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "BTC-USDT-SWAP")
    mapped = map_symbol_config(config)
    assert isinstance(mapped, MappedSymbolConfigs)
    assert mapped.trader_preview.inst_id == "BTC-USDT-SWAP"
    assert mapped.trader_preview.contract_value == Decimal("0.01")
    assert mapped.trader_preview.contract_precision == Decimal("0.01")
    assert mapped.trader_preview.min_contracts == Decimal("0.01")
    assert mapped.trader_preview.live_trading is False
