#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SymbolConfig mapper (A06).

Maps a ``SymbolConfig`` onto the existing runtime config dataclasses
used by monitors, CVD tracker, position sizer and strategy.

Design rules
------------
* Pure functions — no file I/O, no env reads, no network, no side-effects.
* Every public mapper entry point calls ``validate_symbol_config`` first.
* Only config *dataclasses* are created — never monitor / tracker / sizer /
  strategy / Trader *instances*.
* Only fields that exist in ``SymbolConfig`` are mapped; all other fields
  keep their target dataclass defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from config.symbol_config import SymbolConfig
from config.symbol_config_validator import validate_symbol_config
from src.indicators.cvd_tracker import CvdTrackerConfig
from src.monitors.boll_band_breakout_monitor import BollBandBreakoutMonitorConfig
from src.risk.simple_position_sizer import SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def decimal_to_float(value: Decimal) -> float:
    """Convert a ``Decimal`` to ``float``.

    Raises ``TypeError`` if *value* is not a ``Decimal``.
    """
    if not isinstance(value, Decimal):
        raise TypeError(
            f"decimal_to_float expects Decimal, got {type(value).__name__}"
        )
    return float(value)


# ---------------------------------------------------------------------------
# Per-target mappers
# ---------------------------------------------------------------------------


def to_boll_monitor_config(
    config: SymbolConfig,
) -> BollBandBreakoutMonitorConfig:
    """Map *config* → ``BollBandBreakoutMonitorConfig``."""
    validate_symbol_config(config)
    return BollBandBreakoutMonitorConfig(
        inst_id=config.symbol.inst_id,
        bar=config.market.bar,
        boll_window=config.market.boll_window,
        boll_std_multiplier=decimal_to_float(config.market.boll_std_multiplier),
        band_distance_threshold_pct=decimal_to_float(
            config.market.boll_distance_threshold_pct
        ),
        alert_freeze_seconds=config.entry.alert_freeze_seconds,
        tp_boll_enabled=config.tp.tp_boll_enabled,
        tp_boll_window=config.market.tp_boll_window,
    )


def to_cvd_tracker_config(config: SymbolConfig) -> CvdTrackerConfig:
    """Map *config* → ``CvdTrackerConfig``."""
    validate_symbol_config(config)
    return CvdTrackerConfig(
        fast_window_seconds=decimal_to_float(config.cvd.fast_window_seconds),
        price_stall_seconds=decimal_to_float(config.cvd.price_stall_seconds),
        price_stall_tolerance_pct=decimal_to_float(
            config.cvd.price_stall_tolerance_pct
        ),
        burst_window_seconds=decimal_to_float(config.cvd.burst_window_seconds),
        burst_baseline_seconds=decimal_to_float(config.cvd.burst_baseline_seconds),
        burst_min_move_ratio=decimal_to_float(config.cvd.burst_min_move_ratio),
        burst_min_volume_ratio=decimal_to_float(config.cvd.burst_min_volume_ratio),
        burst_min_abs_range_pct=decimal_to_float(
            config.cvd.burst_min_abs_range_pct
        ),
    )


def to_position_sizer_config(
    config: SymbolConfig,
) -> SimplePositionSizerConfig:
    """Map *config* → ``SimplePositionSizerConfig``."""
    validate_symbol_config(config)
    return SimplePositionSizerConfig(
        dry_run_equity_usdt=decimal_to_float(config.capital.dry_run_equity_usdt),
        layer_margin_pct=decimal_to_float(config.capital.layer_margin_pct),
        leverage=decimal_to_float(config.capital.leverage),
        layer_multiplier_step=decimal_to_float(config.capital.layer_multiplier_step),
        sidecar_enabled=config.sidecar.enabled,
        sidecar_margin_pct=decimal_to_float(config.sidecar.margin_pct),
        sidecar_tp_pct=decimal_to_float(config.sidecar.tp_pct),
        sidecar_close_when_core_flat=True,
        sidecar_order_status_check_seconds=decimal_to_float(
            config.sidecar.order_status_check_seconds
        ),
        sidecar_max_legs=config.sidecar.max_legs,
        sidecar_skip_first_layer=config.sidecar.skip_first_layer,
    )


def to_strategy_config(
    config: SymbolConfig,
) -> BollCvdReclaimStrategyConfig:
    """Map *config* → ``BollCvdReclaimStrategyConfig``.

    Only fields covered by ``SymbolConfig`` are mapped; all other fields
    keep their ``BollCvdReclaimStrategyConfig`` defaults.
    """
    validate_symbol_config(config)
    return BollCvdReclaimStrategyConfig(
        add_layer_gap_pct=decimal_to_float(config.entry.add_gap_pct),
        max_layers=config.capital.max_layers,
        tp_min_net_profit_pct=decimal_to_float(config.tp.tp_min_net_profit_pct),
        tp_boll_enabled=config.tp.tp_boll_enabled,
        tp_boll_window=config.market.tp_boll_window,
        three_stage_runner_enabled=config.tp.three_stage_runner_enabled,
        three_stage_tp1_ratio=decimal_to_float(config.tp.three_stage_tp1_ratio),
        three_stage_tp2_ratio=decimal_to_float(config.tp.three_stage_tp2_ratio),
        three_stage_runner_ratio=decimal_to_float(config.tp.three_stage_runner_ratio),
        three_stage_tp2_use_structure_boll=config.tp.three_stage_tp2_use_structure_boll,
        middle_runner_enabled=config.tp.middle_runner_enabled,
        middle_bucket_split_enabled=config.middle_bucket_split.enabled,
        middle_bucket_split_fast_ratio=decimal_to_float(
            config.middle_bucket_split.fast_ratio
        ),
        middle_bucket_split_fast_sl_enabled=config.middle_bucket_split.fast_sl_enabled,
        middle_bucket_split_fast_sl_fee_buffer_pct=decimal_to_float(
            config.middle_bucket_split.fast_sl_fee_buffer_pct
        ),
    )


# ---------------------------------------------------------------------------
# Preview DTO — lightweight, never wired to Trader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolTraderConfigPreview:
    """Lightweight preview of symbol / instrument parameters.

    This is a **preview DTO** only.  It must never be wired into the live
    ``Trader`` — A06 does not instantiate or modify ``Trader``.
    """

    inst_id: str
    td_mode: str
    pos_side_mode: str
    leverage: str
    contract_value: Decimal
    contract_precision: Decimal
    min_contracts: Decimal
    live_trading: bool


def to_trader_config_preview(
    config: SymbolConfig,
) -> SymbolTraderConfigPreview:
    """Build a preview DTO from *config*.

    This is a data-only projection — it does **not** instantiate ``Trader``
    and does **not** feed into any live path.
    """
    validate_symbol_config(config)
    return SymbolTraderConfigPreview(
        inst_id=config.symbol.inst_id,
        td_mode=config.market.td_mode,
        pos_side_mode=config.market.pos_side_mode,
        leverage=str(config.capital.leverage),
        contract_value=config.market.contract_value,
        contract_precision=config.market.contract_precision,
        min_contracts=config.market.min_contracts,
        live_trading=config.symbol.live_trading,
    )


# ---------------------------------------------------------------------------
# Aggregate mapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappedSymbolConfigs:
    """Aggregate of all mapped runtime config objects for a single symbol."""

    monitor: BollBandBreakoutMonitorConfig
    cvd: CvdTrackerConfig
    position_sizer: SimplePositionSizerConfig
    strategy: BollCvdReclaimStrategyConfig
    trader_preview: SymbolTraderConfigPreview


def map_symbol_config(config: SymbolConfig) -> MappedSymbolConfigs:
    """Map *config* into every runtime config object at once.

    This is a convenience aggregator — it delegates to the individual
    ``to_*`` mappers and returns a single frozen DTO.
    """
    validate_symbol_config(config)
    return MappedSymbolConfigs(
        monitor=to_boll_monitor_config(config),
        cvd=to_cvd_tracker_config(config),
        position_sizer=to_position_sizer_config(config),
        strategy=to_strategy_config(config),
        trader_preview=to_trader_config_preview(config),
    )
