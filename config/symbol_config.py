#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-symbol configuration schema for multi-symbol runtime.

This module defines data structures that model the future
``config/symbols/*.toml`` files.  It is a **pure schema / DTO** layer:

* No file I/O.
* No network I/O.
* No environment-variable reads.
* No TOML / YAML parsing.
* No side-effects on import.

Loader, validator and mapper layers are intentionally kept in separate
files so that this module stays dependency-free and testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decimal_from_any(value: object) -> Decimal:
    """Convert an ``int``, ``str`` or ``Decimal`` to ``Decimal``.

    * ``int`` / ``str`` / ``Decimal`` → converted directly.
    * ``float`` → converted via ``str(value)`` to avoid binary floating-point
      contamination.
    * ``None`` → ``ValueError``.
    """
    if value is None:
        raise ValueError("Cannot convert None to Decimal")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        # Go through str to avoid IEEE-754 artefacts.
        return Decimal(str(value))
    if isinstance(value, (int, str)):
        return Decimal(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to Decimal: {value!r}")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolIdentityConfig:
    """Which instrument a strategy runs on and whether it is active."""

    inst_id: str = "ETH-USDT-SWAP"
    enabled: bool = True
    live_trading: bool = False


# ---------------------------------------------------------------------------
# Market / instrument parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolMarketConfig:
    """Instrument-level market parameters for a single symbol."""

    bar: str = "15m"
    td_mode: str = "isolated"
    pos_side_mode: str = "net"
    contract_value: Decimal = Decimal("0.1")
    min_contracts: Decimal = Decimal("0.01")
    contract_precision: Decimal = Decimal("0.01")
    price_precision: Decimal = Decimal("0.01")
    boll_window: int = 20
    boll_std_multiplier: Decimal = Decimal("2.0")
    boll_distance_threshold_pct: Decimal = Decimal("0.005")
    tp_boll_window: int = 15


# ---------------------------------------------------------------------------
# Capital / sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolCapitalConfig:
    """Per-symbol capital and layer-sizing parameters."""

    dry_run_equity_usdt: Decimal = Decimal("1000")
    layer_margin_pct: Decimal = Decimal("0.03")
    leverage: Decimal = Decimal("50")
    max_layers: int = 3
    layer_multiplier_step: Decimal = Decimal("0.15")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolEntryConfig:
    """Add-gap and freeze parameters for staged entry."""

    add_gap_pct: Decimal = Decimal("0.006")
    add_freeze_seconds: int = 3600
    alert_freeze_seconds: int = 3600


# ---------------------------------------------------------------------------
# CVD (cumulative volume delta) trigger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolCvdConfig:
    """CVD-based trigger parameters (per-symbol)."""

    fast_window_seconds: Decimal = Decimal("5")
    price_stall_seconds: Decimal = Decimal("2")
    price_stall_tolerance_pct: Decimal = Decimal("0.0005")
    burst_window_seconds: Decimal = Decimal("3")
    burst_baseline_seconds: Decimal = Decimal("60")
    burst_min_move_ratio: Decimal = Decimal("2.5")
    burst_min_volume_ratio: Decimal = Decimal("2.0")
    burst_min_abs_range_pct: Decimal = Decimal("0.0015")


# ---------------------------------------------------------------------------
# Take-profit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolTpConfig:
    """Take-profit plan parameters for a single symbol."""

    tp_min_net_profit_pct: Decimal = Decimal("0.002")
    tp_boll_enabled: bool = True
    three_stage_runner_enabled: bool = True
    three_stage_tp1_ratio: Decimal = Decimal("0.70")
    three_stage_tp2_ratio: Decimal = Decimal("0.20")
    three_stage_runner_ratio: Decimal = Decimal("0.10")
    three_stage_tp2_use_structure_boll: bool = True
    middle_runner_enabled: bool = False


# ---------------------------------------------------------------------------
# Middle-bucket split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolMiddleBucketSplitConfig:
    """Middle-bucket split parameters."""

    enabled: bool = False
    fast_ratio: Decimal = Decimal("0.60")
    fast_sl_enabled: bool = True
    fast_sl_fee_buffer_pct: Decimal = Decimal("0.001")


# ---------------------------------------------------------------------------
# Sidecar (extra margin / TP layer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolSidecarConfig:
    """Sidecar margin / take-profit parameters.

    .. note::
        Sidecar does not participate in TP1 split.
    """

    enabled: bool = False
    margin_pct: Decimal = Decimal("0.01")
    tp_pct: Decimal = Decimal("0.004")
    skip_first_layer: bool = True
    max_legs: int = 10
    order_status_check_seconds: Decimal = Decimal("5")
    tp_place_retry_count: int = 3
    tp_place_retry_interval_seconds: Decimal = Decimal("0.8")
    tp_place_retry_backoff_multiplier: Decimal = Decimal("1.5")
    tp_rate_limit_fail_action: str = "HALT_ONLY"


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolRiskConfig:
    """Per-symbol risk-management parameters."""

    rolling_loss_guard_enabled: bool = True
    rolling_loss_warn_pct: Decimal = Decimal("0.50")
    rolling_loss_soft_halt_pct: Decimal = Decimal("0.10")
    order_failure_market_exit_delay_seconds: int = 1800


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolExecutionConfig:
    """Per-symbol execution-layer parameters.

    .. note::
        ``private_write_min_interval_seconds`` is a symbol-level placeholder.
        The cross-process ``SharedPrivateWriteLimiter`` will be addressed
        separately.
    """

    private_write_min_interval_seconds: Decimal = Decimal("0.6")
    max_order_retries: int = 3


# ---------------------------------------------------------------------------
# Runtime (queue sizes, sync intervals, heartbeat)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolRuntimeConfig:
    """Per-symbol runtime / telemetry parameters."""

    strategy_tick_queue_maxsize: int = 20000
    execution_queue_maxsize: int = 1000
    position_sync_seconds: Decimal = Decimal("5")
    account_sync_seconds: Decimal = Decimal("60")
    market_tick_heartbeat_seconds: Decimal = Decimal("60")
    account_snapshot_stale_warn_seconds: Decimal = Decimal("30")
    strategy_tick_lag_warn_seconds: Decimal = Decimal("2")
    execution_backlog_log_seconds: Decimal = Decimal("30")


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolConfig:
    """Complete per-symbol configuration.

    Use :meth:`SymbolConfig.default_eth` to obtain the current
    ETH-USDT-SWAP defaults.
    """

    symbol: SymbolIdentityConfig = field(default_factory=SymbolIdentityConfig)
    market: SymbolMarketConfig = field(default_factory=SymbolMarketConfig)
    capital: SymbolCapitalConfig = field(default_factory=SymbolCapitalConfig)
    entry: SymbolEntryConfig = field(default_factory=SymbolEntryConfig)
    cvd: SymbolCvdConfig = field(default_factory=SymbolCvdConfig)
    tp: SymbolTpConfig = field(default_factory=SymbolTpConfig)
    middle_bucket_split: SymbolMiddleBucketSplitConfig = field(
        default_factory=SymbolMiddleBucketSplitConfig
    )
    sidecar: SymbolSidecarConfig = field(default_factory=SymbolSidecarConfig)
    risk: SymbolRiskConfig = field(default_factory=SymbolRiskConfig)
    execution: SymbolExecutionConfig = field(default_factory=SymbolExecutionConfig)
    runtime: SymbolRuntimeConfig = field(default_factory=SymbolRuntimeConfig)

    # -- convenience class-methods ------------------------------------------

    @classmethod
    def default_eth(cls) -> "SymbolConfig":
        """Return the default ETH-USDT-SWAP configuration."""
        return cls()

    # -- convenience properties ---------------------------------------------

    @property
    def inst_id(self) -> str:
        """Shortcut for ``self.symbol.inst_id``."""
        return self.symbol.inst_id

    @property
    def is_enabled(self) -> bool:
        """Shortcut for ``self.symbol.enabled``."""
        return self.symbol.enabled

    @property
    def is_live_trading_enabled(self) -> bool:
        """``True`` only when the symbol is *both* enabled and live-trading."""
        return self.symbol.enabled and self.symbol.live_trading
