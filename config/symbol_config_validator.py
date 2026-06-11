#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SymbolConfig semantic validator (A04).

Design rules
------------
* Validator only — no TOML reading, no .env reading, no file I/O.
* No logging, no printing, no side-effects on import.
* Pure function: ``validate_symbol_config(config)`` inspects the
  frozen dataclass, returns ``None`` on success, and raises
  ``SymbolConfigValidationError`` on failure.
* No live-runtime, strategy, TP, SL, DME, or Sidecar imports.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from config.symbol_config import SymbolConfig


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class SymbolConfigValidationError(ValueError):
    """Raised when a ``SymbolConfig`` fails semantic validation."""


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _fail(section: str, field: str, message: str) -> None:
    """Raise a ``SymbolConfigValidationError`` with a structured message."""
    raise SymbolConfigValidationError(f"[{section}].{field}: {message}")


def _ensure_positive_decimal(
    section: str, field: str, value: Decimal
) -> None:
    """*value* must be > 0 (Decimal)."""
    if not isinstance(value, Decimal):
        _fail(section, field, f"expected Decimal, got {type(value).__name__}")
    if value <= Decimal("0"):
        _fail(section, field, f"must be > 0, got {value}")


def _ensure_non_negative_decimal(
    section: str, field: str, value: Decimal
) -> None:
    """*value* must be >= 0 (Decimal)."""
    if not isinstance(value, Decimal):
        _fail(section, field, f"expected Decimal, got {type(value).__name__}")
    if value < Decimal("0"):
        _fail(section, field, f"must be >= 0, got {value}")


def _ensure_pct_open_closed(
    section: str, field: str, value: Decimal
) -> None:
    """> 0 and < 1 (Decimal)."""
    if not isinstance(value, Decimal):
        _fail(section, field, f"expected Decimal, got {type(value).__name__}")
    if value <= Decimal("0") or value >= Decimal("1"):
        _fail(section, field, f"must be between 0 and 1, got {value}")


def _ensure_int(section: str, field: str, value: object) -> None:
    """*value* must be a strict ``int`` (``bool`` is rejected)."""
    if type(value) is not int:
        _fail(section, field, f"expected int, got {type(value).__name__}")


def _ensure_non_negative_int(
    section: str, field: str, value: object
) -> None:
    """*value* must be a strict ``int`` and >= 0."""
    _ensure_int(section, field, value)
    if value < 0:  # type: ignore[operator]  # guarded by _ensure_int
        _fail(section, field, f"must be >= 0, got {value}")


def _ensure_positive_int(section: str, field: str, value: object) -> None:
    """*value* must be a strict ``int`` and > 0."""
    _ensure_int(section, field, value)
    if value <= 0:  # type: ignore[operator]  # guarded by _ensure_int
        _fail(section, field, f"must be > 0, got {value}")


def _ensure_int_at_least(
    section: str, field: str, value: object, minimum: int
) -> None:
    """*value* must be a strict ``int`` and >= *minimum*."""
    _ensure_int(section, field, value)
    if value < minimum:  # type: ignore[operator]  # guarded by _ensure_int
        _fail(section, field, f"must be >= {minimum}, got {value}")


def _ensure_bool(section: str, field: str, value: Any) -> None:
    """*value* must be bool."""
    if not isinstance(value, bool):
        _fail(section, field, f"expected bool, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Section validators (one per section)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Supported symbols at this stage
# ---------------------------------------------------------------------------

SUPPORTED_SYMBOLS_AT_THIS_STAGE: set[str] = {"ETH-USDT-SWAP", "BTC-USDT-SWAP"}


def _validate_symbol(config: SymbolConfig) -> None:
    s = config.symbol
    sec = "symbol"

    if not isinstance(s.inst_id, str) or not s.inst_id.strip():
        _fail(sec, "inst_id", f"must be a non-empty string, got {s.inst_id!r}")

    # Only allow explicitly supported symbols.
    if s.inst_id not in SUPPORTED_SYMBOLS_AT_THIS_STAGE:
        _fail(
            sec,
            "inst_id",
            f"unsupported symbol {s.inst_id!r}; "
            f"only {sorted(SUPPORTED_SYMBOLS_AT_THIS_STAGE)!r} "
            f"are allowed at this stage",
        )

    _ensure_bool(sec, "enabled", s.enabled)
    _ensure_bool(sec, "live_trading", s.live_trading)

    # NOTE: ``enabled`` and ``live_trading`` are config items —
    # the TOML author controls them.  Neither ETH nor BTC is
    # hard-disabled here.  The real live gate is the supervisor's
    # supported-symbol whitelist and RECLAIM_SYMBOLS in each
    # worker env.


def _validate_market(config: SymbolConfig) -> None:
    m = config.market
    sec = "market"

    if m.bar != "15m":
        _fail(sec, "bar", f"must be '15m', got {m.bar!r}")

    if m.td_mode != "isolated":
        _fail(sec, "td_mode", f"must be 'isolated', got {m.td_mode!r}")

    if m.pos_side_mode != "net":
        _fail(sec, "pos_side_mode", f"must be 'net', got {m.pos_side_mode!r}")

    _ensure_positive_decimal(sec, "contract_value", m.contract_value)
    _ensure_positive_decimal(sec, "min_contracts", m.min_contracts)
    _ensure_positive_decimal(sec, "contract_precision", m.contract_precision)
    _ensure_positive_decimal(sec, "price_precision", m.price_precision)

    _ensure_int_at_least(sec, "boll_window", m.boll_window, 2)

    _ensure_positive_decimal(sec, "boll_std_multiplier", m.boll_std_multiplier)
    _ensure_pct_open_closed(sec, "boll_distance_threshold_pct", m.boll_distance_threshold_pct)

    _ensure_int_at_least(sec, "tp_boll_window", m.tp_boll_window, 2)

    if not isinstance(m.min_outside_pct, Decimal):
        _fail(sec, "min_outside_pct", f"expected Decimal, got {type(m.min_outside_pct).__name__}")
    if m.min_outside_pct < Decimal("0") or m.min_outside_pct > Decimal("0.01"):
        _fail(sec, "min_outside_pct", f"must be between 0 and 0.01, got {m.min_outside_pct}")


def _validate_capital(config: SymbolConfig) -> None:
    c = config.capital
    sec = "capital"

    _ensure_positive_decimal(sec, "dry_run_equity_usdt", c.dry_run_equity_usdt)

    if not isinstance(c.layer_margin_pct, Decimal):
        _fail(sec, "layer_margin_pct", f"expected Decimal, got {type(c.layer_margin_pct).__name__}")
    if c.layer_margin_pct <= Decimal("0") or c.layer_margin_pct > Decimal("1"):
        _fail(sec, "layer_margin_pct", f"must be between 0 and 1, got {c.layer_margin_pct}")

    _ensure_positive_decimal(sec, "leverage", c.leverage)

    _ensure_int_at_least(sec, "max_layers", c.max_layers, 1)

    _ensure_non_negative_decimal(sec, "layer_multiplier_step", c.layer_multiplier_step)


def _validate_entry(config: SymbolConfig) -> None:
    e = config.entry
    sec = "entry"

    if e.add_gap_mode != "linear":
        _fail(sec, "add_gap_mode", f"must be 'linear', got {e.add_gap_mode!r}")
    _ensure_positive_decimal(sec, "add_gap_base_pct", e.add_gap_base_pct)
    _ensure_non_negative_decimal(sec, "add_gap_step_pct", e.add_gap_step_pct)
    _ensure_non_negative_int(sec, "add_freeze_seconds", e.add_freeze_seconds)
    _ensure_non_negative_int(sec, "first_add_block_seconds", e.first_add_block_seconds)
    _ensure_non_negative_int(sec, "add_min_interval_seconds", e.add_min_interval_seconds)
    _ensure_non_negative_int(sec, "alert_freeze_seconds", e.alert_freeze_seconds)


def _validate_cvd(config: SymbolConfig) -> None:
    cvd = config.cvd
    sec = "cvd"

    _ensure_positive_decimal(sec, "fast_window_seconds", cvd.fast_window_seconds)

    if not isinstance(cvd.price_stall_seconds, Decimal):
        _fail(sec, "price_stall_seconds", f"expected Decimal, got {type(cvd.price_stall_seconds).__name__}")
    if cvd.price_stall_seconds < Decimal("0"):
        _fail(sec, "price_stall_seconds", f"must be >= 0, got {cvd.price_stall_seconds}")

    _ensure_non_negative_decimal(sec, "price_stall_tolerance_pct", cvd.price_stall_tolerance_pct)

    _ensure_positive_decimal(sec, "burst_window_seconds", cvd.burst_window_seconds)
    _ensure_positive_decimal(sec, "burst_baseline_seconds", cvd.burst_baseline_seconds)
    _ensure_positive_decimal(sec, "burst_min_move_ratio", cvd.burst_min_move_ratio)
    _ensure_positive_decimal(sec, "burst_min_volume_ratio", cvd.burst_min_volume_ratio)
    _ensure_non_negative_decimal(sec, "burst_min_abs_range_pct", cvd.burst_min_abs_range_pct)


def _validate_tp(config: SymbolConfig) -> None:
    tp = config.tp
    sec = "tp"

    if not isinstance(tp.tp_min_net_profit_pct, Decimal):
        _fail(sec, "tp_min_net_profit_pct", f"expected Decimal, got {type(tp.tp_min_net_profit_pct).__name__}")
    if tp.tp_min_net_profit_pct < Decimal("0") or tp.tp_min_net_profit_pct >= Decimal("1"):
        _fail(sec, "tp_min_net_profit_pct", f"must be between 0 and 1, got {tp.tp_min_net_profit_pct}")

    _ensure_bool(sec, "tp_boll_enabled", tp.tp_boll_enabled)
    _ensure_bool(sec, "three_stage_runner_enabled", tp.three_stage_runner_enabled)

    if not isinstance(tp.three_stage_tp1_ratio, Decimal):
        _fail(sec, "three_stage_tp1_ratio", f"expected Decimal, got {type(tp.three_stage_tp1_ratio).__name__}")
    if tp.three_stage_tp1_ratio < Decimal("0"):
        _fail(sec, "three_stage_tp1_ratio", f"must be >= 0, got {tp.three_stage_tp1_ratio}")

    if not isinstance(tp.three_stage_tp2_ratio, Decimal):
        _fail(sec, "three_stage_tp2_ratio", f"expected Decimal, got {type(tp.three_stage_tp2_ratio).__name__}")
    if tp.three_stage_tp2_ratio < Decimal("0"):
        _fail(sec, "three_stage_tp2_ratio", f"must be >= 0, got {tp.three_stage_tp2_ratio}")

    if not isinstance(tp.three_stage_runner_ratio, Decimal):
        _fail(sec, "three_stage_runner_ratio", f"expected Decimal, got {type(tp.three_stage_runner_ratio).__name__}")
    if tp.three_stage_runner_ratio < Decimal("0"):
        _fail(sec, "three_stage_runner_ratio", f"must be >= 0, got {tp.three_stage_runner_ratio}")

    ratio_sum = tp.three_stage_tp1_ratio + tp.three_stage_tp2_ratio + tp.three_stage_runner_ratio
    if ratio_sum != Decimal("1.00"):
        _fail(
            sec,
            "three_stage_tp1_ratio + three_stage_tp2_ratio + three_stage_runner_ratio",
            f"three-stage ratio sum must equal 1.00, got {ratio_sum}",
        )

    _ensure_bool(sec, "three_stage_tp2_use_structure_boll", tp.three_stage_tp2_use_structure_boll)

    if tp.three_stage_tp2_use_structure_boll is not True:
        _fail(
            sec,
            "three_stage_tp2_use_structure_boll",
            "must be True — three-stage TP2 must use structure BOLL20 outer",
        )

    _ensure_bool(sec, "middle_runner_enabled", tp.middle_runner_enabled)

    _ensure_bool(sec, "split_tp_enabled", tp.split_tp_enabled)


def _validate_middle_bucket_split(config: SymbolConfig) -> None:
    mbs = config.middle_bucket_split
    sec = "middle_bucket_split"

    _ensure_bool(sec, "enabled", mbs.enabled)
    _ensure_pct_open_closed(sec, "fast_ratio", mbs.fast_ratio)
    _ensure_bool(sec, "fast_sl_enabled", mbs.fast_sl_enabled)

    if not isinstance(mbs.fast_sl_fee_buffer_pct, Decimal):
        _fail(sec, "fast_sl_fee_buffer_pct", f"expected Decimal, got {type(mbs.fast_sl_fee_buffer_pct).__name__}")
    if mbs.fast_sl_fee_buffer_pct < Decimal("0") or mbs.fast_sl_fee_buffer_pct >= Decimal("1"):
        _fail(sec, "fast_sl_fee_buffer_pct", f"must be between 0 and 1, got {mbs.fast_sl_fee_buffer_pct}")


def _validate_sidecar(config: SymbolConfig) -> None:
    sc = config.sidecar
    sec = "sidecar"

    _ensure_bool(sec, "enabled", sc.enabled)

    if not isinstance(sc.margin_pct, Decimal):
        _fail(sec, "margin_pct", f"expected Decimal, got {type(sc.margin_pct).__name__}")
    if sc.margin_pct <= Decimal("0") or sc.margin_pct > Decimal("1"):
        _fail(sec, "margin_pct", f"must be between 0 and 1, got {sc.margin_pct}")

    _ensure_pct_open_closed(sec, "tp_pct", sc.tp_pct)
    _ensure_bool(sec, "skip_first_layer", sc.skip_first_layer)
    _ensure_non_negative_int(sec, "max_legs", sc.max_legs)

    _ensure_positive_decimal(sec, "order_status_check_seconds", sc.order_status_check_seconds)
    _ensure_non_negative_int(sec, "tp_place_retry_count", sc.tp_place_retry_count)

    if not isinstance(sc.tp_place_retry_interval_seconds, Decimal):
        _fail(sec, "tp_place_retry_interval_seconds", f"expected Decimal, got {type(sc.tp_place_retry_interval_seconds).__name__}")
    if sc.tp_place_retry_interval_seconds < Decimal("0"):
        _fail(sec, "tp_place_retry_interval_seconds", f"must be >= 0, got {sc.tp_place_retry_interval_seconds}")

    if not isinstance(sc.tp_place_retry_backoff_multiplier, Decimal):
        _fail(sec, "tp_place_retry_backoff_multiplier", f"expected Decimal, got {type(sc.tp_place_retry_backoff_multiplier).__name__}")
    if sc.tp_place_retry_backoff_multiplier < Decimal("1"):
        _fail(sec, "tp_place_retry_backoff_multiplier", f"must be >= 1, got {sc.tp_place_retry_backoff_multiplier}")

    if sc.tp_rate_limit_fail_action != "HALT_ONLY":
        _fail(
            sec,
            "tp_rate_limit_fail_action",
            f"must be 'HALT_ONLY', got {sc.tp_rate_limit_fail_action!r}",
        )


def _validate_risk(config: SymbolConfig) -> None:
    r = config.risk
    sec = "risk"

    _ensure_bool(sec, "rolling_loss_guard_enabled", r.rolling_loss_guard_enabled)

    if not isinstance(r.rolling_loss_warn_pct, Decimal):
        _fail(sec, "rolling_loss_warn_pct", f"expected Decimal, got {type(r.rolling_loss_warn_pct).__name__}")
    if r.rolling_loss_warn_pct < Decimal("0") or r.rolling_loss_warn_pct > Decimal("1"):
        _fail(sec, "rolling_loss_warn_pct", f"must be between 0 and 1, got {r.rolling_loss_warn_pct}")

    if not isinstance(r.rolling_loss_soft_halt_pct, Decimal):
        _fail(sec, "rolling_loss_soft_halt_pct", f"expected Decimal, got {type(r.rolling_loss_soft_halt_pct).__name__}")
    if r.rolling_loss_soft_halt_pct < Decimal("0") or r.rolling_loss_soft_halt_pct > Decimal("1"):
        _fail(sec, "rolling_loss_soft_halt_pct", f"must be between 0 and 1, got {r.rolling_loss_soft_halt_pct}")

    if r.rolling_loss_soft_halt_pct > r.rolling_loss_warn_pct:
        _fail(
            sec,
            "rolling_loss_soft_halt_pct",
            f"rolling_loss_soft_halt_pct ({r.rolling_loss_soft_halt_pct}) "
            f"must be <= rolling_loss_warn_pct ({r.rolling_loss_warn_pct})",
        )

    _ensure_int_at_least(
        sec,
        "order_failure_market_exit_delay_seconds",
        r.order_failure_market_exit_delay_seconds,
        1800,
    )


def _validate_execution(config: SymbolConfig) -> None:
    ex = config.execution
    sec = "execution"

    if not isinstance(ex.private_write_min_interval_seconds, Decimal):
        _fail(sec, "private_write_min_interval_seconds", f"expected Decimal, got {type(ex.private_write_min_interval_seconds).__name__}")
    if ex.private_write_min_interval_seconds < Decimal("0"):
        _fail(sec, "private_write_min_interval_seconds", f"must be >= 0, got {ex.private_write_min_interval_seconds}")

    _ensure_non_negative_int(sec, "max_order_retries", ex.max_order_retries)


def _validate_runtime(config: SymbolConfig) -> None:
    rt = config.runtime
    sec = "runtime"

    _ensure_int_at_least(sec, "strategy_tick_queue_maxsize", rt.strategy_tick_queue_maxsize, 1000)
    _ensure_int_at_least(sec, "execution_queue_maxsize", rt.execution_queue_maxsize, 100)

    _ensure_positive_decimal(sec, "position_sync_seconds", rt.position_sync_seconds)
    _ensure_positive_decimal(sec, "account_sync_seconds", rt.account_sync_seconds)
    _ensure_positive_decimal(sec, "market_tick_heartbeat_seconds", rt.market_tick_heartbeat_seconds)
    _ensure_positive_decimal(sec, "account_snapshot_stale_warn_seconds", rt.account_snapshot_stale_warn_seconds)
    _ensure_positive_decimal(sec, "strategy_tick_lag_warn_seconds", rt.strategy_tick_lag_warn_seconds)
    _ensure_positive_decimal(sec, "execution_backlog_log_seconds", rt.execution_backlog_log_seconds)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_symbol_config(config: SymbolConfig) -> None:
    """Validate *config* against all safety and soundness rules.

    Returns ``None`` on success.  Raises ``SymbolConfigValidationError``
    (a subclass of ``ValueError``) with a message that identifies the
    offending section and field on any violation.

    This function is a pure validator:
    * It reads no files, no environment variables.
    * It mutates nothing.
    * It logs nothing and prints nothing.
    * It does not import strategy / TP / SL / DME / Sidecar modules.
    """
    _validate_symbol(config)
    _validate_market(config)
    _validate_capital(config)
    _validate_entry(config)
    _validate_cvd(config)
    _validate_tp(config)
    _validate_middle_bucket_split(config)
    _validate_sidecar(config)
    _validate_risk(config)
    _validate_execution(config)
    _validate_runtime(config)
