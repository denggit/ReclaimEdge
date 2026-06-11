#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ``config.symbol_config_validator`` — SymbolConfig validator (A04)."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from config.symbol_config import (
    SymbolConfig,
    SymbolIdentityConfig,
    SymbolRiskConfig,
    SymbolTpConfig,
)
from config.symbol_config_loader import load_symbol_config_from_dir
from config.symbol_config_validator import (
    SymbolConfigValidationError,
    validate_symbol_config,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SYMBOLS_DIR = _PROJECT_ROOT / "config" / "symbols"


# ---------------------------------------------------------------------------
# 1. Default config must pass
# ---------------------------------------------------------------------------


def test_default_eth_config_is_valid() -> None:
    """Default ETH config passes validator without error."""
    validate_symbol_config(SymbolConfig.default_eth())


def test_loaded_eth_toml_is_valid() -> None:
    """Loaded ETH TOML config passes validator without error."""
    loaded = load_symbol_config_from_dir(str(_SYMBOLS_DIR), "ETH-USDT-SWAP")
    validate_symbol_config(loaded)


# ---------------------------------------------------------------------------
# 2. inst_id rejection
# ---------------------------------------------------------------------------


def test_rejects_unsupported_inst_id() -> None:
    """Only ETH-USDT-SWAP and BTC-USDT-SWAP inst_ids are allowed at this stage."""
    config = replace(
        SymbolConfig.default_eth(),
        symbol=SymbolIdentityConfig(inst_id="SOL-USDT-SWAP"),
    )
    with pytest.raises(SymbolConfigValidationError, match="symbol.*inst_id"):
        validate_symbol_config(config)


# ---------------------------------------------------------------------------
# 3. TP section rejections
# ---------------------------------------------------------------------------


def test_rejects_tp2_structure_boll_false() -> None:
    """TP2 structure boll must be True."""
    config = replace(
        SymbolConfig.default_eth(),
        tp=replace(SymbolConfig.default_eth().tp, three_stage_tp2_use_structure_boll=False),
    )
    with pytest.raises(SymbolConfigValidationError, match="three_stage_tp2_use_structure_boll"):
        validate_symbol_config(config)


def test_rejects_three_stage_ratios_not_sum_to_one() -> None:
    """TP1 + TP2 + Runner must sum to exactly 1.00."""
    config = replace(
        SymbolConfig.default_eth(),
        tp=SymbolTpConfig(
            three_stage_tp1_ratio=Decimal("0.70"),
            three_stage_tp2_ratio=Decimal("0.20"),
            three_stage_runner_ratio=Decimal("0.20"),
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="ratio sum"):
        validate_symbol_config(config)


# ---------------------------------------------------------------------------
# 4. Risk section rejections
# ---------------------------------------------------------------------------


def test_rejects_order_failure_market_exit_delay_below_1800() -> None:
    """Order failure market exit delay must be >= 1800."""
    config = replace(
        SymbolConfig.default_eth(),
        risk=replace(
            SymbolConfig.default_eth().risk,
            order_failure_market_exit_delay_seconds=60,
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="order_failure_market_exit_delay_seconds"):
        validate_symbol_config(config)


def test_rejects_soft_halt_gt_warn() -> None:
    """soft_halt must be <= warn."""
    config = replace(
        SymbolConfig.default_eth(),
        risk=SymbolRiskConfig(
            rolling_loss_warn_pct=Decimal("0.10"),
            rolling_loss_soft_halt_pct=Decimal("0.20"),
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="rolling_loss_soft_halt_pct"):
        validate_symbol_config(config)


# ---------------------------------------------------------------------------
# 5. Sidecar section rejection
# ---------------------------------------------------------------------------


def test_rejects_sidecar_rate_limit_fail_action_not_halt_only() -> None:
    """Sidecar tp_rate_limit_fail_action must be HALT_ONLY."""
    config = replace(
        SymbolConfig.default_eth(),
        sidecar=replace(
            SymbolConfig.default_eth().sidecar,
            tp_rate_limit_fail_action="MARKET_EXIT",
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="tp_rate_limit_fail_action"):
        validate_symbol_config(config)


# ---------------------------------------------------------------------------
# 6. Runtime section rejection
# ---------------------------------------------------------------------------


def test_rejects_invalid_runtime_queue_size() -> None:
    """strategy_tick_queue_maxsize must be >= 1000."""
    config = replace(
        SymbolConfig.default_eth(),
        runtime=replace(
            SymbolConfig.default_eth().runtime,
            strategy_tick_queue_maxsize=10,
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="strategy_tick_queue_maxsize"):
        validate_symbol_config(config)


# ---------------------------------------------------------------------------
# 7. Market section rejection
# ---------------------------------------------------------------------------


def test_rejects_invalid_market_precision() -> None:
    """price_precision must be > 0."""
    config = replace(
        SymbolConfig.default_eth(),
        market=replace(
            SymbolConfig.default_eth().market,
            price_precision=Decimal("0"),
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="price_precision"):
        validate_symbol_config(config)


# ---------------------------------------------------------------------------
# 8. Int-type rejection (bool / str passed to int fields)
# ---------------------------------------------------------------------------


def test_rejects_bool_for_capital_max_layers() -> None:
    """bool must be rejected for int field max_layers."""
    config = replace(
        SymbolConfig.default_eth(),
        capital=replace(
            SymbolConfig.default_eth().capital,
            max_layers=True,  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="max_layers"):
        validate_symbol_config(config)


def test_rejects_string_for_market_boll_window() -> None:
    """str must be rejected for int field boll_window."""
    config = replace(
        SymbolConfig.default_eth(),
        market=replace(
            SymbolConfig.default_eth().market,
            boll_window="20",  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="boll_window"):
        validate_symbol_config(config)


def test_entry_timing_fields_validate_without_legacy_freeze_field() -> None:
    """Entry timing validation only requires the explicit add gate fields."""
    validate_symbol_config(SymbolConfig.default_eth())


def test_rejects_bool_for_sidecar_max_legs() -> None:
    """bool must be rejected for int field max_legs."""
    config = replace(
        SymbolConfig.default_eth(),
        sidecar=replace(
            SymbolConfig.default_eth().sidecar,
            max_legs=True,  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="max_legs"):
        validate_symbol_config(config)


def test_rejects_string_for_runtime_strategy_tick_queue_maxsize() -> None:
    """str must be rejected for int field strategy_tick_queue_maxsize."""
    config = replace(
        SymbolConfig.default_eth(),
        runtime=replace(
            SymbolConfig.default_eth().runtime,
            strategy_tick_queue_maxsize="20000",  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="strategy_tick_queue_maxsize"):
        validate_symbol_config(config)


def test_rejects_bool_for_execution_max_order_retries() -> None:
    """bool must be rejected for int field max_order_retries."""
    config = replace(
        SymbolConfig.default_eth(),
        execution=replace(
            SymbolConfig.default_eth().execution,
            max_order_retries=True,  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="max_order_retries"):
        validate_symbol_config(config)


# ---------------------------------------------------------------------------
# 9. New field rejections (A08 wiring)
# ---------------------------------------------------------------------------


def test_rejects_negative_min_outside_pct() -> None:
    """min_outside_pct must be >= 0."""
    config = replace(
        SymbolConfig.default_eth(),
        market=replace(
            SymbolConfig.default_eth().market,
            min_outside_pct=Decimal("-0.001"),
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="min_outside_pct"):
        validate_symbol_config(config)


def test_rejects_bool_for_entry_first_add_block_seconds() -> None:
    """bool must be rejected for int field first_add_block_seconds."""
    config = replace(
        SymbolConfig.default_eth(),
        entry=replace(
            SymbolConfig.default_eth().entry,
            first_add_block_seconds=True,  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="first_add_block_seconds"):
        validate_symbol_config(config)


def test_rejects_bool_for_entry_add_min_interval_seconds() -> None:
    """bool must be rejected for int field add_min_interval_seconds."""
    config = replace(
        SymbolConfig.default_eth(),
        entry=replace(
            SymbolConfig.default_eth().entry,
            add_min_interval_seconds=True,  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="add_min_interval_seconds"):
        validate_symbol_config(config)


def test_rejects_add_gap_mode_not_linear() -> None:
    """add_gap_mode must be 'linear'."""
    config = replace(
        SymbolConfig.default_eth(),
        entry=replace(
            SymbolConfig.default_eth().entry,
            add_gap_mode="segmented",
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="add_gap_mode"):
        validate_symbol_config(config)


def test_rejects_add_gap_base_pct_zero() -> None:
    """add_gap_base_pct must be > 0."""
    config = replace(
        SymbolConfig.default_eth(),
        entry=replace(
            SymbolConfig.default_eth().entry,
            add_gap_base_pct=Decimal("0"),
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="add_gap_base_pct"):
        validate_symbol_config(config)


def test_rejects_add_gap_base_pct_negative() -> None:
    """add_gap_base_pct must be > 0."""
    config = replace(
        SymbolConfig.default_eth(),
        entry=replace(
            SymbolConfig.default_eth().entry,
            add_gap_base_pct=Decimal("-0.001"),
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="add_gap_base_pct"):
        validate_symbol_config(config)


def test_rejects_add_gap_step_pct_negative() -> None:
    """add_gap_step_pct must be >= 0."""
    config = replace(
        SymbolConfig.default_eth(),
        entry=replace(
            SymbolConfig.default_eth().entry,
            add_gap_step_pct=Decimal("-0.001"),
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="add_gap_step_pct"):
        validate_symbol_config(config)


def test_accepts_add_gap_step_pct_zero() -> None:
    """add_gap_step_pct = 0 is valid (all layers use base)."""
    config = replace(
        SymbolConfig.default_eth(),
        entry=replace(
            SymbolConfig.default_eth().entry,
            add_gap_step_pct=Decimal("0"),
        ),
    )
    # Should not raise.
    validate_symbol_config(config)


def test_rejects_non_bool_for_tp_split_tp_enabled() -> None:
    """non-bool must be rejected for split_tp_enabled."""
    config = replace(
        SymbolConfig.default_eth(),
        tp=replace(
            SymbolConfig.default_eth().tp,
            split_tp_enabled=1,  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="split_tp_enabled"):
        validate_symbol_config(config)


# ---------------------------------------------------------------------------
# 10. BTC-USDT-SWAP validator rules (G09b: no hard-disable on enabled/live_trading)
# ---------------------------------------------------------------------------


def test_disabled_btc_config_passes() -> None:
    """A disabled BTC config passes validator."""
    config = replace(
        SymbolConfig.default_eth(),
        symbol=SymbolIdentityConfig(
            inst_id="BTC-USDT-SWAP",
            enabled=False,
            live_trading=False,
        ),
        market=replace(
            SymbolConfig.default_eth().market,
            contract_value=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            price_precision=Decimal("0.1"),
        ),
    )
    # Should not raise.
    validate_symbol_config(config)


def test_btc_enabled_true_passes() -> None:
    """BTC enabled=True passes validator — no hard-disable on BTC capability."""
    config = replace(
        SymbolConfig.default_eth(),
        symbol=SymbolIdentityConfig(
            inst_id="BTC-USDT-SWAP",
            enabled=True,
            live_trading=False,
        ),
        market=replace(
            SymbolConfig.default_eth().market,
            contract_value=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            price_precision=Decimal("0.1"),
        ),
    )
    # Must not raise — enabled/live_trading are config items.
    validate_symbol_config(config)


def test_btc_live_trading_true_passes() -> None:
    """BTC live_trading=True passes validator — no hard-disable on BTC capability."""
    config = replace(
        SymbolConfig.default_eth(),
        symbol=SymbolIdentityConfig(
            inst_id="BTC-USDT-SWAP",
            enabled=False,
            live_trading=True,
        ),
        market=replace(
            SymbolConfig.default_eth().market,
            contract_value=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            price_precision=Decimal("0.1"),
        ),
    )
    # Must not raise — enabled/live_trading are config items.
    validate_symbol_config(config)


def test_btc_sidecar_enabled_true_passes() -> None:
    """BTC sidecar.enabled=true passes validator — not hard-disabled by symbol."""
    config = replace(
        SymbolConfig.default_eth(),
        symbol=SymbolIdentityConfig(
            inst_id="BTC-USDT-SWAP",
            enabled=False,
            live_trading=False,
        ),
        sidecar=replace(
            SymbolConfig.default_eth().sidecar,
            enabled=True,
        ),
        market=replace(
            SymbolConfig.default_eth().market,
            contract_value=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            price_precision=Decimal("0.1"),
        ),
    )
    # Must not raise — sidecar.enabled is a config item, not hard-disabled
    # for any supported symbol.
    validate_symbol_config(config)


def test_unsupported_symbol_sol_fails() -> None:
    """SOL-USDT-SWAP (unsupported) must be rejected."""
    config = replace(
        SymbolConfig.default_eth(),
        symbol=SymbolIdentityConfig(inst_id="SOL-USDT-SWAP"),
    )
    with pytest.raises(SymbolConfigValidationError, match="unsupported"):
        validate_symbol_config(config)


def test_eth_live_trading_true_passes() -> None:
    """ETH live_trading=True passes validator — no hard-disable on ETH capability."""
    config = replace(
        SymbolConfig.default_eth(),
        symbol=SymbolIdentityConfig(
            inst_id="ETH-USDT-SWAP",
            enabled=True,
            live_trading=True,
        ),
    )
    # Must not raise — enabled/live_trading are config items.
    validate_symbol_config(config)
