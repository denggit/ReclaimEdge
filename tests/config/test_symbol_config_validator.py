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


def test_rejects_non_eth_inst_id_for_now() -> None:
    """Only ETH-USDT-SWAP inst_id is allowed at this stage."""
    config = replace(
        SymbolConfig.default_eth(),
        symbol=SymbolIdentityConfig(inst_id="BTC-USDT-SWAP"),
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


def test_rejects_bool_for_entry_add_freeze_seconds() -> None:
    """bool must be rejected for int field add_freeze_seconds."""
    config = replace(
        SymbolConfig.default_eth(),
        entry=replace(
            SymbolConfig.default_eth().entry,
            add_freeze_seconds=True,  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(SymbolConfigValidationError, match="add_freeze_seconds"):
        validate_symbol_config(config)


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
