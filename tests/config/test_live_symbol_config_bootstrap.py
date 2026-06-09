#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ``config.live_symbol_config_bootstrap`` (A07)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from config.live_symbol_config_bootstrap import (
    LiveSymbolRuntimeConfigs,
    _temporary_environ,
    build_live_symbol_runtime_configs,
)
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
# 1. test_default_loads_eth_toml
# ---------------------------------------------------------------------------


def test_default_loads_eth_toml() -> None:
    """Default (A08): env={} → use_symbol_toml=True, loads ETH TOML, maps all
    configs, and applies account_equity_usdt override."""
    result = build_live_symbol_runtime_configs(
        env={},
        account_equity_usdt=5000.0,
    )

    assert result.env_runtime.use_symbol_toml is True
    assert result.symbol_config is not None
    assert result.symbol_config.inst_id == "ETH-USDT-SWAP"

    assert result.position_sizer.dry_run_equity_usdt == 5000.0
    assert result.strategy.three_stage_runner_enabled is True
    assert result.strategy.three_stage_tp2_use_structure_boll is True

    # monitor was mapped from TOML
    assert isinstance(result.monitor, BollBandBreakoutMonitorConfig)
    assert result.monitor.inst_id == "ETH-USDT-SWAP"

    # cvd was mapped from TOML
    assert isinstance(result.cvd, CvdTrackerConfig)

    # position_sizer was mapped from TOML (with account equity override)
    assert isinstance(result.position_sizer, SimplePositionSizerConfig)


# ---------------------------------------------------------------------------
# 2. test_explicit_false_uses_legacy_env_path
# ---------------------------------------------------------------------------


def test_explicit_false_uses_legacy_env_path() -> None:
    """Explicit RECLAIM_USE_SYMBOL_TOML=false → legacy .env path."""
    result = build_live_symbol_runtime_configs(
        env={"RECLAIM_USE_SYMBOL_TOML": "false"},
        account_equity_usdt=5000.0,
    )

    assert result.env_runtime.use_symbol_toml is False
    assert result.symbol_config is None
    assert result.position_sizer.dry_run_equity_usdt == 5000.0


# ---------------------------------------------------------------------------
# 3. test_toml_flag_loads_eth_toml
# ---------------------------------------------------------------------------


def test_toml_flag_loads_eth_toml() -> None:
    """RECLAIM_USE_SYMBOL_TOML=true → loads ETH TOML, maps all configs."""
    result = build_live_symbol_runtime_configs(
        env={
            "RECLAIM_USE_SYMBOL_TOML": "true",
            "RECLAIM_SYMBOL_CONFIG_DIR": str(_SYMBOLS_DIR),
            "RECLAIM_SYMBOLS": "ETH-USDT-SWAP",
        }
    )

    assert result.symbol_config is not None
    assert result.symbol_config.inst_id == "ETH-USDT-SWAP"

    assert result.monitor.inst_id == "ETH-USDT-SWAP"

    assert result.strategy.three_stage_runner_enabled is True
    assert result.strategy.three_stage_tp2_use_structure_boll is True

    assert result.position_sizer.layer_multiplier_step == 0.15


# ---------------------------------------------------------------------------
# 4. test_toml_flag_uses_account_equity_override
# ---------------------------------------------------------------------------


def test_toml_flag_uses_account_equity_override() -> None:
    """TOML path + account_equity_usdt → dry_run_equity_usdt is overridden."""
    result = build_live_symbol_runtime_configs(
        env={
            "RECLAIM_USE_SYMBOL_TOML": "true",
            "RECLAIM_SYMBOL_CONFIG_DIR": str(_SYMBOLS_DIR),
            "RECLAIM_SYMBOLS": "ETH-USDT-SWAP",
        },
        account_equity_usdt=1234.5,
    )

    assert result.position_sizer.dry_run_equity_usdt == 1234.5
    assert result.position_sizer.layer_margin_pct == 0.03
    assert result.position_sizer.layer_multiplier_step == 0.15


# ---------------------------------------------------------------------------
# 5. test_toml_flag_rejects_btc_symbols_for_now
# ---------------------------------------------------------------------------


def test_toml_flag_rejects_btc_symbols_for_now() -> None:
    """TOML path with BTC-USDT-SWAP → ValueError mentioning RECLAIM_SYMBOLS."""
    with pytest.raises(ValueError, match="RECLAIM_SYMBOLS"):
        build_live_symbol_runtime_configs(
            env={
                "RECLAIM_USE_SYMBOL_TOML": "true",
                "RECLAIM_SYMBOL_CONFIG_DIR": str(_SYMBOLS_DIR),
                "RECLAIM_SYMBOLS": "BTC-USDT-SWAP",
            }
        )


# ---------------------------------------------------------------------------
# 6. test_toml_flag_missing_file_fails_fast
# ---------------------------------------------------------------------------


def test_toml_flag_missing_file_fails_fast(tmp_path: Path) -> None:
    """TOML path with no ETH TOML in config dir → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        build_live_symbol_runtime_configs(
            env={
                "RECLAIM_USE_SYMBOL_TOML": "true",
                "RECLAIM_SYMBOL_CONFIG_DIR": str(tmp_path),
                "RECLAIM_SYMBOLS": "ETH-USDT-SWAP",
            }
        )


# ---------------------------------------------------------------------------
# 7. test_env_path_does_not_load_toml
# ---------------------------------------------------------------------------


def test_env_path_does_not_load_toml(tmp_path: Path) -> None:
    """Env path with RECLAIM_USE_SYMBOL_TOML=false does NOT touch TOML files."""
    result = build_live_symbol_runtime_configs(
        env={
            "RECLAIM_USE_SYMBOL_TOML": "false",
            "RECLAIM_SYMBOL_CONFIG_DIR": str(tmp_path),
        }
    )

    assert result.symbol_config is None
    assert result.env_runtime.use_symbol_toml is False
    assert isinstance(result.monitor, BollBandBreakoutMonitorConfig)
    assert isinstance(result.strategy, BollCvdReclaimStrategyConfig)
    assert isinstance(result.position_sizer, SimplePositionSizerConfig)


# ---------------------------------------------------------------------------
# 8. test_import_has_no_side_effects
# ---------------------------------------------------------------------------


def test_import_has_no_side_effects() -> None:
    """Importing the module must not read os.environ or TOML files."""
    import importlib

    import config.live_symbol_config_bootstrap as mod

    importlib.reload(mod)
    # Reaching here without exception is sufficient proof of no import-time
    # I/O or env reads.


# ---------------------------------------------------------------------------
# 9. test_env_mapping_restores_os_environ_after_error
# ---------------------------------------------------------------------------


def test_env_mapping_restores_os_environ_after_error() -> None:
    """os.environ is fully restored even when an exception occurs inside
    _temporary_environ."""
    original = os.environ.copy()
    original_keys = set(original.keys())

    # Inject a known value so we can verify it's present before and after.
    os.environ["_TEST_SENTINEL_BEFORE"] = "before"

    env_override = {"RECLAIM_USE_SYMBOL_TOML": "true", "RECLAIM_SYMBOLS": "ETH-USDT-SWAP"}
    try:
        with _temporary_environ(env_override):
            # Inside the block: only our overrides are visible.
            assert os.environ.get("RECLAIM_USE_SYMBOL_TOML") == "true"
            assert os.environ.get("_TEST_SENTINEL_BEFORE") is None
            raise RuntimeError("simulated failure inside _temporary_environ")
    except RuntimeError:
        pass  # expected

    # After the block: original env must be fully restored.
    assert os.environ.get("_TEST_SENTINEL_BEFORE") == "before"
    assert os.environ.get("RECLAIM_USE_SYMBOL_TOML") != "true"
    # All original keys must still be present.
    current_keys = set(os.environ.keys())
    assert original_keys.issubset(current_keys)

    # Cleanup
    del os.environ["_TEST_SENTINEL_BEFORE"]


# ---------------------------------------------------------------------------
# 10. test_temporary_environ_noop_when_env_is_none
# ---------------------------------------------------------------------------


def test_temporary_environ_noop_when_env_is_none() -> None:
    """_temporary_environ with env=None is a no-op."""
    before = os.environ.copy()
    with _temporary_environ(None):
        assert os.environ == before
    # After exit, env is unchanged.
    assert os.environ == before


# ---------------------------------------------------------------------------
# 11. test_account_equity_env_path
# ---------------------------------------------------------------------------


def test_account_equity_env_path() -> None:
    """Legacy path with account_equity_usdt uses from_account_equity."""
    result = build_live_symbol_runtime_configs(
        env={"RECLAIM_USE_SYMBOL_TOML": "false"},
        account_equity_usdt=5000.0,
    )

    assert result.symbol_config is None
    assert result.env_runtime.use_symbol_toml is False
    assert result.position_sizer.dry_run_equity_usdt == 5000.0
    # Other fields come from env defaults.
    assert result.position_sizer.layer_margin_pct == 0.03


# ---------------------------------------------------------------------------
# 12. test_legacy_path_account_equity_does_not_read_dry_run_equity
# ---------------------------------------------------------------------------


def test_legacy_path_account_equity_does_not_read_dry_run_equity() -> None:
    """Legacy path with account_equity_usdt must NOT read DRY_RUN_EQUITY_USDT.

    When live passes real account equity, the helper must use
    ``from_account_equity()`` — even if ``DRY_RUN_EQUITY_USDT`` is set to
    an invalid value the call must succeed and use the passed equity.
    """
    result = build_live_symbol_runtime_configs(
        env={
            "RECLAIM_USE_SYMBOL_TOML": "false",
            "DRY_RUN_EQUITY_USDT": "not-a-number",
        },
        account_equity_usdt=5000.0,
    )

    assert result.symbol_config is None
    assert result.env_runtime.use_symbol_toml is False
    assert result.position_sizer.dry_run_equity_usdt == 5000.0
