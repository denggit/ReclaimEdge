#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for live symbol TOML bootstrap wiring.

G09e intentionally bootstraps TOML runtime config before Trader creation in
live mode so TraderInstrumentMetadata and TraderMarketSettings can be built
from the symbol TOML.  After ``trader.initialize()``, account equity is applied
with the explicit runtime config override helper.

The thin live entry is only responsible for the global LIVE_TRADING gate.

These tests use source inspection — they never import or instantiate live
runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"
_APP_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_app.py"


def _live_source() -> str:
    return _LIVE_SCRIPT.read_text()


def _app_source() -> str:
    return _APP_MODULE.read_text()


# ---------------------------------------------------------------------------
# 1. test_live_path_uses_pre_trader_bootstrap_for_toml_metadata
# ---------------------------------------------------------------------------


def test_live_path_uses_pre_trader_bootstrap_for_toml_metadata() -> None:
    """Live mode bootstraps TOML config before Trader is created."""
    source = _app_source()

    for token in (
        "_build_pre_trader_runtime_configs_for_mode(",
        "_build_live_trader_metadata_from_runtime_configs(",
        "factory.create_trader(",
    ):
        assert token in source

    pre_idx = source.index(
        "pre_runtime_configs = _build_pre_trader_runtime_configs_for_mode("
    )
    metadata_idx = source.index(
        "metadata, market_settings = _build_live_trader_metadata_from_runtime_configs("
    )
    create_trader_idx = source.index("trader = self.factory.create_trader(")
    start_idx = source.index("await trader.start()")
    init_idx = source.index("await trader.initialize()")

    assert pre_idx < metadata_idx < create_trader_idx < start_idx < init_idx


# ---------------------------------------------------------------------------
# 2. test_live_path_overrides_runtime_account_equity_after_initialize
# ---------------------------------------------------------------------------


def test_live_path_overrides_runtime_account_equity_after_initialize() -> None:
    """Live runtime configs are updated with Trader account equity post-init."""
    source = _app_source()

    assert "_override_runtime_config_account_equity(" in source
    assert "pre_runtime_configs" in source
    assert "trader.account_equity_usdt" in source

    init_idx = source.index("await trader.initialize()")
    override_idx = source.index(
        "runtime_configs = _override_runtime_config_account_equity("
    )
    assert init_idx < override_idx


# ---------------------------------------------------------------------------
# 3. test_account_equity_override_is_explicit_helper
# ---------------------------------------------------------------------------


def test_account_equity_override_is_explicit_helper() -> None:
    """Account equity override is centralized in an explicit helper."""
    source = _app_source()
    assert "def _override_runtime_config_account_equity(" in source
    assert "dry_run_equity_usdt=account_equity_usdt" in source


# ---------------------------------------------------------------------------
# 4. test_live_path_pre_trader_bootstrap_occurs_before_create_trader
# ---------------------------------------------------------------------------


def test_live_path_pre_trader_bootstrap_occurs_before_create_trader() -> None:
    """Pre-trader TOML bootstrap feeds Trader construction, then equity override."""
    source = _app_source()

    pre_idx = source.index(
        "pre_runtime_configs = _build_pre_trader_runtime_configs_for_mode("
    )
    create_trader_idx = source.index("trader = self.factory.create_trader(")
    init_idx = source.index("await trader.initialize()")
    override_idx = source.index(
        "runtime_configs = _override_runtime_config_account_equity("
    )
    guard_idx = source.index("_assert_trader_matches_symbol_config(trader,")
    strategy_idx = source.index("factory.create_strategy_objects(")

    assert pre_idx < create_trader_idx
    assert init_idx < override_idx < guard_idx < strategy_idx


# ---------------------------------------------------------------------------
# 5. test_app_has_trader_toml_consistency_guard
# ---------------------------------------------------------------------------


def test_app_has_trader_toml_consistency_guard() -> None:
    """The SymbolWorkerApp must define ``_assert_trader_matches_symbol_config``
    and reference the error message ``TOML/env trader config mismatch``."""
    source = _app_source()
    assert "_assert_trader_matches_symbol_config" in source, (
        "SymbolWorkerApp must define _assert_trader_matches_symbol_config"
    )
    assert "TOML/env trader config mismatch" in source, (
        "SymbolWorkerApp must raise on TOML/env mismatch"
    )


# ---------------------------------------------------------------------------
# 6. test_app_consistency_guard_called_after_bootstrap_before_strategy_creation
# ---------------------------------------------------------------------------


def test_app_consistency_guard_called_after_bootstrap_before_strategy_creation() -> None:
    """The consistency guard must be called after account equity override and
    before strategy objects are created (via factory as of C02)."""
    source = _app_source()

    override_idx = source.index(
        "runtime_configs = _override_runtime_config_account_equity("
    )
    # The call site (not the ``def`` line) — passes ``trader`` as first arg.
    guard_idx = source.index("_assert_trader_matches_symbol_config(trader,")
    strategy_idx = source.index("factory.create_strategy_objects(")

    assert override_idx < guard_idx < strategy_idx, (
        f"Order violation: override={override_idx}, "
        f"guard={guard_idx}, strategy={strategy_idx}"
    )


# ---------------------------------------------------------------------------
# 7. test_live_entry_does_not_use_symbol_live_trading_as_gate
# ---------------------------------------------------------------------------


def test_live_entry_does_not_use_symbol_live_trading_as_gate() -> None:
    """``symbol_config.symbol.live_trading`` must NOT appear in the live
    entrypoint source. ``run_boll_cvd_live.py`` remains responsible only for
    the global ``.env`` LIVE_TRADING gate; per-symbol live gating happens in
    SymbolWorkerApp / startup preflight."""
    source = _live_source()
    assert "symbol_config.symbol.live_trading" not in source, (
        "symbol_config.symbol.live_trading must not be used as a live "
        "trading gate — LIVE_TRADING is still controlled by the .env gate"
    )


# ---------------------------------------------------------------------------
# 8. test_app_sidecar_max_legs_refresh_uses_mapped_config_not_env
# ---------------------------------------------------------------------------


def test_app_sidecar_max_legs_refresh_uses_mapped_config_not_env() -> None:
    """Startup sidecar state refresh must use the mapped
    ``position_sizer_config.sidecar_max_legs`` — NOT a direct
    ``os.getenv("SIDECAR_MAX_LEGS")`` read."""
    source = _app_source()

    # Must NOT contain direct env reads for SIDECAR_MAX_LEGS.
    assert 'os.getenv("SIDECAR_MAX_LEGS"' not in source, (
        "SIDECAR_MAX_LEGS must not be read directly from env"
    )
    assert "os.getenv('SIDECAR_MAX_LEGS'" not in source, (
        "SIDECAR_MAX_LEGS must not be read directly from env"
    )

    # Must still call refresh_sidecar_state_totals.
    assert "refresh_sidecar_state_totals" in source, (
        "refresh_sidecar_state_totals must still be called"
    )

    # The max legs argument must come from the mapped config.
    assert "position_sizer_config.sidecar_max_legs" in source, (
        "sidecar max legs must come from position_sizer_config.sidecar_max_legs"
    )


# ---------------------------------------------------------------------------
# 9. test_live_entry_only_has_live_trading_gate_no_bootstrap
# ---------------------------------------------------------------------------


def test_live_entry_only_has_live_trading_gate_no_bootstrap() -> None:
    """C04 thin live entry must NOT contain symbol config bootstrap —
    that lives in SymbolWorkerApp.run()."""
    source = _live_source()

    assert "live_trading_enabled" in source, (
        "C04 live entry must keep live_trading_enabled gate"
    )
    assert "build_live_symbol_runtime_configs" not in source, (
        "C04 live entry must NOT contain build_live_symbol_runtime_configs"
    )
