"""Tests for Trend Upgrade Add-on state persistence and backward compatibility.

Covers:
1. Old state (no trend upgrade fields) restore doesn't fail
2. New state with trend upgrade fields saves and loads correctly
3. from_strategy_state maps all trend upgrade fields
4. LivePositionState backwards compat with missing fields in JSON
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.reporting.live_state_store import LivePositionState, LiveStateStore


# ======================================================================
# 1. Old state JSON without trend upgrade fields loads correctly
# ======================================================================


def test_old_state_json_loads_without_trend_upgrade_fields():
    """Old JSON without trend_upgrade_* fields should load with defaults."""
    old_json = {
        "position_id": "test-001",
        "symbol": "ETH-USDT-SWAP",
        "side": "LONG",
        "layers": 1,
        "entry_regime": "TREND_BREAKOUT",
        # No trend_upgrade_* fields
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(old_json, f)
        path = f.name

    try:
        store = LiveStateStore(path=path)
        state = store.load()
        assert state is not None
        # Default values for new fields
        assert state.trend_upgrade_active is False
        assert state.trend_upgrade_addon_active is False
        assert state.trend_upgrade_addon_count == 0
        assert state.trend_upgrade_addon_entry_price is None
        assert state.trend_upgrade_addon_qty == 0.0
        assert state.trend_upgrade_addon_risk_budget_usdt == 0.0
        assert state.trend_upgrade_addon_sl_price is None
        assert state.trend_upgrade_last_ts_ms == 0
        assert state.position_management_mode is None
    finally:
        os.unlink(path)


# ======================================================================
# 2. New state with trend upgrade fields saves and loads
# ======================================================================


def test_new_state_saves_and_loads_trend_upgrade_fields():
    """State with trend_upgrade fields round-trips through JSON."""
    new_state = LivePositionState(
        position_id="test-002",
        symbol="ETH-USDT-SWAP",
        side="LONG",
        layers=1,
        entry_regime="TREND_UPGRADE_ADDON",
        trend_upgrade_active=True,
        trend_upgrade_addon_active=True,
        trend_upgrade_addon_count=2,
        trend_upgrade_addon_entry_price=3200.0,
        trend_upgrade_addon_qty=0.05,
        trend_upgrade_addon_risk_budget_usdt=2.0,
        trend_upgrade_addon_sl_price=3096.9,
        trend_upgrade_last_ts_ms=1000000,
        position_management_mode="TREND_UPGRADE_ADDON",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps({}))  # placeholder
        path = f.name

    try:
        store = LiveStateStore(path=path)
        store.save(new_state)
        loaded = store.load()
        assert loaded is not None
        assert loaded.trend_upgrade_active is True
        assert loaded.trend_upgrade_addon_active is True
        assert loaded.trend_upgrade_addon_count == 2
        assert loaded.trend_upgrade_addon_entry_price == 3200.0
        assert loaded.trend_upgrade_addon_qty == 0.05
        assert loaded.trend_upgrade_addon_risk_budget_usdt == 2.0
        assert loaded.trend_upgrade_addon_sl_price == 3096.9
        assert loaded.trend_upgrade_last_ts_ms == 1000000
        assert loaded.position_management_mode == "TREND_UPGRADE_ADDON"
    finally:
        os.unlink(path)


# ======================================================================
# 3. from_strategy_state maps trend upgrade fields
# ======================================================================


def test_from_strategy_state_maps_trend_upgrade_fields():
    """from_strategy_state correctly maps trend upgrade fields from StrategyPositionState."""
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

    strategy_state = StrategyPositionState(
        side="LONG",
        trend_upgrade_active=True,
        trend_upgrade_addon_active=True,
        trend_upgrade_addon_count=1,
        trend_upgrade_addon_entry_price=3100.0,
        trend_upgrade_addon_qty=0.03,
        trend_upgrade_addon_risk_budget_usdt=1.5,
        trend_upgrade_addon_sl_price=3000.0,
        trend_upgrade_last_ts_ms=500000,
        position_management_mode="TREND_UPGRADE",
    )

    live_state = LiveStateStore.from_strategy_state(
        position_id="test-003",
        symbol="ETH-USDT-SWAP",
        strategy_state=strategy_state,
        cash_before_position=None,
    )

    assert live_state.trend_upgrade_active is True
    assert live_state.trend_upgrade_addon_active is True
    assert live_state.trend_upgrade_addon_count == 1
    assert live_state.trend_upgrade_addon_entry_price == 3100.0
    assert live_state.trend_upgrade_addon_qty == 0.03
    assert live_state.trend_upgrade_addon_risk_budget_usdt == 1.5
    assert live_state.trend_upgrade_addon_sl_price == 3000.0
    assert live_state.trend_upgrade_last_ts_ms == 500000
    assert live_state.position_management_mode == "TREND_UPGRADE"


# ======================================================================
# 4. Default state has backward compatible defaults
# ======================================================================


def test_default_state_has_compatible_defaults():
    """Default LivePositionState has backward-compatible defaults."""
    state = LivePositionState()
    assert state.trend_upgrade_active is False
    assert state.trend_upgrade_addon_active is False
    assert state.trend_upgrade_addon_count == 0
    assert state.trend_upgrade_addon_entry_price is None
    assert state.trend_upgrade_addon_qty == 0.0
    assert state.trend_upgrade_addon_risk_budget_usdt == 0.0
    assert state.trend_upgrade_addon_sl_price is None
    assert state.trend_upgrade_last_ts_ms == 0
    assert state.position_management_mode is None


# ======================================================================
# 5. JSON partial fields load with defaults
# ======================================================================


def test_json_partial_fields_load_with_defaults():
    """JSON with only some trend_upgrade fields loads with defaults for others."""
    partial_json = {
        "position_id": "test-004",
        "symbol": "ETH-USDT-SWAP",
        "side": "SHORT",
        "trend_upgrade_active": True,
        "trend_upgrade_addon_count": 3,
        # Missing: addon_active, addon_entry_price, etc.
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(partial_json, f)
        path = f.name

    try:
        store = LiveStateStore(path=path)
        state = store.load()
        assert state is not None
        assert state.trend_upgrade_active is True
        assert state.trend_upgrade_addon_count == 3
        # Missing fields use class defaults
        assert state.trend_upgrade_addon_active is False
        assert state.trend_upgrade_addon_entry_price is None
        assert state.trend_upgrade_addon_qty == 0.0
        assert state.trend_upgrade_addon_risk_budget_usdt == 0.0
        assert state.trend_upgrade_addon_sl_price is None
        assert state.trend_upgrade_last_ts_ms == 0
        assert state.position_management_mode is None
    finally:
        os.unlink(path)


# ======================================================================
# 6. position_management_mode allowed values
# ======================================================================


def test_position_management_mode_values():
    """position_management_mode accepts expected values."""
    valid_modes = [
        "MEAN_REVERSION",
        "TREND_BREAKOUT",
        "TREND_UPGRADE",
        "TREND_UPGRADE_ADDON",
    ]
    for mode in valid_modes:
        state = LivePositionState(position_management_mode=mode)
        assert state.position_management_mode == mode
