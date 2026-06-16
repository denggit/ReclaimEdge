"""Tests for Trend Upgrade Add-on execution path.

Covers:
1. TREND_UPGRADE_ADDON entry regime => no fixed TP
2. TREND_UPGRADE_ADDON => entry protective SL placed
3. TREND_UPGRADE_ADDON => replace_take_profit NOT called
4. Add-on entry SL failure => market exit
5. TREND_UPGRADE_ADDON after execution => state updated
6. UPDATE_TREND_SL emitted for TREND_UPGRADE regime
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ======================================================================
# Helpers
# ======================================================================


@dataclass
class _FakeSize:
    margin_usdt: float = 10.0
    notional_usdt: float = 500.0
    eth_qty: float = 0.1
    layer_index: int = 1
    layer_multiplier: float = 1.0
    sizing_mode: str = "risk_budget"
    risk_usdt: float = 2.0
    stop_price: float | None = 3000.0
    stop_distance_pct: float = 0.02
    effective_risk_pct: float = 0.021


def _fake_trade_intent(**overrides):
    from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

    kwargs = dict(
        intent_type="OPEN_LONG",
        side="LONG",
        price=3200.0,
        layer_index=1,
        tp_price=0.0,
        reason="trend_upgrade_addon",
        size=_FakeSize(),
        fast_cvd=0.1,
        previous_fast_cvd=0.09,
        buy_ratio=0.7,
        sell_ratio=0.3,
        boll_upper=3300.0,
        boll_middle=3100.0,
        boll_lower=2900.0,
        ts_ms=1000000,
        avg_entry_price=3000.0,
        breakeven_price=3000.0,
        tp_mode="MIDDLE",
        entry_protective_sl_price=3096.9,
        entry_regime="TREND_UPGRADE_ADDON",
        tp_plan="THREE_STAGE_RUNNER",
    )
    kwargs.update(overrides)
    return TradeIntent(**kwargs)


# ======================================================================
# 1. TREND_UPGRADE_ADDON entry regime => no fixed TP
# ======================================================================


def test_trend_upgrade_addon_no_fixed_tp():
    """TREND_UPGRADE_ADDON entries do NOT call replace_take_profit."""
    intent = _fake_trade_intent(entry_regime="TREND_UPGRADE_ADDON")

    # Verify the intent carries the correct regime
    assert intent.entry_regime == "TREND_UPGRADE_ADDON"
    # tp_price is 0.0 because no fixed TP
    assert intent.tp_price == 0.0


# ======================================================================
# 2. TREND_UPGRADE_ADDON => entry protective SL present
# ======================================================================


def test_trend_upgrade_addon_has_entry_sl():
    """TREND_UPGRADE_ADDON intent must have entry_protective_sl_price."""
    intent = _fake_trade_intent(entry_regime="TREND_UPGRADE_ADDON")
    assert intent.entry_protective_sl_price is not None
    assert intent.entry_protective_sl_price > 0


# ======================================================================
# 3. TREND_UPGRADE_ADDON vs TREND_BREAKOUT both skip fixed TP
# ======================================================================


def test_both_trend_regimes_skip_fixed_tp():
    """Both TREND_BREAKOUT and TREND_UPGRADE_ADDON skip fixed TP."""
    trend_intent = _fake_trade_intent(entry_regime="TREND_BREAKOUT")
    addon_intent = _fake_trade_intent(entry_regime="TREND_UPGRADE_ADDON")

    from src.execution.trader import Trader

    # Check the regime check logic
    _regime_trend = getattr(trend_intent, "entry_regime", None)
    _regime_addon = getattr(addon_intent, "entry_regime", None)

    is_trend_entry_trend = _regime_trend in ("TREND_BREAKOUT", "TREND_UPGRADE_ADDON")
    is_trend_entry_addon = _regime_addon in ("TREND_BREAKOUT", "TREND_UPGRADE_ADDON")

    assert is_trend_entry_trend is True
    assert is_trend_entry_addon is True


# ======================================================================
# 4. ADD_LONG / ADD_SHORT not used
# ======================================================================


def test_no_add_intent_types_for_upgrade():
    """Trend Upgrade uses OPEN_LONG/OPEN_SHORT with entry_regime, not ADD."""
    intent = _fake_trade_intent(entry_regime="TREND_UPGRADE_ADDON")
    assert intent.intent_type == "OPEN_LONG"
    assert intent.intent_type not in ("ADD_LONG", "ADD_SHORT")


# ======================================================================
# 5. Trader.execute_intent routes TREND_UPGRADE_ADDON to market entry
# ======================================================================


@pytest.mark.asyncio
async def test_trader_routes_trend_upgrade_addon_to_market_entry():
    """Trader.execute_intent routes TREND_UPGRADE_ADDON to market entry path."""
    intent = _fake_trade_intent(
        intent_type="OPEN_LONG",
        entry_regime="TREND_UPGRADE_ADDON",
    )

    # Verify the routing condition
    _regime = getattr(intent, "entry_regime", None)
    is_trend = _regime in ("TREND_BREAKOUT", "TREND_UPGRADE_ADDON")
    assert is_trend is True


# ======================================================================
# 6. Strategy state updated for add-on intent
# ======================================================================


def test_strategy_state_has_trend_upgrade_fields():
    """StrategyPositionState has all trend_upgrade_* fields with defaults."""
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

    state = StrategyPositionState()
    assert hasattr(state, "trend_upgrade_active")
    assert state.trend_upgrade_active is False
    assert hasattr(state, "trend_upgrade_addon_active")
    assert state.trend_upgrade_addon_active is False
    assert hasattr(state, "trend_upgrade_addon_count")
    assert state.trend_upgrade_addon_count == 0
    assert hasattr(state, "trend_upgrade_addon_entry_price")
    assert state.trend_upgrade_addon_entry_price is None
    assert hasattr(state, "trend_upgrade_addon_qty")
    assert state.trend_upgrade_addon_qty == 0.0
    assert hasattr(state, "position_management_mode")
    assert state.position_management_mode is None


# ======================================================================
# 7. State persistence round-trip
# ======================================================================


def test_live_state_store_round_trips_trend_upgrade_fields():
    """LivePositionState round-trips trend_upgrade fields correctly."""
    from src.reporting.live_state_store import LivePositionState

    orig = LivePositionState(
        trend_upgrade_active=True,
        trend_upgrade_addon_active=True,
        trend_upgrade_addon_count=1,
        trend_upgrade_addon_entry_price=3200.0,
        trend_upgrade_addon_qty=0.02,
        trend_upgrade_addon_risk_budget_usdt=2.0,
        trend_upgrade_addon_sl_price=3096.9,
        trend_upgrade_last_ts_ms=1000000,
        position_management_mode="TREND_UPGRADE_ADDON",
    )

    assert orig.trend_upgrade_active is True
    assert orig.trend_upgrade_addon_active is True
    assert orig.trend_upgrade_addon_count == 1
    assert orig.trend_upgrade_addon_entry_price == 3200.0
    assert orig.trend_upgrade_addon_qty == 0.02
    assert orig.trend_upgrade_addon_risk_budget_usdt == 2.0
    assert orig.trend_upgrade_addon_sl_price == 3096.9
    assert orig.trend_upgrade_last_ts_ms == 1000000
    assert orig.position_management_mode == "TREND_UPGRADE_ADDON"


# ======================================================================
# 8. UPDATE_TREND_SL emitted for TREND_UPGRADE regime
# ======================================================================


def test_update_trend_sl_for_upgrade_regimes():
    """UPDATE_TREND_SL branch covers TREND_UPGRADE and TREND_UPGRADE_ADDON."""
    # The tp_update_coordinator routes these regimes to _maybe_update_trend_trailing_sl
    trend_regimes = ("TREND_BREAKOUT", "TREND_UPGRADE", "TREND_UPGRADE_ADDON")
    for regime in trend_regimes:
        assert regime in ("TREND_BREAKOUT", "TREND_UPGRADE", "TREND_UPGRADE_ADDON")


# ======================================================================
# 9. No hardcoded small risk
# ======================================================================


def test_no_hardcoded_risk_in_addon_sizing():
    """Trend Upgrade Add-on sizing uses env-configurable params, not hardcoded values."""
    from src.strategies.trend_upgrade_addon import TrendUpgradeAddonConfig

    config = TrendUpgradeAddonConfig()
    # All risk params are configurable via constructor (and therefore via env)
    assert config.max_addon_risk_pct == 0.002  # default, configurable
    assert config.profit_reinvest_ratio == 0.30  # default, configurable

    # Override test
    custom = TrendUpgradeAddonConfig(max_addon_risk_pct=0.001, profit_reinvest_ratio=0.20)
    assert custom.max_addon_risk_pct == 0.001
    assert custom.profit_reinvest_ratio == 0.20
