"""Tests for Trend Upgrade Add-on pure logic module.

Covers:
1. TP1/TP2 not consumed => no upgrade
2. Same-side trend confirmed => runner upgrade allowed
3. Opposite-side trend confirmed => no upgrade
4. Cooldown blocks same-side upgrade
5. Profit budget unavailable => no add-on
6. Add-on risk budget correctness
7. Add-on uses independent risk budget sizing
8. Trend candidate => no add-on
9. Trend failed => no add-on
10. No trend => no add-on
"""

from __future__ import annotations

import pytest

from src.strategies.trend_upgrade_addon import (
    TrendUpgradeAddonConfig,
    TrendUpgradeAddonDecision,
    assess_trend_upgrade,
)


def _base_config(**overrides) -> TrendUpgradeAddonConfig:
    kwargs = dict(
        enabled=True,
        profit_reinvest_ratio=0.30,
        max_addon_risk_pct=0.002,
        max_total_notional_multiplier=1.0,
        require_tp1_consumed=True,
        require_tp2_consumed=True,
        min_runner_remaining_ratio=0.05,
        min_trend_confidence=0.80,
    )
    kwargs.update(overrides)
    return TrendUpgradeAddonConfig(**kwargs)


def _base_args(**overrides):
    """Return a dict with the default arguments for assess_trend_upgrade."""
    args = dict(
        config=_base_config(),
        has_position=True,
        position_side="LONG",
        entry_regime="MEAN_REVERSION",
        three_stage_runner_enabled_for_position=True,
        three_stage_tp1_consumed=True,
        three_stage_tp2_consumed=True,
        three_stage_tp1_ratio=0.60,
        three_stage_tp2_ratio=0.20,
        three_stage_runner_ratio=0.20,
        trend_runner_active=False,
        trend_confirmed=True,
        trend_direction="LONG",
        trend_confidence=0.90,
        trend_state="TREND_UP_CONFIRMED",
        trend_blocks_mean_reversion=True,
        post_entry_sl_cooldown_active_same_side=False,
        delayed_market_exit_armed=False,
        trading_halt_active=False,
        avg_entry_price=3000.0,
        total_entry_qty=1.0,
        three_stage_tp1_price=3100.0,
        three_stage_tp2_price=3200.0,
        equity_usdt=1000.0,
        leverage=20.0,
        fee_slippage_buffer_pct=0.001,
        max_order_notional_usdt=0.0,
        current_total_notional=3000.0,
        boll_middle=3050.0,
        trend_middle_sl_buffer_pct=0.001,
        price=3200.0,
        ts_ms=1000000,
    )
    args.update(overrides)
    return args


# ======================================================================
# 1. TP1/TP2 not consumed => no upgrade
# ======================================================================


def test_tp1_not_consumed_blocks_upgrade():
    """TP1 consumed=false => no upgrade."""
    decision = assess_trend_upgrade(**_base_args(
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=True,
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_tp1_not_consumed"


def test_tp2_not_consumed_blocks_upgrade():
    """TP2 consumed=false => no upgrade."""
    decision = assess_trend_upgrade(**_base_args(
        three_stage_tp1_consumed=True,
        three_stage_tp2_consumed=False,
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_tp2_not_consumed"


# ======================================================================
# 2. Same-side trend confirmed => runner upgrade allowed
# ======================================================================


def test_same_side_trend_confirmed_allows_runner_upgrade_long():
    """LONG position + TREND_LONG confirmed => runner upgrade allowed."""
    decision = assess_trend_upgrade(**_base_args(
        position_side="LONG",
        trend_direction="LONG",
    ))
    assert decision.allowed
    assert decision.runner_upgrade_allowed
    assert decision.trend_sl_price is not None
    assert decision.side == "LONG"


def test_same_side_trend_confirmed_allows_runner_upgrade_short():
    """SHORT position + TREND_SHORT confirmed => runner upgrade allowed."""
    decision = assess_trend_upgrade(**_base_args(
        position_side="SHORT",
        trend_direction="SHORT",
        boll_middle=3000.0,
        price=2900.0,
        avg_entry_price=3000.0,
        three_stage_tp1_price=2900.0,
        three_stage_tp2_price=2800.0,
    ))
    assert decision.allowed
    assert decision.runner_upgrade_allowed
    assert decision.side == "SHORT"


# ======================================================================
# 3. Opposite-side trend => blocked
# ======================================================================


def test_opposite_side_trend_blocked():
    """LONG position + TREND_SHORT confirmed => no upgrade."""
    decision = assess_trend_upgrade(**_base_args(
        position_side="LONG",
        trend_direction="SHORT",
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_reverse_blocked"


# ======================================================================
# 4. Cooldown blocks same-side upgrade
# ======================================================================


def test_cooldown_blocks_same_side():
    """LONG cooldown active with LONG position => no upgrade."""
    decision = assess_trend_upgrade(**_base_args(
        position_side="LONG",
        trend_direction="LONG",
        post_entry_sl_cooldown_active_same_side=True,
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_post_entry_sl_cooldown_active"


# ======================================================================
# 5. Profit budget unavailable => runner upgrade but no add-on
# ======================================================================


def test_profit_budget_unavailable_blocks_addon():
    """Realized profit cannot be calculated => runner upgrade but no add-on."""
    decision = assess_trend_upgrade(**_base_args(
        avg_entry_price=0.0,  # prevents profit calc
        three_stage_tp1_consumed=True,
        three_stage_tp2_consumed=True,
    ))
    assert decision.allowed  # runner upgrade still allowed
    assert decision.runner_upgrade_allowed
    assert not decision.addon_allowed
    assert decision.reason == "trend_upgrade_profit_budget_unavailable"


def test_profit_budget_unavailable_no_tp_price():
    """No TP prices => profit unavailable."""
    decision = assess_trend_upgrade(**_base_args(
        three_stage_tp1_price=None,
        three_stage_tp2_price=None,
    ))
    assert decision.runner_upgrade_allowed
    assert not decision.addon_allowed
    assert decision.reason == "trend_upgrade_profit_budget_unavailable"


# ======================================================================
# 6. Add-on risk budget correct
# ======================================================================


def test_addon_risk_budget_calculation():
    """normal_risk = equity * max_addon_risk_pct, profit_risk = realized * reinvest_ratio
    risk_budget = min(normal_risk, profit_risk)"""
    config = _base_config(max_addon_risk_pct=0.002, profit_reinvest_ratio=0.30)
    args = _base_args(
        config=config,
        equity_usdt=1000.0,
        avg_entry_price=3000.0,
        total_entry_qty=1.0,
        three_stage_tp1_price=3100.0,  # profit = (3100-3000)*0.60 = 60
        three_stage_tp2_price=3200.0,  # profit = (3200-3000)*0.20 = 40
    )
    decision = assess_trend_upgrade(**args)

    # normal_risk = 1000 * 0.002 = 2.0
    # TP1 profit = (3100-3000)*0.60*1.0 = 60
    # TP2 profit = (3200-3000)*0.20*1.0 = 40
    # realized = 100, profit_risk = 100 * 0.30 = 30
    # risk_budget = min(2.0, 30) = 2.0
    assert decision.addon_allowed
    assert decision.risk_budget_usdt == pytest.approx(2.0, rel=0.01)


def test_addon_risk_budget_limited_by_normal():
    """When normal risk < profit risk, budget = normal risk."""
    config = _base_config(max_addon_risk_pct=0.001, profit_reinvest_ratio=0.30)
    args = _base_args(
        config=config,
        equity_usdt=1000.0,
        avg_entry_price=3000.0,
        total_entry_qty=10.0,
        three_stage_tp1_price=3100.0,  # profit = 100*10*0.60 = 600
        three_stage_tp2_price=3200.0,  # profit = 200*10*0.20 = 400
    )
    decision = assess_trend_upgrade(**args)

    # normal_risk = 1000 * 0.001 = 1.0
    # profit_risk = 1000 * 0.30 = 300
    # risk_budget = min(1.0, 300) = 1.0
    assert decision.addon_allowed
    assert decision.risk_budget_usdt == pytest.approx(1.0, rel=0.01)


def test_addon_risk_budget_limited_by_profit():
    """When profit risk < normal risk, budget = profit risk."""
    config = _base_config(max_addon_risk_pct=0.01, profit_reinvest_ratio=0.30)
    args = _base_args(
        config=config,
        equity_usdt=1000.0,
        avg_entry_price=3000.0,
        total_entry_qty=0.1,
        three_stage_tp1_price=3100.0,  # profit = 100*0.1*0.60 = 6
        three_stage_tp2_price=3200.0,  # profit = 200*0.1*0.20 = 4
    )
    decision = assess_trend_upgrade(**args)

    # normal_risk = 1000 * 0.01 = 10
    # profit_risk = 10 * 0.30 = 3
    # risk_budget = min(10, 3) = 3
    assert decision.addon_allowed
    assert decision.risk_budget_usdt == pytest.approx(3.0, rel=0.01)


# ======================================================================
# 7. Trend candidate => no add-on
# ======================================================================


def test_trend_candidate_no_addon():
    """Trend candidate (not confirmed) => no add-on."""
    decision = assess_trend_upgrade(**_base_args(
        trend_confirmed=False,
        trend_state="TREND_UP_CANDIDATE",
        trend_blocks_mean_reversion=True,
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_trend_not_confirmed"


# ======================================================================
# 8. Trend failed => no add-on
# ======================================================================


def test_trend_failed_no_addon():
    """Trend failed => no add-on."""
    decision = assess_trend_upgrade(**_base_args(
        trend_confirmed=False,
        trend_state="TREND_FAILED",
    ))
    assert not decision.allowed


# ======================================================================
# 9. No trend => no add-on
# ======================================================================


def test_no_trend_no_addon():
    """No trend => no add-on."""
    decision = assess_trend_upgrade(**_base_args(
        trend_confirmed=False,
        trend_direction=None,
        trend_state="NO_TREND",
    ))
    assert not decision.allowed


# ======================================================================
# 10. Already TREND_UPGRADE_ADDON entry_regime => no repeat
# ======================================================================


def test_already_addon_active_no_repeat():
    """entry_regime already TREND_UPGRADE_ADDON => no repeat."""
    decision = assess_trend_upgrade(**_base_args(
        entry_regime="TREND_UPGRADE_ADDON",
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_addon_already_active"


# ======================================================================
# 11. Feature disabled
# ======================================================================


def test_disabled_config_blocks():
    """Config enabled=False => no upgrade."""
    decision = assess_trend_upgrade(**_base_args(
        config=_base_config(enabled=False),
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_addon_disabled"


# ======================================================================
# 12. No position blocks
# ======================================================================


def test_no_position_blocks():
    """has_position=False => no upgrade."""
    decision = assess_trend_upgrade(**_base_args(
        has_position=False,
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_no_position"


# ======================================================================
# 13. Delayed exit blocks
# ======================================================================


def test_delayed_exit_blocks():
    """delayed_market_exit_armed=True => no upgrade."""
    decision = assess_trend_upgrade(**_base_args(
        delayed_market_exit_armed=True,
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_delayed_market_exit_armed"


# ======================================================================
# 14. Trading halt blocks
# ======================================================================


def test_trading_halt_blocks():
    """trading_halt_active=True => no upgrade."""
    decision = assess_trend_upgrade(**_base_args(
        trading_halt_active=True,
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_trading_halt_active"


# ======================================================================
# 15. Low trend confidence blocks
# ======================================================================


def test_low_confidence_blocks():
    """Confidence below min => no upgrade."""
    config = _base_config(min_trend_confidence=0.90)
    decision = assess_trend_upgrade(**_base_args(
        config=config,
        trend_confidence=0.80,
    ))
    assert not decision.allowed
    assert "trend_upgrade_confidence_below_minimum" in decision.reason


# ======================================================================
# 16. Invalid SL blocks
# ======================================================================


def test_sl_above_price_blocks_long():
    """LONG with SL above price => blocked."""
    decision = assess_trend_upgrade(**_base_args(
        position_side="LONG",
        price=3000.0,
        boll_middle=3100.0,  # SL would be 3100*0.999 = 3096.9 > 3000
    ))
    assert not decision.allowed
    assert "sl_above_or_at_price_long" in decision.reason


# ======================================================================
# 17. Runner remaining ratio below minimum
# ======================================================================


def test_runner_ratio_below_minimum():
    """Runner ratio below min => no upgrade."""
    config = _base_config(min_runner_remaining_ratio=0.15)
    decision = assess_trend_upgrade(**_base_args(
        config=config,
        three_stage_runner_ratio=0.05,
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_runner_ratio_below_minimum"


# ======================================================================
# 18. Not blocking mean reversion
# ======================================================================


def test_not_blocking_mean_reversion():
    """Trend doesn't block mean-reversion => no add-on."""
    decision = assess_trend_upgrade(**_base_args(
        trend_blocks_mean_reversion=False,
    ))
    assert not decision.allowed
    assert decision.reason == "trend_upgrade_not_blocking_mean_reversion"


# ======================================================================
# 19. Short-side add-on risk budget test
# ======================================================================


def test_short_addon_risk_budget():
    """SHORT position add-on risk budget calculation."""
    decision = assess_trend_upgrade(**_base_args(
        position_side="SHORT",
        trend_direction="SHORT",
        avg_entry_price=3200.0,
        total_entry_qty=1.0,
        three_stage_tp1_price=3100.0,  # profit = (3200-3100)*0.60 = 60
        three_stage_tp2_price=3000.0,  # profit = (3200-3000)*0.20 = 40
        boll_middle=3050.0,
        price=2900.0,
    ))
    assert decision.addon_allowed
    # normal_risk = 1000 * 0.002 = 2.0
    # realized = 100, profit_risk = 30
    # risk_budget = min(2.0, 30) = 2.0
    assert decision.risk_budget_usdt == pytest.approx(2.0, rel=0.01)


# ======================================================================
# 20. No legacy ADD regression — module should not contain ADD references
# ======================================================================


def test_no_legacy_add_in_module():
    """Verify the module does not reference ADD_LONG / ADD_SHORT."""
    import inspect
    from src.strategies import trend_upgrade_addon

    source = inspect.getsource(trend_upgrade_addon)
    assert "ADD_LONG" not in source
    assert "ADD_SHORT" not in source
    assert "ADD_" not in source.split("TrendUpgrade")[0]  # No legacy ADD prefix
