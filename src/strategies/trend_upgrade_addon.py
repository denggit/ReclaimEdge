"""Trend Upgrade Add-on — pure logic module.

Judges whether a Three-Stage runner position qualifies for Trend Upgrade
management and, when confirmed, computes the add-on risk budget and sizing
input.  No exchange calls, no I/O, no execution.

The module distinguishes two separate steps:

1. **Runner management upgrade** — runner active + same-side TREND_CONFIRMED
   => switch remaining runner to BOLL20 middle trailing SL management.

2. **Trend Upgrade Add-on** — runner already upgraded + TREND_CONFIRMED still
   holds + profit/risk/cooldown/notional all pass => allow an independent
   risk-sized add-on entry.

Neither step is the legacy "ADD" path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

PositionSide = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class TrendUpgradeAddonConfig:
    """Feature and cap configuration for Trend Upgrade Add-on.

    All parameters come from env vars — nothing is hardcoded.
    """

    enabled: bool = False
    profit_reinvest_ratio: float = 0.30
    max_addon_risk_pct: float = 0.002
    max_total_notional_multiplier: float = 1.0
    require_tp1_consumed: bool = True
    require_tp2_consumed: bool = True
    min_runner_remaining_ratio: float = 0.05
    min_trend_confidence: float = 0.80


@dataclass(frozen=True)
class TrendUpgradeAddonDecision:
    """Result of a Trend Upgrade Add-on eligibility assessment.

    *allowed* is the master gate — when False the caller must not proceed
    regardless of other field values.
    """

    allowed: bool
    reason: str
    side: PositionSide | None = None
    # ── Add-on risk budget (USD) ──────────────────────────────────────
    risk_budget_usdt: float = 0.0
    # ── Notional cap from existing position ───────────────────────────
    max_notional_usdt: float | None = None
    # ── Trend middle SL price for entry protective SL ─────────────────
    trend_sl_price: float | None = None
    # ── Trend confidence from regime decision ─────────────────────────
    confidence: float = 0.0
    # ── Whether runner management upgrade is allowed ──────────────────
    runner_upgrade_allowed: bool = False
    # ── Whether add-on entry is allowed (stronger gate) ───────────────
    addon_allowed: bool = False


def _notional_from_risk(
    *,
    price: float,
    stop_price: float,
    risk_budget_usdt: float,
    leverage: float,
    fee_slippage_buffer_pct: float,
    max_order_notional: float,
) -> tuple[float, float]:
    """Calculate notional and qty from a risk budget.

    Returns (notional_usdt, eth_qty).
    """
    if price <= 0 or stop_price <= 0 or price == stop_price:
        return 0.0, 0.0

    stop_distance_pct = abs(price - stop_price) / price
    effective_risk_pct = stop_distance_pct + fee_slippage_buffer_pct
    if effective_risk_pct <= 0:
        return 0.0, 0.0

    notional = risk_budget_usdt / effective_risk_pct
    if max_order_notional > 0:
        notional = min(notional, max_order_notional)
    qty = notional / price
    return notional, qty


def _calculate_realized_profit(
    *,
    side: str,
    avg_entry_price: float,
    total_entry_qty: float,
    tp1_price: float | None,
    tp1_ratio: float,
    tp2_price: float | None,
    tp2_ratio: float,
    tp1_consumed: bool,
    tp2_consumed: bool,
) -> float | None:
    """Calculate realized profit from TP1 and TP2 fills.

    Returns the total realized profit in USDT, or None when insufficient
    data prevents a reliable calculation.
    """
    if avg_entry_price <= 0 or total_entry_qty <= 0:
        return None

    realized = 0.0
    can_calculate = False

    # ── TP1 realized profit ──────────────────────────────────────────
    if tp1_consumed and tp1_price is not None and tp1_ratio > 0:
        tp1_qty = total_entry_qty * tp1_ratio
        if side == "LONG":
            tp1_profit = (tp1_price - avg_entry_price) * tp1_qty
        else:
            tp1_profit = (avg_entry_price - tp1_price) * tp1_qty
        if tp1_profit > 0:
            realized += tp1_profit
            can_calculate = True

    # ── TP2 realized profit ──────────────────────────────────────────
    if tp2_consumed and tp2_price is not None and tp2_ratio > 0:
        tp2_qty = total_entry_qty * tp2_ratio
        if side == "LONG":
            tp2_profit = (tp2_price - avg_entry_price) * tp2_qty
        else:
            tp2_profit = (avg_entry_price - tp2_price) * tp2_qty
        if tp2_profit > 0:
            realized += tp2_profit
            can_calculate = True

    if not can_calculate:
        return None

    return max(realized, 0.0)


def assess_trend_upgrade(
    *,
    config: TrendUpgradeAddonConfig,
    # ── Position state ────────────────────────────────────────────────
    has_position: bool,
    position_side: PositionSide | None,
    entry_regime: str | None,
    three_stage_runner_enabled_for_position: bool,
    three_stage_tp1_consumed: bool,
    three_stage_tp2_consumed: bool,
    three_stage_tp1_ratio: float,
    three_stage_tp2_ratio: float,
    three_stage_runner_ratio: float,
    trend_runner_active: bool,
    # ── Trend assessment ──────────────────────────────────────────────
    trend_confirmed: bool,
    trend_direction: str | None,  # "LONG" | "SHORT" | None
    trend_confidence: float,
    trend_state: str,
    trend_blocks_mean_reversion: bool,
    # ── Cooldown / halt / exit ────────────────────────────────────────
    post_entry_sl_cooldown_active_same_side: bool,
    delayed_market_exit_armed: bool,
    trading_halt_active: bool = False,
    # ── Profit data ───────────────────────────────────────────────────
    avg_entry_price: float,
    total_entry_qty: float,
    three_stage_tp1_price: float | None,
    three_stage_tp2_price: float | None,
    # ── Sizing inputs ─────────────────────────────────────────────────
    equity_usdt: float,
    leverage: float,
    fee_slippage_buffer_pct: float,
    max_order_notional_usdt: float,
    current_total_notional: float,
    # ── Trend SL inputs ───────────────────────────────────────────────
    boll_middle: float,
    trend_middle_sl_buffer_pct: float,
    # ── Current price ─────────────────────────────────────────────────
    price: float,
    ts_ms: int,
) -> TrendUpgradeAddonDecision:
    """Assess whether a Three-Stage runner qualifies for Trend Upgrade management
    and/or an independent risk-sized add-on entry.

    This is pure logic — it does not read/write state, execute orders, or
    call external services.  The caller is responsible for applying the
    decision.

    Returns:
        A ``TrendUpgradeAddonDecision`` with *allowed* gates.
    """
    # ── 1. Feature gate ───────────────────────────────────────────────
    if not config.enabled:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_addon_disabled",
        )

    # ── 2. Position gates ─────────────────────────────────────────────
    if not has_position:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_no_position",
        )
    if position_side is None:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_no_position_side",
        )

    # ── 3. Guard: entry_regime must NOT already be TREND_UPGRADE_ADDON
    #    (prevent repeat add-on triggers from the same upgrade episode) ──
    if entry_regime == "TREND_UPGRADE_ADDON":
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_addon_already_active",
        )

    # ── 4. Three-Stage preconditions ──────────────────────────────────
    if config.require_tp1_consumed and not three_stage_tp1_consumed:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_tp1_not_consumed",
        )
    if config.require_tp2_consumed and not three_stage_tp2_consumed:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_tp2_not_consumed",
        )

    # Check runner still has meaningful remaining size
    runner_remaining_ratio = three_stage_runner_ratio
    if runner_remaining_ratio < config.min_runner_remaining_ratio:
        return TrendUpgradeAddonDecision(
            allowed=False,
            reason="trend_upgrade_runner_ratio_below_minimum",
        )

    # ── 5. Trend direction must be confirmed ───────────────────────────
    if not trend_confirmed:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_trend_not_confirmed",
        )
    if trend_direction is None:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_no_trend_direction",
        )

    # ── 6. Same-side ONLY ──────────────────────────────────────────────
    if trend_direction != position_side:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_reverse_blocked",
        )

    # ── 7. Trend confidence gate ───────────────────────────────────────
    if trend_confidence < config.min_trend_confidence:
        return TrendUpgradeAddonDecision(
            allowed=False,
            reason=f"trend_upgrade_confidence_below_minimum ({trend_confidence:.2f} < {config.min_trend_confidence:.2f})",
        )

    # ── 8. Trend must block mean-reversion and be confirmed ────────────
    if not trend_blocks_mean_reversion:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_not_blocking_mean_reversion",
        )

    # ── 9. Safety gates ────────────────────────────────────────────────
    if post_entry_sl_cooldown_active_same_side:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_post_entry_sl_cooldown_active",
        )
    if delayed_market_exit_armed:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_delayed_market_exit_armed",
        )
    if trading_halt_active:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_trading_halt_active",
        )

    # ── 10. Calculate trend middle SL ──────────────────────────────────
    try:
        from src.strategies.trend_middle_trailing_sl import calculate_trend_middle_sl

        trend_sl = calculate_trend_middle_sl(
            boll_middle=boll_middle,
            buffer_pct=trend_middle_sl_buffer_pct,
            side=position_side,
        )
    except (ValueError, ImportError):
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_sl_calculation_failed",
        )

    # ── 11. Validate SL is on correct side of price ────────────────────
    if position_side == "LONG" and trend_sl >= price:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_sl_above_or_at_price_long",
        )
    if position_side == "SHORT" and trend_sl <= price:
        return TrendUpgradeAddonDecision(
            allowed=False, reason="trend_upgrade_sl_below_or_at_price_short",
        )

    # ── Runner management upgrade is allowed at this point ─────────────
    runner_upgrade_allowed = True

    # ── 12. Calculate realized profit ──────────────────────────────────
    realized_profit = _calculate_realized_profit(
        side=position_side,
        avg_entry_price=avg_entry_price,
        total_entry_qty=total_entry_qty,
        tp1_price=three_stage_tp1_price,
        tp1_ratio=three_stage_tp1_ratio,
        tp2_price=three_stage_tp2_price,
        tp2_ratio=three_stage_tp2_ratio,
        tp1_consumed=three_stage_tp1_consumed,
        tp2_consumed=three_stage_tp2_consumed,
    )

    if realized_profit is None:
        # Runner can still be upgraded, but add-on is blocked
        return TrendUpgradeAddonDecision(
            allowed=True,
            reason="trend_upgrade_profit_budget_unavailable",
            side=position_side,
            trend_sl_price=trend_sl,
            confidence=trend_confidence,
            runner_upgrade_allowed=True,
            addon_allowed=False,
        )

    # ── 13. Calculate add-on risk budget ───────────────────────────────
    normal_risk = equity_usdt * config.max_addon_risk_pct
    profit_risk = realized_profit * config.profit_reinvest_ratio
    risk_budget = min(normal_risk, profit_risk)

    if risk_budget <= 0:
        return TrendUpgradeAddonDecision(
            allowed=True,
            reason="trend_upgrade_risk_budget_zero_or_negative",
            side=position_side,
            risk_budget_usdt=0.0,
            trend_sl_price=trend_sl,
            confidence=trend_confidence,
            runner_upgrade_allowed=True,
            addon_allowed=False,
        )

    # ── 14. Calculate max notional cap ─────────────────────────────────
    max_notional = None
    if config.max_total_notional_multiplier > 0 and current_total_notional > 0:
        max_notional = current_total_notional * config.max_total_notional_multiplier
    if max_order_notional_usdt > 0:
        if max_notional is None:
            max_notional = max_order_notional_usdt
        else:
            max_notional = min(max_notional, max_order_notional_usdt)
    elif max_order_notional_usdt > 0:
        max_notional = max_order_notional_usdt

    # ── 15. Verify add-on qty > 0 ─────────────────────────────────────
    addon_notional, addon_qty = _notional_from_risk(
        price=price,
        stop_price=trend_sl,
        risk_budget_usdt=risk_budget,
        leverage=leverage,
        fee_slippage_buffer_pct=fee_slippage_buffer_pct,
        max_order_notional=max_notional if max_notional is not None else max_order_notional_usdt,
    )

    if addon_qty <= 0 or addon_notional <= 0:
        return TrendUpgradeAddonDecision(
            allowed=True,
            reason="trend_upgrade_addon_qty_zero",
            side=position_side,
            risk_budget_usdt=risk_budget,
            max_notional_usdt=max_notional,
            trend_sl_price=trend_sl,
            confidence=trend_confidence,
            runner_upgrade_allowed=True,
            addon_allowed=False,
        )

    # ── 16. All gates passed — add-on is allowed ───────────────────────
    return TrendUpgradeAddonDecision(
        allowed=True,
        reason="trend_upgrade_addon_allowed",
        side=position_side,
        risk_budget_usdt=risk_budget,
        max_notional_usdt=max_notional,
        trend_sl_price=trend_sl,
        confidence=trend_confidence,
        runner_upgrade_allowed=True,
        addon_allowed=True,
    )
