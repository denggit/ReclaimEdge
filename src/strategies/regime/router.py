from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from src.strategies.regime.types import (
    RegimeDecision,
    RegimeDecisionType,
    RegimeSide,
    TrendState,
)

CooldownScope = Literal["SIDE", "GLOBAL"]


@dataclass(frozen=True)
class RouterInput:
    """All inputs the router needs to produce a single decision."""
    # Trend assessment
    trend_state: TrendState = TrendState.NO_TREND
    trend_confirmed: bool = False
    trend_confirmed_direction: RegimeSide | None = None
    trend_candidate_active: bool = False
    trend_candidate_direction: RegimeSide | None = None
    trend_failed: bool = False
    trend_failure_reason: Optional[str] = None

    # Mean-reversion assessment
    mr_long_allowed: bool = False
    mr_short_allowed: bool = False

    # Cooldown
    cooldown_side: RegimeSide | None = None
    cooldown_until_ts_ms: int = 0
    cooldown_scope: CooldownScope = "SIDE"

    # Current time
    ts_ms: int = 0


class RegimeRouter:
    """Central arbitrator that decides which regime (if any) gets the trade.

    Priority order:
    1. Conflict: opposite directions in same tick → CONFLICT_NO_TRADE
    2. Trend confirmed → TREND_LONG / TREND_SHORT
    3. Trend candidate active (not failed) → NO_TRADE (wait)
    4. Trend failed + mean-reversion allowed → MEAN_REVERSION_LONG/SHORT
    5. Only mean-reversion allowed → MEAN_REVERSION_LONG/SHORT
    6. Else → NO_TRADE

    Cooldown is enforced BEFORE any decision:
    - SIDE: blocks decisions on the same side (LONG blocks all LONG variants,
            SHORT blocks all SHORT variants)
    - GLOBAL: blocks both sides
    """

    def route(self, input_: RouterInput) -> RegimeDecision:
        ts = input_.ts_ms

        # ── Cooldown enforcement ──────────────────────────────────────
        if _is_cooldown_active(input_, ts):
            if input_.cooldown_scope == "GLOBAL":
                return RegimeDecision(
                    decision_type=RegimeDecisionType.NO_TRADE,
                    side=None,
                    reason="cooldown_global_active",
                    confidence=0.0,
                    trend_state=input_.trend_state,
                )
            # SIDE cooldown — we'll filter individual decisions below

        # ── Collect possible decisions this tick ──────────────────────
        possible: list[tuple[RegimeDecisionType, RegimeSide, str, float]] = []

        # Trend confirmed
        if input_.trend_confirmed and input_.trend_confirmed_direction is not None:
            trend_side = input_.trend_confirmed_direction
            dt = (
                RegimeDecisionType.TREND_LONG
                if trend_side == "LONG"
                else RegimeDecisionType.TREND_SHORT
            )
            possible.append((dt, trend_side, "trend_confirmed", 0.9))

        # Trend candidate active but not yet confirmed/failed
        if input_.trend_candidate_active and not input_.trend_confirmed and not input_.trend_failed:
            if input_.trend_candidate_direction is not None:
                # Don't add as possible — it blocks everything else
                return RegimeDecision(
                    decision_type=RegimeDecisionType.NO_TRADE,
                    side=None,
                    reason="trend_candidate_waiting_confirmation",
                    confidence=0.0,
                    trend_state=input_.trend_state,
                )

        # Mean-reversion allowed
        if input_.mr_long_allowed:
            possible.append((
                RegimeDecisionType.MEAN_REVERSION_LONG,
                "LONG",
                "mean_reversion_long_allowed",
                0.7,
            ))
        if input_.mr_short_allowed:
            possible.append((
                RegimeDecisionType.MEAN_REVERSION_SHORT,
                "SHORT",
                "mean_reversion_short_allowed",
                0.7,
            ))

        # ── Conflict detection ────────────────────────────────────────
        long_decision = [p for p in possible if p[1] == "LONG"]
        short_decision = [p for p in possible if p[1] == "SHORT"]

        if long_decision and short_decision:
            return RegimeDecision(
                decision_type=RegimeDecisionType.CONFLICT_NO_TRADE,
                side=None,
                reason="regime_conflict_trend_and_reclaim",
                confidence=0.0,
                trend_state=input_.trend_state,
            )

        # ── No decisions available ────────────────────────────────────
        if not possible:
            return RegimeDecision(
                decision_type=RegimeDecisionType.NO_TRADE,
                side=None,
                reason="no_candidate",
                confidence=0.0,
                trend_state=input_.trend_state,
            )

        # ── Pick the best decision (first is highest priority) ────────
        best = possible[0]
        dt, side, reason, confidence = best

        # ── Apply SIDE cooldown ───────────────────────────────────────
        if _is_cooldown_active(input_, ts) and input_.cooldown_scope == "SIDE":
            if input_.cooldown_side == side:
                # Check if there's an opposite-side decision
                opposite = [p for p in possible if p[1] != side]
                if opposite:
                    best_opposite = opposite[0]
                    dt, side, reason, confidence = best_opposite
                else:
                    return RegimeDecision(
                        decision_type=RegimeDecisionType.NO_TRADE,
                        side=None,
                        reason=f"cooldown_side_{input_.cooldown_side}_blocks_{side}",
                        confidence=0.0,
                        trend_state=input_.trend_state,
                    )

        return RegimeDecision(
            decision_type=dt,
            side=side,
            reason=reason,
            confidence=confidence,
            trend_state=input_.trend_state,
        )


def _is_cooldown_active(input_: RouterInput, ts_ms: int) -> bool:
    if input_.cooldown_side is None:
        return False
    if input_.cooldown_until_ts_ms <= 0:
        return False
    return ts_ms < input_.cooldown_until_ts_ms
