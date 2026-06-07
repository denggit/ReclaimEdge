from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from src.reporting.report_models import (
    ReportPnlMath,
    ResidualPnlBucket,
    _to_float,
)
from src.reporting.trade_journal import JournalEvent

if TYPE_CHECKING:
    from src.reporting.report_models import ReportRuntimeContext


# ---------------------------------------------------------------------------
# net cash transfer (only CASH_TRANSFER, never ACCOUNT_CASH_DRIFT)
# ---------------------------------------------------------------------------


def net_cash_transfer(events: list[JournalEvent]) -> float:
    total = 0.0
    for event in events:
        if event.event_type != "CASH_TRANSFER":
            continue
        amount = _to_float(event.payload.get("amount"))
        if amount is not None:
            total += amount
    return total


# ---------------------------------------------------------------------------
# PnL math
# ---------------------------------------------------------------------------


def calculate_pnl_math(
    events: list[JournalEvent],
    known_closed_pnl: float,
    context: ReportRuntimeContext | None,
) -> ReportPnlMath:
    nct = net_cash_transfer(events)
    period_start_cash = context.period_start_cash if context else None
    current_cash = context.current_cash if context else None
    strategy_total_pnl = None
    residual_pnl = None
    total_pnl = None
    period_start_value: float | None = None
    current_account_value: float | None = None
    current_account_value_source = "cash"

    # --- determine period start value: prefer equity if available ---
    if context is not None and context.period_start_equity is not None:
        period_start_value = context.period_start_equity
    else:
        period_start_value = period_start_cash

    # --- determine current account value: prefer equity when holding a position ---
    if (
        context is not None
        and context.current_has_position
        and context.current_equity is not None
    ):
        current_account_value = context.current_equity
        current_account_value_source = "equity"
    else:
        current_account_value = current_cash
        current_account_value_source = "cash"

    if current_account_value is not None and period_start_value is not None:
        strategy_total_pnl = current_account_value - period_start_value - nct
        residual_pnl = strategy_total_pnl - known_closed_pnl
        total_pnl = strategy_total_pnl

    return ReportPnlMath(
        period_start_cash=period_start_cash,
        current_cash=current_cash,
        net_transfer=nct,
        known_closed_pnl=known_closed_pnl,
        strategy_total_pnl=strategy_total_pnl,
        residual_pnl=residual_pnl,
        total_pnl=total_pnl,
        period_start_value=period_start_value,
        current_account_value=current_account_value,
        current_account_value_source=current_account_value_source,
    )


# ---------------------------------------------------------------------------
# residual bucket builder
# ---------------------------------------------------------------------------


def build_residual_bucket(
    events: list[JournalEvent],
    incomplete_count: int,
    known_closed_pnl: float,
    context: ReportRuntimeContext | None,
) -> ResidualPnlBucket:
    math = calculate_pnl_math(events, known_closed_pnl, context)

    if math.current_account_value_source == "equity":
        formula = "current_equity - period_start_value - net_transfer - known_closed_pnl (有仓, 优先使用 equity)"
    else:
        formula = "current_cash - period_start_cash - net_transfer - known_closed_pnl"

    if context is None or context.current_cash is None or context.period_start_cash is None:
        return ResidualPnlBucket(
            incomplete_count=incomplete_count,
            pnl=None,
            cash_start=context.period_start_cash if context else None,
            cash_end=context.current_cash if context else None,
            net_transfer=math.net_transfer,
            strategy_total_pnl=None,
            known_closed_pnl=known_closed_pnl,
            formula=formula,
            note="missing cash context; incomplete records hidden but not valued",
            period_start_value=math.period_start_value,
            current_account_value=math.current_account_value,
            current_account_value_source=math.current_account_value_source,
        )

    if incomplete_count <= 0 and math.residual_pnl == 0:
        note = "no incomplete records"
    elif incomplete_count <= 0:
        if math.current_account_value_source == "equity":
            note = "no incomplete records; residual shows equity-based unaccounted strategy PnL (有仓, 用 equity 替代 available cash)"
        else:
            note = "no incomplete records; residual shows cash-based unaccounted strategy PnL"
    else:
        if math.current_account_value_source == "equity":
            note = "incomplete records are bucketed, not displayed per position (有仓, 收益估算基于 equity)"
        else:
            note = "incomplete records are bucketed, not displayed per position"

    return ResidualPnlBucket(
        incomplete_count=incomplete_count,
        pnl=math.residual_pnl,
        cash_start=context.period_start_cash,
        cash_end=context.current_cash,
        net_transfer=math.net_transfer,
        strategy_total_pnl=math.strategy_total_pnl,
        known_closed_pnl=known_closed_pnl,
        formula=formula,
        note=note,
        period_start_value=math.period_start_value,
        current_account_value=math.current_account_value,
        current_account_value_source=math.current_account_value_source,
    )


# ---------------------------------------------------------------------------
# cash inference helpers
# ---------------------------------------------------------------------------


def infer_cash_at_or_before(events: list[JournalEvent], target: datetime) -> float | None:
    best_ts: datetime | None = None
    best_cash: float | None = None
    for event in events:
        try:
            ts = datetime.fromisoformat(event.ts_iso)
        except Exception:
            continue
        if ts > target:
            continue
        cash = cash_from_event(event)
        if cash is None:
            continue
        if best_ts is None or ts >= best_ts:
            best_ts = ts
            best_cash = cash
    return best_cash


def infer_first_cash(events: list[JournalEvent]) -> float | None:
    best_ts: datetime | None = None
    best_cash: float | None = None
    for event in events:
        try:
            ts = datetime.fromisoformat(event.ts_iso)
        except Exception:
            continue
        cash = cash_before_or_from_event(event)
        if cash is None:
            continue
        if best_ts is None or ts < best_ts:
            best_ts = ts
            best_cash = cash
    return best_cash


def cash_from_event(event: JournalEvent) -> float | None:
    if event.event_type == "CASH_BASELINE":
        return _to_float(event.payload.get("cash"))
    if event.event_type == "CASH_TRANSFER":
        return _to_float(event.payload.get("cash_after"))
    if event.event_type == "FLAT":
        return _to_float(event.payload.get("cash_after"))
    if event.event_type == "ENTRY":
        return _to_float(event.payload.get("cash_before_position"))
    if event.event_type == "STARTUP_RECOVERY":
        return _to_float(event.payload.get("cash"))
    for key in ("cash_after", "cash_before_position", "cash"):
        value = _to_float(event.payload.get(key))
        if value is not None:
            return value
    return None


def cash_before_or_from_event(event: JournalEvent) -> float | None:
    if event.event_type == "CASH_BASELINE":
        return _to_float(event.payload.get("cash"))
    for key in ("cash_before_position", "cash", "cash_after"):
        value = _to_float(event.payload.get(key))
        if value is not None:
            return value
    return None
