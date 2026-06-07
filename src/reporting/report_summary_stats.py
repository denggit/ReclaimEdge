from __future__ import annotations

from typing import Any

from src.reporting.report_models import ArchivedSummaryStats, _to_float
from src.reporting.trade_journal import JournalEvent


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------


def to_int(value: Any) -> int:
    number = _to_float(value)
    return int(number) if number is not None else 0


def max_non_none(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def min_non_none(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def max_drawdown(equity_points: list[float]) -> tuple[float | None, float | None]:
    if len(equity_points) < 2:
        return None, None
    peak = equity_points[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for equity in equity_points:
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        drawdown_pct = drawdown / peak * 100 if peak > 0 else 0.0
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = drawdown_pct
    return max_dd, max_dd_pct


# ---------------------------------------------------------------------------
# archived summary stats loading
# ---------------------------------------------------------------------------


def load_archived_summary_stats_from_events(summary_events: list[JournalEvent]) -> ArchivedSummaryStats:
    closed_count = 0
    win_count = 0
    loss_count = 0
    breakeven_count = 0
    known_closed_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    best_win: float | None = None
    worst_loss: float | None = None
    archived_event_count = 0
    archived_position_count = 0

    for event in summary_events:
        if event.event_type != "SUMMARY_SNAPSHOT":
            continue
        payload = event.payload
        closed_count += to_int(payload.get("closed_count"))
        win_count += to_int(payload.get("win_count"))
        loss_count += to_int(payload.get("loss_count"))
        breakeven_count += to_int(payload.get("breakeven_count"))
        known_closed_pnl += _to_float(payload.get("known_closed_pnl")) or 0.0
        gross_profit += _to_float(payload.get("gross_profit")) or 0.0
        gross_loss += _to_float(payload.get("gross_loss")) or 0.0
        best_win = max_non_none(best_win, _to_float(payload.get("best_win")))
        worst_loss = min_non_none(worst_loss, _to_float(payload.get("worst_loss")))
        archived_event_count += to_int(payload.get("archived_event_count"))
        archived_position_count += to_int(payload.get("archived_position_count"))

    return ArchivedSummaryStats(
        closed_count=closed_count,
        win_count=win_count,
        loss_count=loss_count,
        breakeven_count=breakeven_count,
        known_closed_pnl=known_closed_pnl,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        best_win=best_win,
        worst_loss=worst_loss,
        archived_event_count=archived_event_count,
        archived_position_count=archived_position_count,
    )
