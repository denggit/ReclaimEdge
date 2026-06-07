from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def fmt(value: Any, digits: int = 4, default: str = "-") -> str:
    try:
        if value is None:
            return default
        return f"{float(value):.{digits}f}"
    except Exception:
        return default


def fmt_pct(value: Any, digits: int = 2, default: str = "-") -> str:
    try:
        if value is None:
            return default
        return f"{float(value):.{digits}f}%"
    except Exception:
        return default


def short_ts(ts_iso: str) -> str:
    try:
        return datetime.fromisoformat(ts_iso).astimezone().strftime("%m-%d %H:%M:%S")
    except Exception:
        return ts_iso


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyReportWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ReportRuntimeContext:
    """Runtime account state used only for report classification.

    Reports must not mutate strategy state or journal files. The context lets us
    distinguish a real open position from stale incomplete records and lets us
    aggregate stale records into one residual PnL bucket.
    """

    current_position_id: str | None = None
    current_has_position: bool = False
    current_cash: float | None = None
    current_equity: float | None = None
    period_start_cash: float | None = None
    period_start_equity: float | None = None


@dataclass(frozen=True)
class ResidualPnlBucket:
    incomplete_count: int
    pnl: float | None
    cash_start: float | None
    cash_end: float | None
    net_transfer: float
    strategy_total_pnl: float | None
    known_closed_pnl: float
    formula: str
    note: str
    # --- value-source fields (task 44) ---
    period_start_value: float | None = None
    period_start_value_source: str = "cash"
    current_account_value: float | None = None
    current_account_value_source: str = "cash"


@dataclass(frozen=True)
class ReportPnlMath:
    period_start_cash: float | None
    current_cash: float | None
    net_transfer: float
    known_closed_pnl: float
    strategy_total_pnl: float | None
    residual_pnl: float | None
    total_pnl: float | None
    period_start_value: float | None = None
    period_start_value_source: str = "cash"
    current_account_value: float | None = None
    current_account_value_source: str = "cash"


@dataclass(frozen=True)
class ArchivedSummaryStats:
    closed_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0
    known_closed_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    best_win: float | None = None
    worst_loss: float | None = None
    archived_event_count: int = 0
    archived_position_count: int = 0
