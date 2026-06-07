from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.execution.trader import PositionSnapshot
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState, TradeIntent


@dataclass(frozen=True)
class SettledFlatBalance:
    cash: float
    equity: float
    attempts: int
    stable: bool
    reason: str


@dataclass
class AccountSnapshot:
    position: PositionSnapshot | None
    cash: float
    equity: float
    updated_monotonic: float
    updated_ts_ms: int
    version: int = 0
    latest_market_price: float | None = None
    latest_market_price_ts_ms: int = 0


@dataclass(frozen=True)
class SidecarPreCoreReconcileResult:
    queried: bool
    changed: bool
    sidecar_tp_filled_count: int = 0
    sidecar_tp_filled_leg_ids: tuple[str, ...] = ()
    sidecar_tp_filled_order_ids: tuple[str, ...] = ()
    sidecar_tp_filled_qty: float = 0.0


@dataclass
class ExecutionState:
    current_position_id: str | None
    cash_before_position: float | None
    trading_halted: bool = False
    last_order_ts_ms: int = 0
    pending_order_count: int = 0
    halt_reason: str | None = None
    halt_until_ts_ms: int | None = None


@dataclass(frozen=True)
class TradeCommand:
    intent: TradeIntent
    strategy_state_snapshot: StrategyPositionState
    tick_ts_ms: int
    created_monotonic: float
    account_snapshot_updated_ts_ms: int
    reason: str


@dataclass(frozen=True)
class ExecutionReport:
    command: TradeCommand
    result: Any | None
    ok: bool
    error: Exception | None
    entry_may_be_live: bool
    created_monotonic: float
    finished_monotonic: float
