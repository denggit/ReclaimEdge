from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from src.execution.trader import PositionSnapshot
from src.live import runtime_types as live_runtime_types
from src.live.account_sync.pre_core_position import run_account_sync_pre_core_position_phase
from src.position_management.sidecar.model import SidecarLegStatus
from src.position_management.sidecar.runtime_state import refresh_sidecar_state_totals
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def append(self, event_name: str, payload: dict, position_id: str | None = None) -> None:
        self.events.append((event_name, dict(payload), position_id))


class FakeStateStore:
    def __init__(self) -> None:
        self.saved: list = []

    def save(self, state) -> None:  # type: ignore[no-untyped-def]
        self.saved.append(state)


class FakeTrader:
    symbol = "ETH-USDT-SWAP"

    def __init__(
            self,
            *,
            position: PositionSnapshot | None = None,
            usdt_equity: float = 100.0,
            cash_balance: float = 100.0,
            sidecar_order_status: str = "OPEN",
            sidecar_order_filled_qty: float | None = None,
            sidecar_order_avg_fill_price: float | None = None,
    ) -> None:
        self._position = position or PositionSnapshot("LONG", Decimal("5"), 3000.0, 0.5, Decimal("5"))
        self._usdt_equity = usdt_equity
        self._cash_balance = cash_balance
        self._sidecar_order_status = sidecar_order_status
        self._sidecar_order_filled_qty = sidecar_order_filled_qty
        self._sidecar_order_avg_fill_price = sidecar_order_avg_fill_price
        self.account_equity_usdt = usdt_equity

    async def fetch_usdt_equity(self) -> float:
        return self._usdt_equity

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        return self._position

    async def request(self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
        return {"data": [{"details": [{"ccy": "USDT", "cashBal": str(self._cash_balance)}]}]}

    async def fetch_sidecar_order_status(self, order_id: str):  # type: ignore[no-untyped-def]
        return {
            "order_id": order_id,
            "status": self._sidecar_order_status,
            "filled_qty": self._sidecar_order_filled_qty,
            "avg_fill_price": self._sidecar_order_avg_fill_price,
        }


@pytest.mark.asyncio
async def test_cash_drift_includes_sidecar_tp_filled_when_detected_in_same_sync() -> None:
    """When a sidecar TP fill is detected in pre-core reconcile and cash drift
    triggers in the same sync cycle, the drift reason must include
    'position_cash_change:sidecar_tp_filled' AND retain 'unsafe_state:'.
    """
    # ── strategy state with OPEN sidecar leg ──
    state = StrategyPositionState(
        side="LONG",
        layers=1,
        total_entry_qty=0.5,
        avg_entry_price=3000.0,
        sidecar_enabled_for_position=True,
        sidecar_margin_pct=0.01,
        sidecar_tp_pct=0.004,
        sidecar_legs=[
            {
                "leg_id": "leg-1",
                "status": SidecarLegStatus.OPEN.value,
                "tp_order_id": "tp-1",
                "qty": 0.1,
                "contracts": "1",
                "entry_price": 3000.0,
                "tp_price": 3012.0,
                "created_ts_ms": 1,
                "updated_ts_ms": 1,
            }
        ],
    )
    refresh_sidecar_state_totals(state)

    # ── trader: OKX has position + sidecar TP filled + cash/equity changed ──
    trader = FakeTrader(
        position=PositionSnapshot("LONG", Decimal("4"), 3000.0, 0.4, Decimal("4")),
        usdt_equity=681.6438,
        cash_balance=584.8172,
        sidecar_order_status="FILLED",
        sidecar_order_filled_qty=0.1,
        sidecar_order_avg_fill_price=3012.0,
    )

    strategy = BollCvdShockReclaimStrategy(
        config=type("C", (), {"breakeven_fee_buffer_pct": 0.001})(),  # type: ignore[arg-type]
        sizer=SimplePositionSizer(SimplePositionSizerConfig()),
    )
    strategy.state = state

    sizer = SimplePositionSizer(SimplePositionSizerConfig())

    execution_state = live_runtime_types.ExecutionState(
        current_position_id="pos-sidecar",
        cash_before_position=568.3372,
    )

    account_snapshot = live_runtime_types.AccountSnapshot(
        position=PositionSnapshot("LONG", Decimal("5"), 3000.0, 0.5, Decimal("5")),
        cash=568.3372,
        equity=681.6438,
        updated_monotonic=0.0,
        updated_ts_ms=0,
    )

    journal = FakeJournal()
    store = FakeStateStore()
    state_lock = asyncio.Lock()

    result = await run_account_sync_pre_core_position_phase(
        state_lock=state_lock,
        account_snapshot=account_snapshot,
        execution_state=execution_state,
        trader=trader,  # type: ignore[arg-type]
        sizer=sizer,
        strategy=strategy,
        journal=journal,  # type: ignore[arg-type]
        state_store=store,  # type: ignore[arg-type]
        now=100.0,
        last_account_sync=-1000.0,
        account_sync_seconds=999,
        cash_transfer_detect_enabled=True,
        cash_transfer_min_delta_usdt=0.5,
        cash_transfer_settle_seconds=120,
        cash_transfer_after_flat_cooldown_seconds=180,
        cash_drift_min_delta_usdt=0.5,
        cash_event_log_interval_seconds=60,
        cash_log_min_delta_usdt=0.5,
        last_logged_cash=568.3372,
        last_logged_equity=568.3372,
        last_cash_event_log=0.0,
        last_flat_detected_monotonic=0.0,
    )

    # ── assert cash_drift_payload exists and reason contains sidecar_tp_filled ──
    assert result.cash_drift_payload is not None, "cash_drift_payload must not be None when cash changes in unsafe state"
    reason = result.cash_drift_payload["reason"]
    assert "position_cash_change:sidecar_tp_filled" in reason, (
        f"reason must contain 'position_cash_change:sidecar_tp_filled', got: {reason}"
    )
    assert "unsafe_state:" in reason, (
        f"reason must still contain 'unsafe_state:', got: {reason}"
    )
    assert result.cash_drift_payload["sidecar_tp_filled_count"] == 1
    assert "leg-1" in result.cash_drift_payload["sidecar_tp_filled_leg_ids"]

    # ── assert cash_transfer_payload is None (NOT CASH_TRANSFER) ──
    assert result.cash_transfer_payload is None, (
        "cash_transfer_payload must be None — sidecar TP cash change is drift, not transfer"
    )

    # ── assert sidecar leg was marked TP_FILLED ──
    assert state.sidecar_legs[0]["status"] == "TP_FILLED"

    # ── assert new fields on result ──
    assert result.sidecar_tp_filled_count == 1
    assert "leg-1" in result.sidecar_tp_filled_leg_ids
    assert "tp-1" in result.sidecar_tp_filled_order_ids


@pytest.mark.asyncio
async def test_cash_drift_no_sidecar_tp_uses_original_unsafe_state_reason() -> None:
    """When no sidecar TP fill is detected, the drift reason must remain
    the original 'unsafe_state:...' without sidecar_tp_filled prefix."""
    state = StrategyPositionState(
        side="LONG",
        layers=1,
        total_entry_qty=0.5,
        avg_entry_price=3000.0,
        sidecar_enabled_for_position=False,
    )

    trader = FakeTrader(
        position=PositionSnapshot("LONG", Decimal("5"), 3000.0, 0.5, Decimal("5")),
        usdt_equity=681.6438,
        cash_balance=584.8172,
        sidecar_order_status="OPEN",
    )

    strategy = BollCvdShockReclaimStrategy(
        config=type("C", (), {"breakeven_fee_buffer_pct": 0.001})(),  # type: ignore[arg-type]
        sizer=SimplePositionSizer(SimplePositionSizerConfig()),
    )
    strategy.state = state

    sizer = SimplePositionSizer(SimplePositionSizerConfig())

    execution_state = live_runtime_types.ExecutionState(
        current_position_id="pos-no-sidecar",
        cash_before_position=568.3372,
    )

    account_snapshot = live_runtime_types.AccountSnapshot(
        position=PositionSnapshot("LONG", Decimal("5"), 3000.0, 0.5, Decimal("5")),
        cash=568.3372,
        equity=681.6438,
        updated_monotonic=0.0,
        updated_ts_ms=0,
    )

    journal = FakeJournal()
    store = FakeStateStore()
    state_lock = asyncio.Lock()

    result = await run_account_sync_pre_core_position_phase(
        state_lock=state_lock,
        account_snapshot=account_snapshot,
        execution_state=execution_state,
        trader=trader,  # type: ignore[arg-type]
        sizer=sizer,
        strategy=strategy,
        journal=journal,  # type: ignore[arg-type]
        state_store=store,  # type: ignore[arg-type]
        now=100.0,
        last_account_sync=-1000.0,
        account_sync_seconds=999,
        cash_transfer_detect_enabled=True,
        cash_transfer_min_delta_usdt=0.5,
        cash_transfer_settle_seconds=120,
        cash_transfer_after_flat_cooldown_seconds=180,
        cash_drift_min_delta_usdt=0.5,
        cash_event_log_interval_seconds=60,
        cash_log_min_delta_usdt=0.5,
        last_logged_cash=568.3372,
        last_logged_equity=568.3372,
        last_cash_event_log=0.0,
        last_flat_detected_monotonic=0.0,
    )

    assert result.cash_drift_payload is not None
    reason = result.cash_drift_payload["reason"]
    assert "position_cash_change:sidecar_tp_filled" not in reason
    assert reason.startswith("unsafe_state:")

    assert result.sidecar_tp_filled_count == 0
