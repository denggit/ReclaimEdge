from __future__ import annotations

from decimal import Decimal

import pytest

from src.execution.trader import PositionSnapshot
from src.live.runtime_types import ExecutionState
from src.live.startup_recovery.order_recovery import apply_main_tp_startup_recovery
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig


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
    def __init__(self, orders: list[dict]) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.tp_order_id = None
        self.orders = orders
        self.cancel_calls: list = []

    async def fetch_pending_orders(self) -> list[dict]:
        return list(self.orders)

    async def request(self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
        self.cancel_calls.append((method, endpoint, payload))
        return {"data": [{"ordId": "cancelled"}]}


def make_strategy() -> BollCvdReclaimStrategy:
    strategy = BollCvdReclaimStrategy(
        BollCvdReclaimStrategyConfig(),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )
    strategy.state = StrategyPositionState(
        side="LONG",
        layers=1,
        avg_entry_price=1670.0,
        tp_price=1700.0,
        total_entry_qty=0.1,
        startup_force_tp_reconcile=True,
    )
    return strategy


@pytest.mark.asyncio
async def test_startup_reconstruct_unique_reduce_only_tp_identity() -> None:
    trader = FakeTrader(
        [
            {
                "instId": "ETH-USDT-SWAP",
                "reduceOnly": "true",
                "side": "sell",
                "ordId": "tp-unique",
                "px": "1700",
                "sz": "0.71",
            }
        ]
    )
    strategy = make_strategy()
    execution_state = ExecutionState("pos-1", 100.0)
    journal = FakeJournal()
    state_store = FakeStateStore()

    await apply_main_tp_startup_recovery(
        execution_state=execution_state,
        saved_state=None,
        startup_position=PositionSnapshot("LONG", Decimal("0.71"), 1670.0, 0.071, Decimal("0.71")),
        trader=trader,  # type: ignore[arg-type]
        journal=journal,  # type: ignore[arg-type]
        strategy=strategy,
        state_store=state_store,  # type: ignore[arg-type]
    )

    assert trader.tp_order_id == "tp-unique"
    assert strategy.state.tp_order_id == "tp-unique"
    assert execution_state.trading_halted is False
    assert trader.cancel_calls == []
    assert [event[0] for event in journal.events] == ["STARTUP_REDUCE_ONLY_TP_IDENTITY_RECONSTRUCTED"]
    assert state_store.saved[-1].tp_order_id == "tp-unique"


@pytest.mark.asyncio
async def test_startup_ambiguous_reduce_only_tp_skips_force_replace_without_halt() -> None:
    trader = FakeTrader(
        [
            {
                "instId": "ETH-USDT-SWAP",
                "reduceOnly": "true",
                "side": "sell",
                "ordId": "tp-a",
                "px": "1700",
                "sz": "0.30",
            },
            {
                "instId": "ETH-USDT-SWAP",
                "reduceOnly": "true",
                "side": "sell",
                "ordId": "tp-b",
                "px": "1710",
                "sz": "0.30",
            },
        ]
    )
    strategy = make_strategy()
    execution_state = ExecutionState("pos-1", 100.0)
    journal = FakeJournal()

    await apply_main_tp_startup_recovery(
        execution_state=execution_state,
        saved_state=None,
        startup_position=PositionSnapshot("LONG", Decimal("0.71"), 1670.0, 0.071, Decimal("0.71")),
        trader=trader,  # type: ignore[arg-type]
        journal=journal,  # type: ignore[arg-type]
        strategy=strategy,
        state_store=FakeStateStore(),  # type: ignore[arg-type]
    )

    assert trader.tp_order_id is None
    assert strategy.state.startup_force_tp_reconcile is False
    assert execution_state.trading_halted is False
    assert trader.cancel_calls == []
    assert [event[0] for event in journal.events] == ["STARTUP_REDUCE_ONLY_TP_IDENTITY_AMBIGUOUS"]
