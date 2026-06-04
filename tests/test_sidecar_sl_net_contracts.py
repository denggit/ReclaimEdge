"""Test that protective SL orders use OKX net position contracts (core + sidecar).

When Sidecar is enabled, protective stop-loss orders must cover the full OKX
net position, not just the core position.  This is a risk-control requirement:
if only core contracts are covered, the sidecar portion is left unprotected.
"""

from __future__ import annotations

from decimal import Decimal

import contextlib
import asyncio
import datetime as dt
import unittest

from scripts.run_boll_cvd_live import (
    AccountSnapshot,
    ExecutionState,
    StrategyPositionState,
    account_position_sync_worker,
    refresh_sidecar_state_totals,
)
from src.execution.trader import PositionSnapshot
from src.position_management.sidecar.model import SidecarLegStatus
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy


def flat_position() -> PositionSnapshot:
    return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))


class FakeJournal:
    def __init__(self) -> None:
        self.entries: list[int] = []
        self.flats: list[dict] = []
        self.events: list[tuple[str, dict, str | None]] = []

    def new_position_id(self, symbol: str, side: str, ts_ms: int | None = None) -> str:
        return f"{symbol}:{side}:{ts_ms}"

    def record_entry(self, **kwargs) -> None:
        self.entries.append(kwargs["intent"].ts_ms)

    def record_tp_update(self, **kwargs) -> None:
        pass

    def record_error(self, **kwargs) -> None:
        pass

    def record_flat(self, **kwargs) -> None:
        self.flats.append(kwargs)

    def record_cash_transfer(self, **kwargs) -> None:
        pass

    def record_account_cash_drift(self, **kwargs) -> None:
        pass

    def record_rolling_loss_guard(self, **kwargs) -> None:
        pass

    def append(self, event_name: str, payload: dict, position_id: str | None = None) -> None:
        self.events.append((event_name, dict(payload), position_id))


class FakeStateStore:
    def __init__(self) -> None:
        self.saved_states: list = []
        self.clear_calls = 0

    def save(self, state) -> None:
        self.saved_states.append(state)

    def clear(self) -> None:
        self.clear_calls += 1


class FullProtectiveTrader:
    """Trader that records all protective SL orders for inspection."""
    symbol = "ETH-USDT-SWAP"
    account_equity_usdt = 1000.0

    def __init__(self) -> None:
        self.position_contracts = Decimal("0")
        self.post_tp1_stop_orders: list[dict] = []
        self.cancelled_post_tp1_stop_ids: list[str | None] = []
        self.cancel_post_tp1_ok = True
        self.middle_runner_stop_orders: list[dict] = []
        self.cancelled_middle_runner_stop_ids: list[str | None] = []
        self.cancel_middle_runner_ok = True
        self.sidecar_tps: list[tuple] = []
        self.sidecar_order_status: dict[str, str] = {}
        self.cancelled_sidecar_tps: list[str] = []
        self.market_exits: list[tuple] = []
        self._equity = 1000.0

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        return flat_position()

    async def fetch_usdt_equity(self) -> float:
        return self._equity

    async def request(self, method: str, endpoint: str, payload=None):
        return {"data": [{"details": [{"ccy": "USDT", "cashBal": str(self._equity)}]}]}

    def mark_flat(self) -> None:
        self.position_contracts = Decimal("0")

    async def place_three_stage_post_tp1_protective_stop_with_retries(
        self, side, contracts, stop_price, retry_count, retry_interval_seconds
    ):
        order_id = f"post-tp1-{len(self.post_tp1_stop_orders) + 1}"
        self.post_tp1_stop_orders.append({
            "side": side, "contracts": contracts, "stop_price": stop_price,
            "retry_count": retry_count, "retry_interval_seconds": retry_interval_seconds,
            "order_id": order_id,
        })
        return True, order_id, "protective_sl_placed"

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id: str | None) -> bool:
        self.cancelled_post_tp1_stop_ids.append(order_id)
        return self.cancel_post_tp1_ok

    async def place_middle_runner_protective_stop_with_retries(
        self, side, contracts, stop_price, retry_count, retry_interval_seconds
    ):
        order_id = f"mr-sl-{len(self.middle_runner_stop_orders) + 1}"
        self.middle_runner_stop_orders.append({
            "side": side, "contracts": contracts, "stop_price": stop_price,
            "retry_count": retry_count, "retry_interval_seconds": retry_interval_seconds,
            "order_id": order_id,
        })
        return True, order_id, "protective_sl_placed"

    async def cancel_middle_runner_protective_stop(self, order_id: str | None) -> bool:
        self.cancelled_middle_runner_stop_ids.append(order_id)
        return self.cancel_middle_runner_ok

    async def place_sidecar_market_order(self, *, side, eth_qty):
        return {"order_id": "sc-market", "contracts": "2", "qty": eth_qty}

    async def place_sidecar_fixed_take_profit(self, *, side, contracts, tp_price, client_order_id=None):
        self.sidecar_tps.append((side, contracts, tp_price, client_order_id))
        return f"sc-tp-{len(self.sidecar_tps)}"

    async def fetch_sidecar_order_status(self, order_id: str):
        return {"order_id": order_id, "status": self.sidecar_order_status.get(order_id, "OPEN"),
                "filled_qty": None, "avg_fill_price": None}

    async def cancel_sidecar_take_profit(self, order_id: str):
        self.cancelled_sidecar_tps.append(order_id)
        return True

    async def market_exit_remaining_position_with_retries(self, side, retry_count):
        self.market_exits.append((side, retry_count))
        return True, "ok"

    async def fetch_pending_orders(self):
        return []


class SidecarSLNetContractsTest(unittest.IsolatedAsyncioTestCase):

    async def run_account_sync_until(
        self, predicate, *, account_snapshot, execution_state, trader,
        strategy, journal, state_store, timeout: float = 1.0
    ):
        task = asyncio.create_task(
            account_position_sync_worker(
                state_lock=asyncio.Lock(),
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                trader=trader,
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy,
                journal=journal,
                state_store=state_store,
                position_sync_seconds=0,
                account_sync_seconds=999,
                cash_log_min_delta_usdt=999,
            )
        )
        try:
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.01)
            self.fail("account sync predicate was not satisfied")
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def three_stage_strategy_with_sidecar(self, side: str = "LONG") -> BollCvdShockReclaimStrategy:
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(
                three_stage_runner_enabled=True,
                three_stage_post_tp1_protective_sl_enabled=True,
                three_stage_post_tp1_sl_extension_trigger_ratio=0.6,
            ),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side=side, layers=1, total_entry_qty=1.0, total_entry_notional=100.0,
            avg_entry_price=100.0, tp_price=110.0 if side == "LONG" else 90.0,
            tp_plan="THREE_STAGE_RUNNER", three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0 if side == "LONG" else 99.0,
            three_stage_tp2_price=110.0 if side == "LONG" else 90.0,
            three_stage_tp1_ratio=0.6, three_stage_tp2_ratio=0.2, three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=False, three_stage_tp2_consumed=False,
            sidecar_enabled_for_position=True, sidecar_margin_pct=0.01, sidecar_tp_pct=0.004,
            sidecar_legs=[{
                "leg_id": "sc-leg-1", "position_id": "pos-1", "layer_index": 1,
                "side": side, "entry_price": 100.0, "qty": 0.2, "contracts": "2",
                "margin_pct": 0.01, "layer_multiplier": 1.0, "tp_pct": 0.004,
                "tp_price": 100.4 if side == "LONG" else 99.6, "tp_order_id": "sc-tp-1",
                "status": SidecarLegStatus.OPEN.value, "created_ts_ms": 1000, "updated_ts_ms": 1000,
            }],
        )
        refresh_sidecar_state_totals(strategy.state, 10)
        return strategy

    def middle_runner_strategy_with_sidecar(self, side: str = "LONG") -> BollCvdShockReclaimStrategy:
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(middle_runner_protective_sl_enabled=True),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side=side, layers=1, total_entry_qty=1.0, total_entry_notional=100.0,
            avg_entry_price=100.0, tp_price=110.0 if side == "LONG" else 90.0,
            tp_plan="SINGLE", partial_tp_consumed=False,
            middle_runner_enabled_for_position=True, middle_runner_pending=True,
            middle_runner_active=False, middle_runner_first_close_ratio=0.6,
            middle_runner_keep_ratio=0.4,
            middle_runner_first_tp_price=101.0 if side == "LONG" else 99.0,
            middle_runner_final_tp_price=110.0 if side == "LONG" else 90.0,
            middle_runner_protective_sl_price=None, middle_runner_protective_sl_order_id=None,
            middle_runner_add_disabled=False,
            sidecar_enabled_for_position=True, sidecar_margin_pct=0.01, sidecar_tp_pct=0.004,
            sidecar_legs=[{
                "leg_id": "sc-leg-1", "position_id": "pos-1", "layer_index": 1,
                "side": side, "entry_price": 100.0, "qty": 0.2, "contracts": "2",
                "margin_pct": 0.01, "layer_multiplier": 1.0, "tp_pct": 0.004,
                "tp_price": 100.4 if side == "LONG" else 99.6, "tp_order_id": "sc-tp-1",
                "status": SidecarLegStatus.OPEN.value, "created_ts_ms": 1000, "updated_ts_ms": 1000,
            }],
        )
        refresh_sidecar_state_totals(strategy.state, 10)
        return strategy

    # ── Test 1: Three-Stage TP1 post-TP1 SL uses net contracts ─────────

    async def test_three_stage_tp1_sl_uses_net_contracts_with_sidecar(self) -> None:
        """post-TP1 protective SL must cover OKX net position, not just core."""
        net_contracts = Decimal("12")
        core_contracts = Decimal("10")

        class Tp1Trader(FullProtectiveTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", net_contracts, 100.0, 0.6, net_contracts)

        strategy = self.three_stage_strategy_with_sidecar("LONG")
        trader = Tp1Trader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        latest_ts_ms = int(dt.datetime.now().timestamp() * 1000)
        account_snapshot = AccountSnapshot(
            None, 1000.0, 1000.0, asyncio.get_running_loop().time(), 0, 1,
            latest_market_price=102.0, latest_market_price_ts_ms=latest_ts_ms,
        )
        execution_state = ExecutionState("pos-1", 1000.0)

        await self.run_account_sync_until(
            lambda: len(trader.post_tp1_stop_orders) >= 1,
            account_snapshot=account_snapshot, execution_state=execution_state,
            trader=trader, strategy=strategy, journal=journal, state_store=state_store,
        )

        self.assertEqual(len(trader.post_tp1_stop_orders), 1)
        self.assertEqual(trader.post_tp1_stop_orders[0]["contracts"], net_contracts,
                         f"post-TP1 SL must use net contracts {net_contracts}, "
                         f"not core contracts {core_contracts}")

        placed_events = [e for e in journal.events if e[0] == "THREE_STAGE_TP1_PROTECTIVE_SL_PLACED"]
        self.assertEqual(len(placed_events), 1)
        pp = placed_events[0][1]
        self.assertEqual(str(pp.get("core_contracts")), str(core_contracts))
        self.assertEqual(str(pp.get("net_contracts")), str(net_contracts))
        self.assertEqual(str(pp.get("sl_contracts")), str(net_contracts))

    # ── Test 2: Middle Runner partial TP SL uses net contracts ─────────

    async def test_middle_runner_sl_uses_net_contracts_with_sidecar(self) -> None:
        """Middle runner protective SL must cover OKX net position, not just core."""
        net_contracts = Decimal("12")
        core_contracts = Decimal("10")

        class MRTrader(FullProtectiveTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", net_contracts, 100.0, 0.6, net_contracts)

        strategy = self.middle_runner_strategy_with_sidecar("LONG")
        trader = MRTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        latest_ts_ms = int(dt.datetime.now().timestamp() * 1000)
        account_snapshot = AccountSnapshot(
            None, 1000.0, 1000.0, asyncio.get_running_loop().time(), 0, 1,
            latest_market_price=102.0, latest_market_price_ts_ms=latest_ts_ms,
        )
        execution_state = ExecutionState("pos-1", 1000.0)

        await self.run_account_sync_until(
            lambda: len(trader.middle_runner_stop_orders) >= 1,
            account_snapshot=account_snapshot, execution_state=execution_state,
            trader=trader, strategy=strategy, journal=journal, state_store=state_store,
        )

        self.assertEqual(len(trader.middle_runner_stop_orders), 1)
        self.assertEqual(trader.middle_runner_stop_orders[0]["contracts"], net_contracts,
                         f"middle runner SL must use net contracts {net_contracts}, "
                         f"not core contracts {core_contracts}")

        placed_events = [e for e in journal.events
                         if e[0] in ("MIDDLE_RUNNER_ACTIVATED", "MIDDLE_RUNNER_SIZE_MISMATCH_PROTECTED")]
        self.assertGreaterEqual(len(placed_events), 1)
        pp = placed_events[0][1]
        self.assertEqual(str(pp.get("core_contracts")), str(core_contracts))
        self.assertEqual(str(pp.get("net_contracts")), str(net_contracts))
        self.assertEqual(str(pp.get("sl_contracts")), str(net_contracts))

    # ── Test 3: Three-Stage TP1 SL SHORT side ──────────────────────────

    async def test_three_stage_tp1_sl_uses_net_contracts_short_sidecar(self) -> None:
        """Same net-contracts requirement for SHORT side."""
        net_contracts = Decimal("12")

        class Tp1Trader(FullProtectiveTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("SHORT", net_contracts, 100.0, 0.6, Decimal("-12"))

        strategy = self.three_stage_strategy_with_sidecar("SHORT")
        trader = Tp1Trader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        latest_ts_ms = int(dt.datetime.now().timestamp() * 1000)
        account_snapshot = AccountSnapshot(
            None, 1000.0, 1000.0, asyncio.get_running_loop().time(), 0, 1,
            latest_market_price=98.0, latest_market_price_ts_ms=latest_ts_ms,
        )
        execution_state = ExecutionState("pos-1", 1000.0)

        await self.run_account_sync_until(
            lambda: len(trader.post_tp1_stop_orders) >= 1,
            account_snapshot=account_snapshot, execution_state=execution_state,
            trader=trader, strategy=strategy, journal=journal, state_store=state_store,
        )

        self.assertEqual(len(trader.post_tp1_stop_orders), 1)
        self.assertEqual(trader.post_tp1_stop_orders[0]["contracts"], net_contracts)

    # ── Test 4: post-TP1 SL payload includes core_contracts / net_contracts in journal ──

    async def test_post_tp1_sl_journal_includes_core_and_net_contracts(self) -> None:
        """Journal for THREE_STAGE_TP1_PROTECTIVE_SL_PLACED includes core/net/sl_contracts."""
        net_contracts = Decimal("12")
        core_contracts = Decimal("10")

        class Tp1Trader(FullProtectiveTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", net_contracts, 100.0, 0.6, net_contracts)

        strategy = self.three_stage_strategy_with_sidecar("LONG")
        trader = Tp1Trader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        latest_ts_ms = int(dt.datetime.now().timestamp() * 1000)
        account_snapshot = AccountSnapshot(
            None, 1000.0, 1000.0, asyncio.get_running_loop().time(), 0, 1,
            latest_market_price=102.0, latest_market_price_ts_ms=latest_ts_ms,
        )
        execution_state = ExecutionState("pos-1", 1000.0)

        await self.run_account_sync_until(
            lambda: len(trader.post_tp1_stop_orders) >= 1,
            account_snapshot=account_snapshot, execution_state=execution_state,
            trader=trader, strategy=strategy, journal=journal, state_store=state_store,
        )

        sl_events = [e for e in journal.events
                     if e[0] == "THREE_STAGE_TP1_PROTECTIVE_SL_PLACED"]
        self.assertEqual(len(sl_events), 1)
        pp = sl_events[0][1]
        self.assertIn("core_contracts", pp)
        self.assertIn("net_contracts", pp)
        self.assertIn("sl_contracts", pp)
        self.assertEqual(str(pp["core_contracts"]), str(core_contracts))
        self.assertEqual(str(pp["net_contracts"]), str(net_contracts))

    # ── Test 5: Middle Runner SL journal includes core_contracts / net_contracts ──

    async def test_middle_runner_sl_journal_includes_core_and_net_contracts(self) -> None:
        """Journal for MIDDLE_RUNNER_ACTIVATED includes core/net/sl_contracts."""
        net_contracts = Decimal("12")

        class MRTrader(FullProtectiveTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", net_contracts, 100.0, 0.6, net_contracts)

        strategy = self.middle_runner_strategy_with_sidecar("LONG")
        trader = MRTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        latest_ts_ms = int(dt.datetime.now().timestamp() * 1000)
        account_snapshot = AccountSnapshot(
            None, 1000.0, 1000.0, asyncio.get_running_loop().time(), 0, 1,
            latest_market_price=102.0, latest_market_price_ts_ms=latest_ts_ms,
        )
        execution_state = ExecutionState("pos-1", 1000.0)

        await self.run_account_sync_until(
            lambda: len(trader.middle_runner_stop_orders) >= 1,
            account_snapshot=account_snapshot, execution_state=execution_state,
            trader=trader, strategy=strategy, journal=journal, state_store=state_store,
        )

        sl_events = [e for e in journal.events
                     if e[0] in ("MIDDLE_RUNNER_ACTIVATED", "MIDDLE_RUNNER_SIZE_MISMATCH_PROTECTED")]
        self.assertGreaterEqual(len(sl_events), 1)
        pp = sl_events[0][1]
        self.assertIn("core_contracts", pp)
        self.assertIn("net_contracts", pp)
        self.assertIn("sl_contracts", pp)
