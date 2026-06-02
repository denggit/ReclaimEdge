from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("dotenv") is None:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)

from scripts.run_boll_cvd_live import (  # noqa: E402
    AccountSnapshot,
    ExecutionState,
    account_position_sync_worker,
    apply_rolling_loss_guard_startup_state,
    rolling_loss_halt_reason,
)
from src.execution.trader import PositionSnapshot  # noqa: E402
from src.reporting.trade_journal import JournalEvent  # noqa: E402
from src.risk.rolling_loss_guard import (  # noqa: E402
    MS_PER_HOUR,
    RollingLossGuard,
    RollingLossGuardConfig,
)
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig  # noqa: E402
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState  # noqa: E402


NOW_MS = 1_800_000_000_000


def flat_position() -> PositionSnapshot:
    return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))


def live_position() -> PositionSnapshot:
    return PositionSnapshot("LONG", Decimal("1"), 100.0, 1.0, Decimal("1"))


def event(event_type: str, ts_ms: int, payload: dict) -> JournalEvent:  # type: ignore[type-arg]
    ts_iso = dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).isoformat()
    return JournalEvent("event-id", event_type, ts_iso, None, payload)


class FakeStrategy:
    def __init__(self) -> None:
        self.state = StrategyPositionState()


class FakeTrader:
    def __init__(self, position: PositionSnapshot | None = None, equity: float = 100.0) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.account_equity_usdt = equity
        self.position_contracts = Decimal("0")
        self.position = position or flat_position()
        self.fetched = asyncio.Event()

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        self.fetched.set()
        return self.position

    async def fetch_usdt_equity(self) -> float:
        return self.account_equity_usdt

    async def request(self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
        return {"data": [{"details": [{"ccy": "USDT", "cashBal": str(self.account_equity_usdt)}]}]}

    def mark_flat(self) -> None:
        self.position_contracts = Decimal("0")


class FakeJournal:
    def __init__(self) -> None:
        self.rolling_loss_events: list[dict] = []  # type: ignore[type-arg]
        self.events: list[JournalEvent] = []
        self.recorded = asyncio.Event()

    def record_cash_transfer(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def record_account_cash_drift(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def record_rolling_loss_guard(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.rolling_loss_events.append(kwargs)
        self.recorded.set()

    def record_flat(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        payload = dict(kwargs)
        if kwargs.get("cash_before_position") is not None and kwargs.get("cash_after") is not None:
            payload["realized_pnl_usdt_est"] = kwargs["cash_after"] - kwargs["cash_before_position"]
        ts_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        self.events.append(JournalEvent("flat-id", "FLAT", ts_iso, kwargs.get("position_id"), payload))

    def load_events(self) -> list[JournalEvent]:
        return list(self.events)


class FakeStateStore:
    def save(self, state) -> None:  # type: ignore[no-untyped-def]
        pass

    def clear(self) -> None:
        pass


class FakeEmailSender:
    def __init__(self) -> None:
        self.subjects: list[str] = []

    async def send_email_async(self, subject: str, content: str, content_type: str = "html") -> bool:
        self.subjects.append(subject)
        return True


class RollingLossGuardTest(unittest.IsolatedAsyncioTestCase):
    def make_guard(self, root: Path, **overrides) -> RollingLossGuard:  # type: ignore[no-untyped-def]
        values = {
            "enabled": True,
            "window_hours": 24,
            "warn_pct": 0.10,
            "soft_halt_pct": 0.15,
            "soft_halt_hours": 6,
            "hard_halt_pct": 0.20,
            "hard_halt_hours": 12,
            "email_enabled": False,
            "event_time_tolerance_ms": 5000,
        }
        values.update(overrides)
        return RollingLossGuard(root / "rolling_loss_guard_state.json", RollingLossGuardConfig(**values))

    def test_initialize_window_uses_current_equity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            state = guard.load_or_initialize(NOW_MS, 100.0)

            self.assertEqual(state.baseline_equity, 100.0)
            self.assertEqual(state.window_start_ts_ms, NOW_MS)
            self.assertEqual(state.window_end_ts_ms, NOW_MS + 24 * MS_PER_HOUR)

    def test_warn_at_10_percent_loss_no_halt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)

            decision = guard.evaluate_after_flat(
                now_ms=NOW_MS,
                journal_events=[event("FLAT", NOW_MS - 1_000, {"realized_pnl_usdt_est": -10})],
            )

            self.assertEqual(decision.action, "WARN")
            self.assertFalse(decision.should_halt)
            self.assertIsNone(rolling_loss_halt_reason(decision.action))

    def test_current_flat_event_with_same_or_slightly_future_timestamp_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)

            decision = guard.evaluate_after_flat(
                now_ms=NOW_MS,
                journal_events=[event("FLAT", NOW_MS + 1_000, {"realized_pnl_usdt_est": -10})],
            )

            self.assertEqual(decision.action, "WARN")
            self.assertEqual(decision.rolling_realized_pnl, -10.0)

    def test_far_future_flat_event_is_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)

            decision = guard.evaluate_after_flat(
                now_ms=NOW_MS,
                journal_events=[event("FLAT", NOW_MS + 60_000, {"realized_pnl_usdt_est": -20})],
            )

            self.assertIsNone(decision.action)
            self.assertEqual(decision.rolling_realized_pnl, 0.0)

    def test_soft_halt_at_15_percent_loss_after_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)

            decision = guard.evaluate_after_flat(
                now_ms=NOW_MS,
                journal_events=[event("FLAT", NOW_MS - 1_000, {"realized_pnl_usdt_est": -15})],
            )

            self.assertEqual(decision.action, "SOFT_HALT")
            self.assertTrue(decision.should_halt)
            self.assertEqual(decision.halt_until_ts_ms, NOW_MS + 6 * MS_PER_HOUR)
            self.assertEqual(rolling_loss_halt_reason(decision.action), "rolling_loss_soft_halt")

    def test_hard_halt_at_20_percent_loss_after_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)

            decision = guard.evaluate_after_flat(
                now_ms=NOW_MS,
                journal_events=[event("FLAT", NOW_MS - 1_000, {"realized_pnl_usdt_est": -20})],
            )

            self.assertEqual(decision.action, "HARD_HALT")
            self.assertEqual(decision.halt_until_ts_ms, NOW_MS + 12 * MS_PER_HOUR)
            self.assertEqual(rolling_loss_halt_reason(decision.action), "rolling_loss_hard_halt")
            self.assertTrue(guard.state.warn_triggered)
            self.assertTrue(guard.state.soft_halt_triggered)

    def test_hard_halt_overrides_soft_halt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)
            first = guard.evaluate_after_flat(
                now_ms=NOW_MS,
                journal_events=[event("FLAT", NOW_MS - 1_000, {"realized_pnl_usdt_est": -15})],
            )
            self.assertEqual(first.action, "SOFT_HALT")

            later = NOW_MS + 60_000
            decision = guard.evaluate_after_flat(
                now_ms=later,
                journal_events=[
                    event("FLAT", NOW_MS - 1_000, {"realized_pnl_usdt_est": -15}),
                    event("FLAT", later - 1_000, {"realized_pnl_usdt_est": -5}),
                ],
            )

            self.assertEqual(decision.action, "HARD_HALT")
            self.assertEqual(decision.halt_until_ts_ms, later + 12 * MS_PER_HOUR)
            self.assertEqual(guard.state.halt_level, "HARD")

    def test_positive_pnl_offsets_losses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)

            decision = guard.evaluate_after_flat(
                now_ms=NOW_MS,
                journal_events=[
                    event("FLAT", NOW_MS - 2_000, {"realized_pnl_usdt_est": -12}),
                    event("FLAT", NOW_MS - 1_000, {"realized_pnl_usdt_est": 5}),
                ],
            )

            self.assertIsNone(decision.action)
            self.assertEqual(decision.rolling_realized_pnl, -7.0)

    def test_missing_pnl_flat_event_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)

            decision = guard.evaluate_after_flat(
                now_ms=NOW_MS,
                journal_events=[
                    event("FLAT", NOW_MS - 2_000, {"realized_pnl_usdt_est": "not-a-number"}),
                    event("FLAT", NOW_MS - 1_000, {}),
                ],
            )

            self.assertIsNone(decision.action)
            self.assertEqual(decision.rolling_realized_pnl, 0.0)

    def test_no_forced_close_while_position_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)
            events = [event("FLAT", NOW_MS - 1_000, {"realized_pnl_usdt_est": -20})]

            open_decision = guard.evaluate_after_flat(now_ms=NOW_MS, journal_events=events, has_position=True)
            flat_decision = guard.evaluate_after_flat(now_ms=NOW_MS, journal_events=events, has_position=False)

            self.assertIsNone(open_decision.action)
            self.assertEqual(flat_decision.action, "HARD_HALT")

    def test_window_expired_resets_only_when_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - 25 * MS_PER_HOUR, 100.0)

            self.assertFalse(guard.should_reset_expired_window(NOW_MS, has_position=True))
            self.assertTrue(guard.should_reset_expired_window(NOW_MS, has_position=False))
            guard.reset_window(NOW_MS, 120.0)
            self.assertEqual(guard.state.baseline_equity, 120.0)

    async def test_cooldown_persists_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)
            guard.state.halt_active = True
            guard.state.halt_level = "SOFT"
            guard.state.halt_until_ts_ms = NOW_MS + MS_PER_HOUR
            guard.save()

            restored = self.make_guard(Path(tmp))
            restored.load_or_initialize(NOW_MS, 100.0)
            execution_state = ExecutionState(None, None)
            await apply_rolling_loss_guard_startup_state(
                rolling_loss_guard=restored,
                execution_state=execution_state,
                has_position=False,
                equity=100.0,
                now_ms=NOW_MS,
                journal=FakeJournal(),  # type: ignore[arg-type]
                email_sender=None,
            )

            self.assertTrue(execution_state.trading_halted)
            self.assertEqual(execution_state.halt_reason, "rolling_loss_soft_halt")
            self.assertEqual(execution_state.halt_until_ts_ms, NOW_MS + MS_PER_HOUR)

    async def test_cooldown_resume_resets_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp), email_enabled=True)
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)
            guard.state.halt_active = True
            guard.state.halt_level = "HARD"
            guard.state.halt_until_ts_ms = 1
            guard.save()
            journal = FakeJournal()
            email_sender = FakeEmailSender()
            execution_state = ExecutionState(None, None, trading_halted=True, halt_reason="rolling_loss_hard_halt", halt_until_ts_ms=1)
            trader = FakeTrader(equity=125.0)
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=AccountSnapshot(flat_position(), 125.0, 125.0, asyncio.get_running_loop().time(), 0, 1),
                    execution_state=execution_state,
                    trader=trader,  # type: ignore[arg-type]
                    sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                    strategy=FakeStrategy(),  # type: ignore[arg-type]
                    journal=journal,  # type: ignore[arg-type]
                    state_store=FakeStateStore(),  # type: ignore[arg-type]
                    position_sync_seconds=0,
                    account_sync_seconds=0,
                    cash_log_min_delta_usdt=999,
                    rolling_loss_guard=guard,
                    email_sender=email_sender,  # type: ignore[arg-type]
                )
            )
            await asyncio.wait_for(journal.recorded.wait(), timeout=1)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

            self.assertFalse(execution_state.trading_halted)
            self.assertIsNone(execution_state.halt_reason)
            self.assertEqual(guard.state.baseline_equity, 125.0)
            self.assertEqual(journal.rolling_loss_events[-1]["action"], "RESUME")
            self.assertIn("Rolling loss guard cooldown ended; trading resumed", email_sender.subjects)

    async def test_critical_halt_not_resumed_by_rolling_loss_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS - MS_PER_HOUR, 100.0)
            guard.state.halt_active = True
            guard.state.halt_level = "SOFT"
            guard.state.halt_until_ts_ms = 1
            guard.save()
            execution_state = ExecutionState(None, None, trading_halted=True, halt_reason="near_tp_reduce_failure", halt_until_ts_ms=1)
            trader = FakeTrader(equity=125.0)
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=AccountSnapshot(flat_position(), 125.0, 125.0, asyncio.get_running_loop().time(), 0, 1),
                    execution_state=execution_state,
                    trader=trader,  # type: ignore[arg-type]
                    sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                    strategy=FakeStrategy(),  # type: ignore[arg-type]
                    journal=FakeJournal(),  # type: ignore[arg-type]
                    state_store=FakeStateStore(),  # type: ignore[arg-type]
                    position_sync_seconds=0,
                    account_sync_seconds=0,
                    cash_log_min_delta_usdt=999,
                    rolling_loss_guard=guard,
                    email_sender=None,
                )
            )
            await asyncio.wait_for(trader.fetched.wait(), timeout=1)
            await asyncio.sleep(0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

            self.assertTrue(execution_state.trading_halted)
            self.assertEqual(execution_state.halt_reason, "near_tp_reduce_failure")

    async def test_rolling_loss_hard_halt_does_not_override_or_email_when_critical_halt_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
            guard = self.make_guard(Path(tmp), email_enabled=True)
            guard.load_or_initialize(now_ms - MS_PER_HOUR, 100.0)
            journal = FakeJournal()
            email_sender = FakeEmailSender()
            strategy = FakeStrategy()
            strategy.state = StrategyPositionState(
                side="LONG",
                layers=1,
                last_entry_price=100.0,
                tp_price=101.0,
                total_entry_qty=1.0,
                total_entry_notional=100.0,
                avg_entry_price=100.0,
            )
            execution_state = ExecutionState(
                "pos-1",
                100.0,
                trading_halted=True,
                halt_reason="near_tp_reduce_failure",
            )

            with patch.dict(
                os.environ,
                {
                    "FLAT_BALANCE_CONFIRM_ATTEMPTS": "2",
                    "FLAT_BALANCE_CONFIRM_INTERVAL_SECONDS": "0",
                    "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                    "CASH_TRANSFER_SETTLE_SECONDS": "0",
                    "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "180",
                    "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                },
            ):
                task = asyncio.create_task(
                    account_position_sync_worker(
                        state_lock=asyncio.Lock(),
                        account_snapshot=AccountSnapshot(live_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                        execution_state=execution_state,
                        trader=FakeTrader(equity=80.0),  # type: ignore[arg-type]
                        sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                        strategy=strategy,  # type: ignore[arg-type]
                        journal=journal,  # type: ignore[arg-type]
                        state_store=FakeStateStore(),  # type: ignore[arg-type]
                        position_sync_seconds=0,
                        account_sync_seconds=999,
                        cash_log_min_delta_usdt=999,
                        rolling_loss_guard=guard,
                        email_sender=email_sender,  # type: ignore[arg-type]
                    )
                )
                await asyncio.wait_for(journal.recorded.wait(), timeout=1)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            self.assertTrue(execution_state.trading_halted)
            self.assertEqual(execution_state.halt_reason, "near_tp_reduce_failure")
            self.assertIsNone(execution_state.halt_until_ts_ms)
            self.assertEqual(journal.rolling_loss_events[-1]["action"], "HARD_HALT")
            self.assertTrue(journal.rolling_loss_events[-1]["critical_halt_preserved"])
            self.assertEqual(journal.rolling_loss_events[-1]["existing_halt_reason"], "near_tp_reduce_failure")
            self.assertTrue(journal.rolling_loss_events[-1]["rolling_loss_halt_not_applied"])
            self.assertNotIn("Rolling loss guard hard halt: 20% realized loss reached", email_sender.subjects)

    async def test_rolling_loss_hard_halt_emails_when_no_critical_halt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
            guard = self.make_guard(Path(tmp), email_enabled=True)
            guard.load_or_initialize(now_ms - MS_PER_HOUR, 100.0)
            journal = FakeJournal()
            email_sender = FakeEmailSender()
            strategy = FakeStrategy()
            strategy.state = StrategyPositionState(
                side="LONG",
                layers=1,
                last_entry_price=100.0,
                tp_price=101.0,
                total_entry_qty=1.0,
                total_entry_notional=100.0,
                avg_entry_price=100.0,
            )
            execution_state = ExecutionState("pos-1", 100.0)

            with patch.dict(
                os.environ,
                {
                    "FLAT_BALANCE_CONFIRM_ATTEMPTS": "2",
                    "FLAT_BALANCE_CONFIRM_INTERVAL_SECONDS": "0",
                    "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                    "CASH_TRANSFER_SETTLE_SECONDS": "0",
                    "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "180",
                    "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                },
            ):
                task = asyncio.create_task(
                    account_position_sync_worker(
                        state_lock=asyncio.Lock(),
                        account_snapshot=AccountSnapshot(live_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                        execution_state=execution_state,
                        trader=FakeTrader(equity=80.0),  # type: ignore[arg-type]
                        sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                        strategy=strategy,  # type: ignore[arg-type]
                        journal=journal,  # type: ignore[arg-type]
                        state_store=FakeStateStore(),  # type: ignore[arg-type]
                        position_sync_seconds=0,
                        account_sync_seconds=999,
                        cash_log_min_delta_usdt=999,
                        rolling_loss_guard=guard,
                        email_sender=email_sender,  # type: ignore[arg-type]
                    )
                )
                await asyncio.wait_for(journal.recorded.wait(), timeout=1)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            self.assertTrue(execution_state.trading_halted)
            self.assertEqual(execution_state.halt_reason, "rolling_loss_hard_halt")
            self.assertIn("Rolling loss guard hard halt: 20% realized loss reached", email_sender.subjects)


if __name__ == "__main__":
    unittest.main()
