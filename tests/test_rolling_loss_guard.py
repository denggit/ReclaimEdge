from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import importlib.util
import json
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

from scripts.run_boll_cvd_live import account_position_sync_worker  # noqa: E402
from src.execution.trader import PositionSnapshot  # noqa: E402
from src.live.runtime_types import AccountSnapshot, ExecutionState  # noqa: E402
from src.reporting.trade_journal import JournalEvent  # noqa: E402
from src.risk.rolling_loss_guard import (  # noqa: E402
    MS_PER_HOUR,
    RollingLossGuard,
    RollingLossGuardConfig,
)
from src.risk.rolling_loss_live import (  # noqa: E402
    apply_rolling_loss_guard_startup_state,
    rolling_loss_guard_payload,
    rolling_loss_guard_state_payload,
    rolling_loss_halt_reason,
)
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig  # noqa: E402
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState  # noqa: E402

NOW_MS = 1_800_000_000_000


def flat_position() -> PositionSnapshot:
    return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))


def live_position() -> PositionSnapshot:
    return PositionSnapshot("LONG", Decimal("1"), 100.0, 1.0, Decimal("1"))


def old_flat_event(loss: float) -> JournalEvent:
    return JournalEvent(
        "old-event",
        "FLAT",
        dt.datetime.fromtimestamp(NOW_MS / 1000, tz=dt.timezone.utc).isoformat(),
        "old-position",
        {"realized_pnl_usdt_est": loss},
    )


class FakeStrategy:
    def __init__(self) -> None:
        self.state = StrategyPositionState()


class FakeTrader:
    def __init__(self, position: PositionSnapshot | None = None, equity: float = 100.0,
                 cash: float | None = None) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.account_equity_usdt = equity
        self.account_cash_usdt = equity if cash is None else cash
        self.position_contracts = Decimal("0")
        self.position = position or flat_position()
        self.fetched = asyncio.Event()

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        self.fetched.set()
        return self.position

    async def fetch_usdt_equity(self) -> float:
        return self.account_equity_usdt

    async def request(self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
        return {"data": [{"details": [{"ccy": "USDT", "cashBal": str(self.account_cash_usdt)}]}]}

    def mark_flat(self) -> None:
        self.position_contracts = Decimal("0")


class FakeJournal:
    def __init__(self) -> None:
        self.rolling_loss_events: list[dict] = []  # type: ignore[type-arg]
        self.flats: list[dict] = []  # type: ignore[type-arg]
        self.cash_transfers: list[dict] = []  # type: ignore[type-arg]
        self.cash_drifts: list[dict] = []  # type: ignore[type-arg]
        self.recorded = asyncio.Event()

    def record_cash_transfer(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.cash_transfers.append(kwargs)

    def record_account_cash_drift(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.cash_drifts.append(kwargs)

    def record_rolling_loss_guard(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.rolling_loss_events.append(kwargs)
        self.recorded.set()

    def record_flat(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        payload = dict(kwargs)
        if kwargs.get("cash_before_position") is not None and kwargs.get("cash_after") is not None:
            payload["realized_pnl_usdt_est"] = kwargs["cash_after"] - kwargs["cash_before_position"]
        self.flats.append(payload)

    def append(self, event_name: str, payload: dict, position_id: str | None = None) -> None:
        if event_name == "ROLLING_LOSS_GUARD":
            self.rolling_loss_events.append(payload)
            self.recorded.set()


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
            "warn_pct": 0.50,
            "soft_halt_pct": 0.15,
            "soft_halt_hours": 6,
            "hard_halt_pct": 0.20,
            "hard_halt_hours": 12,
            "email_enabled": False,
            "event_time_tolerance_ms": 5000,
        }
        values.update(overrides)
        return RollingLossGuard(root / "rolling_loss_guard_state.json", RollingLossGuardConfig(**values))

    def test_rolling_loss_halt_reason_maps_only_halt_actions(self) -> None:
        self.assertEqual(rolling_loss_halt_reason("SOFT_HALT"), "rolling_loss_soft_halt")
        self.assertEqual(rolling_loss_halt_reason("HARD_HALT"), "rolling_loss_hard_halt")
        self.assertIsNone(rolling_loss_halt_reason("WARN"))
        self.assertIsNone(rolling_loss_halt_reason("RESUME"))
        self.assertIsNone(rolling_loss_halt_reason("UNKNOWN"))

    def test_rolling_loss_guard_payload_preserves_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS, 100.0)
            decision = guard.evaluate_after_flat(now_ms=NOW_MS + 1, flat_equity=80.0, flat_event_id="flat-1")

            payload = rolling_loss_guard_payload(decision.action, decision)

            self.assertEqual(
                set(payload),
                {
                    "action",
                    "mode",
                    "window_start_ts_ms",
                    "window_end_ts_ms",
                    "baseline_equity",
                    "reference_flat_equity",
                    "flat_equity",
                    "segment_retention",
                    "segment_return_pct",
                    "cumulative_retention",
                    "drawdown_pct",
                    "max_drawdown_pct",
                    "rolling_realized_pnl",
                    "loss_usdt",
                    "loss_pct",
                    "threshold_pct",
                    "halt_hours",
                    "halt_until_ts_ms",
                    "reason",
                },
            )
            self.assertEqual(payload["action"], "HARD_HALT")
            self.assertEqual(payload["mode"], "flat_to_flat_drawdown")
            self.assertIsNone(payload["window_start_ts_ms"])
            self.assertIsNone(payload["window_end_ts_ms"])
            self.assertEqual(payload["reference_flat_equity"], decision.reference_flat_equity)
            self.assertEqual(payload["flat_equity"], decision.flat_equity)
            self.assertEqual(payload["halt_until_ts_ms"], decision.halt_until_ts_ms)

    def test_rolling_loss_guard_state_payload_preserves_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            with self.assertRaises(RuntimeError):
                rolling_loss_guard_state_payload("RESUME", guard, "not_loaded")

            guard.load_or_initialize(NOW_MS, 100.0)
            guard.evaluate_after_flat(now_ms=NOW_MS + 1, flat_equity=80.0, flat_event_id="flat-1")

            payload = rolling_loss_guard_state_payload("RESUME", guard, "cooldown_elapsed")

            self.assertEqual(
                set(payload),
                {
                    "action",
                    "mode",
                    "window_start_ts_ms",
                    "window_end_ts_ms",
                    "baseline_equity",
                    "reference_flat_equity",
                    "flat_equity",
                    "segment_retention",
                    "segment_return_pct",
                    "cumulative_retention",
                    "drawdown_pct",
                    "max_drawdown_pct",
                    "rolling_realized_pnl",
                    "loss_usdt",
                    "loss_pct",
                    "threshold_pct",
                    "halt_hours",
                    "halt_until_ts_ms",
                    "reason",
                },
            )
            self.assertEqual(payload["action"], "RESUME")
            self.assertEqual(payload["mode"], "flat_to_flat_drawdown")
            self.assertIsNone(payload["window_start_ts_ms"])
            self.assertIsNone(payload["window_end_ts_ms"])
            self.assertEqual(payload["reference_flat_equity"], guard.state.reference_flat_equity)
            self.assertEqual(payload["flat_equity"], guard.state.last_flat_equity)
            self.assertEqual(payload["reason"], "cooldown_elapsed")

    def test_flat_to_flat_drawdown_soft_halt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp), soft_halt_pct=0.15)
            guard.load_or_initialize(NOW_MS, 100.0)

            decision = guard.evaluate_after_flat(now_ms=NOW_MS + 1, flat_equity=85.0, flat_event_id="flat-1")

            self.assertEqual(decision.action, "SOFT_HALT")
            self.assertAlmostEqual(decision.segment_retention, 0.85)
            self.assertAlmostEqual(decision.cumulative_retention, 0.85)
            self.assertAlmostEqual(decision.drawdown_pct, 0.15)
            self.assertEqual(decision.halt_until_ts_ms, NOW_MS + 1 + 6 * MS_PER_HOUR)
            self.assertEqual(rolling_loss_halt_reason(decision.action), "rolling_loss_soft_halt")

    def test_flat_to_flat_drawdown_hard_halt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS, 100.0)

            decision = guard.evaluate_after_flat(now_ms=NOW_MS + 1, flat_equity=80.0, flat_event_id="flat-1")

            self.assertEqual(decision.action, "HARD_HALT")
            self.assertAlmostEqual(decision.drawdown_pct, 0.20)
            self.assertEqual(decision.halt_until_ts_ms, NOW_MS + 1 + 12 * MS_PER_HOUR)
            self.assertEqual(rolling_loss_halt_reason(decision.action), "rolling_loss_hard_halt")

    def test_transfer_while_flat_does_not_reset_drawdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp), soft_halt_pct=0.15, hard_halt_pct=0.25)
            guard.load_or_initialize(NOW_MS, 100.0)
            first = guard.evaluate_after_flat(now_ms=NOW_MS + 1, flat_equity=90.0, flat_event_id="flat-1")
            self.assertAlmostEqual(first.drawdown_pct, 0.10)

            guard.adjust_flat_reference_for_cash_transfer(now_ms=NOW_MS + 2, new_flat_equity=100.0, reason="deposit")
            self.assertAlmostEqual(guard.state.cumulative_retention, 0.90)
            self.assertAlmostEqual(guard.state.drawdown_pct, 0.10)

            second = guard.evaluate_after_flat(now_ms=NOW_MS + 3, flat_equity=90.0, flat_event_id="flat-2")

            self.assertEqual(second.action, "SOFT_HALT")
            self.assertAlmostEqual(second.cumulative_retention, 0.81)
            self.assertAlmostEqual(second.drawdown_pct, 0.19)

    async def test_safe_flat_cash_transfer_adjusts_rolling_drawdown_reference_in_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp), soft_halt_pct=0.15, hard_halt_pct=0.25)
            guard.load_or_initialize(NOW_MS, 100.0)
            first = guard.evaluate_after_flat(now_ms=NOW_MS + 1, flat_equity=90.0, flat_event_id="flat-1")
            self.assertAlmostEqual(first.drawdown_pct, 0.10)
            self.assertAlmostEqual(guard.state.reference_flat_equity, 90.0)

            transfer_recorded = asyncio.Event()

            class RecordingJournal(FakeJournal):
                def record_cash_transfer(inner_self, **kwargs) -> None:  # type: ignore[no-untyped-def]
                    super().record_cash_transfer(**kwargs)
                    transfer_recorded.set()

            trader = FakeTrader(position=flat_position(), equity=100.0, cash=100.0)
            journal = RecordingJournal()

            with patch.dict(
                    os.environ,
                    {
                        "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                        "CASH_TRANSFER_SETTLE_SECONDS": "0",
                        "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "0",
                        "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                    },
            ):
                task = asyncio.create_task(
                    account_position_sync_worker(
                        state_lock=asyncio.Lock(),
                        account_snapshot=AccountSnapshot(flat_position(), 90.0, 90.0, asyncio.get_running_loop().time(),
                                                         0, 1),
                        execution_state=ExecutionState(None, None),
                        trader=trader,  # type: ignore[arg-type]
                        sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                        strategy=FakeStrategy(),  # type: ignore[arg-type]
                        journal=journal,  # type: ignore[arg-type]
                        state_store=FakeStateStore(),  # type: ignore[arg-type]
                        position_sync_seconds=0,
                        account_sync_seconds=0,
                        cash_log_min_delta_usdt=999,
                        rolling_loss_guard=guard,
                        email_sender=None,
                    )
                )
                await asyncio.wait_for(transfer_recorded.wait(), timeout=1)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            self.assertEqual(len(journal.cash_transfers), 1)
            self.assertEqual(journal.cash_transfers[0]["cash_after"], 100.0)
            self.assertAlmostEqual(guard.state.reference_flat_equity, 100.0)
            self.assertAlmostEqual(guard.state.cumulative_retention, 0.90)
            self.assertAlmostEqual(guard.state.drawdown_pct, 0.10)

            second = guard.evaluate_after_flat(now_ms=NOW_MS + 2, flat_equity=90.0, flat_event_id="flat-2")
            self.assertAlmostEqual(second.cumulative_retention, 0.81)
            self.assertAlmostEqual(second.drawdown_pct, 0.19)

    def test_profit_recovers_drawdown_and_resets_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp), warn_pct=0.10, soft_halt_pct=0.15)
            guard.load_or_initialize(NOW_MS, 100.0)
            first = guard.evaluate_after_flat(now_ms=NOW_MS + 1, flat_equity=90.0, flat_event_id="flat-1")
            self.assertEqual(first.action, "WARN")
            self.assertTrue(guard.state.warn_triggered)

            recovered = guard.evaluate_after_flat(now_ms=NOW_MS + 2, flat_equity=100.0, flat_event_id="flat-2")

            self.assertIsNone(recovered.action)
            self.assertAlmostEqual(recovered.cumulative_retention, 1.0)
            self.assertAlmostEqual(recovered.drawdown_pct, 0.0)
            self.assertFalse(guard.state.warn_triggered)
            self.assertFalse(guard.state.soft_halt_triggered)
            self.assertFalse(guard.state.hard_halt_triggered)

    def test_no_time_window_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS, 100.0)

            self.assertFalse(guard.should_reset_expired_window(NOW_MS + 999 * MS_PER_HOUR, has_position=False))
            self.assertFalse(guard.should_reset_expired_window(NOW_MS + 999 * MS_PER_HOUR, has_position=True))

    def test_evaluate_after_flat_does_not_scan_journal_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp), warn_pct=0.10)
            guard.load_or_initialize(NOW_MS, 100.0)

            decision = guard.evaluate_after_flat(
                now_ms=NOW_MS + 1,
                flat_equity=100.0,
                flat_event_id="flat-1",
                journal_events=[old_flat_event(-90.0)],
            )

            self.assertIsNone(decision.action)
            self.assertEqual(decision.reason, "drawdown_not_worsened")
            self.assertAlmostEqual(decision.drawdown_pct, 0.0)

    def test_duplicate_flat_event_id_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp), warn_pct=0.10, hard_halt_pct=0.50)
            guard.load_or_initialize(NOW_MS, 100.0)
            first = guard.evaluate_after_flat(now_ms=NOW_MS + 1, flat_equity=90.0, flat_event_id="flat-1")
            second = guard.evaluate_after_flat(now_ms=NOW_MS + 2, flat_equity=90.0, flat_event_id="flat-1")

            self.assertEqual(first.action, "WARN")
            self.assertIsNone(second.action)
            self.assertEqual(second.reason, "duplicate_flat_event")
            self.assertAlmostEqual(guard.state.cumulative_retention, 0.90)

    async def test_cash_transfer_with_position_does_not_adjust_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS, 100.0)
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
            trader = FakeTrader(position=live_position(), equity=120.0, cash=120.0)
            execution_state = ExecutionState("pos-1", 100.0)
            journal = FakeJournal()

            with patch.dict(
                    os.environ,
                    {
                        "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                        "CASH_TRANSFER_SETTLE_SECONDS": "0",
                        "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "0",
                        "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                    },
            ):
                task = asyncio.create_task(
                    account_position_sync_worker(
                        state_lock=asyncio.Lock(),
                        account_snapshot=AccountSnapshot(live_position(), 100.0, 100.0,
                                                         asyncio.get_running_loop().time(), 0, 1),
                        execution_state=execution_state,
                        trader=trader,  # type: ignore[arg-type]
                        sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                        strategy=strategy,  # type: ignore[arg-type]
                        journal=journal,  # type: ignore[arg-type]
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

            self.assertAlmostEqual(guard.state.reference_flat_equity, 100.0)
            self.assertEqual(journal.cash_transfers, [])
            self.assertEqual(len(journal.cash_drifts), 1)

    def test_old_state_migrates_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "rolling_loss_guard_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "window_start_ts_ms": NOW_MS - 24 * MS_PER_HOUR,
                        "window_end_ts_ms": NOW_MS,
                        "baseline_equity": 100.0,
                        "last_loss_pct": 0.1,
                    }
                ),
                encoding="utf-8",
            )
            guard = RollingLossGuard(state_path, RollingLossGuardConfig())

            state = guard.load_or_initialize(NOW_MS, 100.0)

            self.assertAlmostEqual(state.reference_flat_equity, 100.0)
            self.assertAlmostEqual(state.cumulative_retention, 0.9)
            self.assertAlmostEqual(state.drawdown_pct, 0.1)

    def test_mark_resumed_does_not_reset_drawdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS, 100.0)
            guard.evaluate_after_flat(now_ms=NOW_MS + 1, flat_equity=80.0, flat_event_id="flat-1")
            self.assertTrue(guard.state.halt_active)

            guard.mark_resumed(NOW_MS + 13 * MS_PER_HOUR, 125.0)

            self.assertFalse(guard.state.halt_active)
            self.assertIsNone(guard.state.halt_level)
            self.assertIsNone(guard.state.halt_until_ts_ms)
            self.assertAlmostEqual(guard.state.cumulative_retention, 0.8)
            self.assertAlmostEqual(guard.state.drawdown_pct, 0.2)
            self.assertAlmostEqual(guard.state.reference_flat_equity, 80.0)

    async def test_cooldown_persists_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS, 100.0)
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

    async def test_cooldown_resume_preserves_drawdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp), email_enabled=True)
            guard.load_or_initialize(NOW_MS, 100.0)
            guard.evaluate_after_flat(now_ms=NOW_MS + 1, flat_equity=80.0, flat_event_id="flat-1")
            guard.state.halt_until_ts_ms = 1
            guard.save()
            journal = FakeJournal()
            email_sender = FakeEmailSender()
            execution_state = ExecutionState(None, None, trading_halted=True, halt_reason="rolling_loss_hard_halt",
                                             halt_until_ts_ms=1)
            trader = FakeTrader(equity=125.0)
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=AccountSnapshot(flat_position(), 125.0, 125.0, asyncio.get_running_loop().time(),
                                                     0, 1),
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
            self.assertAlmostEqual(guard.state.reference_flat_equity, 80.0)
            self.assertAlmostEqual(guard.state.cumulative_retention, 0.8)
            self.assertAlmostEqual(guard.state.drawdown_pct, 0.2)
            self.assertEqual(journal.rolling_loss_events[-1]["action"], "RESUME")
            self.assertIn("Rolling loss guard cooldown ended; trading resumed", email_sender.subjects)

    async def test_critical_halt_not_resumed_by_rolling_loss_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.make_guard(Path(tmp))
            guard.load_or_initialize(NOW_MS, 100.0)
            guard.state.halt_active = True
            guard.state.halt_level = "SOFT"
            guard.state.halt_until_ts_ms = 1
            guard.save()
            execution_state = ExecutionState(None, None, trading_halted=True, halt_reason="near_tp_reduce_failure",
                                             halt_until_ts_ms=1)
            trader = FakeTrader(equity=125.0)
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=AccountSnapshot(flat_position(), 125.0, 125.0, asyncio.get_running_loop().time(),
                                                     0, 1),
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
            guard.load_or_initialize(now_ms, 100.0)
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
                        account_snapshot=AccountSnapshot(live_position(), 100.0, 100.0,
                                                         asyncio.get_running_loop().time(), 0, 1),
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
            guard.load_or_initialize(now_ms, 100.0)
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
                        account_snapshot=AccountSnapshot(live_position(), 100.0, 100.0,
                                                         asyncio.get_running_loop().time(), 0, 1),
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
            self.assertEqual(journal.rolling_loss_events[-1]["mode"], "flat_to_flat_drawdown")
            self.assertAlmostEqual(journal.rolling_loss_events[-1]["drawdown_pct"], 0.20)
            self.assertIn("Rolling loss guard hard halt: 20% realized loss reached", email_sender.subjects)


if __name__ == "__main__":
    unittest.main()
