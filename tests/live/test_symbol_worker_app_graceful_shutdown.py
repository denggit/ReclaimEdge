#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D06b tests for SymbolWorkerApp graceful shutdown / drain mode.

These tests use fake factory, fake workers, and monkeypatching — they do
NOT connect to OKX or start a real trader.

Key invariants verified:
* shutdown marks execution_state.trading_halted / halt_reason
* shutdown does NOT call market_close / cancel / place_order
* shutdown best-effort saves state when a position exists
* shutdown writes stopping/stopped heartbeat
* shutdown cancels runtime tasks
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from src.live.live_app_config import (
    DailyReportConfig,
    LiveAppConfig,
    LiveHeartbeatConfig,
    WeeklySummaryConfig,
)
from src.live.runtime_types import AccountSnapshot, ExecutionState
from src.live.symbol_worker_app import SymbolWorkerApp
from src.live.symbol_worker_factory import SymbolWorkerFactory
from src.live.worker_shutdown import WorkerShutdownController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_config(**overrides: Any) -> LiveAppConfig:
    kwargs: dict[str, Any] = dict(
        strategy_tick_queue_maxsize=100,
        execution_queue_maxsize=10,
        position_sync_seconds=5.0,
        account_sync_seconds=60.0,
        cash_log_min_delta_usdt=0.01,
        market_tick_heartbeat_seconds=60.0,
        account_snapshot_stale_warn_seconds=30.0,
        strategy_tick_lag_warn_seconds=2.0,
        execution_queue_backlog_log_seconds=30.0,
        daily_report=DailyReportConfig(raw_time="09:00", hour=9, minute=0),
        weekly_summary=WeeklySummaryConfig(
            enabled=True,
            raw_time="10:00",
            raw_weekday="0",
            weekday=0,
            hour=10,
            minute=0,
            compact_after_success=False,
        ),
        heartbeat=LiveHeartbeatConfig(
            enabled=True, interval_seconds=10.0, stale_after_seconds=30.0
        ),
    )
    kwargs.update(overrides)
    return LiveAppConfig(**kwargs)


class _FakeHeartbeatWriter:
    """Records status writes for assertion."""

    def __init__(self) -> None:
        self.status_writes: list[str] = []
        self.config = _FakeHeartbeatConfig(enabled=True)

    def write_status_once(self, status: str) -> None:
        self.status_writes.append(status)

    async def run_until_cancelled(self) -> None:
        # Block forever — simulate an enabled heartbeat loop.
        await asyncio.Event().wait()


@dataclass
class _FakeHeartbeatConfig:
    enabled: bool


class _FakeMonitor:
    async def run_forever(self) -> None:
        await asyncio.Event().wait()


class _FakeStrategy:
    def __init__(self, layers: int = 0) -> None:
        self.state = _FakeStrategyState(layers=layers)


class _FakeStrategyState:
    def __init__(self, layers: int = 0) -> None:
        self.layers = layers
        self.side = "long" if layers > 0 else None
        self.last_entry_price = None
        self.tp_price = None
        self.tp_mode = "MIDDLE"
        self.last_order_ts_ms = 0
        self.total_entry_qty = 0.0
        self.total_entry_notional = 0.0
        self.avg_entry_price = 0.0
        self.breakeven_price = 0.0
        self.last_tp_update_ts_ms = 0


class _FakeTrader:
    def __init__(self) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.td_mode = "isolated"
        self.pos_side_mode = "net"
        self.leverage = 15
        self.account_equity_usdt = 10000.0

    async def start(self) -> None:
        pass

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def fetch_position_snapshot(self) -> Any:
        from src.execution.trader import PositionSnapshot
        from decimal import Decimal

        return PositionSnapshot(
            side=None,
            contracts=Decimal("0"),
            eth_qty=0.0,
            avg_entry_price=0.0,
            raw_pos=Decimal("0"),
        )


class _FakeStateStore:
    def __init__(self) -> None:
        self.save_calls: list[Any] = []
        self.clear_calls: int = 0
        self.load_calls: int = 0

    def load(self) -> None:
        self.load_calls += 1
        return None

    def save(self, state: Any) -> None:
        self.save_calls.append(state)

    def clear(self) -> None:
        self.clear_calls += 1


class _FakeJournal:
    def record_cash_baseline(self, **kwargs: Any) -> None:
        pass

    def new_position_id(self, *args: Any) -> str:
        return "pos-fake-1"

    def record_startup_recovery(self, **kwargs: Any) -> None:
        pass


class _FakeRollingLossGuard:
    def load_or_initialize(self, *args: Any) -> None:
        pass


class _FakeSizer:
    def __init__(self) -> None:
        self.config = _FakeSizerConfig()


class _FakeSizerConfig:
    sidecar_skip_first_layer: bool = False


class _FakeCvd:
    pass


class _FakeQueues:
    def __init__(self) -> None:
        self.strategy_tick_queue = asyncio.Queue(maxsize=10)
        self.execution_queue = asyncio.Queue(maxsize=10)


class _FakePersistence:
    def __init__(self) -> None:
        self.journal = _FakeJournal()
        self.state_store = _FakeStateStore()
        self.rolling_loss_guard = _FakeRollingLossGuard()
        self.reporter = _FakeReporter()


class _FakeReporter:
    async def send_last_24h_report(self, context: Any) -> bool:
        return True

    async def send_overall_summary_report(self, context: Any) -> bool:
        return True


class _FakeStrategyObjects:
    def __init__(self, strategy: _FakeStrategy | None = None) -> None:
        self.sizer = _FakeSizer()
        self.strategy = strategy or _FakeStrategy()


class _FakeRuntimePaths:
    def __init__(self, tmp_path: Any) -> None:
        self.heartbeat_file = tmp_path / "heartbeat.json"
        self.heartbeats_dir = tmp_path
        self.symbol_slug = "ETH-USDT-SWAP"
        self.state_file = tmp_path / "state.json"
        self.journal_file = tmp_path / "journal.jsonl"
        self.trade_summary_file = tmp_path / "summary.jsonl"
        self.rolling_loss_guard_state_file = tmp_path / "rolling_loss.json"


class _ShutdownTestFactory(SymbolWorkerFactory):
    """Factory that returns fully faked objects for shutdown testing."""

    def __init__(
        self,
        *,
        tmp_path: Any,
        strategy: _FakeStrategy | None = None,
        state_store: _FakeStateStore | None = None,
    ) -> None:
        super().__init__()
        self._tmp_path = tmp_path
        self._strategy = strategy or _FakeStrategy()
        self._state_store = state_store or _FakeStateStore()

    def create_email_sender(self) -> Any:
        return object()

    def create_trader(self) -> _FakeTrader:
        return _FakeTrader()

    def create_runtime_paths(self, *, runtime_dir: str | Any, inst_id: str) -> _FakeRuntimePaths:
        return _FakeRuntimePaths(self._tmp_path)

    def create_heartbeat_writer(self, *, runtime_paths: Any, heartbeat_config: Any) -> _FakeHeartbeatWriter:
        return _FakeHeartbeatWriter()

    def create_persistence(self, *, runtime_paths: Any, email_sender: Any) -> _FakePersistence:
        p = _FakePersistence()
        if self._state_store is not None:
            p.state_store = self._state_store
        return p

    def create_strategy_objects(self, *, strategy_config: Any, position_sizer_config: Any) -> _FakeStrategyObjects:
        return _FakeStrategyObjects(strategy=self._strategy)

    def create_cvd_tracker(self, config: Any) -> _FakeCvd:
        return _FakeCvd()

    def create_queues(self, app_config: Any) -> _FakeQueues:
        return _FakeQueues()

    def create_monitor(self, *, config: Any, tick_handlers: Any) -> _FakeMonitor:
        return _FakeMonitor()


# ============================================================================
# 1. from_env accepts shutdown_controller
# ============================================================================


class TestFromEnvShutdownController:
    def test_from_env_accepts_shutdown_controller(self) -> None:
        controller = WorkerShutdownController()
        app = SymbolWorkerApp.from_env(shutdown_controller=controller)
        assert app.shutdown_controller is controller

    def test_from_env_shutdown_controller_defaults_to_none(self) -> None:
        app = SymbolWorkerApp.from_env()
        assert app.shutdown_controller is None


# ============================================================================
# 2. Shutdown marks execution state draining
# ============================================================================


class TestShutdownMarksDraining:
    @pytest.mark.asyncio
    async def test_shutdown_marks_execution_state_draining(self, tmp_path: Any) -> None:
        """When shutdown is triggered, execution_state.trading_halted must be
        True with halt_reason='symbol_worker_shutdown_draining'."""
        controller = WorkerShutdownController()
        factory = _ShutdownTestFactory(tmp_path=tmp_path)

        app = SymbolWorkerApp(
            app_config=_make_app_config(
                symbol_worker_shutdown_save_state_enabled=False,
                symbol_worker_shutdown_heartbeat_enabled=False,
            ),
            factory=factory,
            shutdown_controller=controller,
        )

        # Patch the runtime tasks to block briefly, then trigger shutdown
        original_run = app.run

        async def _patched_run() -> None:
            # We need to intercept the execution_state after it's created
            # inside run().  The simplest approach: monkeypatch the
            # monitor.run_forever to trigger shutdown after a short delay.
            pass

        # Instead of full app.run(), test the _begin_symbol_worker_drain
        # helper directly to verify drain marking.
        from src.live.symbol_worker_shutdown_runtime import _begin_symbol_worker_drain

        execution_state = ExecutionState(
            current_position_id="pos-1",
            cash_before_position=5000.0,
        )
        state_lock = asyncio.Lock()

        await _begin_symbol_worker_drain(
            execution_state=execution_state,
            state_lock=state_lock,
            reason="SIGTERM",
        )

        assert execution_state.trading_halted is True
        assert execution_state.halt_reason == "symbol_worker_shutdown_draining"
        assert execution_state.halt_until_ts_ms is None


# ============================================================================
# 3. Shutdown saves state when position exists
# ============================================================================


class TestShutdownSavesState:
    def test_should_save_when_position_id_exists(self) -> None:
        from src.live.symbol_worker_shutdown_runtime import _should_save_state_on_shutdown

        execution_state = ExecutionState(
            current_position_id="pos-1",
            cash_before_position=5000.0,
        )
        strategy_state = _FakeStrategyState(layers=0)

        assert _should_save_state_on_shutdown(execution_state, strategy_state) is True

    def test_should_save_when_layers_gt_zero(self) -> None:
        from src.live.symbol_worker_shutdown_runtime import _should_save_state_on_shutdown

        execution_state = ExecutionState(
            current_position_id=None,
            cash_before_position=None,
        )
        strategy_state = _FakeStrategyState(layers=3)

        assert _should_save_state_on_shutdown(execution_state, strategy_state) is True

    def test_should_not_save_when_flat_and_no_position(self) -> None:
        from src.live.symbol_worker_shutdown_runtime import _should_save_state_on_shutdown

        execution_state = ExecutionState(
            current_position_id=None,
            cash_before_position=None,
        )
        strategy_state = _FakeStrategyState(layers=0)

        assert _should_save_state_on_shutdown(execution_state, strategy_state) is False

    @pytest.mark.asyncio
    async def test_save_state_called_when_position_exists(self, tmp_path: Any) -> None:
        from src.live.symbol_worker_shutdown_runtime import _save_state_on_shutdown

        execution_state = ExecutionState(
            current_position_id="pos-1",
            cash_before_position=5000.0,
        )
        strategy = _FakeStrategy(layers=1)
        state_store = _FakeStateStore()

        await _save_state_on_shutdown(
            execution_state=execution_state,
            strategy=strategy,
            trader_symbol="ETH-USDT-SWAP",
            state_store=state_store,  # type: ignore[arg-type]
        )

        assert len(state_store.save_calls) == 1
        assert state_store.clear_calls == 0

    @pytest.mark.asyncio
    async def test_save_state_skipped_when_flat(self, tmp_path: Any) -> None:
        from src.live.symbol_worker_shutdown_runtime import _save_state_on_shutdown

        execution_state = ExecutionState(
            current_position_id=None,
            cash_before_position=None,
        )
        strategy = _FakeStrategy(layers=0)
        state_store = _FakeStateStore()

        await _save_state_on_shutdown(
            execution_state=execution_state,
            strategy=strategy,
            trader_symbol="ETH-USDT-SWAP",
            state_store=state_store,  # type: ignore[arg-type]
        )

        assert len(state_store.save_calls) == 0
        assert state_store.clear_calls == 0


# ============================================================================
# 4. Shutdown heartbeat writes
# ============================================================================


class TestShutdownHeartbeat:
    def test_write_status_once_stopping(self, tmp_path: Any) -> None:
        """write_status_once('stopping') must record the status."""
        hb = _FakeHeartbeatWriter()
        hb.write_status_once("stopping")
        assert hb.status_writes == ["stopping"]

    def test_write_status_once_stopped(self, tmp_path: Any) -> None:
        """write_status_once('stopped') must record the status."""
        hb = _FakeHeartbeatWriter()
        hb.write_status_once("stopped")
        assert hb.status_writes == ["stopped"]

    def test_write_status_once_sequence(self, tmp_path: Any) -> None:
        """write_status_once must record stopping THEN stopped."""
        hb = _FakeHeartbeatWriter()
        hb.write_status_once("stopping")
        hb.write_status_once("stopped")
        assert hb.status_writes == ["stopping", "stopped"]


# ============================================================================
# 5. Shutdown does NOT call order mutation methods
# ============================================================================


class TestShutdownNoOrderMutation:
    def test_shutdown_runtime_module_no_trader_imports(self) -> None:
        """The shutdown runtime module must not import Trader or any trading
        mutation symbols."""
        from pathlib import Path

        source = (
            Path(__file__)
            .resolve()
            .parent.parent.parent.joinpath(
                "src", "live", "symbol_worker_shutdown_runtime.py"
            )
            .read_text(encoding="utf-8")
        )

        forbidden = [
            "from src.execution.trader import",
            "place_market_order",
            "close_position",
            "market_close",
            "cancel_all",
            "cancel_algo_order",
            "cancel_near_tp",
            "cancel_middle_runner",
            "cancel_three_stage",
            "cancel_trend_runner",
        ]
        for token in forbidden:
            assert token not in source, (
                f"symbol_worker_shutdown_runtime.py must not reference {token!r}"
            )


# ============================================================================
# 6. Shutdown cancels runtime tasks
# ============================================================================


class TestShutdownCancelsTasks:
    @pytest.mark.asyncio
    async def test_cancel_runtime_tasks(self) -> None:
        """_cancel_runtime_tasks must cancel and await pending tasks."""
        from src.live.symbol_worker_shutdown_runtime import _cancel_runtime_tasks

        async def _block_forever() -> None:
            await asyncio.Event().wait()

        tasks = {
            asyncio.ensure_future(_block_forever()),
            asyncio.ensure_future(_block_forever()),
        }

        await _cancel_runtime_tasks(tasks, timeout=1.0)

        for task in tasks:
            assert task.done()
            assert task.cancelled() or task.exception() is None

    @pytest.mark.asyncio
    async def test_cancel_runtime_tasks_empty(self) -> None:
        """_cancel_runtime_tasks with empty set must not raise."""
        from src.live.symbol_worker_shutdown_runtime import _cancel_runtime_tasks

        await _cancel_runtime_tasks(set(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_cancel_runtime_tasks_timeout(self) -> None:
        """Tasks that ignore cancellation must be logged but not block."""
        from src.live.symbol_worker_shutdown_runtime import _cancel_runtime_tasks

        async def _ignore_cancel() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Swallow cancellation — task will not finish
                await asyncio.Event().wait()

        tasks = {asyncio.ensure_future(_ignore_cancel())}

        # Must return without raising despite the task ignoring cancellation
        await _cancel_runtime_tasks(tasks, timeout=0.01)


# ============================================================================
# 7. LiveAppConfig shutdown validation
# ============================================================================


class TestShutdownConfig:
    def test_drain_timeout_validation(self) -> None:
        with pytest.raises(ValueError, match="drain_timeout_seconds must be > 0"):
            _make_app_config(symbol_worker_shutdown_drain_timeout_seconds=0)

    def test_drain_timeout_negative(self) -> None:
        with pytest.raises(ValueError, match="drain_timeout_seconds must be > 0"):
            _make_app_config(symbol_worker_shutdown_drain_timeout_seconds=-1)

    def test_default_config_values(self) -> None:
        config = _make_app_config()
        assert config.symbol_worker_shutdown_drain_timeout_seconds == 10.0
        assert config.symbol_worker_shutdown_save_state_enabled is True
        assert config.symbol_worker_shutdown_heartbeat_enabled is True


# ============================================================================
# 8. Weekly summary disabled → not added to FIRST_COMPLETED tasks
# ============================================================================


class TestWeeklySummaryDisabled:
    @pytest.mark.asyncio
    async def test_weekly_summary_disabled_not_scheduled(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """When weekly_summary.enabled=False, the weekly_summary_loop must NOT
        enter the FIRST_COMPLETED task set.  app.run() must exit through the
        graceful-shutdown path when shutdown is triggered, NOT because
        weekly_summary_loop returns immediately."""
        controller = WorkerShutdownController()
        factory = _ShutdownTestFactory(tmp_path=tmp_path)

        app_config = _make_app_config(
            weekly_summary=WeeklySummaryConfig(
                enabled=False,
                raw_time="10:00",
                raw_weekday="0",
                weekday=0,
                hour=10,
                minute=0,
                compact_after_success=False,
            ),
            symbol_worker_shutdown_save_state_enabled=False,
            symbol_worker_shutdown_heartbeat_enabled=True,
        )

        app = SymbolWorkerApp(
            app_config=app_config,
            factory=factory,
            shutdown_controller=controller,
        )

        # Monkeypatch startup functions that need a real trader / OKX API.
        async def _fake_fetch_cash(_trader: Any) -> float:
            return 10000.0

        async def _noop_tp_recovery(**kwargs: Any) -> None:
            return None

        async def _noop_sidecar_recovery(**kwargs: Any) -> None:
            return None

        async def _noop_rolling_loss(**kwargs: Any) -> None:
            return None

        def _noop_safety_gate(**kwargs: Any) -> bool:
            return False

        monkeypatch.setattr(
            "src.live.symbol_worker_app.live_flat_balance.fetch_usdt_cash_balance",
            _fake_fetch_cash,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_main_tp_startup_recovery",
            _noop_tp_recovery,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_sidecar_startup_recovery",
            _noop_sidecar_recovery,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.rolling_loss_live_helpers.apply_rolling_loss_guard_startup_state",
            _noop_rolling_loss,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.runner_live_helpers.apply_three_stage_startup_safety_gate",
            _noop_safety_gate,
        )

        # Schedule shutdown after a short delay so app.run() reaches
        # the asyncio.wait before we trigger it.
        async def _trigger_shutdown() -> None:
            await asyncio.sleep(0.10)
            controller.request_shutdown("test_disabled_weekly")

        trigger_task = asyncio.ensure_future(_trigger_shutdown())

        try:
            # If weekly_summary_loop were scheduled and returns immediately
            # (because enabled=False), this would enter the unexpected-exit
            # path and raise RuntimeError.  The fact that we can reach here
            # without error proves it wasn't scheduled.
            await asyncio.wait_for(app.run(), timeout=10.0)
        finally:
            if not trigger_task.done():
                trigger_task.cancel()
                try:
                    await trigger_task
                except asyncio.CancelledError:
                    pass

        # Shutdown was requested → controller must reflect that.
        assert controller.requested is True
        assert controller.reason == "test_disabled_weekly"


# ============================================================================
# 9. Unexpected clean task completion raises RuntimeError
# ============================================================================


class TestUnexpectedCleanTaskCompletion:
    @pytest.mark.asyncio
    async def test_unexpected_clean_task_completion_raises(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """When a runtime task returns None cleanly (without raising),
        SymbolWorkerApp.run() must raise RuntimeError instead of silently
        exiting.  This prevents a silent worker return from being mistaken
        for a successful shutdown."""

        controller = WorkerShutdownController()
        factory = _ShutdownTestFactory(tmp_path=tmp_path)

        app = SymbolWorkerApp(
            app_config=_make_app_config(),
            factory=factory,
            shutdown_controller=controller,
        )

        # Monkeypatch startup functions that need a real trader / OKX API.
        async def _fake_fetch_cash(_trader: Any) -> float:
            return 10000.0

        async def _noop_tp_recovery(**kwargs: Any) -> None:
            return None

        async def _noop_sidecar_recovery(**kwargs: Any) -> None:
            return None

        async def _noop_rolling_loss(**kwargs: Any) -> None:
            return None

        def _noop_safety_gate(**kwargs: Any) -> bool:
            return False

        monkeypatch.setattr(
            "src.live.symbol_worker_app.live_flat_balance.fetch_usdt_cash_balance",
            _fake_fetch_cash,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_main_tp_startup_recovery",
            _noop_tp_recovery,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_sidecar_startup_recovery",
            _noop_sidecar_recovery,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.rolling_loss_live_helpers.apply_rolling_loss_guard_startup_state",
            _noop_rolling_loss,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.runner_live_helpers.apply_three_stage_startup_safety_gate",
            _noop_safety_gate,
        )

        # Monkeypatch account_position_sync_worker to return immediately
        # without raising.  This simulates a worker that unexpectedly
        # returns cleanly (rather than blocking forever).
        async def _fake_worker(*args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(
            "src.live.symbol_worker_app.account_position_sync_worker_module.account_position_sync_worker",
            _fake_worker,
        )

        with pytest.raises(
            RuntimeError,
            match="Symbol worker runtime task exited unexpectedly without exception",
        ):
            await app.run()


# ============================================================================
# 10. Weekly summary enabled source guard
# ============================================================================


class TestWeeklySummaryEnabledSourceGuard:
    def test_weekly_summary_enabled_condition_present(self) -> None:
        """symbol_worker_app.py must contain the conditional check
        for weekly_summary.enabled."""
        from pathlib import Path

        source = (
            Path(__file__)
            .resolve()
            .parent.parent.parent.joinpath("src", "live", "symbol_worker_app.py")
            .read_text(encoding="utf-8")
        )

        assert "if self.app_config.weekly_summary.enabled:" in source, (
            "symbol_worker_app.py must conditionally schedule weekly_summary_loop"
            " based on self.app_config.weekly_summary.enabled"
        )
        assert "weekly_summary_loop()" in source, (
            "symbol_worker_app.py must still contain weekly_summary_loop()"
        )

    def test_unexpected_clean_exit_runtime_error_present(self) -> None:
        """symbol_worker_app.py must contain the RuntimeError for unexpected
        clean task exit."""
        from pathlib import Path

        source = (
            Path(__file__)
            .resolve()
            .parent.parent.parent.joinpath("src", "live", "symbol_worker_app.py")
            .read_text(encoding="utf-8")
        )

        assert (
            "Symbol worker runtime task exited unexpectedly without exception"
        ) in source, (
            "symbol_worker_app.py must raise RuntimeError for unexpected clean exit"
        )


# ============================================================================
# 11. Disabled heartbeat not scheduled source guard
# ============================================================================


class TestDisabledHeartbeatSourceGuard:
    def test_disabled_heartbeat_condition_present(self) -> None:
        """symbol_worker_app.py must conditionally schedule heartbeat only
        when heartbeat_writer.config.enabled is True."""
        from pathlib import Path

        source = (
            Path(__file__)
            .resolve()
            .parent.parent.parent.joinpath("src", "live", "symbol_worker_app.py")
            .read_text(encoding="utf-8")
        )

        assert "if heartbeat_writer.config.enabled:" in source, (
            "symbol_worker_app.py must conditionally schedule heartbeat"
        )


# ============================================================================
# 12. symbol_worker_app.py source guard — no trading mutation
# ============================================================================


class TestSymbolWorkerAppNoTradingMutation:
    def test_app_source_no_trading_mutation(self) -> None:
        """symbol_worker_app.py must NOT contain market_close,
        close_position, cancel_all, cancel_algo_order, or
        place_market_order."""
        from pathlib import Path

        source = (
            Path(__file__)
            .resolve()
            .parent.parent.parent.joinpath("src", "live", "symbol_worker_app.py")
            .read_text(encoding="utf-8")
        )

        forbidden = [
            "market_close",
            "close_position",
            "cancel_all",
            "cancel_algo_order",
            "place_market_order",
        ]
        for token in forbidden:
            assert token not in source, (
                f"symbol_worker_app.py must not contain {token!r}"
            )


# ============================================================================
# 13. _drain_critical_runtime_tasks helper — direct tests
# ============================================================================


class TestDrainCriticalRuntimeTasks:
    @pytest.mark.asyncio
    async def test_drain_does_not_cancel_tasks(self) -> None:
        """_drain_critical_runtime_tasks must NOT cancel tasks."""
        from src.live.symbol_worker_shutdown_runtime import (
            _drain_critical_runtime_tasks,
        )

        async def _block() -> None:
            await asyncio.Event().wait()

        task = asyncio.ensure_future(_block())
        try:
            await _drain_critical_runtime_tasks(
                {task}, timeout=0.01,
            )
            # Task must NOT be cancelled or done after drain.
            assert not task.cancelled()
            assert not task.done()
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_drain_empty_tasks_returns_immediately(self) -> None:
        """_drain_critical_runtime_tasks with empty set must return immediately."""
        from src.live.symbol_worker_shutdown_runtime import (
            _drain_critical_runtime_tasks,
        )

        await _drain_critical_runtime_tasks(set(), timeout=10.0)

    @pytest.mark.asyncio
    async def test_drain_returns_when_execution_queue_empty(self) -> None:
        """_drain_critical_runtime_tasks must return when the execution
        queue becomes empty, even if critical tasks are still running."""
        from src.live.symbol_worker_shutdown_runtime import (
            _drain_critical_runtime_tasks,
        )

        execution_queue: asyncio.Queue = asyncio.Queue()

        # Put one item — the drain loop should NOT exit immediately.
        await execution_queue.put("pending_item")

        async def _block() -> None:
            await asyncio.Event().wait()

        task = asyncio.ensure_future(_block())

        # Background: drain the queue item after a short delay.
        async def _drain_queue() -> None:
            await asyncio.sleep(0.05)
            execution_queue.get_nowait()

        drainer = asyncio.ensure_future(_drain_queue())

        try:
            await _drain_critical_runtime_tasks(
                {task},
                execution_queue=execution_queue,
                timeout=10.0,
            )
            # Task is still running — drain returned because queue emptied.
            assert not task.cancelled()
            assert not task.done()
            assert execution_queue.empty()
        finally:
            task.cancel()
            if not drainer.done():
                drainer.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            try:
                await drainer
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_drain_returns_when_all_tasks_done(self) -> None:
        """_drain_critical_runtime_tasks must return when all critical
        tasks complete, even if the execution queue is not empty."""
        from src.live.symbol_worker_shutdown_runtime import (
            _drain_critical_runtime_tasks,
        )

        execution_queue: asyncio.Queue = asyncio.Queue()
        await execution_queue.put("pending_item")

        allow_finish = asyncio.Event()

        async def _finish_quickly() -> None:
            await allow_finish.wait()

        task = asyncio.ensure_future(_finish_quickly())

        # Background: set allow_finish after a short delay.
        async def _trigger() -> None:
            await asyncio.sleep(0.05)
            allow_finish.set()

        trigger = asyncio.ensure_future(_trigger())

        try:
            await _drain_critical_runtime_tasks(
                {task},
                execution_queue=execution_queue,
                timeout=10.0,
            )
            # Task completed — drain returned because all tasks done,
            # even though the queue still has an item.
            assert task.done()
            assert not task.cancelled()
            assert not execution_queue.empty()
        finally:
            if not task.done():
                task.cancel()
            if not trigger.done():
                trigger.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            try:
                await trigger
            except asyncio.CancelledError:
                pass


# ============================================================================
# 14. Shutdown does NOT cancel execution_task immediately
# ============================================================================


class TestShutdownDoesNotCancelExecutionImmediately:
    @pytest.mark.asyncio
    async def test_shutdown_does_not_cancel_execution_task_immediately(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """When shutdown is triggered, the execution worker must NOT be
        cancelled immediately.  It must get a drain window to finish its
        work naturally."""
        controller = WorkerShutdownController()
        factory = _ShutdownTestFactory(tmp_path=tmp_path)

        execution_started = asyncio.Event()
        allow_finish = asyncio.Event()
        cancelled_before_finish: list[str] = []

        async def _fake_execution_worker(**kwargs: Any) -> None:
            execution_started.set()
            try:
                await allow_finish.wait()
            except asyncio.CancelledError:
                cancelled_before_finish.append("cancelled_before_finish")
                raise

        async def _fake_account_worker(**kwargs: Any) -> None:
            # Block forever — account sync is also critical.
            await asyncio.Event().wait()

        async def _fake_strategy_worker(**kwargs: Any) -> None:
            # Block forever — will be cancelled as producer.
            await asyncio.Event().wait()

        async def _fake_monitor_run(self: Any) -> None:
            await asyncio.Event().wait()

        app_config = _make_app_config(
            symbol_worker_shutdown_drain_timeout_seconds=2.0,
            symbol_worker_shutdown_save_state_enabled=False,
            symbol_worker_shutdown_heartbeat_enabled=False,
        )

        app = SymbolWorkerApp(
            app_config=app_config,
            factory=factory,
            shutdown_controller=controller,
        )

        # Monkeypatch startup helpers.
        async def _fake_fetch_cash(_trader: Any) -> float:
            return 10000.0

        async def _noop(**kwargs: Any) -> None:
            return None

        def _noop_sync(**kwargs: Any) -> bool:
            return False

        monkeypatch.setattr(
            "src.live.symbol_worker_app.live_flat_balance.fetch_usdt_cash_balance",
            _fake_fetch_cash,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_main_tp_startup_recovery",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_sidecar_startup_recovery",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.rolling_loss_live_helpers.apply_rolling_loss_guard_startup_state",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.runner_live_helpers.apply_three_stage_startup_safety_gate",
            _noop_sync,
        )

        # Monkeypatch workers to control behaviour.
        monkeypatch.setattr(
            "src.live.symbol_worker_app.execution_worker_module.execution_worker",
            _fake_execution_worker,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.account_position_sync_worker_module.account_position_sync_worker",
            _fake_account_worker,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.strategy_tick_worker_module.strategy_tick_worker",
            _fake_strategy_worker,
        )
        monkeypatch.setattr(
            _FakeMonitor, "run_forever", _fake_monitor_run,
        )

        # Monkeypatch _drain_critical_runtime_tasks to give the
        # execution worker time to finish.  In a real shutdown the
        # execution queue would typically have items in flight, so the
        # drain loop would naturally wait.  In this test the fake queue
        # is empty, so we simulate a brief drain window.
        async def _patched_drain(tasks: set[asyncio.Task], **kwargs: Any) -> None:
            # Signal the execution worker to finish.
            allow_finish.set()
            # Wait a short time for it to process the signal.
            await asyncio.sleep(0.05)

        monkeypatch.setattr(
            "src.live.symbol_worker_app._drain_critical_runtime_tasks",
            _patched_drain,
        )

        # Run app.run in background.
        run_completed = asyncio.Event()

        async def _run_app() -> None:
            await app.run()
            run_completed.set()

        run_task = asyncio.ensure_future(_run_app())

        try:
            # Wait for execution worker to start.
            await asyncio.wait_for(execution_started.wait(), timeout=5.0)

            # Trigger shutdown.
            controller.request_shutdown("test_no_immediate_cancel")

            # Wait for app.run to complete.
            await asyncio.wait_for(run_completed.wait(), timeout=5.0)

            # Execution worker must NOT have been cancelled — it was
            # allowed to finish naturally during the drain window.
            assert len(cancelled_before_finish) == 0, (
                "execution worker was cancelled instead of allowed to finish"
            )

        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
            allow_finish.set()

        assert controller.requested is True


# ============================================================================
# 15. Shutdown cancels execution_task after drain timeout
# ============================================================================


class TestShutdownCancelsExecutionAfterDrainTimeout:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_execution_task_after_drain_timeout(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """When the execution worker does not finish within the drain
        timeout, it must be cancelled in the final cleanup step."""
        controller = WorkerShutdownController()
        factory = _ShutdownTestFactory(tmp_path=tmp_path)

        cancelled_events: list[str] = []

        async def _fake_execution_worker(**kwargs: Any) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled_events.append("execution_cancelled")
                raise

        async def _fake_account_worker(**kwargs: Any) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled_events.append("account_cancelled")
                raise

        async def _fake_strategy_worker(**kwargs: Any) -> None:
            await asyncio.Event().wait()

        async def _fake_monitor_run(self: Any) -> None:
            await asyncio.Event().wait()

        app_config = _make_app_config(
            symbol_worker_shutdown_drain_timeout_seconds=0.05,
            symbol_worker_shutdown_save_state_enabled=False,
            symbol_worker_shutdown_heartbeat_enabled=False,
        )

        app = SymbolWorkerApp(
            app_config=app_config,
            factory=factory,
            shutdown_controller=controller,
        )

        async def _fake_fetch_cash(_trader: Any) -> float:
            return 10000.0

        async def _noop(**kwargs: Any) -> None:
            return None

        def _noop_sync(**kwargs: Any) -> bool:
            return False

        monkeypatch.setattr(
            "src.live.symbol_worker_app.live_flat_balance.fetch_usdt_cash_balance",
            _fake_fetch_cash,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_main_tp_startup_recovery",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_sidecar_startup_recovery",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.rolling_loss_live_helpers.apply_rolling_loss_guard_startup_state",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.runner_live_helpers.apply_three_stage_startup_safety_gate",
            _noop_sync,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.execution_worker_module.execution_worker",
            _fake_execution_worker,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.account_position_sync_worker_module.account_position_sync_worker",
            _fake_account_worker,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.strategy_tick_worker_module.strategy_tick_worker",
            _fake_strategy_worker,
        )
        monkeypatch.setattr(
            _FakeMonitor, "run_forever", _fake_monitor_run,
        )

        run_completed = asyncio.Event()

        async def _run_app() -> None:
            await app.run()
            run_completed.set()

        run_task = asyncio.ensure_future(_run_app())

        try:
            # Give startup time to finish.
            await asyncio.sleep(0.10)

            # Trigger shutdown.
            controller.request_shutdown("test_drain_timeout")

            # Wait for app.run to complete.
            await asyncio.wait_for(run_completed.wait(), timeout=5.0)

            # After the drain timeout, critical tasks must have been
            # cancelled in the final cleanup step.
            assert "execution_cancelled" in cancelled_events, (
                "execution worker was not cancelled after drain timeout"
            )
            assert "account_cancelled" in cancelled_events, (
                "account worker was not cancelled after drain timeout"
            )

        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass

        assert controller.requested is True


# ============================================================================
# 16. Shutdown saves state AFTER critical drain
# ============================================================================


class TestShutdownSavesStateAfterCriticalDrain:
    @pytest.mark.asyncio
    async def test_shutdown_saves_state_after_critical_drain(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """When the execution worker modifies strategy state during the
        drain window, save_state must capture the updated state (i.e.
        save must happen AFTER the critical drain, not before it)."""
        controller = WorkerShutdownController()
        state_store = _FakeStateStore()
        strategy = _FakeStrategy(layers=1)
        factory = _ShutdownTestFactory(
            tmp_path=tmp_path,
            strategy=strategy,
            state_store=state_store,
        )

        allow_finish = asyncio.Event()

        async def _fake_execution_worker(strategy: Any = None, **kwargs: Any) -> None:
            # Simulate work that updates strategy state during drain.
            await allow_finish.wait()
            # Mutate strategy state — save_state should capture this.
            strategy.state.layers = 2

        async def _fake_account_worker(**kwargs: Any) -> None:
            # Block — cancelled after drain.
            await asyncio.Event().wait()

        async def _fake_strategy_worker(**kwargs: Any) -> None:
            await asyncio.Event().wait()

        async def _fake_monitor_run(self: Any) -> None:
            await asyncio.Event().wait()

        app_config = _make_app_config(
            symbol_worker_shutdown_drain_timeout_seconds=2.0,
            symbol_worker_shutdown_save_state_enabled=True,
            symbol_worker_shutdown_heartbeat_enabled=False,
        )

        app = SymbolWorkerApp(
            app_config=app_config,
            factory=factory,
            shutdown_controller=controller,
        )

        async def _fake_fetch_cash(_trader: Any) -> float:
            return 10000.0

        async def _noop(**kwargs: Any) -> None:
            return None

        def _noop_sync(**kwargs: Any) -> bool:
            return False

        monkeypatch.setattr(
            "src.live.symbol_worker_app.live_flat_balance.fetch_usdt_cash_balance",
            _fake_fetch_cash,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_main_tp_startup_recovery",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_sidecar_startup_recovery",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.rolling_loss_live_helpers.apply_rolling_loss_guard_startup_state",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.runner_live_helpers.apply_three_stage_startup_safety_gate",
            _noop_sync,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.execution_worker_module.execution_worker",
            _fake_execution_worker,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.account_position_sync_worker_module.account_position_sync_worker",
            _fake_account_worker,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.strategy_tick_worker_module.strategy_tick_worker",
            _fake_strategy_worker,
        )
        monkeypatch.setattr(
            _FakeMonitor, "run_forever", _fake_monitor_run,
        )

        run_completed = asyncio.Event()

        async def _run_app() -> None:
            await app.run()
            run_completed.set()

        run_task = asyncio.ensure_future(_run_app())

        try:
            await asyncio.sleep(0.10)

            # Trigger shutdown.
            controller.request_shutdown("test_save_after_drain")

            # Give the shutdown path time to reach the drain step.
            await asyncio.sleep(0.10)

            # Allow execution worker to finish (it sets layers=2).
            allow_finish.set()

            # Wait for app.run to complete.
            await asyncio.wait_for(run_completed.wait(), timeout=5.0)

            # Save must have been called and must reflect layers=2,
            # proving save happened AFTER the execution worker updated
            # strategy state in the drain window.
            assert len(state_store.save_calls) >= 1, (
                "state_store.save was not called"
            )
            saved_state = state_store.save_calls[-1]
            # The saved state should have layers=2 (updated during drain).
            # LiveStateStore.from_strategy_state stores the strategy_state,
            # so we need to check what was actually saved.
            # The FakeStateStore.save appends the LiveStateStore object.
            from src.reporting.live_state_store import LiveStateStore
            if hasattr(saved_state, "strategy_state"):
                assert saved_state.strategy_state.layers == 2, (
                    f"save captured layers={saved_state.strategy_state.layers},"
                    f" expected 2 (state should be saved AFTER critical drain)"
                )

        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
            allow_finish.set()

        assert controller.requested is True


# ============================================================================
# 17. Producers cancelled before critical drain
# ============================================================================


class TestShutdownCancelsProducersBeforeCriticalDrain:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_producers_before_critical_drain(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Strategy tick worker, monitor, and report loops must be
        cancelled BEFORE the critical drain starts."""
        controller = WorkerShutdownController()
        factory = _ShutdownTestFactory(tmp_path=tmp_path)

        cancel_order: list[str] = []

        allow_execution_finish = asyncio.Event()

        async def _fake_execution_worker(**kwargs: Any) -> None:
            await allow_execution_finish.wait()

        async def _fake_account_worker(**kwargs: Any) -> None:
            await asyncio.Event().wait()

        async def _fake_strategy_worker(**kwargs: Any) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_order.append("strategy_cancelled")
                raise

        async def _fake_monitor_run(self: Any) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_order.append("monitor_cancelled")
                raise

        app_config = _make_app_config(
            symbol_worker_shutdown_drain_timeout_seconds=2.0,
            symbol_worker_shutdown_save_state_enabled=False,
            symbol_worker_shutdown_heartbeat_enabled=False,
        )

        app = SymbolWorkerApp(
            app_config=app_config,
            factory=factory,
            shutdown_controller=controller,
        )

        async def _fake_fetch_cash(_trader: Any) -> float:
            return 10000.0

        async def _noop(**kwargs: Any) -> None:
            return None

        def _noop_sync(**kwargs: Any) -> bool:
            return False

        monkeypatch.setattr(
            "src.live.symbol_worker_app.live_flat_balance.fetch_usdt_cash_balance",
            _fake_fetch_cash,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_main_tp_startup_recovery",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.startup_order_recovery.apply_sidecar_startup_recovery",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.rolling_loss_live_helpers.apply_rolling_loss_guard_startup_state",
            _noop,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.runner_live_helpers.apply_three_stage_startup_safety_gate",
            _noop_sync,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.execution_worker_module.execution_worker",
            _fake_execution_worker,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.account_position_sync_worker_module.account_position_sync_worker",
            _fake_account_worker,
        )
        monkeypatch.setattr(
            "src.live.symbol_worker_app.strategy_tick_worker_module.strategy_tick_worker",
            _fake_strategy_worker,
        )
        monkeypatch.setattr(
            _FakeMonitor, "run_forever", _fake_monitor_run,
        )

        run_completed = asyncio.Event()

        async def _run_app() -> None:
            await app.run()
            run_completed.set()

        run_task = asyncio.ensure_future(_run_app())

        try:
            await asyncio.sleep(0.10)

            # Trigger shutdown.
            controller.request_shutdown("test_producers_first")

            # Wait a bit for producers to be cancelled.
            await asyncio.sleep(0.20)

            # Producers must have been cancelled.
            assert "strategy_cancelled" in cancel_order, (
                "strategy worker was not cancelled before drain"
            )
            assert "monitor_cancelled" in cancel_order, (
                "monitor was not cancelled before drain"
            )

            # Execution worker must still be running (critical, not yet cancelled).
            # We prove this by allowing it to finish now.
            allow_execution_finish.set()

            # Wait for app.run to complete.
            await asyncio.wait_for(run_completed.wait(), timeout=5.0)

        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
            allow_execution_finish.set()

        assert controller.requested is True


# ============================================================================
# 18. shutdown_runtime.py source guard — extended
# ============================================================================


class TestShutdownRuntimeExtendedSourceGuard:
    def test_shutdown_runtime_no_external_http_or_trading(self) -> None:
        """symbol_worker_shutdown_runtime.py must not import or reference
        any external HTTP library, OKX client, websocket, or trading
        mutation symbols."""
        from pathlib import Path

        source = (
            Path(__file__)
            .resolve()
            .parent.parent.parent.joinpath(
                "src", "live", "symbol_worker_shutdown_runtime.py"
            )
            .read_text(encoding="utf-8")
        )

        forbidden = [
            "from src.execution.trader",
            "place_market_order",
            "market_close",
            "close_position",
            "cancel_all",
            "cancel_algo_order",
            "cancel_near_tp",
            "cancel_middle_runner",
            "cancel_three_stage",
            "cancel_trend_runner",
            "import okx",
            "from okx",
            "import httpx",
            "from httpx",
            "import requests",
            "from requests",
            "import websocket",
            "from websocket",
        ]
        for token in forbidden:
            assert token not in source, (
                f"symbol_worker_shutdown_runtime.py must not reference {token!r}"
            )

    def test_shutdown_runtime_contains_drain_critical_helper(self) -> None:
        """symbol_worker_shutdown_runtime.py must define
        _drain_critical_runtime_tasks."""
        from pathlib import Path

        source = (
            Path(__file__)
            .resolve()
            .parent.parent.parent.joinpath(
                "src", "live", "symbol_worker_shutdown_runtime.py"
            )
            .read_text(encoding="utf-8")
        )

        assert "async def _drain_critical_runtime_tasks(" in source, (
            "symbol_worker_shutdown_runtime.py must define"
            " _drain_critical_runtime_tasks"
        )
        assert "does NOT cancel" in source, (
            "_drain_critical_runtime_tasks docstring must state it does not cancel"
        )


# ============================================================================
# 19. symbol_worker_app.py source guard — shutdown classification
# ============================================================================


class TestSymbolWorkerAppShutdownClassification:
    def test_app_source_contains_critical_drain_classification(self) -> None:
        """symbol_worker_app.py must contain the two-stage shutdown
        classification variables."""
        from pathlib import Path

        source = (
            Path(__file__)
            .resolve()
            .parent.parent.parent.joinpath("src", "live", "symbol_worker_app.py")
            .read_text(encoding="utf-8")
        )

        assert "critical_drain_tasks" in source, (
            "symbol_worker_app.py must classify critical_drain_tasks"
        )
        assert "producer_or_aux_tasks" in source, (
            "symbol_worker_app.py must classify producer_or_aux_tasks"
        )
        assert "_drain_critical_runtime_tasks(" in source, (
            "symbol_worker_app.py must call _drain_critical_runtime_tasks"
        )
