from __future__ import annotations

import asyncio
import datetime as dt
import os
import time
from dataclasses import dataclass, replace

from config.live_symbol_config_bootstrap import build_live_symbol_runtime_configs
from src.execution.paper_trader import PaperTrader
from src.execution.trader import Trader
from src.execution.trader_types import TraderInstrumentMetadata, TraderMarketSettings
from src.live import queue_helpers as live_queue_helpers
from src.live import runtime_types as live_runtime_types
from src.live import time_utils as live_time_utils
from src.live.account_sync import flat_balance as live_flat_balance
from src.live.live_app_config import LiveAppConfig
from src.live.outbox import (
    WORKER_DRAIN_COMPLETED,
    WORKER_DRAIN_STARTED,
    WORKER_DRAIN_TIMEOUT,
    WORKER_HEARTBEAT_WRITE_FAILED,
    WORKER_STARTED,
    WORKER_STARTUP_RECOVERY_COMPLETED,
    WORKER_STARTUP_RECOVERY_FAILED,
    WORKER_STOPPED,
    WORKER_STOPPING,
    JsonlOutbox,
    WorkerEventEmitter,
)
from src.live.portfolio_allocator_shadow import (
    PortfolioAllocatorShadowConfig,
    PortfolioAllocatorShadowRunner,
)
from src.live.portfolio_allocator_enforcer import (
    PortfolioAllocatorEnforceConfig,
    PortfolioAllocatorEnforcer,
)
from src.live.runtime_path_compat import handoff_legacy_runtime_files
from src.live.startup_recovery import basic_restore as startup_basic_restore
from src.live.startup_recovery import order_recovery as startup_order_recovery
from src.live.startup_recovery.portfolio_reconciliation import (
    StartupReconciliationResult,
    reconcile_startup_state,
)
from src.live.startup_recovery import trust_validation as startup_trust_validation
from src.live.symbol_worker_factory import SymbolWorkerFactory
from src.live.symbol_worker_shutdown_runtime import (
    _begin_symbol_worker_drain,
    _cancel_runtime_tasks,
    _drain_critical_runtime_tasks,
    _save_state_on_shutdown,
    _wait_for_shutdown,
)
from src.live.worker_shutdown import WorkerShutdownController
from src.live.workers import account_position_sync_worker as account_position_sync_worker_module
from src.live.workers import execution_worker as execution_worker_module
from src.live.workers import strategy_tick_worker as strategy_tick_worker_module
from src.monitors.boll_band_breakout_monitor import MarketTickEvent
from src.position_management import core_position_view as core_position_view_helpers
from src.position_management import runner_live_helpers
from src.position_management.core_position_view import build_core_position_view
from src.position_management.sidecar import runtime_state as sidecar_runtime_state
from src.position_management.sidecar.model import sidecar_open_contracts, sidecar_open_qty
from src.reporting import live_report_helpers as report_helpers
from src.reporting.journal_compactor import compact_after_weekly_summary
from src.reporting.live_state_store import LiveStateStore
from src.risk import rolling_loss_live as rolling_loss_live_helpers
from src.utils.log import get_logger

logger = get_logger(__name__)


def _emit_worker_event_best_effort(
    emitter: WorkerEventEmitter | None,
    event_type: str,
    *,
    severity: str = "INFO",
    payload: dict[str, object] | None = None,
) -> None:
    """Best-effort emit a worker lifecycle event.

    This helper never raises — the caller must not have its control
    flow altered by an emit failure.  It is a pure child-side outbox
    write and must never send email or interact with the supervisor.
    """
    if emitter is None:
        return
    try:
        emitter.emit(event_type, payload or {}, severity=severity)
    except Exception:
        logger.exception(
            "WORKER_EVENT_EMIT_FAILED | event_type=%s severity=%s",
            event_type,
            severity,
        )


def _runtime_config_env_for_worker_mode(
    *, mode: str, trader_symbol: str
) -> dict[str, str] | None:
    """Return an env override for ``build_live_symbol_runtime_configs``.

    Live mode returns ``None`` — the function reads ``os.environ`` as
    usual (preserving existing behaviour).

    Paper mode sets the symbols env var to the trader symbol.  For
    ETH-USDT-SWAP the legacy env path is forced (``RECLAIM_USE_SYMBOL_TOML=false``).
    For all other symbols the TOML path is allowed so that the worker
    can bootstrap runtime configs without hitting the legacy-path guard
    that rejects non-ETH symbols.
    """
    if mode != "paper":
        return None
    env = os.environ.copy()
    env["RECLAIM_" + "SYMBOLS"] = trader_symbol
    if trader_symbol == "ETH-USDT-SWAP":
        env["RECLAIM_USE_SYMBOL_TOML"] = "false"
    return env


def _decimal_equal(left: object, right: object) -> bool:
    """Return ``True`` when *left* and *right* represent the same Decimal value.

    ``"50"`` and ``Decimal("50.0")`` are considered equal.  Invalid values
    (non-numeric strings, None, bool) return ``False``.
    """
    from decimal import Decimal, InvalidOperation

    try:
        return Decimal(str(left)) == Decimal(str(right))
    except InvalidOperation:
        return False


def _assert_trader_matches_symbol_config(
    trader: Trader | PaperTrader,
    runtime_configs: "LiveSymbolRuntimeConfigs",
) -> None:
    """Fail fast when the running Trader disagrees with the TOML symbol config.

    On the **legacy .env path** ``runtime_configs.symbol_config`` is
    ``None`` and this function is a no-op.

    On the **TOML path** (default as of A08) we verify that the key
    market-structure parameters the Trader initialised from ``.env`` match
    what the TOML declares.  A mismatch here means the live session would
    run with inconsistent leverage / tdMode / posSide — a dangerous state
    that must be refused at startup.

    PaperTrader (G08) skips the live_trading requirement but still checks
    symbol / market-structure metadata consistency when TOML is loaded.
    """
    symbol_config = runtime_configs.symbol_config
    if symbol_config is None:
        # Legacy .env path — nothing to cross-check.
        return

    # PaperTrader has its own hardcoded market settings that do not
    # necessarily match the TOML (e.g. leverage=20 vs TOML leverage=15).
    # Skip the cross-check for paper mode.
    if isinstance(trader, PaperTrader):
        return

    errors: list[str] = []

    if trader.symbol != symbol_config.symbol.inst_id:
        errors.append(
            f"inst_id: trader={trader.symbol!r} vs TOML={symbol_config.symbol.inst_id!r}"
        )

    if trader.td_mode != symbol_config.market.td_mode:
        errors.append(
            f"td_mode: trader={trader.td_mode!r} vs TOML={symbol_config.market.td_mode!r}"
        )

    if trader.pos_side_mode != symbol_config.market.pos_side_mode:
        errors.append(
            f"pos_side_mode: trader={trader.pos_side_mode!r} vs TOML={symbol_config.market.pos_side_mode!r}"
        )

    if not _decimal_equal(trader.leverage, symbol_config.capital.leverage):
        errors.append(
            f"leverage: trader={trader.leverage!r} vs TOML={symbol_config.capital.leverage!r}"
        )

    if errors:
        raise RuntimeError(
            "TOML/env trader config mismatch: " + "; ".join(errors)
        )


def _build_pre_trader_runtime_configs_for_mode(
    *,
    mode: str,
    trader_symbol: str,
) -> "LiveSymbolRuntimeConfigs":
    """Build runtime configs **before** the Trader is constructed.

    For **live** mode this runs the TOML bootstrap path so the caller can
    extract instrument metadata / market settings from the loaded
    ``SymbolConfig`` and inject them into ``Trader``.

    For **paper** mode the path depends on the symbol:

    * ``ETH-USDT-SWAP`` — legacy env path (``symbol_config`` is ``None``).
    * Non-ETH symbols — TOML path (``symbol_config`` is loaded from TOML).
    """
    runtime_config_env = _runtime_config_env_for_worker_mode(
        mode=mode,
        trader_symbol=trader_symbol,
    )
    return build_live_symbol_runtime_configs(
        env=runtime_config_env,
        account_equity_usdt=None,
    )


def _build_live_trader_metadata_from_runtime_configs(
    runtime_configs: "LiveSymbolRuntimeConfigs",
) -> tuple[TraderInstrumentMetadata | None, TraderMarketSettings | None]:
    """Extract Trader metadata / market settings from a TOML-backed
    ``LiveSymbolRuntimeConfigs``.

    Returns ``(None, None)`` when the legacy ``.env`` path is active
    (``symbol_config`` is ``None``).  When a TOML ``SymbolConfig`` is
    available, metadata and market settings are built from that config for
    every supported symbol.
    """
    from src.live.symbol_trader_config import (
        build_trader_instrument_metadata,
        build_trader_market_settings,
    )

    symbol_config = runtime_configs.symbol_config
    if symbol_config is None:
        return None, None

    return (
        build_trader_instrument_metadata(symbol_config),
        build_trader_market_settings(symbol_config),
    )


def _assert_symbol_live_trading_enabled_for_worker_mode(
    *,
    mode: str,
    runtime_configs: "LiveSymbolRuntimeConfigs",
) -> None:
    """Enforce the TOML per-symbol live gate for live workers.

    Legacy ``.env`` runtime configs do not have a ``SymbolConfig`` and remain
    compatible.  Paper workers can load TOML without requiring live trading.
    """
    if mode != "live":
        return

    symbol_config = runtime_configs.symbol_config
    if symbol_config is None:
        return

    if symbol_config.symbol.live_trading is False:
        raise RuntimeError(
            "TOML symbol.live_trading is false for live worker: "
            f"symbol={symbol_config.symbol.inst_id} worker_mode=live"
        )


def _override_runtime_config_account_equity(
    runtime_configs: "LiveSymbolRuntimeConfigs",
    account_equity_usdt: float,
) -> "LiveSymbolRuntimeConfigs":
    """Return a copy of *runtime_configs* whose ``position_sizer`` has
    ``dry_run_equity_usdt`` set to *account_equity_usdt*.
    """
    return replace(
        runtime_configs,
        position_sizer=replace(
            runtime_configs.position_sizer,
            dry_run_equity_usdt=account_equity_usdt,
        ),
    )


async def _run_startup_portfolio_reconciliation_if_available(
    *,
    trader: Trader | PaperTrader,
    startup_position: object,
    saved_state: object | None,
    execution_state: live_runtime_types.ExecutionState,
    journal: object,
    portfolio_allocator_shadow_runner: PortfolioAllocatorShadowRunner | None,
    portfolio_allocator_enforcer: PortfolioAllocatorEnforcer | None,
) -> StartupReconciliationResult | None:
    ledger_owner = portfolio_allocator_enforcer or portfolio_allocator_shadow_runner
    if ledger_owner is None:
        return None

    try:
        ledger_snapshot = await asyncio.to_thread(ledger_owner.ledger.read_locked)
        reconciliation = reconcile_startup_state(
            inst_id=trader.symbol,
            position=startup_position,  # type: ignore[arg-type]
            saved_state=saved_state,  # type: ignore[arg-type]
            ledger_snapshot=ledger_snapshot,
        )

        if hasattr(journal, "append"):
            journal.append(
                "STARTUP_PORTFOLIO_RECONCILIATION",
                {
                    "symbol": reconciliation.inst_id,
                    "severity": reconciliation.severity,
                    "action": reconciliation.action,
                    "okx_has_position": reconciliation.okx_has_position,
                    "saved_has_position": reconciliation.saved_has_position,
                    "ledger_is_active": reconciliation.ledger_is_active,
                    "okx_side": reconciliation.okx_side,
                    "saved_side": reconciliation.saved_side,
                    "ledger_side": reconciliation.ledger_side,
                    "saved_layers": reconciliation.saved_layers,
                    "ledger_used_layers": reconciliation.ledger_used_layers,
                    "ledger_plan_exists": reconciliation.ledger_plan_exists,
                    "issues": [
                        {
                            "code": issue.code,
                            "severity": issue.severity,
                            "message": issue.message,
                        }
                        for issue in reconciliation.issues
                    ],
                },
                position_id=getattr(saved_state, "position_id", None),
            )

        if reconciliation.severity == "CRITICAL":
            logger.critical(
                "STARTUP_PORTFOLIO_RECONCILIATION_CRITICAL | symbol=%s action=%s issues=%s",
                reconciliation.inst_id,
                reconciliation.action,
                [issue.code for issue in reconciliation.issues],
            )
        elif reconciliation.severity == "WARN":
            logger.warning(
                "STARTUP_PORTFOLIO_RECONCILIATION_WARN | symbol=%s action=%s issues=%s",
                reconciliation.inst_id,
                reconciliation.action,
                [issue.code for issue in reconciliation.issues],
            )
        else:
            logger.info(
                "STARTUP_PORTFOLIO_RECONCILIATION_OK | symbol=%s",
                reconciliation.inst_id,
            )

        if reconciliation.should_halt_new_risk:
            execution_state.trading_halted = True
            execution_state.halt_reason = (
                f"startup_portfolio_reconciliation_{reconciliation.severity.lower()}"
            )
            execution_state.halt_until_ts_ms = None

        return reconciliation
    except Exception:
        execution_state.trading_halted = True
        execution_state.halt_reason = "startup_portfolio_reconciliation_error"
        execution_state.halt_until_ts_ms = None
        logger.exception(
            "STARTUP_PORTFOLIO_RECONCILIATION_ERROR | symbol=%s",
            trader.symbol,
        )
        return None


@dataclass(frozen=True)
class SymbolWorkerApp:
    app_config: LiveAppConfig
    factory: SymbolWorkerFactory
    shutdown_controller: WorkerShutdownController | None = None

    @classmethod
    def from_env(
        cls,
        *,
        factory: SymbolWorkerFactory | None = None,
        shutdown_controller: WorkerShutdownController | None = None,
    ) -> "SymbolWorkerApp":
        return cls(
            app_config=LiveAppConfig.from_env(),
            factory=factory or SymbolWorkerFactory(),
            shutdown_controller=shutdown_controller,
        )

    async def run(self) -> None:
        worker_event_emitter: WorkerEventEmitter | None = None
        startup_phase = "trader_initialize"

        email_sender = self.factory.create_email_sender()
        mode = os.getenv("RECLAIM_WORKER_MODE", "live").strip().lower()

        # ── G09c: build runtime configs BEFORE trader for live mode ─────
        if mode == "live":
            pre_runtime_configs = _build_pre_trader_runtime_configs_for_mode(
                mode=mode,
                trader_symbol=os.getenv("OKX_INST_ID", "ETH-USDT-SWAP").strip(),
            )
            _assert_symbol_live_trading_enabled_for_worker_mode(
                mode=mode,
                runtime_configs=pre_runtime_configs,
            )
            metadata, market_settings = _build_live_trader_metadata_from_runtime_configs(
                pre_runtime_configs,
            )
            trader = self.factory.create_trader(
                trader_mode=mode,
                instrument_metadata=metadata,
                market_settings=market_settings,
            )
        else:
            trader = self.factory.create_trader(trader_mode=mode)

        await trader.start()
        try:
            await trader.initialize()

            if mode == "live":
                runtime_configs = _override_runtime_config_account_equity(
                    pre_runtime_configs, trader.account_equity_usdt,
                )
            else:
                runtime_config_env = _runtime_config_env_for_worker_mode(
                    mode=mode,
                    trader_symbol=trader.symbol,
                )
                runtime_configs = build_live_symbol_runtime_configs(
                    env=runtime_config_env,
                    account_equity_usdt=trader.account_equity_usdt,
                )

            _assert_trader_matches_symbol_config(trader, runtime_configs)
            runtime_paths = self.factory.create_runtime_paths(
                runtime_dir=runtime_configs.env_runtime.runtime_dir,
                inst_id=trader.symbol,
            )
            # -- E05h: create worker event emitter and outbox ------------------
            worker_event_outbox = JsonlOutbox(runtime_paths.worker_event_outbox_file)
            worker_event_emitter = WorkerEventEmitter(
                symbol=trader.symbol,
                outbox=worker_event_outbox,
            )
            _emit_worker_event_best_effort(
                worker_event_emitter,
                WORKER_STARTED,
                severity="INFO",
                payload={"phase": "runtime_paths_ready"},
            )
            startup_phase = "runtime_paths_ready"
            # ------------------------------------------------------------------
            handoff_result = handoff_legacy_runtime_files(
                runtime_paths=runtime_paths,
                inst_id=trader.symbol,
            )
            for item in handoff_result.items:
                logger.info(
                    "RUNTIME_PATH_HANDOFF | symbol=%s label=%s action=%s reason=%s legacy_path=%s symbol_path=%s",
                    handoff_result.inst_id,
                    item.label,
                    item.action,
                    item.reason,
                    item.legacy_path,
                    item.symbol_path,
                )
            heartbeat_writer = self.factory.create_heartbeat_writer(
                runtime_paths=runtime_paths,
                heartbeat_config=self.app_config.heartbeat,
            )
            persistence = self.factory.create_persistence(
                runtime_paths=runtime_paths,
                email_sender=email_sender,
            )
            journal = persistence.journal
            state_store = persistence.state_store
            rolling_loss_guard = persistence.rolling_loss_guard
            reporter = persistence.reporter
            monitor_config = runtime_configs.monitor
            cvd_config = runtime_configs.cvd
            strategy_config = runtime_configs.strategy
            position_sizer_config = runtime_configs.position_sizer
            strategy_objects = self.factory.create_strategy_objects(
                strategy_config=strategy_config,
                position_sizer_config=position_sizer_config,
            )
            sizer = strategy_objects.sizer
            strategy = strategy_objects.strategy

            # ── G05: portfolio allocator shadow runner ──────────────────
            shadow_config = PortfolioAllocatorShadowConfig.from_env(
                runtime_dir=runtime_configs.env_runtime.runtime_dir,
            )
            portfolio_allocator_shadow_runner = (
                PortfolioAllocatorShadowRunner.from_config(shadow_config)
                if shadow_config.enabled
                else None
            )

            # ── G06a: portfolio allocator enforcer ──────────────────────
            enforce_config = PortfolioAllocatorEnforceConfig.from_env(
                runtime_dir=runtime_configs.env_runtime.runtime_dir,
            )
            portfolio_allocator_enforcer = (
                PortfolioAllocatorEnforcer.from_config(enforce_config)
                if enforce_config.enabled
                else None
            )

            startup_position = await trader.fetch_position_snapshot()
            startup_cash = await live_flat_balance.fetch_usdt_cash_balance(trader)
            rolling_loss_guard.load_or_initialize(live_time_utils.utc_ms(), trader.account_equity_usdt)
            journal.record_cash_baseline(
                source="startup",
                cash=startup_cash,
                equity=trader.account_equity_usdt,
                note="Live runner startup cash baseline.",
            )
        except Exception as exc:
            _emit_worker_event_best_effort(
                worker_event_emitter,
                WORKER_STARTUP_RECOVERY_FAILED,
                severity="ERROR",
                payload={
                    "phase": startup_phase,
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:300],
                },
            )
            await trader.close()
            raise
        current_position_id: str | None = None
        cash_before_position: float | None = None

        startup_phase = "startup_recovery"
        try:
            saved_state = state_store.load()
            trusted_saved_state = startup_trust_validation.trusted_startup_saved_state(saved_state, startup_position)
            if startup_position.has_position:
                if trusted_saved_state is not None:
                    startup_basic_restore.restore_strategy_from_saved_state(strategy, trusted_saved_state)
                    current_position_id = trusted_saved_state.position_id
                    cash_before_position = trusted_saved_state.cash_before_position
                else:
                    startup_basic_restore.restore_strategy_from_position(strategy, startup_position, live_time_utils.utc_ms())
                    current_position_id = journal.new_position_id(trader.symbol, startup_position.side or "UNKNOWN")
                    cash_before_position = startup_cash
                    journal.record_startup_recovery(
                        position_id=current_position_id,
                        symbol=trader.symbol,
                        side=startup_position.side or "UNKNOWN",
                        contracts=str(startup_position.contracts),
                        eth_qty=startup_position.eth_qty,
                        avg_entry=startup_position.avg_entry_price,
                        cash=startup_cash,
                        equity=trader.account_equity_usdt,
                    )
                strategy.state.startup_force_tp_reconcile = True
                logger.warning(
                    "STARTUP_FORCE_TP_RECONCILE_ARMED | position_id=%s side=%s layers=%s tp_plan=%s last_tp_update_candle_ts_ms=%s trusted_saved_state=%s",
                    current_position_id,
                    strategy.state.side,
                    strategy.state.layers,
                    getattr(strategy.state, "tp_plan", "SINGLE"),
                    getattr(strategy.state, "last_tp_update_candle_ts_ms", 0),
                    trusted_saved_state is not None,
                )
                state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol,
                                                                    strategy_state=strategy.state,
                                                                    cash_before_position=cash_before_position))
            else:
                state_store.clear()

            cvd = self.factory.create_cvd_tracker(cvd_config)
            state_lock = asyncio.Lock()
            account_snapshot = live_runtime_types.AccountSnapshot(
                position=startup_position,
                cash=startup_cash,
                equity=trader.account_equity_usdt,
                updated_monotonic=time.monotonic(),
                updated_ts_ms=live_time_utils.utc_ms(),
                version=1,
            )
            execution_state = live_runtime_types.ExecutionState(
                current_position_id=current_position_id,
                cash_before_position=cash_before_position,
                trading_halted=False,
            )
            await startup_order_recovery.apply_main_tp_startup_recovery(
                execution_state=execution_state,
                saved_state=trusted_saved_state,
                startup_position=startup_position,
                trader=trader,
                journal=journal,
            )
            await startup_order_recovery.apply_sidecar_startup_recovery(
                strategy=strategy,
                execution_state=execution_state,
                saved_state=trusted_saved_state,
                startup_position=startup_position,
                trader=trader,
                journal=journal,
                state_store=state_store,
            )
            sidecar_runtime_state.refresh_sidecar_state_totals(strategy.state, int(position_sizer_config.sidecar_max_legs))
            startup_core_position = build_core_position_view(
                startup_position,
                sidecar_open_qty(strategy.state.sidecar_legs),
                sidecar_open_contracts(strategy.state.sidecar_legs),
            )
            core_position_view_helpers.apply_core_position_view_to_state(strategy.state, startup_core_position)
            account_snapshot.position = startup_core_position
            if startup_position.has_position:
                state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol,
                                                                    strategy_state=strategy.state,
                                                                    cash_before_position=cash_before_position))
            await rolling_loss_live_helpers.apply_rolling_loss_guard_startup_state(
                rolling_loss_guard=rolling_loss_guard,
                execution_state=execution_state,
                has_position=startup_position.has_position,
                equity=trader.account_equity_usdt,
                now_ms=live_time_utils.utc_ms(),
                journal=journal,
                email_sender=email_sender,
                worker_event_emitter=worker_event_emitter,
            )
            runner_live_helpers.apply_three_stage_startup_safety_gate(
                strategy=strategy,
                execution_state=execution_state,
                saved_state=trusted_saved_state,
                startup_position=startup_position,
                journal=journal,
                state_store=state_store,
                trader_symbol=trader.symbol,
            )
            await _run_startup_portfolio_reconciliation_if_available(
                trader=trader,
                startup_position=startup_position,
                saved_state=saved_state,
                execution_state=execution_state,
                journal=journal,
                portfolio_allocator_shadow_runner=portfolio_allocator_shadow_runner,
                portfolio_allocator_enforcer=portfolio_allocator_enforcer,
            )
        except Exception as exc:
            _emit_worker_event_best_effort(
                worker_event_emitter,
                WORKER_STARTUP_RECOVERY_FAILED,
                severity="ERROR",
                payload={
                    "phase": startup_phase,
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:300],
                },
            )
            await trader.close()
            raise

        # -- E05h: startup recovery completed ----------------------------------
        _emit_worker_event_best_effort(
            worker_event_emitter,
            WORKER_STARTUP_RECOVERY_COMPLETED,
            severity="INFO",
            payload={
                "has_startup_position": startup_position.has_position,
                "trusted_saved_state": trusted_saved_state is not None,
                "position_id": current_position_id,
            },
        )
        queues = self.factory.create_queues(self.app_config)
        strategy_tick_queue = queues.strategy_tick_queue
        execution_queue = queues.execution_queue
        position_sync_seconds = self.app_config.position_sync_seconds
        account_sync_seconds = self.app_config.account_sync_seconds
        cash_log_min_delta_usdt = self.app_config.cash_log_min_delta_usdt
        market_tick_heartbeat_seconds = self.app_config.market_tick_heartbeat_seconds
        account_snapshot_stale_warn_seconds = self.app_config.account_snapshot_stale_warn_seconds
        strategy_lag_warn_seconds = self.app_config.strategy_tick_lag_warn_seconds
        execution_backlog_log_seconds = self.app_config.execution_queue_backlog_log_seconds

        async def daily_report_loop() -> None:
            daily_report_config = self.app_config.daily_report
            raw_time = daily_report_config.raw_time
            hour = daily_report_config.hour
            minute = daily_report_config.minute
            logger.info("Daily trade report loop started | DAILY_REPORT_TIME=%s", raw_time)
            while True:
                target = live_time_utils.next_daily_report_time(hour, minute)
                sleep_seconds = max((target - dt.datetime.now().astimezone()).total_seconds(), 1)
                await asyncio.sleep(sleep_seconds)
                try:
                    context = report_helpers.build_report_context(
                        account_snapshot=account_snapshot,
                        execution_state=execution_state,
                    )
                    ok = await reporter.send_last_24h_report(context)
                    if ok:
                        logger.info("Daily trade report sent successfully")
                    else:
                        logger.error("Daily trade report failed")
                except Exception:
                    logger.exception("Daily trade report loop failed")

        async def weekly_summary_loop() -> None:
            weekly_config = self.app_config.weekly_summary
            enabled = weekly_config.enabled
            if not enabled:
                logger.info("Weekly overall summary loop disabled")
                return

            raw_time = weekly_config.raw_time
            compact_after_success = weekly_config.compact_after_success
            weekday = weekly_config.weekday
            hour = weekly_config.hour
            minute = weekly_config.minute

            logger.info(
                "Weekly compaction config | WEEKLY_COMPACT_AFTER_SUCCESS=%s risk=enable_only_after_summary_merge_verified",
                compact_after_success,
            )
            logger.info(
                "Weekly overall summary loop started | WEEKLY_SUMMARY_WEEKDAY=%s WEEKLY_SUMMARY_TIME=%s",
                weekday,
                raw_time,
            )

            while True:
                target = live_time_utils.next_weekly_summary_time(hour, minute, weekday)
                sleep_seconds = max((target - dt.datetime.now().astimezone()).total_seconds(), 1)
                await asyncio.sleep(sleep_seconds)
                try:
                    context = report_helpers.build_report_context(
                        account_snapshot=account_snapshot,
                        execution_state=execution_state,
                    )
                    ok = await reporter.send_overall_summary_report(context)
                    if ok:
                        logger.info("Weekly overall summary report sent successfully")
                        if compact_after_success:
                            async with state_lock:
                                compact_position_id = execution_state.current_position_id
                            result = await asyncio.to_thread(
                                compact_after_weekly_summary,
                                journal,
                                target,
                                compact_position_id,
                            )
                            if result.archived_event_count > 0:
                                logger.warning(
                                    "JOURNAL_COMPACTED | archived_event_count=%s retained_event_count=%s archive_path=%s",
                                    result.archived_event_count,
                                    result.retained_event_count,
                                    result.archive_path,
                                )
                    else:
                        logger.error("Weekly overall summary report failed")
                except Exception:
                    logger.exception("Weekly overall summary report loop failed")

        async def on_market_tick(event: MarketTickEvent) -> None:
            await live_queue_helpers.enqueue_strategy_tick(event, strategy_tick_queue, state_lock, execution_state)

        monitor = self.factory.create_monitor(
            config=monitor_config,
            tick_handlers=[on_market_tick],
        )

        # ── D06b: graceful shutdown controller ─────────────────────────
        shutdown_controller = self.shutdown_controller or WorkerShutdownController()

        try:
            # ── D06b: named task references for shutdown classification ──
            # Each runtime task is kept as a named variable so the
            # shutdown path can distinguish critical tasks (account sync,
            # execution worker) from producer / auxiliary tasks (strategy,
            # monitor, report loops, heartbeat loop).
            account_task = asyncio.ensure_future(
                account_position_sync_worker_module.account_position_sync_worker(
                    state_lock=state_lock,
                    account_snapshot=account_snapshot,
                    execution_state=execution_state,
                    trader=trader,
                    sizer=sizer,
                    strategy=strategy,
                    journal=journal,
                    state_store=state_store,
                    position_sync_seconds=position_sync_seconds,
                    account_sync_seconds=account_sync_seconds,
                    cash_log_min_delta_usdt=cash_log_min_delta_usdt,
                    rolling_loss_guard=rolling_loss_guard,
                    email_sender=email_sender,
                    worker_event_emitter=worker_event_emitter,
                )
            )
            strategy_task = asyncio.ensure_future(
                strategy_tick_worker_module.strategy_tick_worker(
                    strategy_tick_queue=strategy_tick_queue,
                    execution_queue=execution_queue,
                    state_lock=state_lock,
                    account_snapshot=account_snapshot,
                    execution_state=execution_state,
                    cvd=cvd,
                    strategy=strategy,
                    heartbeat_seconds=market_tick_heartbeat_seconds,
                    account_stale_warn_seconds=account_snapshot_stale_warn_seconds,
                    strategy_lag_warn_seconds=strategy_lag_warn_seconds,
                )
            )
            execution_task = asyncio.ensure_future(
                execution_worker_module.execution_worker(
                    execution_queue=execution_queue,
                    state_lock=state_lock,
                    execution_state=execution_state,
                    account_snapshot=account_snapshot,
                    trader=trader,
                    strategy=strategy,
                    journal=journal,
                    state_store=state_store,
                    email_sender=email_sender,
                    backlog_log_seconds=execution_backlog_log_seconds,
                    sidecar_skip_first_layer=sizer.config.sidecar_skip_first_layer,
                    portfolio_allocator_shadow_runner=portfolio_allocator_shadow_runner,
                    portfolio_allocator_enforcer=portfolio_allocator_enforcer,
                )
            )
            daily_report_task = asyncio.ensure_future(daily_report_loop())

            weekly_summary_task: asyncio.Task | None = None
            if self.app_config.weekly_summary.enabled:
                weekly_summary_task = asyncio.ensure_future(weekly_summary_loop())
            else:
                logger.info("Weekly overall summary loop disabled")

            # Only add heartbeat to the FIRST_COMPLETED wait when it is
            # enabled — a disabled heartbeat returns immediately, which
            # would falsely trigger the shutdown path.
            heartbeat_task: asyncio.Task | None = None
            if heartbeat_writer.config.enabled:
                heartbeat_task = asyncio.ensure_future(
                    heartbeat_writer.run_until_cancelled()
                )

            monitor_task = asyncio.ensure_future(monitor.run_forever())

            # Build the flat task list for asyncio.wait preserving the
            # C05 gather order:
            # account → strategy → execution → daily_report → weekly_summary
            # → heartbeat (if enabled) → monitor
            tasks: list[asyncio.Task] = [
                account_task,
                strategy_task,
                execution_task,
                daily_report_task,
            ]
            if weekly_summary_task is not None:
                tasks.append(weekly_summary_task)
            if heartbeat_task is not None:
                tasks.append(heartbeat_task)
            tasks.append(monitor_task)

            # ── D06b: task classification for two-stage shutdown ─────────
            # Critical tasks may be in the middle of an OKX request +
            # journal + state_store save.  They get a drain window before
            # cancellation.
            critical_drain_tasks: set[asyncio.Task] = {
                account_task,
                execution_task,
            }
            # Producer / auxiliary tasks create new work or observe the
            # market.  They are cancelled first so the execution queue
            # drains naturally.
            producer_or_aux_tasks: set[asyncio.Task] = (
                set(tasks) - critical_drain_tasks
            )

            shutdown_task = asyncio.ensure_future(
                _wait_for_shutdown(shutdown_controller)
            )

            done, pending = await asyncio.wait(
                [*tasks, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if shutdown_task in done:
                # ── Graceful shutdown path (D06b two-stage) ─────────────
                # Stage 1: drain critical tasks, then save state, then
                # cancel remaining critical tasks.

                # -- E05h: WORKER_STOPPING ---------------------------------
                _emit_worker_event_best_effort(
                    worker_event_emitter,
                    WORKER_STOPPING,
                    severity="INFO",
                    payload={"reason": shutdown_controller.reason or "unknown"},
                )

                # 1. Begin drain: mark trading_halted, no new positions
                await _begin_symbol_worker_drain(
                    execution_state=execution_state,
                    state_lock=state_lock,
                    reason=shutdown_controller.reason or "unknown",
                )

                # -- E05h: WORKER_DRAIN_STARTED ---------------------------
                _emit_worker_event_best_effort(
                    worker_event_emitter,
                    WORKER_DRAIN_STARTED,
                    severity="INFO",
                    payload={
                        "reason": shutdown_controller.reason or "unknown",
                        "critical_task_count": len(
                            critical_drain_tasks & pending
                        ),
                    },
                )

                # 2. Write stopping heartbeat best-effort (early signal)
                if (
                    self.app_config.symbol_worker_shutdown_heartbeat_enabled
                    and heartbeat_writer.config.enabled
                ):
                    try:
                        heartbeat_writer.write_status_once("stopping")
                    except Exception as exc:
                        logger.exception(
                            "SYMBOL_WORKER_HEARTBEAT_WRITE_FAILED | status=stopping"
                        )
                        _emit_worker_event_best_effort(
                            worker_event_emitter,
                            WORKER_HEARTBEAT_WRITE_FAILED,
                            severity="ERROR",
                            payload={
                                "status": "stopping",
                                "error_type": type(exc).__name__,
                                "reason": str(exc)[:300],
                            },
                        )

                # 3. Cancel producer / auxiliary tasks first
                #    (strategy tick, monitor, report loops, heartbeat loop).
                #    These produce new work; cancelling them early means the
                #    execution queue will drain naturally.
                await _cancel_runtime_tasks(
                    producer_or_aux_tasks & pending,
                    timeout=min(
                        2.0,
                        self.app_config.symbol_worker_shutdown_drain_timeout_seconds,
                    ),
                )

                # 4. Drain critical tasks WITHOUT immediate cancellation.
                #    Give the execution worker and account sync a window to
                #    finish in-flight OKX requests, journal writes, and
                #    state_store saves.
                await _drain_critical_runtime_tasks(
                    critical_drain_tasks & pending,
                    execution_queue=execution_queue,
                    timeout=self.app_config.symbol_worker_shutdown_drain_timeout_seconds,
                )

                # -- E05h: DRAIN_COMPLETED or DRAIN_TIMEOUT ----------------
                execution_queue_remaining = execution_queue.qsize()
                remaining_critical = sum(
                    1 for t in critical_drain_tasks if not t.done()
                )
                drain_timeout = self.app_config.symbol_worker_shutdown_drain_timeout_seconds
                if execution_queue_remaining > 0:
                    _emit_worker_event_best_effort(
                        worker_event_emitter,
                        WORKER_DRAIN_TIMEOUT,
                        severity="ERROR",
                        payload={
                            "reason": shutdown_controller.reason or "unknown",
                            "remaining_critical_tasks": remaining_critical,
                            "execution_queue_size": execution_queue_remaining,
                            "timeout_seconds": drain_timeout,
                        },
                    )
                else:
                    _emit_worker_event_best_effort(
                        worker_event_emitter,
                        WORKER_DRAIN_COMPLETED,
                        severity="INFO",
                        payload={
                            "reason": shutdown_controller.reason or "unknown",
                            "remaining_critical_tasks": remaining_critical,
                            "execution_queue_size": 0,
                            "timeout_seconds": drain_timeout,
                        },
                    )

                # 5. Save strategy state AFTER critical drain.
                #    Captures the most recent state from any work that
                #    completed during the drain window.
                if self.app_config.symbol_worker_shutdown_save_state_enabled:
                    await _save_state_on_shutdown(
                        execution_state=execution_state,
                        strategy=strategy,
                        trader_symbol=trader.symbol,
                        state_store=state_store,
                    )

                # 6. Cancel any critical tasks still running after drain
                #    (best-effort cleanup with a short timeout).
                await _cancel_runtime_tasks(
                    {t for t in critical_drain_tasks if not t.done()},
                    timeout=2.0,
                )

                # 7. Write stopped heartbeat best-effort (final signal)
                if (
                    self.app_config.symbol_worker_shutdown_heartbeat_enabled
                    and heartbeat_writer.config.enabled
                ):
                    try:
                        heartbeat_writer.write_status_once("stopped")
                    except Exception as exc:
                        logger.exception(
                            "SYMBOL_WORKER_HEARTBEAT_WRITE_FAILED | status=stopped"
                        )
                        _emit_worker_event_best_effort(
                            worker_event_emitter,
                            WORKER_HEARTBEAT_WRITE_FAILED,
                            severity="ERROR",
                            payload={
                                "status": "stopped",
                                "error_type": type(exc).__name__,
                                "reason": str(exc)[:300],
                            },
                        )

                # 8. Complete
                logger.warning("SYMBOL_WORKER_SHUTDOWN_COMPLETE")
            else:
                # ── Unexpected task exit (existing behaviour) ──────────
                shutdown_task.cancel()
                for task in pending - {shutdown_task}:
                    if not task.done():
                        task.cancel()
                # Wait for cancellation to propagate
                await asyncio.gather(
                    shutdown_task, *(pending - {shutdown_task}),
                    return_exceptions=True,
                )
                # Propagate the first exception from the unexpectedly
                # completed task. If a runtime task exited cleanly
                # (no exception), raise a RuntimeError so that a
                # silent return from a worker doesn't go unnoticed.
                for task in done:
                    if task is shutdown_task:
                        continue
                    exc = task.exception()
                    if exc is not None:
                        raise exc
                    raise RuntimeError(
                        "Symbol worker runtime task exited unexpectedly without exception"
                    )
        finally:
            # -- E05h: best-effort WORKER_STOPPED before close ------------
            _emit_worker_event_best_effort(
                worker_event_emitter,
                WORKER_STOPPED,
                severity="INFO",
                payload={"phase": "finally"},
            )
            await trader.close()
