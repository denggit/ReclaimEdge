from __future__ import annotations

import asyncio
import datetime as dt
import time
from dataclasses import dataclass

from config.live_symbol_config_bootstrap import build_live_symbol_runtime_configs
from src.execution.trader import Trader
from src.live import queue_helpers as live_queue_helpers
from src.live import runtime_types as live_runtime_types
from src.live import time_utils as live_time_utils
from src.live.account_sync import flat_balance as live_flat_balance
from src.live.live_app_config import LiveAppConfig
from src.live.runtime_path_compat import handoff_legacy_runtime_files
from src.live.startup_recovery import basic_restore as startup_basic_restore
from src.live.startup_recovery import order_recovery as startup_order_recovery
from src.live.startup_recovery import trust_validation as startup_trust_validation
from src.live.symbol_worker_factory import SymbolWorkerFactory
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


def _assert_trader_matches_symbol_config(
    trader: Trader,
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
    """
    symbol_config = runtime_configs.symbol_config
    if symbol_config is None:
        # Legacy .env path — nothing to cross-check.
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

    if str(trader.leverage) != str(symbol_config.capital.leverage):
        errors.append(
            f"leverage: trader={trader.leverage!r} vs TOML={symbol_config.capital.leverage!r}"
        )

    if errors:
        raise RuntimeError(
            "TOML/env trader config mismatch: " + "; ".join(errors)
        )


@dataclass(frozen=True)
class SymbolWorkerApp:
    app_config: LiveAppConfig
    factory: SymbolWorkerFactory

    @classmethod
    def from_env(cls, *, factory: SymbolWorkerFactory | None = None) -> "SymbolWorkerApp":
        return cls(
            app_config=LiveAppConfig.from_env(),
            factory=factory or SymbolWorkerFactory(),
        )

    async def run(self) -> None:
        email_sender = self.factory.create_email_sender()
        trader = self.factory.create_trader()
        await trader.start()
        try:
            await trader.initialize()
            runtime_configs = build_live_symbol_runtime_configs(
                account_equity_usdt=trader.account_equity_usdt,
            )
            _assert_trader_matches_symbol_config(trader, runtime_configs)
            runtime_paths = self.factory.create_runtime_paths(
                runtime_dir=runtime_configs.env_runtime.runtime_dir,
                inst_id=trader.symbol,
            )
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
            startup_position = await trader.fetch_position_snapshot()
            startup_cash = await live_flat_balance.fetch_usdt_cash_balance(trader)
            rolling_loss_guard.load_or_initialize(live_time_utils.utc_ms(), trader.account_equity_usdt)
            journal.record_cash_baseline(
                source="startup",
                cash=startup_cash,
                equity=trader.account_equity_usdt,
                note="Live runner startup cash baseline.",
            )
        except Exception:
            await trader.close()
            raise
        current_position_id: str | None = None
        cash_before_position: float | None = None

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
        try:
            await asyncio.gather(
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
                ),
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
                ),
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
                ),
                daily_report_loop(),
                weekly_summary_loop(),
                monitor.run_forever(),
            )
        finally:
            await trader.close()
