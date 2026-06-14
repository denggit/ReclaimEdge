from __future__ import annotations

import asyncio
import datetime as dt
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from src.execution.trader import Trader  # noqa: E402
from src.execution.live_trader_factory import create_live_trader  # noqa: E402
from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig  # noqa: E402
from src.live import config_helpers as live_config_helpers  # noqa: E402
from src.live import queue_helpers as live_queue_helpers  # noqa: E402
from src.live import runtime_types as live_runtime_types  # noqa: E402
from src.live import time_utils as live_time_utils  # noqa: E402
from src.live.workers import account_position_sync_worker as account_position_sync_worker_module  # noqa: E402
from src.live.workers import execution_worker as execution_worker_module  # noqa: E402
from src.live.workers import strategy_tick_worker as strategy_tick_worker_module  # noqa: E402
from src.live.startup_recovery import basic_restore as startup_basic_restore  # noqa: E402
from src.live.startup_recovery import order_recovery as startup_order_recovery  # noqa: E402
from src.live.startup_recovery import trust_validation as startup_trust_validation  # noqa: E402
from src.live.account_sync import flat_balance as live_flat_balance  # noqa: E402
from src.position_management import core_position_view as core_position_view_helpers  # noqa: E402
from src.position_management import runner_live_helpers  # noqa: E402
from src.monitors.boll_band_breakout_monitor import (  # noqa: E402
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    MarketTickEvent,
)

from src.position_management.sidecar import runtime_state as sidecar_runtime_state  # noqa: E402
from src.position_management.sidecar.model import (  # noqa: E402
    sidecar_open_contracts,
    sidecar_open_qty,
)

from src.position_management.sidecar.reconciler import build_core_position_view  # noqa: E402
from src.reporting import live_report_helpers as report_helpers  # noqa: E402
from src.reporting.daily_trade_reporter import DailyTradeReporter  # noqa: E402
from src.reporting.journal_compactor import compact_after_weekly_summary  # noqa: E402
from src.reporting.live_state_store import LiveStateStore  # noqa: E402
from src.reporting.trade_journal import LiveTradeJournal  # noqa: E402
from src.risk.simple_position_sizer import (  # noqa: E402
    SimplePositionSizer,
    SimplePositionSizerConfig,
)
from src.risk.rolling_loss_guard import RollingLossGuard  # noqa: E402
from src.risk import rolling_loss_live as rolling_loss_live_helpers  # noqa: E402
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig  # noqa: E402
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy  # noqa: E402
from src.utils.email_sender import EmailSender  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

logger = get_logger(__name__)


async def main() -> None:
    load_dotenv()

    from src.live.live_runtime_selector import LiveRuntimeKind, select_live_runtime

    selection = select_live_runtime(os.environ)

    if selection.kind == LiveRuntimeKind.BINANCE_SIGNAL_ONLY:
        from src.live.binance_signal_only_runtime import run_binance_signal_only

        await run_binance_signal_only()
        return

    if selection.kind == LiveRuntimeKind.BINANCE_LIVE_BLOCKED:
        from src.live.binance_main_live_runtime import run_binance_main_live

        await run_binance_main_live()
        return

    # ── OKX legacy path continues below ─────────────────────────────────

    if not live_config_helpers.live_trading_enabled():
        raise RuntimeError("LIVE_TRADING is not true. Refusing to start live runner.")

    monitor_config = BollBandBreakoutMonitorConfig.from_env()
    cvd_config = CvdTrackerConfig.from_env()
    email_sender = EmailSender()
    journal = LiveTradeJournal()
    rolling_loss_guard = RollingLossGuard.from_env()
    state_store = LiveStateStore()
    reporter = DailyTradeReporter(journal, email_sender)
    trader = create_live_trader(os.environ)
    await trader.start()
    try:
        await trader.initialize()
        sizer = SimplePositionSizer(SimplePositionSizerConfig.from_account_equity(trader.account_equity_usdt))
        strategy = BollCvdShockReclaimStrategy(BollCvdReclaimStrategyConfig.from_env(), sizer)
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

    cvd = CvdTracker(cvd_config)
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
    sidecar_runtime_state.refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
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
    strategy_tick_queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue(
        maxsize=int(os.getenv("STRATEGY_TICK_QUEUE_MAXSIZE", "20000")))
    execution_queue: asyncio.Queue[live_runtime_types.TradeCommand] = asyncio.Queue(
        maxsize=int(os.getenv("EXECUTION_QUEUE_MAXSIZE", "1000")))
    position_sync_seconds = float(os.getenv("POSITION_SYNC_SECONDS", "5"))
    account_sync_seconds = float(os.getenv("ACCOUNT_SYNC_SECONDS", "60"))
    cash_log_min_delta_usdt = float(os.getenv("ACCOUNT_LOG_MIN_DELTA_USDT", "0.01"))
    market_tick_heartbeat_seconds = float(os.getenv("MARKET_TICK_HEARTBEAT_SECONDS", "60"))
    account_snapshot_stale_warn_seconds = float(os.getenv("ACCOUNT_SNAPSHOT_STALE_WARN_SECONDS", "30"))
    strategy_lag_warn_seconds = float(os.getenv("STRATEGY_TICK_LAG_WARN_SECONDS", "2"))
    execution_backlog_log_seconds = float(os.getenv("EXECUTION_QUEUE_BACKLOG_LOG_SECONDS", "30"))

    async def daily_report_loop() -> None:
        raw_time = os.getenv("DAILY_REPORT_TIME", "09:00")
        hour, minute = live_time_utils.parse_daily_report_time(raw_time)
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
        enabled = os.getenv("WEEKLY_SUMMARY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
        if not enabled:
            logger.info("Weekly overall summary loop disabled")
            return

        raw_time = os.getenv("WEEKLY_SUMMARY_TIME", "10:00")
        raw_weekday = os.getenv("WEEKLY_SUMMARY_WEEKDAY", "0")
        compact_after_success = os.getenv("WEEKLY_COMPACT_AFTER_SUCCESS", "false").strip().lower() in {"1", "true",
                                                                                                       "yes", "y", "on"}
        hour, minute = live_time_utils.parse_weekly_report_time(raw_time)
        weekday = int(raw_weekday)
        if weekday < 0 or weekday > 6:
            raise ValueError(f"Invalid WEEKLY_SUMMARY_WEEKDAY={raw_weekday}")

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

    monitor = BollBandBreakoutMonitor(
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


if __name__ == "__main__":
    asyncio.run(main())
