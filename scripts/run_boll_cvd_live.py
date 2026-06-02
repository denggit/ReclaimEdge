from __future__ import annotations

import asyncio
import copy
import datetime as dt
import html
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from src.execution.trader import PositionSnapshot, Trader  # noqa: E402
from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig  # noqa: E402
from src.monitors.boll_band_breakout_monitor import (  # noqa: E402
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    MarketTickEvent,
)
from src.reporting.daily_trade_reporter import DailyTradeReporter  # noqa: E402
from src.reporting.live_state_store import LiveStateStore  # noqa: E402
from src.reporting.trade_journal import LiveTradeJournal  # noqa: E402
from src.risk.simple_position_sizer import (  # noqa: E402
    SimplePositionSizer,
    SimplePositionSizerConfig,
)
from src.strategies.boll_cvd_reclaim_strategy import (  # noqa: E402
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy  # noqa: E402
from src.utils.email_sender import EmailSender  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

logger = get_logger(__name__)


def live_trading_enabled() -> bool:
    return os.getenv("LIVE_TRADING", "false").strip().lower() in {"1", "true", "yes", "y", "on"}


def format_ts_ms(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def parse_daily_report_time(value: str) -> tuple[int, int]:
    hour_text, minute_text = value.strip().split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid DAILY_REPORT_TIME={value}")
    return hour, minute


def parse_weekly_report_time(value: str) -> tuple[int, int]:
    return parse_daily_report_time(value)


def next_daily_report_time(hour: int, minute: int) -> dt.datetime:
    now = dt.datetime.now().astimezone()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return target


def next_weekly_summary_time(hour: int, minute: int, weekday: int = 0) -> dt.datetime:
    now = dt.datetime.now().astimezone()
    target = now + dt.timedelta(days=weekday - now.weekday())
    target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=7)
    return target


def build_live_failure_email(intent: TradeIntent, error: Exception, rolled_back: bool, halted: bool) -> tuple[str, str]:
    subject = f"LIVE order failed | ETH-USDT-SWAP | {intent.intent_type} | layer {intent.layer_index}"
    event_time = format_ts_ms(intent.ts_ms)
    state_text = "Strategy state has been rolled back." if rolled_back else "Entry may be live. Strategy state was NOT rolled back."
    halt_text = "Trading has been halted. Please check OKX manually." if halted else "Trading is not halted."
    content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <h2>LIVE order failed</h2>
  <p><strong>{html.escape(state_text)}</strong></p>
  <p><strong>{html.escape(halt_text)}</strong></p>
  <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">intent_type</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.intent_type)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">side</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.side)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">layer</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.layer_index}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.tp_price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_mode</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.tp_mode)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">avg_entry</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.avg_entry_price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">breakeven</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.breakeven_price:.4f}</td></tr>
  </table>
  <p><strong>Reason:</strong> {html.escape(intent.reason)}</p>
  <p><strong>Error:</strong> {html.escape(str(error))}</p>
  <p><strong>Event time:</strong> {html.escape(event_time)}</p>
</div>
""".strip()
    return subject, content


def restore_strategy_from_position(strategy: BollCvdReclaimStrategy, position: PositionSnapshot) -> None:
    if not position.has_position or position.side is None or position.avg_entry_price <= 0:
        return
    strategy.state = StrategyPositionState(
        side=position.side,
        layers=1,
        last_entry_price=position.avg_entry_price,
        tp_price=None,
        last_order_ts_ms=0,
        last_tp_update_ts_ms=0,
        total_entry_qty=position.eth_qty,
        total_entry_notional=position.avg_entry_price * position.eth_qty,
        avg_entry_price=position.avg_entry_price,
    )
    logger.warning(
        "Recovered existing position into strategy state | side=%s contracts=%s eth_qty=%.6f avg_entry=%.4f",
        position.side,
        position.contracts,
        position.eth_qty,
        position.avg_entry_price,
    )


def restore_strategy_from_saved_state(strategy: BollCvdReclaimStrategy, saved_state) -> None:  # type: ignore[no-untyped-def]
    strategy.state = StrategyPositionState(
        side=saved_state.side,
        layers=saved_state.layers,
        last_entry_price=saved_state.last_entry_price,
        tp_price=saved_state.tp_price,
        last_order_ts_ms=saved_state.last_order_ts_ms,
        last_tp_update_ts_ms=saved_state.last_tp_update_ts_ms,
        last_tp_update_candle_ts_ms=saved_state.last_tp_update_candle_ts_ms,
        total_entry_qty=saved_state.total_entry_qty,
        total_entry_notional=saved_state.total_entry_notional,
        avg_entry_price=saved_state.avg_entry_price,
        breakeven_price=saved_state.breakeven_price,
        tp_mode=saved_state.tp_mode,
    )
    logger.warning(
        "Recovered strategy state from local disk | position_id=%s side=%s layers=%s avg_entry=%.4f tp=%s",
        saved_state.position_id,
        saved_state.side,
        saved_state.layers,
        saved_state.avg_entry_price,
        saved_state.tp_price,
    )


def sync_strategy_cost_from_position(strategy: BollCvdReclaimStrategy, position: PositionSnapshot) -> None:
    if not position.has_position or position.side is None or position.avg_entry_price <= 0:
        return
    if strategy.state.side is None or strategy.state.side != position.side or strategy.state.layers <= 0:
        restore_strategy_from_position(strategy, position)
        return
    strategy.state.total_entry_qty = position.eth_qty
    strategy.state.total_entry_notional = position.avg_entry_price * position.eth_qty
    strategy.state.avg_entry_price = position.avg_entry_price
    strategy.state.last_entry_price = strategy.state.last_entry_price or position.avg_entry_price


def position_log_key(position: PositionSnapshot) -> tuple[str, str, float]:
    if not position.has_position or position.side is None:
        return ("FLAT", "0", 0.0)
    return (position.side, str(position.contracts), round(position.avg_entry_price, 2))


async def fetch_usdt_cash_balance(trader: Trader) -> float:
    res = await trader.request("GET", "/api/v5/account/balance?ccy=USDT")
    data = res.get("data", [])
    if not data:
        return 0.0
    for item in data[0].get("details", []):
        if item.get("ccy") == "USDT":
            return float(item.get("cashBal") or item.get("availBal") or item.get("availEq") or item.get("eq") or 0.0)
    return float(data[0].get("totalEq") or 0.0)


@dataclass
class AccountSnapshot:
    position: PositionSnapshot | None
    cash: float
    equity: float
    updated_monotonic: float
    updated_ts_ms: int
    version: int = 0


@dataclass
class ExecutionState:
    current_position_id: str | None
    cash_before_position: float | None
    trading_halted: bool = False
    last_order_ts_ms: int = 0
    pending_order_count: int = 0


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


def utc_ms() -> int:
    return int(time.time() * 1000)


def queue_log_level(queue_size: int) -> int | None:
    if queue_size < 500:
        return None
    if queue_size < 2000:
        return logging.INFO
    if queue_size < 8000:
        return logging.WARNING
    return logging.ERROR


def queue_oldest_command_age_seconds(queue: asyncio.Queue[TradeCommand]) -> float:
    try:
        oldest = queue._queue[0]  # type: ignore[attr-defined]
    except Exception:
        return 0.0
    return max(time.monotonic() - oldest.created_monotonic, 0.0)


async def enqueue_strategy_tick(
    event: MarketTickEvent,
    strategy_tick_queue: asyncio.Queue[MarketTickEvent],
    state_lock: asyncio.Lock,
    execution_state: ExecutionState,
) -> None:
    if event.boll is None:
        return
    try:
        strategy_tick_queue.put_nowait(event)
    except asyncio.QueueFull:
        logger.error(
            "STRATEGY_TICK_QUEUE_FULL | price=%.4f tick_ts_ms=%s queue_size=%s",
            event.tick.price,
            event.tick.ts_ms,
            strategy_tick_queue.qsize(),
        )
        async with state_lock:
            execution_state.trading_halted = True


async def enqueue_execution_command(
    command: TradeCommand,
    execution_queue: asyncio.Queue[TradeCommand],
    state_lock: asyncio.Lock,
    execution_state: ExecutionState,
) -> bool:
    async with state_lock:
        if execution_queue.full():
            logger.error(
                "EXECUTION_QUEUE_FULL | intent_type=%s side=%s tick_ts_ms=%s queue_size=%s",
                command.intent.intent_type,
                command.intent.side,
                command.tick_ts_ms,
                execution_queue.qsize(),
            )
            execution_state.trading_halted = True
            return False
        execution_state.pending_order_count += 1
        try:
            execution_queue.put_nowait(command)
        except asyncio.QueueFull:
            execution_state.pending_order_count = max(execution_state.pending_order_count - 1, 0)
            execution_state.trading_halted = True
            logger.error(
                "EXECUTION_QUEUE_FULL | intent_type=%s side=%s tick_ts_ms=%s queue_size=%s",
                command.intent.intent_type,
                command.intent.side,
                command.tick_ts_ms,
                execution_queue.qsize(),
            )
            return False
    return True


async def strategy_tick_worker(
    *,
    strategy_tick_queue: asyncio.Queue[MarketTickEvent],
    execution_queue: asyncio.Queue[TradeCommand],
    state_lock: asyncio.Lock,
    account_snapshot: AccountSnapshot,
    execution_state: ExecutionState,
    cvd: CvdTracker,
    strategy: BollCvdShockReclaimStrategy,
    heartbeat_seconds: float,
    account_stale_warn_seconds: float,
    strategy_lag_warn_seconds: float,
) -> None:
    last_heartbeat = 0.0
    last_lag_log = 0.0
    last_account_stale_log = 0.0
    latest_tick_ts_ms = 0
    while True:
        event = await strategy_tick_queue.get()
        try:
            if event.boll is None:
                continue
            latest_tick_ts_ms = max(latest_tick_ts_ms, event.tick.ts_ms)
            now = time.monotonic()
            tick_lag_seconds = max(time.time() - event.tick.ts_ms / 1000, 0.0)
            queue_size = strategy_tick_queue.qsize()
            level = queue_log_level(queue_size)
            if (level is not None or tick_lag_seconds >= strategy_lag_warn_seconds) and now - last_lag_log >= 30:
                logger.log(
                    level or logging.WARNING,
                    "STRATEGY_TICK_LAG | tick_lag_seconds=%.3f strategy_queue_size=%s latest_tick_ts_ms=%s processed_tick_ts_ms=%s",
                    tick_lag_seconds,
                    queue_size,
                    latest_tick_ts_ms,
                    event.tick.ts_ms,
                )
                last_lag_log = now

            account_age_seconds = max(now - account_snapshot.updated_monotonic, 0.0) if account_snapshot.updated_monotonic > 0 else float("inf")
            if account_age_seconds >= account_stale_warn_seconds and now - last_account_stale_log >= 60:
                logger.warning(
                    "ACCOUNT_SNAPSHOT_STALE | age_seconds=%.1f threshold=%.1f",
                    account_age_seconds,
                    account_stale_warn_seconds,
                )
                last_account_stale_log = now

            cvd_snapshot = cvd.update(
                side=event.tick.side,
                size=event.tick.size,
                price=event.tick.price,
                ts_ms=event.tick.ts_ms,
            )
            if now - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = now
                logger.info(
                    "MARKET_TICK_HEARTBEAT | price=%.4f tick_ts_ms=%s side=%s size=%.8f boll_lower=%.4f boll_middle=%.4f boll_upper=%.4f switch=%s fast_cvd=%.8f previous_fast_cvd=%.8f buy_ratio=%.4f sell_ratio=%.4f burst_net_move_pct=%.6f move_ratio=%.2f volume_ratio=%.2f burst_range_pct=%.6f baseline_range_pct=%.6f burst_volume=%.8f baseline_volume=%.8f up_burst=%s down_burst=%s",
                    event.tick.price,
                    event.tick.ts_ms,
                    event.tick.side,
                    event.tick.size,
                    event.boll.lower,
                    event.boll.middle,
                    event.boll.upper,
                    event.boll.alert_switch_on,
                    cvd_snapshot.fast_cvd,
                    cvd_snapshot.previous_fast_cvd,
                    cvd_snapshot.buy_ratio,
                    cvd_snapshot.sell_ratio,
                    cvd_snapshot.burst_net_move_pct,
                    cvd_snapshot.burst_move_ratio,
                    cvd_snapshot.burst_volume_ratio,
                    cvd_snapshot.burst_range_pct,
                    cvd_snapshot.baseline_range_pct,
                    cvd_snapshot.burst_volume,
                    cvd_snapshot.baseline_volume,
                    cvd_snapshot.up_burst,
                    cvd_snapshot.down_burst,
                )

            async with state_lock:
                trading_halted = execution_state.trading_halted
                pending_order_count = execution_state.pending_order_count
            if trading_halted or pending_order_count > 0:
                continue

            backup_state = copy.deepcopy(strategy.state)
            intents = strategy.on_tick(
                price=event.tick.price,
                ts_ms=event.tick.ts_ms,
                boll=event.boll,
                cvd=cvd_snapshot,
            )
            for intent in intents:
                command = TradeCommand(
                    intent=intent,
                    strategy_state_snapshot=backup_state,
                    tick_ts_ms=event.tick.ts_ms,
                    created_monotonic=time.monotonic(),
                    account_snapshot_updated_ts_ms=account_snapshot.updated_ts_ms,
                    reason=intent.reason,
                )
                ok = await enqueue_execution_command(command, execution_queue, state_lock, execution_state)
                if not ok:
                    async with state_lock:
                        strategy.state = backup_state
                    break
        except Exception:
            logger.exception("Strategy tick worker failed")
        finally:
            strategy_tick_queue.task_done()


async def execution_worker(
    *,
    execution_queue: asyncio.Queue[TradeCommand],
    state_lock: asyncio.Lock,
    execution_state: ExecutionState,
    account_snapshot: AccountSnapshot,
    trader: Trader,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    email_sender: EmailSender,
    backlog_log_seconds: float,
) -> None:
    last_backlog_log = 0.0
    while True:
        command = await execution_queue.get()
        result = None
        try:
            queue_size = execution_queue.qsize()
            level = queue_log_level(queue_size)
            now = time.monotonic()
            if level is not None and now - last_backlog_log >= backlog_log_seconds:
                logger.log(
                    level,
                    "EXECUTION_QUEUE_BACKLOG | queue_size=%s maxsize=%s oldest_command_age_seconds=%.3f",
                    queue_size,
                    execution_queue.maxsize,
                    queue_oldest_command_age_seconds(execution_queue),
                )
                last_backlog_log = now

            async with state_lock:
                if execution_state.trading_halted:
                    logger.warning(
                        "EXECUTION_SKIPPED | reason=trading_halted intent_type=%s side=%s tick_ts_ms=%s",
                        command.intent.intent_type,
                        command.intent.side,
                        command.tick_ts_ms,
                    )
                    continue
                current_position_id = execution_state.current_position_id
                cash_before_position = execution_state.cash_before_position

            entry_cash_before = cash_before_position
            if command.intent.intent_type != "UPDATE_TP" and current_position_id is None:
                entry_cash_before = await fetch_usdt_cash_balance(trader)

            result = await trader.execute_intent(command.intent)
            if not result.ok:
                raise RuntimeError(result.message)

            if command.intent.intent_type == "UPDATE_TP":
                async with state_lock:
                    current_position_id = execution_state.current_position_id
                    cash_before_position = execution_state.cash_before_position
                    execution_state.last_order_ts_ms = command.intent.ts_ms
                    strategy_state_for_save = copy.deepcopy(strategy.state)
                    equity = account_snapshot.equity
                journal.record_tp_update(position_id=current_position_id, intent=command.intent, result=result, equity=equity)
                state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy_state_for_save, cash_before_position=cash_before_position))
                logger.warning(
                    "LIVE TP update success | side=%s layer=%s price=%.4f contracts=%s new_tp_price=%s tp_mode=%s avg_entry=%.4f breakeven=%.4f tp_order_id=%s",
                    command.intent.side,
                    command.intent.layer_index,
                    command.intent.price,
                    result.contracts,
                    result.tp_price,
                    command.intent.tp_mode,
                    command.intent.avg_entry_price,
                    command.intent.breakeven_price,
                    result.tp_order_id,
                )
            else:
                new_position_id = None
                async with state_lock:
                    if execution_state.current_position_id is None:
                        new_position_id = journal.new_position_id(trader.symbol, command.intent.side, command.intent.ts_ms)
                        execution_state.current_position_id = new_position_id
                        execution_state.cash_before_position = entry_cash_before
                    current_position_id = execution_state.current_position_id
                    cash_before_position = execution_state.cash_before_position
                    execution_state.last_order_ts_ms = command.intent.ts_ms
                    strategy_state_for_save = copy.deepcopy(strategy.state)
                    equity = account_snapshot.equity
                journal.record_entry(
                    position_id=current_position_id or new_position_id or "",
                    intent=command.intent,
                    result=result,
                    cash_before_position=cash_before_position,
                    equity=equity,
                    extra={"symbol": trader.symbol},
                )
                state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy_state_for_save, cash_before_position=cash_before_position))
                logger.warning(
                    "LIVE entry success | intent_type=%s side=%s layer=%s price=%.4f contracts=%s tp_price=%s tp_mode=%s avg_entry=%.4f breakeven=%.4f order_id=%s tp_order_id=%s",
                    command.intent.intent_type,
                    command.intent.side,
                    command.intent.layer_index,
                    command.intent.price,
                    result.contracts,
                    result.tp_price,
                    command.intent.tp_mode,
                    command.intent.avg_entry_price,
                    command.intent.breakeven_price,
                    result.order_id,
                    result.tp_order_id,
                )
        except Exception as exc:
            await handle_execution_failure(
                command=command,
                result=result,
                error=exc,
                state_lock=state_lock,
                execution_state=execution_state,
                trader=trader,
                strategy=strategy,
                journal=journal,
                email_sender=email_sender,
            )
        finally:
            async with state_lock:
                execution_state.pending_order_count = max(execution_state.pending_order_count - 1, 0)
            execution_queue.task_done()


async def handle_execution_failure(
    *,
    command: TradeCommand,
    result: Any | None,
    error: Exception,
    state_lock: asyncio.Lock,
    execution_state: ExecutionState,
    trader: Trader,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    email_sender: EmailSender,
) -> ExecutionReport:
    contracts = trader.position_contracts
    if result is None or getattr(result, "entry_filled", False):
        try:
            position = await trader.fetch_position_snapshot()
            contracts = position.contracts
        except Exception:
            contracts = trader.position_contracts

    entry_may_be_live = bool(getattr(result, "entry_filled", False)) or contracts > 0
    rolled_back = False
    async with state_lock:
        current_position_id = execution_state.current_position_id
        if entry_may_be_live:
            execution_state.trading_halted = True
            trader.position_contracts = contracts
        else:
            strategy.state = copy.deepcopy(command.strategy_state_snapshot)
            rolled_back = True
        halted = execution_state.trading_halted

    if entry_may_be_live:
        logger.exception("LIVE execution failed after/possibly after entry. Trading halted; strategy state NOT rolled back.")
    else:
        logger.exception("LIVE execution failed before entry; strategy state has been rolled back")

    try:
        journal.record_error(position_id=current_position_id, intent=command.intent, error=error, rolled_back=rolled_back, halted=halted)
    except Exception:
        logger.exception("Failed to write trade journal error event")

    subject, content = build_live_failure_email(command.intent, error, rolled_back=rolled_back, halted=halted)
    ok = await email_sender.send_email_async(subject, content, content_type="html")
    if not ok:
        logger.error("Failed to send live execution failure email")

    return ExecutionReport(
        command=command,
        result=result,
        ok=False,
        error=error,
        entry_may_be_live=entry_may_be_live,
        created_monotonic=command.created_monotonic,
        finished_monotonic=time.monotonic(),
    )


async def account_position_sync_worker(
    *,
    state_lock: asyncio.Lock,
    account_snapshot: AccountSnapshot,
    execution_state: ExecutionState,
    trader: Trader,
    sizer: SimplePositionSizer,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    position_sync_seconds: float,
    account_sync_seconds: float,
    cash_log_min_delta_usdt: float,
) -> None:
    last_account_sync = 0.0
    last_logged_cash = account_snapshot.cash
    last_logged_position_key = position_log_key(account_snapshot.position) if account_snapshot.position is not None else ("FLAT", "0", 0.0)
    consecutive_failures = 0
    first_failure_monotonic = 0.0
    last_failure_log = 0.0
    last_stale_log = 0.0
    sync_failure_log_interval_seconds = float(os.getenv("ACCOUNT_SYNC_FAILURE_LOG_INTERVAL_SECONDS", "60"))
    sync_stale_warn_seconds = float(os.getenv("ACCOUNT_SYNC_STALE_WARN_SECONDS", "180"))
    while True:
        try:
            await asyncio.sleep(position_sync_seconds)
            now = time.monotonic()
            cash = account_snapshot.cash
            equity = account_snapshot.equity
            if now - last_account_sync >= account_sync_seconds:
                equity = await trader.fetch_usdt_equity()
                cash = await fetch_usdt_cash_balance(trader)
                last_account_sync = now

            position = await trader.fetch_position_snapshot()
            current_position_key = position_log_key(position)
            cash_after = cash
            equity_after = equity
            async with state_lock:
                pending_order_count = execution_state.pending_order_count
                should_fetch_flat_balances = (
                    pending_order_count == 0 and not position.has_position and strategy.state.layers > 0
                )
            if should_fetch_flat_balances:
                cash_after = await fetch_usdt_cash_balance(trader)
                equity_after = await trader.fetch_usdt_equity()

            record_flat_payload: dict[str, Any] | None = None
            save_state_payload: tuple[str | None, StrategyPositionState, float | None] | None = None
            clear_state = False
            async with state_lock:
                account_snapshot.position = position
                account_snapshot.cash = cash
                account_snapshot.equity = equity
                account_snapshot.updated_monotonic = time.monotonic()
                account_snapshot.updated_ts_ms = utc_ms()
                account_snapshot.version += 1
                trader.account_equity_usdt = equity
                sizer.update_account_equity(equity)

                if abs(cash - last_logged_cash) >= cash_log_min_delta_usdt:
                    logger.info(
                        "CASH_SYNC_CHANGED | cash=%.4f previous=%.4f equity=%.4f layer_margin_pct=%.4f leverage=%.2f",
                        cash,
                        last_logged_cash,
                        equity,
                        sizer.config.layer_margin_pct,
                        sizer.config.leverage,
                    )
                    last_logged_cash = cash

                pending_order_count = execution_state.pending_order_count
                if should_fetch_flat_balances and pending_order_count == 0 and not position.has_position and strategy.state.layers > 0:
                    record_flat_payload = {
                        "position_id": execution_state.current_position_id,
                        "symbol": trader.symbol,
                        "side": strategy.state.side,
                        "cash_before_position": execution_state.cash_before_position,
                        "cash_after": cash_after,
                        "equity_after": equity_after,
                        "reason": "OKX position is flat. TP filled or manual close detected.",
                        "layers": strategy.state.layers,
                        "avg_entry_price": strategy.state.avg_entry_price,
                        "last_tp_price": strategy.state.tp_price,
                    }
                    logger.warning("POSITION_SYNC_CHANGED | flat_on_okx=true. Resetting strategy and trader state.")
                    strategy.state = StrategyPositionState()
                    trader.mark_flat()
                    execution_state.trading_halted = False
                    execution_state.current_position_id = None
                    execution_state.cash_before_position = None
                    clear_state = True
                    last_logged_cash = cash_after
                    last_logged_position_key = current_position_key
                elif position.has_position:
                    trader.position_contracts = position.contracts
                    if pending_order_count == 0:
                        sync_strategy_cost_from_position(strategy, position)
                        save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state), execution_state.cash_before_position)
                        if current_position_key != last_logged_position_key:
                            logger.info(
                                "POSITION_SYNC_CHANGED | side=%s contracts=%s avg_entry=%.4f eth_qty=%.6f strategy_layers=%s",
                                position.side,
                                position.contracts,
                                position.avg_entry_price,
                                position.eth_qty,
                                strategy.state.layers,
                            )
                            last_logged_position_key = current_position_key

            if record_flat_payload is not None:
                journal.record_flat(**record_flat_payload)
            if clear_state:
                state_store.clear()
            if save_state_payload is not None:
                position_id, strategy_state, cash_before_position = save_state_payload
                state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader.symbol, strategy_state=strategy_state, cash_before_position=cash_before_position))
            if consecutive_failures > 0:
                logger.warning("ACCOUNT_SYNC_RECOVERED | failures=%s", consecutive_failures)
            consecutive_failures = 0
            first_failure_monotonic = 0.0
        except Exception as exc:
            now = time.monotonic()
            consecutive_failures += 1
            if first_failure_monotonic <= 0:
                first_failure_monotonic = now
            last_success_age_seconds = (
                max(now - account_snapshot.updated_monotonic, 0.0)
                if account_snapshot.updated_monotonic > 0
                else float("inf")
            )
            if now - last_failure_log >= sync_failure_log_interval_seconds:
                logger.warning(
                    "ACCOUNT_SYNC_FAILED | failures=%s error_type=%s error=%s last_success_age_seconds=%.1f",
                    consecutive_failures,
                    type(exc).__name__,
                    str(exc),
                    last_success_age_seconds,
                )
                last_failure_log = now
            if now - first_failure_monotonic >= sync_stale_warn_seconds and now - last_stale_log >= sync_failure_log_interval_seconds:
                logger.warning(
                    "ACCOUNT_SYNC_STALE | failures=%s last_success_age_seconds=%.1f risk=account_snapshot_may_be_stale",
                    consecutive_failures,
                    last_success_age_seconds,
                )
                last_stale_log = now


async def main() -> None:
    load_dotenv()
    if not live_trading_enabled():
        raise RuntimeError("LIVE_TRADING is not true. Refusing to start live runner.")

    monitor_config = BollBandBreakoutMonitorConfig.from_env()
    cvd_config = CvdTrackerConfig.from_env()
    email_sender = EmailSender()
    journal = LiveTradeJournal()
    state_store = LiveStateStore()
    reporter = DailyTradeReporter(journal, email_sender)
    trader = Trader()
    await trader.start()
    try:
        await trader.initialize()
        sizer = SimplePositionSizer(SimplePositionSizerConfig.from_account_equity(trader.account_equity_usdt))
        strategy = BollCvdShockReclaimStrategy(BollCvdReclaimStrategyConfig.from_env(), sizer)
        startup_position = await trader.fetch_position_snapshot()
        startup_cash = await fetch_usdt_cash_balance(trader)
    except Exception:
        await trader.close()
        raise
    current_position_id: str | None = None
    cash_before_position: float | None = None

    saved_state = state_store.load()
    if startup_position.has_position:
        if saved_state and saved_state.side == startup_position.side and saved_state.layers > 0:
            restore_strategy_from_saved_state(strategy, saved_state)
            current_position_id = saved_state.position_id
            cash_before_position = saved_state.cash_before_position
        else:
            restore_strategy_from_position(strategy, startup_position)
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
        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy.state, cash_before_position=cash_before_position))
    else:
        state_store.clear()

    cvd = CvdTracker(cvd_config)
    state_lock = asyncio.Lock()
    account_snapshot = AccountSnapshot(
        position=startup_position,
        cash=startup_cash,
        equity=trader.account_equity_usdt,
        updated_monotonic=time.monotonic(),
        updated_ts_ms=utc_ms(),
        version=1,
    )
    execution_state = ExecutionState(
        current_position_id=current_position_id,
        cash_before_position=cash_before_position,
        trading_halted=False,
    )
    strategy_tick_queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue(maxsize=int(os.getenv("STRATEGY_TICK_QUEUE_MAXSIZE", "20000")))
    execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=int(os.getenv("EXECUTION_QUEUE_MAXSIZE", "1000")))
    position_sync_seconds = float(os.getenv("POSITION_SYNC_SECONDS", "5"))
    account_sync_seconds = float(os.getenv("ACCOUNT_SYNC_SECONDS", "60"))
    cash_log_min_delta_usdt = float(os.getenv("ACCOUNT_LOG_MIN_DELTA_USDT", "0.01"))
    market_tick_heartbeat_seconds = float(os.getenv("MARKET_TICK_HEARTBEAT_SECONDS", "60"))
    account_snapshot_stale_warn_seconds = float(os.getenv("ACCOUNT_SNAPSHOT_STALE_WARN_SECONDS", "30"))
    strategy_lag_warn_seconds = float(os.getenv("STRATEGY_TICK_LAG_WARN_SECONDS", "2"))
    execution_backlog_log_seconds = float(os.getenv("EXECUTION_QUEUE_BACKLOG_LOG_SECONDS", "30"))

    async def daily_report_loop() -> None:
        raw_time = os.getenv("DAILY_REPORT_TIME", "09:00")
        hour, minute = parse_daily_report_time(raw_time)
        logger.info("Daily trade report loop started | DAILY_REPORT_TIME=%s", raw_time)
        while True:
            target = next_daily_report_time(hour, minute)
            sleep_seconds = max((target - dt.datetime.now().astimezone()).total_seconds(), 1)
            await asyncio.sleep(sleep_seconds)
            try:
                ok = await reporter.send_last_24h_report()
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
        hour, minute = parse_weekly_report_time(raw_time)
        weekday = int(raw_weekday)
        if weekday < 0 or weekday > 6:
            raise ValueError(f"Invalid WEEKLY_SUMMARY_WEEKDAY={raw_weekday}")

        logger.info(
            "Weekly overall summary loop started | WEEKLY_SUMMARY_WEEKDAY=%s WEEKLY_SUMMARY_TIME=%s",
            weekday,
            raw_time,
        )

        while True:
            target = next_weekly_summary_time(hour, minute, weekday)
            sleep_seconds = max((target - dt.datetime.now().astimezone()).total_seconds(), 1)
            await asyncio.sleep(sleep_seconds)
            try:
                ok = await reporter.send_overall_summary_report()
                if ok:
                    logger.info("Weekly overall summary report sent successfully")
                else:
                    logger.error("Weekly overall summary report failed")
            except Exception:
                logger.exception("Weekly overall summary report loop failed")

    async def on_market_tick(event: MarketTickEvent) -> None:
        await enqueue_strategy_tick(event, strategy_tick_queue, state_lock, execution_state)

    monitor = BollBandBreakoutMonitor(
        config=monitor_config,
        tick_handlers=[on_market_tick],
    )
    try:
        await asyncio.gather(
            account_position_sync_worker(
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
            ),
            strategy_tick_worker(
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
            execution_worker(
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
            ),
            daily_report_loop(),
            weekly_summary_loop(),
            monitor.run_forever(),
        )
    finally:
        await trader.close()


if __name__ == "__main__":
    asyncio.run(main())
