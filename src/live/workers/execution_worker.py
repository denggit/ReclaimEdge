from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.execution.trader import Trader
from src.live import queue_helpers as live_queue_helpers
from src.live import runtime_types as live_runtime_types
from src.live.workers import execution_failure as execution_failure_handler
from src.live.workers.execution_command_processor import ExecutionCommandProcessor
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.email_sender import EmailSender
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.live.portfolio_allocator_shadow import PortfolioAllocatorShadowRunner
    from src.live.portfolio_allocator_enforcer import PortfolioAllocatorEnforcer

logger = get_logger(__name__)


async def execution_worker(
    *,
    execution_queue: asyncio.Queue[live_runtime_types.TradeCommand],
    state_lock: asyncio.Lock,
    execution_state: live_runtime_types.ExecutionState,
    account_snapshot: live_runtime_types.AccountSnapshot,
    trader: Trader,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    email_sender: EmailSender,
    backlog_log_seconds: float,
    sidecar_skip_first_layer: bool = True,
    portfolio_allocator_shadow_runner: "PortfolioAllocatorShadowRunner | None" = None,
    portfolio_allocator_enforcer: "PortfolioAllocatorEnforcer | None" = None,
) -> None:
    processor = ExecutionCommandProcessor(
        state_lock=state_lock,
        execution_state=execution_state,
        account_snapshot=account_snapshot,
        trader=trader,
        strategy=strategy,
        journal=journal,
        state_store=state_store,
        email_sender=email_sender,
        sidecar_skip_first_layer=sidecar_skip_first_layer,
        portfolio_allocator_shadow_runner=portfolio_allocator_shadow_runner,
        portfolio_allocator_enforcer=portfolio_allocator_enforcer,
    )

    last_backlog_log = 0.0
    while True:
        command = await execution_queue.get()
        result = None
        try:
            queue_size = execution_queue.qsize()
            level = live_queue_helpers.queue_log_level(queue_size)
            now = time.monotonic()
            if level is not None and now - last_backlog_log >= backlog_log_seconds:
                logger.log(
                    level,
                    "EXECUTION_QUEUE_BACKLOG | queue_size=%s maxsize=%s oldest_command_age_seconds=%.3f",
                    queue_size,
                    execution_queue.maxsize,
                    live_queue_helpers.queue_oldest_command_age_seconds(execution_queue),
                )
                last_backlog_log = now

            result = await processor.process(command)
            if result is not None and not result.ok:
                raise RuntimeError(result.message)
        except Exception as exc:
            await execution_failure_handler.handle_execution_failure(
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
