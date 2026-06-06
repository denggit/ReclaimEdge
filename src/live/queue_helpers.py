from __future__ import annotations

import asyncio
import logging
import time

from src.live.runtime_types import ExecutionState, TradeCommand
from src.monitors.boll_band_breakout_monitor import MarketTickEvent
from src.utils.log import get_logger

logger = get_logger(__name__)


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
