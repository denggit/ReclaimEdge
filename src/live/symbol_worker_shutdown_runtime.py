from __future__ import annotations

import asyncio

from src.live import runtime_types as live_runtime_types
from src.live.worker_shutdown import WorkerShutdownController
from src.reporting.live_state_store import LiveStateStore
from src.utils.log import get_logger

logger = get_logger(__name__)


async def _wait_for_shutdown(shutdown_controller: WorkerShutdownController) -> None:
    """Coroutine that blocks until the shutdown event is set.

    This is used as a sentinel task in :meth:`SymbolWorkerApp.run` so the
    runtime can distinguish a requested graceful shutdown from an unexpected
    task exit.
    """
    await shutdown_controller.event.wait()
    logger.warning(
        "SYMBOL_WORKER_DRAINING | reason=%s",
        shutdown_controller.reason,
    )


async def _begin_symbol_worker_drain(
    execution_state: live_runtime_types.ExecutionState,
    state_lock: asyncio.Lock,
    reason: str,
) -> None:
    """Mark the execution state as draining so no new positions are opened.

    This is the **only** mutation D06b makes to the trading pipeline.  It
    sets ``trading_halted=True`` with a ``halt_reason`` of
    ``"symbol_worker_shutdown_draining"`` and clears ``halt_until_ts_ms`` so
    the halt is indefinite.

    It does **not**:
    * close positions
    * cancel TP / SL / Algo / Sidecar orders
    * place any OKX private write requests
    * modify the strategy state
    """
    async with state_lock:
        execution_state.trading_halted = True
        execution_state.halt_reason = "symbol_worker_shutdown_draining"
        execution_state.halt_until_ts_ms = None
    logger.warning(
        "SYMBOL_WORKER_DRAIN_MARKED | reason=%s trading_halted=True halt_reason=symbol_worker_shutdown_draining",
        reason,
    )


def _should_save_state_on_shutdown(
    execution_state: live_runtime_types.ExecutionState,
    strategy_state: object,
) -> bool:
    """Return ``True`` when the strategy state warrants a best-effort save.

    We save when there is a current position OR the strategy believes it has
    layers.  We do **not** save when the strategy is flat unless there is
    ambiguity.
    """
    if execution_state.current_position_id:
        return True
    layers = int(getattr(strategy_state, "layers", 0) or 0)
    return layers > 0


async def _save_state_on_shutdown(
    *,
    execution_state: live_runtime_types.ExecutionState,
    strategy: object,
    trader_symbol: str,
    state_store: LiveStateStore,
) -> None:
    """Best-effort persist of the current strategy state to ``state_store``.

    Silently degrades on failure — a failed state save must not prevent the
    worker from shutting down.
    """
    try:
        if not _should_save_state_on_shutdown(execution_state, strategy.state):
            logger.info(
                "SYMBOL_WORKER_SHUTDOWN_STATE_SKIP | reason=no_position_or_layers position_id=%s layers=%s",
                execution_state.current_position_id,
                int(getattr(strategy.state, "layers", 0) or 0),
            )
            return
        live_state = LiveStateStore.from_strategy_state(
            position_id=execution_state.current_position_id,
            symbol=trader_symbol,
            strategy_state=strategy.state,
            cash_before_position=execution_state.cash_before_position,
        )
        state_store.save(live_state)
        logger.warning(
            "SYMBOL_WORKER_SHUTDOWN_STATE_SAVED | position_id=%s layers=%s",
            execution_state.current_position_id,
            int(getattr(strategy.state, "layers", 0) or 0),
        )
    except Exception:
        logger.exception("SYMBOL_WORKER_SHUTDOWN_STATE_SAVE_FAILED")


async def _cancel_runtime_tasks(
    tasks: set[asyncio.Task],
    timeout: float,
) -> None:
    """Cancel a set of asyncio tasks with a bounded grace period.

    Each task is cancelled.  We then wait up to *timeout* seconds for them
    to finish.  Tasks that do not finish within the timeout are logged and
    abandoned — the event loop will clean them up when the process exits.
    """
    if not tasks:
        return

    for task in tasks:
        if not task.done():
            task.cancel()

    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "SYMBOL_WORKER_SHUTDOWN_DRAIN_TIMEOUT | timeout_seconds=%s remaining_tasks=%s",
            timeout,
            sum(1 for t in tasks if not t.done()),
        )
