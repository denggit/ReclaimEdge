from __future__ import annotations

import asyncio
import copy
import logging
import time

from src.indicators.cvd_tracker import CvdTracker
from src.live import queue_helpers as live_queue_helpers
from src.live import runtime_types as live_runtime_types
from src.live.halt_modes import (
    FULL_HALT,
    ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED,
    SIDECAR_DIRTY_HALT,
    allowed_intents_for_halt_mode,
    allows_core_position_management,
    resolve_halt_mode,
)
from src.monitors.boll_band_breakout_monitor import MarketTickEvent
from src.position_management import core_position_view as core_position_view_helpers
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)

# Retained for backward-compat references from other modules.
POSITION_MANAGEMENT_INTENTS = {"UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"}

# Intents that open or add to a position.
ENTRY_INTENTS = {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}


def _monotonic() -> float:
    return time.monotonic()


def _should_coalesce_strategy_ticks(
        *,
        enabled: bool,
        queue_size: int,
        threshold: int,
) -> bool:
    return enabled and queue_size >= threshold


def _drain_strategy_tick_queue_nowait(
        queue: asyncio.Queue[MarketTickEvent],
        *,
        max_drain: int,
) -> list[MarketTickEvent]:
    drained: list[MarketTickEvent] = []
    for _ in range(max_drain):
        try:
            drained.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return drained


def _halt_filter_intents(
    intents: list,
    halt_mode: str,
) -> list:
    """Filter intents based on halt_mode using the unified halt_modes helper.

    - FULL_HALT: drop everything.
    - ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED: keep POSITION_MANAGEMENT_INTENTS.
    - SIDECAR_DIRTY_HALT: keep only CORE_POSITION_MANAGEMENT_INTENTS (UPDATE_TP, MARKET_EXIT_RUNNER).
    """
    allowed = allowed_intents_for_halt_mode(halt_mode)
    return [intent for intent in intents if intent.intent_type in allowed]


async def strategy_tick_worker(
        *,
        strategy_tick_queue: asyncio.Queue[MarketTickEvent],
        execution_queue: asyncio.Queue[live_runtime_types.TradeCommand],
        state_lock: asyncio.Lock,
        account_snapshot: live_runtime_types.AccountSnapshot,
        execution_state: live_runtime_types.ExecutionState,
        cvd: CvdTracker,
        strategy: BollCvdShockReclaimStrategy,
        heartbeat_seconds: float,
        account_stale_warn_seconds: float,
        strategy_lag_warn_seconds: float,
        strategy_tick_coalesce_enabled: bool = True,
        strategy_tick_coalesce_queue_threshold: int = 50,
        strategy_tick_coalesce_min_decision_interval_seconds: float = 0.1,
        strategy_tick_coalesce_max_drain: int = 5000,
) -> None:
    last_heartbeat = 0.0
    last_lag_log = 0.0
    last_account_stale_log = 0.0
    last_coalesce_log = 0.0
    last_coalesce_decision_skip_log = 0.0
    last_strategy_decision_monotonic = 0.0
    latest_tick_ts_ms = 0
    while True:
        event = await strategy_tick_queue.get()
        drained_events: list[MarketTickEvent] = []
        try:
            queue_size_before = strategy_tick_queue.qsize()
            if _should_coalesce_strategy_ticks(
                    enabled=strategy_tick_coalesce_enabled,
                    queue_size=queue_size_before,
                    threshold=strategy_tick_coalesce_queue_threshold,
            ):
                drained_events = _drain_strategy_tick_queue_nowait(
                    strategy_tick_queue,
                    max_drain=strategy_tick_coalesce_max_drain,
                )

            batch = [event, *drained_events]
            coalesced = len(batch) > 1
            latest_event = batch[-1]
            latest_valid_event: MarketTickEvent | None = None
            latest_cvd_snapshot = None

            for batch_event in batch:
                latest_tick_ts_ms = max(latest_tick_ts_ms, batch_event.tick.ts_ms)
                latest_cvd_snapshot = cvd.update(
                    side=batch_event.tick.side,
                    size=batch_event.tick.size,
                    price=batch_event.tick.price,
                    ts_ms=batch_event.tick.ts_ms,
                )
                if batch_event.boll is not None:
                    latest_valid_event = batch_event

            async with state_lock:
                account_snapshot.latest_market_price = latest_event.tick.price
                account_snapshot.latest_market_price_ts_ms = latest_event.tick.ts_ms

            now = _monotonic()
            tick_lag_seconds = max(time.time() - latest_event.tick.ts_ms / 1000, 0.0)
            queue_size = strategy_tick_queue.qsize()
            level = live_queue_helpers.queue_log_level(queue_size)
            if (level is not None or tick_lag_seconds >= strategy_lag_warn_seconds) and now - last_lag_log >= 30:
                logger.log(
                    level or logging.WARNING,
                    "STRATEGY_TICK_LAG | tick_lag_seconds=%.3f strategy_queue_size=%s latest_tick_ts_ms=%s processed_tick_ts_ms=%s",
                    tick_lag_seconds,
                    queue_size,
                    latest_tick_ts_ms,
                    latest_event.tick.ts_ms,
                )
                last_lag_log = now

            account_age_seconds = max(now - account_snapshot.updated_monotonic,
                                      0.0) if account_snapshot.updated_monotonic > 0 else float("inf")
            if account_age_seconds >= account_stale_warn_seconds and now - last_account_stale_log >= 60:
                logger.warning(
                    "ACCOUNT_SNAPSHOT_STALE | age_seconds=%.1f threshold=%.1f",
                    account_age_seconds,
                    account_stale_warn_seconds,
                )
                last_account_stale_log = now

            decision_skipped = False
            if coalesced and last_strategy_decision_monotonic > 0:
                elapsed = now - last_strategy_decision_monotonic
                if elapsed < strategy_tick_coalesce_min_decision_interval_seconds:
                    decision_skipped = True

            if coalesced and now - last_coalesce_log >= 30:
                logger.warning(
                    "STRATEGY_TICK_COALESCED | queue_size_before=%s coalesced_count=%s latest_tick_lag_seconds=%.3f oldest_tick_ts_ms=%s latest_tick_ts_ms=%s decision_skipped=%s min_decision_interval_seconds=%.3f",
                    queue_size_before,
                    len(batch),
                    tick_lag_seconds,
                    batch[0].tick.ts_ms,
                    latest_event.tick.ts_ms,
                    str(decision_skipped).lower(),
                    strategy_tick_coalesce_min_decision_interval_seconds,
                )
                last_coalesce_log = now

            if decision_skipped and now - last_coalesce_decision_skip_log >= 30:
                logger.warning(
                    "STRATEGY_TICK_COALESCED_DECISION_SKIPPED | queue_size_before=%s coalesced_count=%s elapsed_seconds=%.3f min_decision_interval_seconds=%.3f latest_tick_ts_ms=%s",
                    queue_size_before,
                    len(batch),
                    now - last_strategy_decision_monotonic,
                    strategy_tick_coalesce_min_decision_interval_seconds,
                    latest_event.tick.ts_ms,
                )
                last_coalesce_decision_skip_log = now

            if decision_skipped:
                continue

            if latest_valid_event is None or latest_cvd_snapshot is None:
                continue

            if now - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = now
                logger.info(
                    "MARKET_TICK_HEARTBEAT | price=%.4f tick_ts_ms=%s side=%s size=%.8f boll_lower=%.4f boll_middle=%.4f boll_upper=%.4f switch=%s fast_cvd=%.8f previous_fast_cvd=%.8f buy_ratio=%.4f sell_ratio=%.4f burst_net_move_pct=%.6f move_ratio=%.2f volume_ratio=%.2f burst_range_pct=%.6f baseline_range_pct=%.6f burst_volume=%.8f baseline_volume=%.8f up_burst=%s down_burst=%s",
                    latest_valid_event.tick.price,
                    latest_valid_event.tick.ts_ms,
                    latest_valid_event.tick.side,
                    latest_valid_event.tick.size,
                    latest_valid_event.boll.lower,
                    latest_valid_event.boll.middle,
                    latest_valid_event.boll.upper,
                    latest_valid_event.boll.alert_switch_on,
                    latest_cvd_snapshot.fast_cvd,
                    latest_cvd_snapshot.previous_fast_cvd,
                    latest_cvd_snapshot.buy_ratio,
                    latest_cvd_snapshot.sell_ratio,
                    latest_cvd_snapshot.burst_net_move_pct,
                    latest_cvd_snapshot.burst_move_ratio,
                    latest_cvd_snapshot.burst_volume_ratio,
                    latest_cvd_snapshot.burst_range_pct,
                    latest_cvd_snapshot.baseline_range_pct,
                    latest_cvd_snapshot.burst_volume,
                    latest_cvd_snapshot.baseline_volume,
                    latest_cvd_snapshot.up_burst,
                    latest_cvd_snapshot.down_burst,
                )

            async with state_lock:
                trading_halted = execution_state.trading_halted
                halt_reason = execution_state.halt_reason
                pending_order_count = execution_state.pending_order_count
                has_position = bool(account_snapshot.position and account_snapshot.position.has_position)

            # ── Resolve halt mode ──────────────────────────────────────────
            halt_mode = resolve_halt_mode(halt_reason) if trading_halted else None

            if pending_order_count > 0:
                continue

            if halt_mode == FULL_HALT:
                # Complete stop — no on_tick, no intents, no emails.
                continue

            # ── Run strategy on_tick ───────────────────────────────────────
            backup_state = copy.deepcopy(strategy.state)
            intents = strategy.on_tick(
                price=latest_valid_event.tick.price,
                ts_ms=latest_valid_event.tick.ts_ms,
                boll=latest_valid_event.boll,
                cvd=latest_cvd_snapshot,
            )
            last_strategy_decision_monotonic = _monotonic()

            # ── Filter intents based on halt mode ──────────────────────────
            if halt_mode is not None:
                intents = _halt_filter_intents(intents, halt_mode)

            for intent in intents:
                if getattr(strategy.state, "sidecar_enabled_for_position", False):
                    intent = core_position_view_helpers.with_runtime_managed_core(intent, account_snapshot.position)
                command = live_runtime_types.TradeCommand(
                    intent=intent,
                    strategy_state_snapshot=backup_state,
                    tick_ts_ms=latest_valid_event.tick.ts_ms,
                    created_monotonic=_monotonic(),
                    account_snapshot_updated_ts_ms=account_snapshot.updated_ts_ms,
                    reason=intent.reason,
                )
                ok = await live_queue_helpers.enqueue_execution_command(command, execution_queue, state_lock,
                                                                        execution_state)
                if not ok:
                    async with state_lock:
                        strategy.state = backup_state
                    break
        except Exception:
            logger.exception("Strategy tick worker failed")
        finally:
            for _ in drained_events:
                strategy_tick_queue.task_done()
            strategy_tick_queue.task_done()
