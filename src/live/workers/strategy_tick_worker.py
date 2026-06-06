from __future__ import annotations

import asyncio
import copy
import logging
import time

from src.indicators.cvd_tracker import CvdTracker
from src.live import queue_helpers as live_queue_helpers
from src.live import runtime_types as live_runtime_types
from src.monitors.boll_band_breakout_monitor import MarketTickEvent
from src.position_management import core_position_view as core_position_view_helpers
from src.risk.rolling_loss_guard import ROLLING_LOSS_HALT_REASONS
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)

POSITION_MANAGEMENT_INTENTS = {"UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"}


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
            async with state_lock:
                account_snapshot.latest_market_price = event.tick.price
                account_snapshot.latest_market_price_ts_ms = event.tick.ts_ms
            now = time.monotonic()
            tick_lag_seconds = max(time.time() - event.tick.ts_ms / 1000, 0.0)
            queue_size = strategy_tick_queue.qsize()
            level = live_queue_helpers.queue_log_level(queue_size)
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

            account_age_seconds = max(now - account_snapshot.updated_monotonic,
                                      0.0) if account_snapshot.updated_monotonic > 0 else float("inf")
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
                halt_reason = execution_state.halt_reason
                pending_order_count = execution_state.pending_order_count
                has_position = bool(account_snapshot.position and account_snapshot.position.has_position)
            allow_position_management_only = (
                    trading_halted
                    and halt_reason in ROLLING_LOSS_HALT_REASONS
                    and has_position
            )
            if pending_order_count > 0:
                continue
            if trading_halted and not allow_position_management_only:
                continue

            backup_state = copy.deepcopy(strategy.state)
            intents = strategy.on_tick(
                price=event.tick.price,
                ts_ms=event.tick.ts_ms,
                boll=event.boll,
                cvd=cvd_snapshot,
            )
            if allow_position_management_only:
                intents = [intent for intent in intents if intent.intent_type in POSITION_MANAGEMENT_INTENTS]
            for intent in intents:
                if getattr(strategy.state, "sidecar_enabled_for_position", False):
                    intent = core_position_view_helpers.with_runtime_managed_core(intent, account_snapshot.position)
                command = live_runtime_types.TradeCommand(
                    intent=intent,
                    strategy_state_snapshot=backup_state,
                    tick_ts_ms=event.tick.ts_ms,
                    created_monotonic=time.monotonic(),
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
            strategy_tick_queue.task_done()
