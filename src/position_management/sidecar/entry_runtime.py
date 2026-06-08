from __future__ import annotations

import asyncio
import os
from typing import Any

from src.execution.trader import Trader
from src.live import runtime_types as live_runtime_types
from src.live.alerts.halt_alerts import (
    HaltAlertDeduper,
    HaltAlertPayload,
    send_halt_alert_once,
)
from src.live.halt_modes import resolve_halt_mode
from src.position_management import cost_runtime as position_cost_runtime
from src.position_management.sidecar import runtime_state as sidecar_runtime_state
from src.position_management.sidecar.model import SidecarLegStatus
from src.position_management.sidecar.planner import SidecarExecutionPlan
from src.position_management.sidecar.reconciler import (
    mark_sidecar_leg_open_unprotected,
    sidecar_leg_from_fill,
)
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState, TradeIntent
from src.utils.log import get_logger

logger = get_logger(__name__)


def _is_okx_rate_limit_error(exc: Exception) -> bool:
    """Detect OKX rate limit errors (code 50011)."""
    msg = str(exc).lower()
    return "50011" in msg or "rate limit reached" in msg


def _sidecar_tp_place_retry_count() -> int:
    return max(1, int(os.getenv("SIDECAR_TP_PLACE_RETRY_COUNT", "3")))


def _sidecar_tp_place_retry_interval() -> float:
    return max(0.1, float(os.getenv("SIDECAR_TP_PLACE_RETRY_INTERVAL_SECONDS", "0.8")))


def _sidecar_tp_place_retry_backoff() -> float:
    return max(1.0, float(os.getenv("SIDECAR_TP_PLACE_RETRY_BACKOFF_MULTIPLIER", "1.5")))


def _sidecar_tp_rate_limit_fail_action() -> str:
    return os.getenv("SIDECAR_TP_RATE_LIMIT_FAIL_ACTION", "HALT_ONLY").strip().upper()


async def _place_sidecar_tp_with_rate_limit_retry(
    trader: Trader,
    side: str,
    contracts: str,
    tp_price: float,
    client_order_id: str,
) -> tuple[str | None, str]:
    """Place sidecar TP with rate-limit-aware retry and backoff.

    Returns:
        (tp_order_id, error_message).  tp_order_id is None on failure.
    """
    retry_count = _sidecar_tp_place_retry_count()
    interval = _sidecar_tp_place_retry_interval()
    backoff = _sidecar_tp_place_retry_backoff()
    last_error = ""

    for attempt in range(1, retry_count + 1):
        try:
            tp_order_id = await trader.place_sidecar_fixed_take_profit(
                side=side,
                contracts=contracts,
                tp_price=tp_price,
                client_order_id=client_order_id,
            )
            if attempt > 1:
                logger.warning(
                    "SIDECAR_TP_PLACE_RETRY_SUCCESS | attempt=%s/%s client_order_id=%s",
                    attempt,
                    retry_count,
                    client_order_id,
                )
            return tp_order_id, ""
        except Exception as exc:
            last_error = str(exc)
            is_rate_limit = _is_okx_rate_limit_error(exc)
            if attempt < retry_count:
                wait = interval * (backoff ** (attempt - 1))
                logger.warning(
                    "SIDECAR_TP_PLACE_RETRY | attempt=%s/%s error_type=%s wait=%.3fs client_order_id=%s error=%s",
                    attempt,
                    retry_count,
                    "rate_limit" if is_rate_limit else "other",
                    wait,
                    client_order_id,
                    exc,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "SIDECAR_TP_PLACE_RETRY_EXHAUSTED | attempts=%s error_type=%s client_order_id=%s last_error=%s",
                    retry_count,
                    "rate_limit" if is_rate_limit else "other",
                    client_order_id,
                    last_error,
                )

    return None, last_error


async def _send_sidecar_failure_halt_alert(
    *,
    email_sender: Any,
    halt_alert_deduper: HaltAlertDeduper | None,
    trader_symbol: str,
    position_id: str | None,
    halt_reason: str,
    side: str,
    layer: int | None,
    sidecar_dirty: bool,
    manual_intervention_required: bool,
    message: str,
    extra: dict | None = None,
) -> None:
    """Send a rate-limited critical email when sidecar TP fails."""
    if email_sender is None or halt_alert_deduper is None:
        return
    try:
        await send_halt_alert_once(
            email_sender=email_sender,
            deduper=halt_alert_deduper,
            payload=HaltAlertPayload(
                symbol=trader_symbol,
                position_id=position_id,
                halt_reason=halt_reason,
                halt_mode=resolve_halt_mode(halt_reason),
                side=side,
                layer=layer,
                has_position=True,
                sidecar_dirty=sidecar_dirty,
                manual_intervention_required=manual_intervention_required,
                message=message,
                extra=extra or {},
            ),
        )
    except Exception:
        logger.exception("SIDECAR_HALT_ALERT_EXCEPTION | position_id=%s halt_reason=%s", position_id, halt_reason)


async def attach_sidecar_after_combined_entry(
        *,
        trader: Trader,
        strategy_state: StrategyPositionState,
        execution_state: live_runtime_types.ExecutionState,
        intent: TradeIntent,
        sidecar_plan: SidecarExecutionPlan,
        journal: LiveTradeJournal,
        state_store: LiveStateStore,
        trader_symbol: str,
        fee_buffer_pct: float = position_cost_runtime.DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
        email_sender: Any = None,
        halt_alert_deduper: HaltAlertDeduper | None = None,
) -> bool:
    if not getattr(strategy_state, "sidecar_enabled_for_position", False):
        return True
    if intent.intent_type not in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}:
        return True
    position_id = execution_state.current_position_id
    contracts = str(sidecar_plan.sidecar_contracts)
    filled_qty = float(sidecar_plan.sidecar_qty)
    position_cost_runtime.record_remaining_entry_notional(
        strategy_state,
        qty=filled_qty,
        price=float(intent.price),
        fee_buffer_pct=fee_buffer_pct,
    )
    leg = sidecar_leg_from_fill(
        leg_id=f"{position_id}:SC:{intent.layer_index}:{intent.ts_ms}",
        position_id=str(position_id or ""),
        layer_index=intent.layer_index,
        side=intent.side,
        entry_price=float(intent.price),
        qty=filled_qty,
        contracts=contracts,
        margin_pct=float(sidecar_plan.sidecar_margin_pct),
        layer_multiplier=float(sidecar_plan.layer_multiplier),
        tp_pct=float(strategy_state.sidecar_tp_pct or 0.0),
        tp_order_id=None,
        ts_ms=int(intent.ts_ms),
    )
    leg["tp_price"] = float(sidecar_plan.sidecar_tp_price)
    leg["sidecar_client_order_id"] = sidecar_plan.client_order_id

    # ── Place sidecar TP with rate-limit-aware retry ──────────────────
    # The leg must never appear in strategy_state.sidecar_legs with
    # status=OPEN + tp_order_id=None.  If the pre-core reconcile runs
    # concurrently it would flag that intermediate state as dirty and
    # halt the position unnecessarily.
    tp_order_id, tp_error = await _place_sidecar_tp_with_rate_limit_retry(
        trader=trader,
        side=intent.side,
        contracts=contracts,
        tp_price=float(leg["tp_price"]),
        client_order_id=sidecar_plan.client_order_id,
    )

    if tp_order_id is not None:
        # ── TP placed successfully → append leg with tp_order_id set ──
        leg["tp_order_id"] = tp_order_id
        leg["updated_ts_ms"] = int(intent.ts_ms)
        strategy_state.sidecar_legs.append(leg)
        sidecar_runtime_state.refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        state_store.save(
            LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol,
                                               strategy_state=strategy_state,
                                               cash_before_position=execution_state.cash_before_position))
        journal.append("SIDECAR_LEG_OPENED", dict(leg), position_id=position_id)
        journal.append("SIDECAR_TP_PLACED", dict(leg), position_id=position_id)
        return True

    # ── TP placement failed after all retries ─────────────────────────
    is_rate_limit = _is_okx_rate_limit_error(Exception(tp_error)) if tp_error else False

    if is_rate_limit:
        # ── Rate limit failure ────────────────────────────────────────
        fail_action = _sidecar_tp_rate_limit_fail_action()
        leg = mark_sidecar_leg_open_unprotected(leg, int(intent.ts_ms), warning_recorded=True)
        strategy_state.sidecar_legs.append(leg)
        strategy_state.sidecar_dirty = True
        execution_state.trading_halted = True
        execution_state.halt_reason = "sidecar_tp_place_rate_limited_unprotected"
        strategy_state.sidecar_halt_reason = "sidecar_tp_place_rate_limited_unprotected"

        sidecar_runtime_state.refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        state_store.save(
            LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol,
                                               strategy_state=strategy_state,
                                               cash_before_position=execution_state.cash_before_position))

        if fail_action == "MARKET_EXIT":
            exit_ok, exit_message = await trader.market_exit_remaining_position_with_retries(
                intent.side,
                retry_count=int(os.getenv("SIDECAR_TP_FAIL_MARKET_EXIT_RETRY_COUNT",
                                          os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3"))),
                context="sidecar_tp_place_rate_limited",
                retry_interval_seconds=float(os.getenv("SIDECAR_TP_FAIL_MARKET_EXIT_RETRY_INTERVAL_SECONDS", "0.5")),
            )
            if exit_ok:
                execution_state.halt_reason = "sidecar_tp_rate_limited_market_exit_waiting_flat"
                strategy_state.sidecar_halt_reason = "sidecar_tp_rate_limited_market_exit_waiting_flat"
            manual_intervention_required = not exit_ok
            journal.append(
                "SIDECAR_TP_PLACE_RATE_LIMITED",
                {
                    **dict(leg),
                    "error": tp_error,
                    "error_type": "rate_limit",
                    "market_exit_attempted": True,
                    "market_exit_ok": exit_ok,
                    "market_exit_message": exit_message,
                    "sidecar_contracts": str(sidecar_plan.sidecar_contracts),
                    "sidecar_qty": sidecar_plan.sidecar_qty,
                    "core_contracts": str(sidecar_plan.core_contracts),
                    "net_contracts": str(sidecar_plan.total_contracts),
                    "total_contracts": str(sidecar_plan.total_contracts),
                    "sidecar_status": SidecarLegStatus.OPEN_UNPROTECTED.value,
                    "manual_intervention_required": manual_intervention_required,
                },
                position_id=position_id,
            )
            logger.error(
                "SIDECAR_TP_PLACE_RATE_LIMITED | position_id=%s leg_id=%s error=%s fail_action=MARKET_EXIT market_exit_ok=%s manual_intervention_required=%s",
                position_id,
                leg.get("leg_id"),
                tp_error,
                exit_ok,
                manual_intervention_required,
            )
            await _send_sidecar_failure_halt_alert(
                email_sender=email_sender,
                halt_alert_deduper=halt_alert_deduper,
                trader_symbol=trader_symbol,
                position_id=position_id,
                halt_reason=execution_state.halt_reason or "sidecar_tp_place_rate_limited_unprotected",
                side=intent.side,
                layer=intent.layer_index,
                sidecar_dirty=True,
                manual_intervention_required=manual_intervention_required,
                message=f"Sidecar TP rate-limited; MARKET_EXIT attempted. exit_ok={exit_ok}.",
                extra={"tp_error": tp_error, "fail_action": "MARKET_EXIT", "exit_ok": exit_ok, "exit_message": exit_message},
            )
        else:
            # HALT_ONLY (default)
            manual_intervention_required = True
            journal.append(
                "SIDECAR_TP_PLACE_RATE_LIMITED",
                {
                    **dict(leg),
                    "error": tp_error,
                    "error_type": "rate_limit",
                    "market_exit_attempted": False,
                    "sidecar_contracts": str(sidecar_plan.sidecar_contracts),
                    "sidecar_qty": sidecar_plan.sidecar_qty,
                    "core_contracts": str(sidecar_plan.core_contracts),
                    "net_contracts": str(sidecar_plan.total_contracts),
                    "total_contracts": str(sidecar_plan.total_contracts),
                    "sidecar_status": SidecarLegStatus.OPEN_UNPROTECTED.value,
                    "manual_intervention_required": True,
                    "fail_action": "HALT_ONLY",
                },
                position_id=position_id,
            )
            logger.error(
                "SIDECAR_TP_PLACE_RATE_LIMITED | position_id=%s leg_id=%s error=%s fail_action=HALT_ONLY trading_halted=true halt_reason=sidecar_tp_place_rate_limited_unprotected manual_intervention_required=true",
                position_id,
                leg.get("leg_id"),
                tp_error,
            )
            await _send_sidecar_failure_halt_alert(
                email_sender=email_sender,
                halt_alert_deduper=halt_alert_deduper,
                trader_symbol=trader_symbol,
                position_id=position_id,
                halt_reason=execution_state.halt_reason or "sidecar_tp_place_rate_limited_unprotected",
                side=intent.side,
                layer=intent.layer_index,
                sidecar_dirty=True,
                manual_intervention_required=True,
                message="Sidecar TP rate-limited after all retries; HALT_ONLY. Core position may be unprotected.",
                extra={"tp_error": tp_error, "fail_action": "HALT_ONLY"},
            )
        return False

    # ── Non-rate-limit (irrecoverable) failure ────────────────────────
    leg = mark_sidecar_leg_open_unprotected(leg, int(intent.ts_ms), warning_recorded=True)
    strategy_state.sidecar_legs.append(leg)
    execution_state.trading_halted = True
    strategy_state.sidecar_dirty = True
    exit_ok, exit_message = await trader.market_exit_remaining_position_with_retries(
        intent.side,
        retry_count=int(os.getenv("SIDECAR_TP_FAIL_MARKET_EXIT_RETRY_COUNT",
                                  os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3"))),
        context="sidecar_tp_place_failed",
        retry_interval_seconds=float(os.getenv("SIDECAR_TP_FAIL_MARKET_EXIT_RETRY_INTERVAL_SECONDS", "0.5")),
    )
    if exit_ok:
        execution_state.halt_reason = "sidecar_tp_place_failed_market_exit_waiting_flat"
        strategy_state.sidecar_halt_reason = "sidecar_tp_place_failed_market_exit_waiting_flat"
    else:
        execution_state.halt_reason = "sidecar_tp_place_failed"
        strategy_state.sidecar_halt_reason = "sidecar_tp_place_failed"
    sidecar_runtime_state.refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
    state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol,
                                                        strategy_state=strategy_state,
                                                        cash_before_position=execution_state.cash_before_position))
    manual_intervention_required = not exit_ok
    journal.append(
        "SIDECAR_TP_PLACE_FAILED",
        {
            **dict(leg),
            "error": tp_error,
            "error_type": "irrecoverable",
            "market_exit_attempted": True,
            "market_exit_ok": exit_ok,
            "market_exit_message": exit_message,
            "sidecar_contracts": str(sidecar_plan.sidecar_contracts),
            "sidecar_qty": sidecar_plan.sidecar_qty,
            "core_contracts": str(sidecar_plan.core_contracts),
            "net_contracts": str(sidecar_plan.total_contracts),
            "total_contracts": str(sidecar_plan.total_contracts),
            "sidecar_status": SidecarLegStatus.OPEN_UNPROTECTED.value,
            "manual_intervention_required": manual_intervention_required,
        },
        position_id=position_id,
    )
    logger.error(
        "SIDECAR_TP_PLACE_FAILED | position_id=%s leg_id=%s error=%s error_type=irrecoverable market_exit_attempted=true market_exit_ok=%s manual_intervention_required=%s",
        position_id,
        leg.get("leg_id"),
        tp_error,
        exit_ok,
        manual_intervention_required,
    )
    await _send_sidecar_failure_halt_alert(
        email_sender=email_sender,
        halt_alert_deduper=halt_alert_deduper,
        trader_symbol=trader_symbol,
        position_id=position_id,
        halt_reason=execution_state.halt_reason or "sidecar_tp_place_failed",
        side=intent.side,
        layer=intent.layer_index,
        sidecar_dirty=True,
        manual_intervention_required=manual_intervention_required,
        message="Sidecar TP placement failed (irrecoverable error); market exit attempted. Core entry may remain open.",
        extra={"tp_error": tp_error, "exit_ok": exit_ok, "exit_message": exit_message},
    )
    return False


async def execute_sidecar_after_core_entry(
        *,
        trader: Trader,
        strategy_state: StrategyPositionState,
        execution_state: live_runtime_types.ExecutionState,
        intent: TradeIntent,
        journal: LiveTradeJournal,
        state_store: LiveStateStore,
        trader_symbol: str,
) -> bool:
    logger.error(
        "SIDECAR_LEGACY_AFTER_CORE_ENTRY_DISABLED | position_id=%s intent_type=%s side=%s layer=%s",
        execution_state.current_position_id,
        intent.intent_type,
        intent.side,
        intent.layer_index,
    )
    return False
