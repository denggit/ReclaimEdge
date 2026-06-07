from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from src.execution.trader import PositionSnapshot, Trader
from src.live import runtime_types as live_runtime_types
from src.live import time_utils as live_time_utils
from src.live.account_sync import flat_balance as live_flat_balance
from src.position_management import core_position_view as core_position_view_helpers
from src.position_management.sidecar import pre_core_reconcile as sidecar_pre_core_reconcile
from src.position_management.sidecar import runtime_state as sidecar_runtime_state
from src.position_management.sidecar.model import (
    SidecarLegStatus,
    sidecar_open_contracts,
    sidecar_open_qty,
)
from src.position_management.sidecar.reconciler import build_core_position_view
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.risk.simple_position_sizer import SimplePositionSizer
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AccountSyncPreCorePositionResult:
    cash: float
    equity: float
    position: PositionSnapshot
    core_position: PositionSnapshot
    current_position_key: Any
    pending_order_count: int
    force_close_sidecar: bool
    pending_flat_payload: dict[str, Any] | None
    cash_transfer_payload: dict[str, Any] | None
    cash_drift_payload: dict[str, Any] | None
    sidecar_reconciled_this_sync: bool
    sidecar_state_changed_this_sync: bool
    last_account_sync: float
    last_logged_cash: float
    last_logged_equity: float
    last_cash_event_log: float
    last_flat_detected_monotonic: float
    sidecar_tp_filled_count: int = 0
    sidecar_tp_filled_leg_ids: tuple[str, ...] = ()
    sidecar_tp_filled_order_ids: tuple[str, ...] = ()
    sidecar_tp_filled_qty: float = 0.0


async def run_account_sync_pre_core_position_phase(
        *,
        state_lock: asyncio.Lock,
        account_snapshot: live_runtime_types.AccountSnapshot,
        execution_state: live_runtime_types.ExecutionState,
        trader: Trader,
        sizer: SimplePositionSizer,
        strategy: BollCvdShockReclaimStrategy,
        journal: LiveTradeJournal,
        state_store: LiveStateStore,
        now: float,
        last_account_sync: float,
        account_sync_seconds: float,
        cash_transfer_detect_enabled: bool,
        cash_transfer_min_delta_usdt: float,
        cash_transfer_settle_seconds: float,
        cash_transfer_after_flat_cooldown_seconds: float,
        cash_drift_min_delta_usdt: float,
        cash_event_log_interval_seconds: float,
        cash_log_min_delta_usdt: float,
        last_logged_cash: float,
        last_logged_equity: float,
        last_cash_event_log: float,
        last_flat_detected_monotonic: float,
) -> AccountSyncPreCorePositionResult:
    cash = account_snapshot.cash
    equity = account_snapshot.equity
    if now - last_account_sync >= account_sync_seconds:
        equity = await trader.fetch_usdt_equity()
        cash = await live_flat_balance.fetch_usdt_cash_balance(trader)
        last_account_sync = now

    position = await trader.fetch_position_snapshot()
    core_position = position
    current_position_key: Any = core_position_view_helpers.position_log_key(core_position)
    pending_flat_payload: dict[str, Any] | None = None
    cash_transfer_payload: dict[str, Any] | None = None
    cash_drift_payload: dict[str, Any] | None = None
    force_close_sidecar = False
    # ── Pre-core sidecar reconciliation ──────────────────────────
    # Sidecar TP may have already filled on OKX but local state still
    # counts it as open.  If we compute core_position = OKX_net -
    # stale_sidecar_open_qty first and discover the fill later via
    # monitor_sidecar_orders_once, the stale core view can incorrectly
    # trigger TP progress markers or pollute strategy cost.
    #
    # Reconcile sidecar orders NOW so that refresh_sidecar_state_totals
    # and build_core_position_view inside the main state_lock block
    # always see up-to-date sidecar_open_qty.
    _sidecar_pre_core_result = await sidecar_pre_core_reconcile.reconcile_sidecar_orders_before_core_view(
        trader=trader,
        strategy=strategy,
        execution_state=execution_state,
        journal=journal,
        state_store=state_store,
        trader_symbol=trader.symbol,
        ts_ms=live_time_utils.utc_ms(),
        state_lock=state_lock,
    )
    sidecar_reconciled_this_sync = _sidecar_pre_core_result.queried
    sidecar_state_changed_this_sync = _sidecar_pre_core_result.changed
    _sidecar_tp_filled_count = _sidecar_pre_core_result.sidecar_tp_filled_count
    _sidecar_tp_filled_leg_ids = _sidecar_pre_core_result.sidecar_tp_filled_leg_ids
    _sidecar_tp_filled_order_ids = _sidecar_pre_core_result.sidecar_tp_filled_order_ids
    _sidecar_tp_filled_qty = _sidecar_pre_core_result.sidecar_tp_filled_qty
    # ── End pre-core reconciliation ──────────────────────────────
    async with state_lock:
        pending_order_count = execution_state.pending_order_count
        sidecar_runtime_state.refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        if sidecar_runtime_state.open_sidecar_legs_exceed_limit(strategy.state,
                                                                int(os.getenv("SIDECAR_MAX_LEGS", "10"))):
            execution_state.trading_halted = True
            execution_state.halt_reason = "sidecar_open_legs_exceed_max"
            strategy.state.sidecar_dirty = True
            strategy.state.sidecar_halt_reason = "sidecar_open_legs_exceed_max"
            if hasattr(journal, "append"):
                journal.append(
                    "SIDECAR_OPEN_LEGS_EXCEED_MAX",
                    {
                        "open_leg_count": sum(
                            1
                            for leg in strategy.state.sidecar_legs
                            if
                            leg.get("status") in {SidecarLegStatus.OPEN.value, SidecarLegStatus.OPEN_UNPROTECTED.value}
                        ),
                        "sidecar_max_legs": int(os.getenv("SIDECAR_MAX_LEGS", "10")),
                        "manual_intervention_required": True,
                    },
                    position_id=execution_state.current_position_id,
                )
        open_sidecar_qty = sidecar_open_qty(strategy.state.sidecar_legs)
        core_position = build_core_position_view(position, open_sidecar_qty,
                                                 sidecar_open_contracts(strategy.state.sidecar_legs))
        core_position_view_helpers.apply_core_position_view_to_state(strategy.state, core_position)
        current_position_key = core_position_view_helpers.position_log_key(core_position)
        if core_position_view_helpers.sidecar_position_mismatch(position, strategy.state):
            execution_state.trading_halted = True
            execution_state.halt_reason = "core_sidecar_position_mismatch"
            strategy.state.sidecar_dirty = True
            strategy.state.sidecar_halt_reason = "core_sidecar_position_mismatch"
            if hasattr(journal, "append"):
                journal.append(
                    "CORE_SIDECAR_POSITION_MISMATCH",
                    {
                        "okx_eth_qty": position.eth_qty,
                        "core_eth_qty": core_position.eth_qty,
                        "sidecar_open_qty": open_sidecar_qty,
                        "manual_intervention_required": True,
                    },
                    position_id=execution_state.current_position_id,
                )
            logger.error(
                "CORE_SIDECAR_POSITION_MISMATCH | position_id=%s okx_eth_qty=%.8f core_eth_qty=%.8f sidecar_open_qty=%.8f trading_halted=true manual_intervention_required=true",
                execution_state.current_position_id,
                position.eth_qty,
                core_position.eth_qty,
                open_sidecar_qty,
            )
        force_close_sidecar = bool(
            pending_order_count == 0
            and not core_position.has_position
            and open_sidecar_qty > 0
            and getattr(strategy.state, "sidecar_enabled_for_position", False)
            and getattr(sizer.config, "sidecar_close_when_core_flat", True)
        )
        flat_transition_detected = (
                pending_order_count == 0
                and not core_position.has_position
                and not force_close_sidecar
                and strategy.state.layers > 0
        )
        if flat_transition_detected:
            pending_flat_payload = {
                "position_id": execution_state.current_position_id,
                "symbol": trader.symbol,
                "side": strategy.state.side,
                "cash_before_position": execution_state.cash_before_position,
                "reason": "OKX position is flat. TP filled or manual close detected.",
                "layers": strategy.state.layers,
                "avg_entry_price": strategy.state.avg_entry_price,
                "last_tp_price": strategy.state.tp_price,
                "last_partial_tp_price": getattr(strategy.state, "partial_tp_price", None),
                "last_tp_plan": getattr(strategy.state, "tp_plan", "SINGLE"),
                "partial_tp_consumed": getattr(strategy.state, "partial_tp_consumed", False),
                "near_tp_protective_sl_order_id": getattr(strategy.state, "near_tp_protective_sl_order_id", None),
                "middle_runner_protective_sl_order_id": getattr(strategy.state, "middle_runner_protective_sl_order_id",
                                                                None),
                "three_stage_post_tp1_protective_sl_order_id": getattr(strategy.state,
                                                                       "three_stage_post_tp1_protective_sl_order_id",
                                                                       None),
                "trend_runner_sl_order_id": getattr(strategy.state, "trend_runner_sl_order_id", None),
                "trend_runner_exit_reason": getattr(strategy.state, "trend_runner_exit_reason", None),
            }
            execution_state.trading_halted = True
            last_flat_detected_monotonic = now
            logger.warning("POSITION_SYNC_CHANGED | flat_on_okx=true. Confirming settled balance before FLAT journal.")
        else:
            account_snapshot.position = core_position
            account_snapshot.cash = cash
            account_snapshot.equity = equity
            account_snapshot.updated_monotonic = time.monotonic()
            account_snapshot.updated_ts_ms = live_time_utils.utc_ms()
            account_snapshot.version += 1
            trader.account_equity_usdt = equity
            sizer.update_account_equity(equity)

            cash_delta = cash - last_logged_cash
            seconds_since_last_order = (
                cash_transfer_settle_seconds
                if execution_state.last_order_ts_ms == 0
                else max((live_time_utils.utc_ms() - execution_state.last_order_ts_ms) / 1000, 0.0)
            )
            unsafe_reasons = []
            if pending_order_count > 0:
                unsafe_reasons.append("pending_order")
            if core_position.has_position:
                unsafe_reasons.append("has_position")
            if strategy.state.layers != 0:
                unsafe_reasons.append("strategy_layers")
            if execution_state.current_position_id is not None:
                unsafe_reasons.append("current_position_id")
            if seconds_since_last_order < cash_transfer_settle_seconds:
                unsafe_reasons.append("order_settle")
            in_flat_settle_cooldown = (
                    last_flat_detected_monotonic > 0
                    and now - last_flat_detected_monotonic < cash_transfer_after_flat_cooldown_seconds
            )
            if in_flat_settle_cooldown:
                unsafe_reasons.append("flat_settle_cooldown")
            safe_for_cash_transfer = (
                    cash_transfer_detect_enabled
                    and pending_order_count == 0
                    and not core_position.has_position
                    and strategy.state.layers == 0
                    and execution_state.current_position_id is None
                    and seconds_since_last_order >= cash_transfer_settle_seconds
                    and not in_flat_settle_cooldown
                    and abs(cash_delta) >= cash_transfer_min_delta_usdt
            )
            if safe_for_cash_transfer:
                direction = "DEPOSIT" if cash_delta > 0 else "WITHDRAWAL"
                cash_transfer_payload = {
                    "direction": direction,
                    "amount": cash_delta,
                    "cash_before": last_logged_cash,
                    "cash_after": cash,
                    "equity_before": last_logged_equity,
                    "equity_after": equity,
                    "reason": "safe_flat_account_sync",
                }
                if now - last_cash_event_log >= cash_event_log_interval_seconds:
                    logger.warning(
                        "CASH_TRANSFER_DETECTED | direction=%s amount=%.4f cash_before=%.4f cash_after=%.4f",
                        direction,
                        cash_delta,
                        last_logged_cash,
                        cash,
                    )
                    last_cash_event_log = now
            elif unsafe_reasons and abs(cash_delta) >= cash_drift_min_delta_usdt:
                if _sidecar_tp_filled_count > 0:
                    drift_reason = "position_cash_change:sidecar_tp_filled;unsafe_state:" + ",".join(unsafe_reasons)
                    cash_drift_payload = {
                        "amount": cash_delta,
                        "cash_before": last_logged_cash,
                        "cash_after": cash,
                        "equity_before": last_logged_equity,
                        "equity_after": equity,
                        "reason": drift_reason,
                        "sidecar_tp_filled_count": _sidecar_tp_filled_count,
                        "sidecar_tp_filled_leg_ids": list(_sidecar_tp_filled_leg_ids),
                        "sidecar_tp_filled_order_ids": list(_sidecar_tp_filled_order_ids),
                        "sidecar_tp_filled_qty": _sidecar_tp_filled_qty,
                    }
                else:
                    drift_reason = "unsafe_state:" + ",".join(unsafe_reasons)
                    cash_drift_payload = {
                        "amount": cash_delta,
                        "cash_before": last_logged_cash,
                        "cash_after": cash,
                        "equity_before": last_logged_equity,
                        "equity_after": equity,
                        "reason": drift_reason,
                    }
                if now - last_cash_event_log >= cash_event_log_interval_seconds:
                    logger.warning(
                        "ACCOUNT_CASH_DRIFT | amount=%.4f cash_before=%.4f cash_after=%.4f reason=%s",
                        cash_delta,
                        last_logged_cash,
                        cash,
                        drift_reason,
                    )
                    last_cash_event_log = now

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
                last_logged_equity = equity
            elif cash_transfer_payload is not None or cash_drift_payload is not None:
                last_logged_cash = cash
                last_logged_equity = equity

    return AccountSyncPreCorePositionResult(
        cash=cash,
        equity=equity,
        position=position,
        core_position=core_position,
        current_position_key=current_position_key,
        pending_order_count=pending_order_count,
        force_close_sidecar=force_close_sidecar,
        pending_flat_payload=pending_flat_payload,
        cash_transfer_payload=cash_transfer_payload,
        cash_drift_payload=cash_drift_payload,
        sidecar_reconciled_this_sync=sidecar_reconciled_this_sync,
        sidecar_state_changed_this_sync=sidecar_state_changed_this_sync,
        last_account_sync=last_account_sync,
        last_logged_cash=last_logged_cash,
        last_logged_equity=last_logged_equity,
        last_cash_event_log=last_cash_event_log,
        last_flat_detected_monotonic=last_flat_detected_monotonic,
        sidecar_tp_filled_count=_sidecar_tp_filled_count,
        sidecar_tp_filled_leg_ids=_sidecar_tp_filled_leg_ids,
        sidecar_tp_filled_order_ids=_sidecar_tp_filled_order_ids,
        sidecar_tp_filled_qty=_sidecar_tp_filled_qty,
    )
