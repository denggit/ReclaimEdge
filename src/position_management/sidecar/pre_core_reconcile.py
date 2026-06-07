from __future__ import annotations

import asyncio
import os
from typing import Any

from src.execution.trader import Trader
from src.live import runtime_types as live_runtime_types
from src.position_management import cost_runtime as position_cost_runtime
from src.position_management.sidecar import runtime_state as sidecar_runtime_state
from src.position_management.sidecar.fill_telemetry import (
    SidecarFillTelemetry,
    build_sidecar_fill_telemetry,
    log_sidecar_tp_filled,
    merge_sidecar_fill_telemetry,
    normalized_sidecar_fill_payload,
)
from src.position_management.sidecar.model import SidecarLegStatus, sidecar_open_qty
from src.position_management.sidecar.reconciler import (
    is_sidecar_dirty_missing_tp_order,
    mark_sidecar_leg_tp_filled,
    mark_sidecar_leg_unknown_halted,
)
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)


async def reconcile_sidecar_orders_before_core_view(
        *,
        trader: Trader,
        strategy: BollCvdShockReclaimStrategy,
        execution_state: live_runtime_types.ExecutionState,
        journal: LiveTradeJournal,
        state_store: LiveStateStore,
        trader_symbol: str,
        ts_ms: int,
        state_lock: asyncio.Lock,
) -> live_runtime_types.SidecarPreCoreReconcileResult:
    """Reconcile sidecar TP order status BEFORE constructing core_position view.

    When Sidecar is enabled and OPEN sidecar legs exist, this must be called
    before computing core_position = OKX_net - sidecar_open_qty. Otherwise a
    sidecar TP that already filled on OKX (but not yet reflected in local state)
    would cause core_position to be understated, which can incorrectly trigger
    TP progress markers or pollute strategy average entry.

    Returns live_runtime_types.SidecarPreCoreReconcileResult:
      - queried: True if we performed any REST order status fetch for OPEN legs.
      - changed: True if any sidecar state was modified and saved.

    Sets trading_halted if unrecoverable state is detected.
    """
    # Pending orders mean core position is in flux — do not reconcile sidecar
    # orders or advance core state.  Return False / False to allow the caller
    # to fall through safely without blocking the sync cycle.
    if execution_state.pending_order_count > 0:
        return live_runtime_types.SidecarPreCoreReconcileResult(queried=False, changed=False)

    if not getattr(strategy.state, "sidecar_enabled_for_position", False):
        return live_runtime_types.SidecarPreCoreReconcileResult(queried=False, changed=False)

    # --- Phase 1: handle dirty / missing TP orders under lock (no network) ---
    dirty_changed = False
    async with state_lock:
        for index, leg in enumerate(list(strategy.state.sidecar_legs)):
            if leg.get("status") != SidecarLegStatus.OPEN.value:
                continue
            if is_sidecar_dirty_missing_tp_order(leg):
                execution_state.trading_halted = True
                execution_state.halt_reason = "sidecar_tp_order_missing_or_unknown"
                strategy.state.sidecar_dirty = True
                strategy.state.sidecar_halt_reason = "sidecar_tp_order_missing_or_unknown"
                if not leg.get("warning_recorded") and hasattr(journal, "append"):
                    journal.append("SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN", dict(leg),
                                   position_id=execution_state.current_position_id)
                strategy.state.sidecar_legs[index] = mark_sidecar_leg_unknown_halted(leg, ts_ms)
                dirty_changed = True
        if dirty_changed:
            sidecar_runtime_state.refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
            state_store.save(LiveStateStore.from_strategy_state(
                position_id=execution_state.current_position_id,
                symbol=trader_symbol,
                strategy_state=strategy.state,
                cash_before_position=execution_state.cash_before_position,
            ))
        # Snapshot remaining OPEN legs for network queries
        open_legs: list[tuple[int, str, str]] = []
        for index, leg in enumerate(strategy.state.sidecar_legs):
            if leg.get("status") != SidecarLegStatus.OPEN.value:
                continue
            if is_sidecar_dirty_missing_tp_order(leg):
                continue
            tp_order_id = leg.get("tp_order_id")
            if not tp_order_id:
                continue
            open_legs.append((index, str(tp_order_id), str(leg.get("leg_id", ""))))
        position_id = execution_state.current_position_id
        cash_before_position = execution_state.cash_before_position

    if not open_legs:
        return live_runtime_types.SidecarPreCoreReconcileResult(queried=False, changed=dirty_changed)

    # --- Phase 2: query order status (outside lock) ---
    leg_updates: list[tuple[int, str, dict[str, Any], str]] = []
    for index, order_id, leg_id in open_legs:
        status = await trader.fetch_sidecar_order_status(order_id)
        order_status = status.get("status")
        if order_status != "OPEN":
            leg_updates.append((index, order_status, status, leg_id))

    if not leg_updates:
        # We queried open legs but found no status changes.
        return live_runtime_types.SidecarPreCoreReconcileResult(queried=True, changed=False)

    # --- Phase 3: apply updates under lock ---
    changed = dirty_changed
    _telemetry_items: list[SidecarFillTelemetry] = []
    async with state_lock:
        for index, order_status, status_dict, expected_leg_id in leg_updates:
            if index >= len(strategy.state.sidecar_legs):
                continue
            leg = strategy.state.sidecar_legs[index]
            if leg.get("status") != SidecarLegStatus.OPEN.value:
                continue
            if leg.get("leg_id") != expected_leg_id:
                continue

            if order_status == "FILLED":
                position_cost_runtime.record_sidecar_tp_fill_exit(
                    strategy.state,
                    leg,
                    status_dict,
                    fee_buffer_pct=getattr(
                        getattr(strategy, "config", None),
                        "breakeven_fee_buffer_pct",
                        position_cost_runtime.DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
                    ),
                )
                strategy.state.sidecar_legs[index] = mark_sidecar_leg_tp_filled(leg, ts_ms)
                if hasattr(journal, "append"):
                    _normalized = normalized_sidecar_fill_payload(leg, status_dict)
                    journal.append(
                        "SIDECAR_TP_FILLED",
                        {
                            **dict(leg),
                            **status_dict,
                            "filled_contracts": _normalized["filled_contracts"],
                            "filled_eth_qty": _normalized["filled_eth_qty"],
                            "filled_notional_usdt": _normalized["filled_notional_usdt"],
                            "filled_qty_unit": "contracts_from_okx_accFillSz",
                        },
                        position_id=position_id,
                    )
                changed = True

                # Compute real remaining sidecar open qty after this fill
                _open_qty_after = sidecar_open_qty(strategy.state.sidecar_legs)

                # Build and accumulate fill telemetry
                _tp_telemetry = build_sidecar_fill_telemetry(
                    source="pre_core_reconcile",
                    leg=leg,
                    status=status_dict,
                )
                _telemetry_items.append(_tp_telemetry)
                log_sidecar_tp_filled(
                    logger,
                    position_id=position_id,
                    telemetry=_tp_telemetry,
                    leg=leg,
                    status=status_dict,
                    sidecar_open_qty_after=_open_qty_after,
                )

                # Sidecar TP reduces OKX net position → existing global SL orders
                # may now exceed current net position. Must halt for manual reconciliation.
                active_global_sl_orders: list[str] = []
                for sl_field in (
                        "near_tp_protective_sl_order_id",
                        "middle_runner_protective_sl_order_id",
                        "three_stage_post_tp1_protective_sl_order_id",
                        "trend_runner_sl_order_id",
                ):
                    # Use trader fallback (same as monitor_sidecar_orders_once)
                    # because SL orders may have been placed by startup recovery
                    # or by a previous session and only tracked on the trader.
                    sl_order_id = (
                            getattr(strategy.state, sl_field, None)
                            or getattr(trader, sl_field, None)
                    )
                    if sl_order_id:
                        active_global_sl_orders.append(f"{sl_field}={sl_order_id}")
                if active_global_sl_orders:
                    execution_state.trading_halted = True
                    execution_state.halt_reason = "sidecar_tp_filled_requires_global_sl_reconcile"
                    strategy.state.sidecar_dirty = True
                    strategy.state.sidecar_halt_reason = "sidecar_tp_filled_requires_global_sl_reconcile"
                    if hasattr(journal, "append"):
                        journal.append(
                            "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE",
                            {
                                "active_global_sl_orders": active_global_sl_orders,
                                "trading_halted": True,
                                "halt_reason": "sidecar_tp_filled_requires_global_sl_reconcile",
                                "manual_intervention_required": True,
                            },
                            position_id=position_id,
                        )
                    logger.error(
                        "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE | position_id=%s leg_id=%s active_global_sl_orders=%s trading_halted=true halt_reason=sidecar_tp_filled_requires_global_sl_reconcile manual_intervention_required=true",
                        position_id,
                        leg.get("leg_id"),
                        active_global_sl_orders,
                    )
                continue

            if order_status in {"CANCELED", "NOT_FOUND", "UNKNOWN"}:
                # Without a verified core view we cannot determine whether the
                # remaining OKX position is core-only or core+sidecar. Halt.
                execution_state.trading_halted = True
                execution_state.halt_reason = "sidecar_tp_order_missing_or_unknown"
                strategy.state.sidecar_dirty = True
                strategy.state.sidecar_halt_reason = "sidecar_tp_order_missing_or_unknown"
                if not leg.get("warning_recorded") and hasattr(journal, "append"):
                    journal.append(
                        "SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN",
                        {**dict(leg), **status_dict, "manual_intervention_required": True},
                        position_id=position_id,
                    )
                strategy.state.sidecar_legs[index] = mark_sidecar_leg_unknown_halted(leg, ts_ms)
                changed = True
                logger.error(
                    "SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN | position_id=%s leg_id=%s status=%s manual_intervention_required=true",
                    position_id,
                    leg.get("leg_id"),
                    order_status,
                )

        if changed:
            sidecar_runtime_state.refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
            state_store.save(LiveStateStore.from_strategy_state(
                position_id=position_id,
                symbol=trader_symbol,
                strategy_state=strategy.state,
                cash_before_position=cash_before_position,
            ))

    _merged = merge_sidecar_fill_telemetry(_telemetry_items)
    return live_runtime_types.SidecarPreCoreReconcileResult(
        queried=True,
        changed=changed,
        sidecar_tp_filled_count=_merged.filled_count,
        sidecar_tp_filled_leg_ids=_merged.filled_leg_ids,
        sidecar_tp_filled_order_ids=_merged.filled_order_ids,
        sidecar_tp_filled_qty=_merged.filled_eth_qty,
        sidecar_tp_filled_contracts=_merged.filled_contracts,
    )
