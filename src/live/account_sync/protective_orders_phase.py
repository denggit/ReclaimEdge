from __future__ import annotations

import asyncio
import copy
import os
from dataclasses import dataclass
from typing import Any

from src.execution.trader import Trader
from src.live import runtime_types as live_runtime_types
from src.position_management import runner_live_helpers
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AccountSyncProtectiveOrdersResult:
    save_state_payload: tuple[str | None, StrategyPositionState, float | None] | None


async def run_account_sync_protective_orders_phase(
        *,
        state_lock: asyncio.Lock,
        execution_state: live_runtime_types.ExecutionState,
        trader: Trader,
        strategy: BollCvdShockReclaimStrategy,
        journal: LiveTradeJournal,
        state_store: LiveStateStore,
        save_state_payload: tuple[str | None, StrategyPositionState, float | None] | None,
        three_stage_post_tp1_cancel_payload: dict[str, Any] | None,
        three_stage_post_tp1_sl_payload: dict[str, Any] | None,
        middle_runner_sl_payload: dict[str, Any] | None,
        middle_runner_activation_payload: dict[str, Any] | None,
) -> AccountSyncProtectiveOrdersResult:
    if three_stage_post_tp1_cancel_payload is not None:
        old_order_id = three_stage_post_tp1_cancel_payload.get("protective_sl_order_id")
        cancel_ok = True
        if old_order_id:
            try:
                cancel_ok = await trader.cancel_three_stage_post_tp1_protective_stop(old_order_id)
            except Exception:
                cancel_ok = False
                logger.exception(
                    "THREE_STAGE_TP1_PROTECTIVE_SL_CANCEL_FAILED_ON_TP2 | position_id=%s algoId=%s cancel_exception=true manual_intervention_required=true",
                    three_stage_post_tp1_cancel_payload.get("position_id"),
                    old_order_id,
                )
        if cancel_ok:
            async with state_lock:
                strategy.state.three_stage_post_tp1_protective_sl_order_id = None
                strategy.state.three_stage_post_tp1_protective_sl_price = None
                strategy.state.three_stage_post_tp1_protected = False
                if (
                        three_stage_post_tp1_cancel_payload.get("pending_halt_applied")
                        and execution_state.trading_halted
                        and execution_state.halt_reason == runner_live_helpers.THREE_STAGE_CANCEL_PENDING_HALT_REASON
                ):
                    execution_state.trading_halted = False
                    execution_state.halt_reason = None
                save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state),
                                      execution_state.cash_before_position)
            if hasattr(journal, "append"):
                journal.append(
                    "THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED_ON_TP2",
                    {
                        **three_stage_post_tp1_cancel_payload,
                        "cancel_ok": True,
                        "reason": "three_stage_tp2_filled",
                    },
                    position_id=three_stage_post_tp1_cancel_payload.get("position_id"),
                )
            logger.warning(
                "THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED_ON_TP2 | position_id=%s algoId=%s cancel_ok=true",
                three_stage_post_tp1_cancel_payload.get("position_id"),
                old_order_id,
            )
        else:
            async with state_lock:
                strategy.state.three_stage_post_tp1_protective_sl_order_id = old_order_id
                strategy.state.three_stage_post_tp1_protective_sl_price = three_stage_post_tp1_cancel_payload.get(
                    "protective_sl_price")
                strategy.state.three_stage_post_tp1_protected = True
                execution_state.trading_halted = True
                execution_state.halt_reason = "three_stage_post_tp1_sl_cancel_failed_on_tp2"
                save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state),
                                      execution_state.cash_before_position)
            if hasattr(journal, "append"):
                journal.append(
                    "THREE_STAGE_TP1_PROTECTIVE_SL_CANCEL_FAILED_ON_TP2",
                    {
                        **three_stage_post_tp1_cancel_payload,
                        "critical": True,
                        "cancel_ok": False,
                        "trading_halted": True,
                        "halt_reason": "three_stage_post_tp1_sl_cancel_failed_on_tp2",
                        "reason": "manual_intervention_required_old_post_tp1_sl_may_remain_on_exchange",
                    },
                    position_id=three_stage_post_tp1_cancel_payload.get("position_id"),
                )
            logger.error(
                "THREE_STAGE_TP1_PROTECTIVE_SL_CANCEL_FAILED_ON_TP2 | position_id=%s algoId=%s protective_sl_price=%s trading_halted=true halt_reason=three_stage_post_tp1_sl_cancel_failed_on_tp2 manual_intervention_required=true",
                three_stage_post_tp1_cancel_payload.get("position_id"),
                old_order_id,
                three_stage_post_tp1_cancel_payload.get("protective_sl_price"),
            )
    if three_stage_post_tp1_sl_payload is not None:
        sl_price = three_stage_post_tp1_sl_payload.get("protective_sl_price")
        sl_order_id = None
        sl_ok = False
        sl_message = "protective_sl_price_missing"
        if sl_price is not None and three_stage_post_tp1_sl_payload.get("side") is not None:
            try:
                sl_ok, sl_order_id, sl_message = await trader.place_three_stage_post_tp1_protective_stop_with_retries(
                    three_stage_post_tp1_sl_payload["side"],
                    three_stage_post_tp1_sl_payload["contracts"],
                    float(sl_price),
                    retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                    retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
                )
            except Exception as exc:
                sl_ok = False
                sl_order_id = None
                sl_message = f"trader_exception: {type(exc).__name__}: {exc}"
        if sl_ok:
            old_sl_order_id = three_stage_post_tp1_sl_payload.get("old_sl_order_id")
            if old_sl_order_id and old_sl_order_id != sl_order_id:
                await trader.cancel_three_stage_post_tp1_protective_stop(old_sl_order_id)
            async with state_lock:
                strategy.state.three_stage_post_tp1_protective_sl_order_id = sl_order_id
                strategy.state.three_stage_post_tp1_protective_sl_price = float(sl_price)
                strategy.state.three_stage_post_tp1_protected = True
                save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state),
                                      execution_state.cash_before_position)
            if hasattr(journal, "append"):
                journal.append(
                    "THREE_STAGE_TP1_PROTECTIVE_SL_PLACED",
                    {
                        "position_id": three_stage_post_tp1_sl_payload.get("position_id"),
                        "side": three_stage_post_tp1_sl_payload.get("side"),
                        "contracts": str(three_stage_post_tp1_sl_payload.get("contracts")),
                        "core_contracts": str(three_stage_post_tp1_sl_payload.get("core_contracts")),
                        "net_contracts": str(three_stage_post_tp1_sl_payload.get("net_contracts")),
                        "sl_contracts": str(three_stage_post_tp1_sl_payload.get("contracts")),
                        "protective_sl_price": sl_price,
                        "protective_sl_order_id": sl_order_id,
                        "current_price": three_stage_post_tp1_sl_payload.get("current_price"),
                        "current_price_source": three_stage_post_tp1_sl_payload.get("current_price_source"),
                        "avg_entry_price": getattr(strategy.state, "avg_entry_price", None),
                        "tp1_price": getattr(strategy.state, "three_stage_tp1_price", None),
                        "tp1_ratio": getattr(strategy.state, "three_stage_tp1_ratio", 0.0),
                        "tp2_price": getattr(strategy.state, "three_stage_tp2_price", None),
                        "tp2_ratio": getattr(strategy.state, "three_stage_tp2_ratio", 0.0),
                        "runner_ratio": getattr(strategy.state, "three_stage_runner_ratio", 0.0),
                        "reason": "three_stage_tp1_filled",
                        "retry_config": "NEAR_TP_PROTECTIVE_SL_RETRY_COUNT/NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS",
                    },
                    position_id=three_stage_post_tp1_sl_payload.get("position_id"),
                )
            logger.warning(
                "THREE_STAGE_TP1_PROTECTIVE_SL_PLACED | position_id=%s side=%s core_contracts=%s net_contracts=%s sl_contracts=%s protective_sl_price=%s protective_sl_order_id=%s retry_config=near_tp",
                three_stage_post_tp1_sl_payload.get("position_id"),
                three_stage_post_tp1_sl_payload.get("side"),
                three_stage_post_tp1_sl_payload.get("core_contracts"),
                three_stage_post_tp1_sl_payload.get("net_contracts"),
                three_stage_post_tp1_sl_payload.get("contracts"),
                sl_price,
                sl_order_id,
            )
        else:
            # ── Risk control: protective SL failed → immediate full market exit ──
            side = three_stage_post_tp1_sl_payload.get("side")
            exit_ok, exit_message = (False, "side_missing")
            if side is not None:
                exit_ok, exit_message = await trader.market_exit_remaining_position_with_retries(
                    side,
                    retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
                )
            core_contracts = three_stage_post_tp1_sl_payload.get("core_contracts")
            net_contracts = three_stage_post_tp1_sl_payload.get("net_contracts")
            sl_contracts = three_stage_post_tp1_sl_payload.get("contracts")
            manual_intervention_required = not exit_ok
            if exit_ok:
                halt_reason = "three_stage_post_tp1_sl_failed_market_exit_waiting_flat"
            else:
                halt_reason = "three_stage_post_tp1_protective_sl_failure"
            async with state_lock:
                execution_state.trading_halted = True
                execution_state.halt_reason = halt_reason
            state_store.save(
                LiveStateStore.from_strategy_state(
                    position_id=execution_state.current_position_id,
                    symbol=trader.symbol,
                    strategy_state=strategy.state,
                    cash_before_position=execution_state.cash_before_position,
                )
            )
            if hasattr(journal, "append"):
                journal.append(
                    "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED",
                    {
                        "position_id": three_stage_post_tp1_sl_payload.get("position_id"),
                        "side": side,
                        "protective_sl_price": sl_price,
                        "reason": sl_message,
                        "trading_halted": True,
                        "halt_reason": halt_reason,
                        "retry_config": "NEAR_TP_PROTECTIVE_SL_RETRY_COUNT/NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS",
                        "market_exit_attempted": True,
                        "market_exit_ok": exit_ok,
                        "market_exit_message": exit_message,
                        "core_contracts": str(core_contracts) if core_contracts is not None else None,
                        "net_contracts": str(net_contracts) if net_contracts is not None else None,
                        "sl_contracts": str(sl_contracts) if sl_contracts is not None else None,
                        "manual_intervention_required": manual_intervention_required,
                    },
                    position_id=three_stage_post_tp1_sl_payload.get("position_id"),
                )
            logger.error(
                "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED | position_id=%s side=%s sl_price=%s sl_message=%s market_exit_attempted=true market_exit_ok=%s market_exit_message=%s core_contracts=%s net_contracts=%s sl_contracts=%s manual_intervention_required=%s",
                three_stage_post_tp1_sl_payload.get("position_id"),
                side,
                sl_price,
                sl_message,
                exit_ok,
                exit_message,
                core_contracts,
                net_contracts,
                sl_contracts,
                manual_intervention_required,
            )
    middle_runner_activation_recorded = False
    if middle_runner_sl_payload is not None:
        sl_price = middle_runner_sl_payload.get("protective_sl_price")
        sl_order_id = None
        sl_ok = False
        sl_message = "protective_sl_price_missing"
        if sl_price is not None and middle_runner_sl_payload.get("side") is not None:
            try:
                sl_ok, sl_order_id, sl_message = await trader.place_middle_runner_protective_stop_with_retries(
                    middle_runner_sl_payload["side"],
                    middle_runner_sl_payload["contracts"],
                    float(sl_price),
                    retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                    retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
                )
            except Exception as exc:
                sl_ok = False
                sl_order_id = None
                sl_message = f"trader_exception: {type(exc).__name__}: {exc}"
        if sl_ok:
            old_sl_order_id = middle_runner_sl_payload.get("old_sl_order_id")
            if old_sl_order_id and old_sl_order_id != sl_order_id:
                await trader.cancel_middle_runner_protective_stop(old_sl_order_id)
            async with state_lock:
                strategy.state.middle_runner_protective_sl_order_id = sl_order_id
                strategy.state.middle_runner_protective_sl_price = float(sl_price)
                if middle_runner_sl_payload.get("reason") == "partial_size_mismatch_degraded":
                    strategy.state.middle_runner_size_mismatch_protected = True
                save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state),
                                      execution_state.cash_before_position)
            if hasattr(journal, "append"):
                event_name = (
                    "MIDDLE_RUNNER_SIZE_MISMATCH_PROTECTED"
                    if middle_runner_sl_payload.get("reason") == "partial_size_mismatch_degraded"
                    else "MIDDLE_RUNNER_ACTIVATED"
                )
                journal.append(
                    event_name,
                    {
                        **(middle_runner_activation_payload or {}),
                        "side": middle_runner_sl_payload.get("side"),
                        "contracts": str(middle_runner_sl_payload.get("contracts")),
                        "core_contracts": str(middle_runner_sl_payload.get("core_contracts")),
                        "net_contracts": str(middle_runner_sl_payload.get("net_contracts")),
                        "sl_contracts": str(middle_runner_sl_payload.get("contracts")),
                        "protective_sl_price": sl_price,
                        "protective_sl_order_id": sl_order_id,
                        "reason": middle_runner_sl_payload.get("reason", "partial_tp_filled"),
                    },
                    position_id=middle_runner_sl_payload.get("position_id"),
                )
                if event_name == "MIDDLE_RUNNER_ACTIVATED":
                    middle_runner_activation_recorded = True
            logger.warning(
                "%s | position_id=%s side=%s core_contracts=%s net_contracts=%s sl_contracts=%s protective_sl_price=%s protective_sl_order_id=%s",
                "MIDDLE_RUNNER_SIZE_MISMATCH_PROTECTED" if middle_runner_sl_payload.get(
                    "reason") == "partial_size_mismatch_degraded" else "MIDDLE_RUNNER_ACTIVATED",
                middle_runner_sl_payload.get("position_id"),
                middle_runner_sl_payload.get("side"),
                middle_runner_sl_payload.get("core_contracts"),
                middle_runner_sl_payload.get("net_contracts"),
                middle_runner_sl_payload.get("contracts"),
                sl_price,
                sl_order_id,
            )
        else:
            side = middle_runner_sl_payload.get("side")
            exit_ok, exit_message = (False, "side_missing")
            if side is not None:
                exit_ok, exit_message = await trader.market_exit_remaining_position_with_retries(
                    side,
                    retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
                )
            async with state_lock:
                execution_state.trading_halted = True
                execution_state.halt_reason = "middle_runner_protective_sl_failure"
            if hasattr(journal, "append"):
                journal.append(
                    "MIDDLE_RUNNER_ORDER_WARNING",
                    {
                        "side": side,
                        "protective_sl_price": sl_price,
                        "reason": f"protective_sl_failed:{sl_message};market_exit_ok={exit_ok};{exit_message}",
                    },
                    position_id=middle_runner_sl_payload.get("position_id"),
                )
            logger.error(
                "MIDDLE_RUNNER_ORDER_WARNING | reason=protective_sl_failed side=%s sl_price=%s sl_message=%s market_exit_ok=%s market_exit_message=%s",
                side,
                sl_price,
                sl_message,
                exit_ok,
                exit_message,
            )
    if (
            middle_runner_activation_payload is not None
            and not middle_runner_activation_recorded
            and middle_runner_sl_payload is None
            and hasattr(journal, "append")
    ):
        journal.append(
            "MIDDLE_RUNNER_ACTIVATED",
            {
                **middle_runner_activation_payload,
                "protective_sl_price": None,
                "protective_sl_order_id": None,
                "reason": "partial_tp_filled_protective_sl_disabled",
            },
            position_id=middle_runner_activation_payload.get("position_id"),
        )
    return AccountSyncProtectiveOrdersResult(
        save_state_payload=save_state_payload,
    )
