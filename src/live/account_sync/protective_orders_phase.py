from __future__ import annotations

import asyncio
import copy
import os
from dataclasses import dataclass
from typing import Any

from typing import Any

import time

from src.execution.trader import Trader
from src.live import delayed_market_exit as dme
from src.live import runtime_types as live_runtime_types
from src.live.alerts.halt_alerts import (
    HaltAlertDeduper,
    HaltAlertPayload,
    send_halt_alert_once,
)
from src.live.halt_modes import resolve_halt_mode
from src.position_management import runner_live_helpers
from src.position_management.protective_sl_strength import should_replace_sl
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)


async def _send_protective_orders_halt_alert(
    *,
    email_sender: Any,
    halt_alert_deduper: HaltAlertDeduper | None,
    symbol: str,
    position_id: str | None,
    halt_reason: str,
    side: str | None = None,
    layer: int | None = None,
    has_position: bool = True,
    sidecar_dirty: bool = False,
    manual_intervention_required: bool = True,
    message: str = "",
    extra: dict | None = None,
) -> None:
    """Send a rate-limited halt alert from the protective orders phase."""
    if email_sender is None or halt_alert_deduper is None:
        return
    try:
        await send_halt_alert_once(
            email_sender=email_sender,
            deduper=halt_alert_deduper,
            payload=HaltAlertPayload(
                symbol=symbol,
                position_id=position_id,
                halt_reason=halt_reason,
                halt_mode=resolve_halt_mode(halt_reason),
                side=side,
                layer=layer,
                has_position=has_position,
                sidecar_dirty=sidecar_dirty,
                manual_intervention_required=manual_intervention_required,
                message=message,
                extra=extra or {},
            ),
        )
    except Exception:
        logger.exception("PROTECTIVE_ORDERS_HALT_ALERT_EXCEPTION | halt_reason=%s", halt_reason)


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
        middle_bucket_split_event_payload: dict[str, Any] | None = None,
        middle_bucket_split_fast_protection_payload: dict[str, Any] | None = None,
        email_sender: Any = None,
        halt_alert_deduper: HaltAlertDeduper | None = None,
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
        side = three_stage_post_tp1_sl_payload.get("side")
        old_sl_order_id = three_stage_post_tp1_sl_payload.get("old_sl_order_id")

        # ── No-loosen check: only replace existing SL if candidate is stronger ──
        # Priority for existing_sl: payload (set by tp_progress_phase before
        # any candidate was written to state) → state fallback.
        existing_sl_price: float | None = three_stage_post_tp1_sl_payload.get("existing_sl_price")
        if existing_sl_price is None:
            existing_sl_price = getattr(strategy.state, "three_stage_post_tp1_protective_sl_price", None)
        if existing_sl_price is None:
            existing_sl_price = getattr(strategy.state, "middle_bucket_split_fast_sl_price", None)
        # Priority for existing_sl_order_id: payload → state fallback
        existing_sl_order_id: str | None = three_stage_post_tp1_sl_payload.get("existing_sl_order_id")
        if existing_sl_order_id is None:
            existing_sl_order_id = old_sl_order_id
        if existing_sl_order_id is None:
            existing_sl_order_id = getattr(strategy.state, "three_stage_post_tp1_protective_sl_order_id", None)
        if existing_sl_order_id is None:
            existing_sl_order_id = getattr(strategy.state, "middle_bucket_split_fast_sl_order_id", None)
        candidate_sl_price: float | None = three_stage_post_tp1_sl_payload.get("candidate_sl_price")
        if candidate_sl_price is None:
            candidate_sl_price = float(sl_price) if sl_price is not None else None

        _should_replace = should_replace_sl(
            side=side,
            existing_sl_price=existing_sl_price,
            candidate_sl_price=candidate_sl_price,
        )

        if not _should_replace and existing_sl_price is not None and existing_sl_order_id:
            # Existing SL is stronger or equal — keep it, skip new placement
            async with state_lock:
                strategy.state.three_stage_post_tp1_protective_sl_price = existing_sl_price
                strategy.state.three_stage_post_tp1_protective_sl_order_id = existing_sl_order_id
                strategy.state.three_stage_post_tp1_protected = True
                save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state),
                                      execution_state.cash_before_position)
            logger.warning(
                "PROTECTIVE_SL_KEEP_EXISTING_STRONGER | target=three_stage_post_tp1 "
                "side=%s existing_sl=%.4f candidate_sl=%s chosen_sl=%.4f "
                "old_sl_order_id=%s reason=no_loosen",
                side,
                existing_sl_price,
                candidate_sl_price,
                existing_sl_price,
                existing_sl_order_id,
            )
            if hasattr(journal, "append"):
                journal.append(
                    "PROTECTIVE_SL_KEEP_EXISTING_STRONGER",
                    {
                        "position_id": three_stage_post_tp1_sl_payload.get("position_id"),
                        "target": "three_stage_post_tp1",
                        "side": side,
                        "existing_sl": existing_sl_price,
                        "candidate_sl": candidate_sl_price,
                        "chosen_sl": existing_sl_price,
                        "old_sl_order_id": existing_sl_order_id,
                        "reason": "no_loosen",
                    },
                    position_id=three_stage_post_tp1_sl_payload.get("position_id"),
                )
            # Skip to end — do NOT place new SL, do NOT trigger DME
        elif sl_price is not None and side is not None:
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
            # Also cancel any stale middle bucket fast SL (from split slow fill)
            fast_sl_order_id = getattr(strategy.state, "middle_bucket_split_fast_sl_order_id", None)
            if fast_sl_order_id and fast_sl_order_id != sl_order_id:
                await trader.cancel_middle_bucket_fast_protective_stop(fast_sl_order_id)
            async with state_lock:
                strategy.state.three_stage_post_tp1_protective_sl_order_id = sl_order_id
                strategy.state.three_stage_post_tp1_protective_sl_price = float(sl_price)
                strategy.state.three_stage_post_tp1_protected = True
                # Clear stale fast SL state after slow fill (replaced by post-TP1 SL)
                strategy.state.middle_bucket_split_fast_sl_order_id = None
                strategy.state.middle_bucket_split_fast_sl_protected = False
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
            # ── candidate None but existing SL valid → keep existing, skip DME ──
            if candidate_sl_price is None and existing_sl_price is not None and existing_sl_order_id:
                async with state_lock:
                    strategy.state.three_stage_post_tp1_protective_sl_price = existing_sl_price
                    strategy.state.three_stage_post_tp1_protective_sl_order_id = existing_sl_order_id
                    strategy.state.three_stage_post_tp1_protected = True
                    save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state),
                                          execution_state.cash_before_position)
                logger.warning(
                    "PROTECTIVE_SL_KEEP_EXISTING_STRONGER | target=three_stage_post_tp1 "
                    "side=%s existing_sl=%.4f candidate_sl=None chosen_sl=%.4f "
                    "old_sl_order_id=%s reason=candidate_missing_keep_existing",
                    side,
                    existing_sl_price,
                    existing_sl_price,
                    existing_sl_order_id,
                )
                if hasattr(journal, "append"):
                    journal.append(
                        "PROTECTIVE_SL_KEEP_EXISTING_STRONGER",
                        {
                            "position_id": three_stage_post_tp1_sl_payload.get("position_id"),
                            "target": "three_stage_post_tp1",
                            "side": side,
                            "existing_sl": existing_sl_price,
                            "candidate_sl": None,
                            "chosen_sl": existing_sl_price,
                            "old_sl_order_id": existing_sl_order_id,
                            "reason": "candidate_missing_keep_existing",
                        },
                        position_id=three_stage_post_tp1_sl_payload.get("position_id"),
                    )
            else:
                # ── Risk control: protective SL failed → delayed market exit (NO immediate exit) ──
                _dme_side = three_stage_post_tp1_sl_payload.get("side")
                _dme_core_contracts = three_stage_post_tp1_sl_payload.get("core_contracts")
                _dme_net_contracts = three_stage_post_tp1_sl_payload.get("net_contracts")
                _dme_sl_contracts = three_stage_post_tp1_sl_payload.get("contracts")
                _dme_halt_reason = "three_stage_post_tp1_sl_failed_delayed_market_exit_armed"
                _dme_position_id = three_stage_post_tp1_sl_payload.get("position_id")

                async with state_lock:
                    arm_payload = dme.arm_delayed_market_exit(
                        strategy_state=strategy.state,
                        execution_state=execution_state,
                        position_id=_dme_position_id,
                        side=_dme_side or "UNKNOWN",
                        reason=_dme_halt_reason,
                        context="three_stage_post_tp1_protective_sl_failed",
                        source_event="THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED",
                        now_ms=int(time.time() * 1000),
                        error=sl_message,
                    )

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
                            "position_id": _dme_position_id,
                            "side": _dme_side,
                            "protective_sl_price": sl_price,
                            "reason": sl_message,
                            "trading_halted": True,
                            "halt_reason": _dme_halt_reason,
                            "retry_config": "NEAR_TP_PROTECTIVE_SL_RETRY_COUNT/NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS",
                            "market_exit_attempted": False,
                            "delayed_market_exit_armed": True,
                            "core_contracts": str(_dme_core_contracts) if _dme_core_contracts is not None else None,
                            "net_contracts": str(_dme_net_contracts) if _dme_net_contracts is not None else None,
                            "sl_contracts": str(_dme_sl_contracts) if _dme_sl_contracts is not None else None,
                            "manual_intervention_required": True,
                            **arm_payload,
                        },
                        position_id=_dme_position_id,
                    )
                logger.error(
                    "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED | position_id=%s side=%s sl_price=%s sl_message=%s delayed_market_exit_armed=true core_contracts=%s net_contracts=%s sl_contracts=%s manual_intervention_required=true",
                    _dme_position_id,
                    _dme_side,
                    sl_price,
                    sl_message,
                    _dme_core_contracts,
                    _dme_net_contracts,
                    _dme_sl_contracts,
                )
                await _send_protective_orders_halt_alert(
                    email_sender=email_sender,
                    halt_alert_deduper=halt_alert_deduper,
                    symbol=trader.symbol,
                    position_id=_dme_position_id,
                    halt_reason=_dme_halt_reason,
                    side=_dme_side,
                    manual_intervention_required=True,
                    message=(
                        f"Three-stage post-TP1 protective SL failed: {sl_message}. "
                        f"Delayed market exit armed (30 min countdown). NO immediate market exit."
                    ),
                    extra={"sl_price": str(sl_price), "delayed_market_exit_armed": True},
                )
    middle_runner_activation_recorded = False
    if middle_runner_sl_payload is not None:
        sl_price = middle_runner_sl_payload.get("protective_sl_price")
        sl_order_id = None
        sl_ok = False
        sl_message = "protective_sl_price_missing"
        mr_side = middle_runner_sl_payload.get("side")
        mr_old_sl_order_id = middle_runner_sl_payload.get("old_sl_order_id")

        # ── No-loosen check: only replace existing SL if candidate is stronger ──
        # Priority for existing_sl: payload → state fallback.
        mr_existing_sl_price: float | None = middle_runner_sl_payload.get("existing_sl_price")
        if mr_existing_sl_price is None:
            mr_existing_sl_price = getattr(strategy.state, "middle_runner_protective_sl_price", None)
        if mr_existing_sl_price is None:
            mr_existing_sl_price = getattr(strategy.state, "middle_bucket_split_fast_sl_price", None)
        # Priority for existing_sl_order_id: payload → state fallback
        mr_existing_sl_order_id: str | None = middle_runner_sl_payload.get("existing_sl_order_id")
        if mr_existing_sl_order_id is None:
            mr_existing_sl_order_id = mr_old_sl_order_id
        if mr_existing_sl_order_id is None:
            mr_existing_sl_order_id = getattr(strategy.state, "middle_runner_protective_sl_order_id", None)
        if mr_existing_sl_order_id is None:
            mr_existing_sl_order_id = getattr(strategy.state, "middle_bucket_split_fast_sl_order_id", None)
        mr_candidate_sl_price: float | None = middle_runner_sl_payload.get("candidate_sl_price")
        if mr_candidate_sl_price is None:
            mr_candidate_sl_price = float(sl_price) if sl_price is not None else None

        _mr_should_replace = should_replace_sl(
            side=mr_side,
            existing_sl_price=mr_existing_sl_price,
            candidate_sl_price=mr_candidate_sl_price,
        )

        if not _mr_should_replace and mr_existing_sl_price is not None and mr_existing_sl_order_id:
            # Existing SL is stronger or equal — keep it, skip new placement
            async with state_lock:
                strategy.state.middle_runner_protective_sl_price = mr_existing_sl_price
                strategy.state.middle_runner_protective_sl_order_id = mr_existing_sl_order_id
                if middle_runner_sl_payload.get("reason") == "partial_size_mismatch_degraded":
                    strategy.state.middle_runner_size_mismatch_protected = True
                save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state),
                                      execution_state.cash_before_position)
            logger.warning(
                "PROTECTIVE_SL_KEEP_EXISTING_STRONGER | target=middle_runner "
                "side=%s existing_sl=%.4f candidate_sl=%s chosen_sl=%.4f "
                "old_sl_order_id=%s reason=no_loosen",
                mr_side,
                mr_existing_sl_price,
                mr_candidate_sl_price,
                mr_existing_sl_price,
                mr_existing_sl_order_id,
            )
            if hasattr(journal, "append"):
                journal.append(
                    "PROTECTIVE_SL_KEEP_EXISTING_STRONGER",
                    {
                        "position_id": middle_runner_sl_payload.get("position_id"),
                        "target": "middle_runner",
                        "side": mr_side,
                        "existing_sl": mr_existing_sl_price,
                        "candidate_sl": mr_candidate_sl_price,
                        "chosen_sl": mr_existing_sl_price,
                        "old_sl_order_id": mr_existing_sl_order_id,
                        "reason": "no_loosen",
                    },
                    position_id=middle_runner_sl_payload.get("position_id"),
                )
            # Skip to end — do NOT place new SL, do NOT trigger DME
        elif sl_price is not None and mr_side is not None:
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
            # Also cancel any stale middle bucket fast SL (from split slow fill)
            fast_sl_order_id = getattr(strategy.state, "middle_bucket_split_fast_sl_order_id", None)
            if fast_sl_order_id and fast_sl_order_id != sl_order_id:
                await trader.cancel_middle_bucket_fast_protective_stop(fast_sl_order_id)
            async with state_lock:
                strategy.state.middle_runner_protective_sl_order_id = sl_order_id
                strategy.state.middle_runner_protective_sl_price = float(sl_price)
                # Clear stale fast SL state after slow fill (replaced by middle runner SL)
                strategy.state.middle_bucket_split_fast_sl_order_id = None
                strategy.state.middle_bucket_split_fast_sl_protected = False
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
            # ── candidate None but existing SL valid → keep existing, skip DME ──
            if mr_candidate_sl_price is None and mr_existing_sl_price is not None and mr_existing_sl_order_id:
                async with state_lock:
                    strategy.state.middle_runner_protective_sl_price = mr_existing_sl_price
                    strategy.state.middle_runner_protective_sl_order_id = mr_existing_sl_order_id
                    if middle_runner_sl_payload.get("reason") == "partial_size_mismatch_degraded":
                        strategy.state.middle_runner_size_mismatch_protected = True
                    save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state),
                                          execution_state.cash_before_position)
                logger.warning(
                    "PROTECTIVE_SL_KEEP_EXISTING_STRONGER | target=middle_runner "
                    "side=%s existing_sl=%.4f candidate_sl=None chosen_sl=%.4f "
                    "old_sl_order_id=%s reason=candidate_missing_keep_existing",
                    mr_side,
                    mr_existing_sl_price,
                    mr_existing_sl_price,
                    mr_existing_sl_order_id,
                )
                if hasattr(journal, "append"):
                    journal.append(
                        "PROTECTIVE_SL_KEEP_EXISTING_STRONGER",
                        {
                            "position_id": middle_runner_sl_payload.get("position_id"),
                            "target": "middle_runner",
                            "side": mr_side,
                            "existing_sl": mr_existing_sl_price,
                            "candidate_sl": None,
                            "chosen_sl": mr_existing_sl_price,
                            "old_sl_order_id": mr_existing_sl_order_id,
                            "reason": "candidate_missing_keep_existing",
                        },
                        position_id=middle_runner_sl_payload.get("position_id"),
                    )
            else:
                _mr_dme_side = middle_runner_sl_payload.get("side")
                _mr_dme_halt_reason = "middle_runner_sl_failed_delayed_market_exit_armed"
                _mr_dme_position_id = middle_runner_sl_payload.get("position_id")

                async with state_lock:
                    arm_payload = dme.arm_delayed_market_exit(
                        strategy_state=strategy.state,
                        execution_state=execution_state,
                        position_id=_mr_dme_position_id,
                        side=_mr_dme_side or "UNKNOWN",
                        reason=_mr_dme_halt_reason,
                        context="middle_runner_protective_sl_failed",
                        source_event="MIDDLE_RUNNER_ORDER_WARNING",
                        now_ms=int(time.time() * 1000),
                        error=sl_message,
                    )
                if hasattr(journal, "append"):
                    journal.append(
                        "MIDDLE_RUNNER_ORDER_WARNING",
                        {
                            "side": _mr_dme_side,
                            "protective_sl_price": sl_price,
                            "reason": f"protective_sl_failed:{sl_message}",
                            "delayed_market_exit_armed": True,
                            "halt_reason": _mr_dme_halt_reason,
                            **arm_payload,
                        },
                        position_id=_mr_dme_position_id,
                    )
                logger.error(
                    "MIDDLE_RUNNER_ORDER_WARNING | reason=protective_sl_failed side=%s sl_price=%s sl_message=%s delayed_market_exit_armed=true halt_reason=%s",
                    _mr_dme_side,
                    sl_price,
                    sl_message,
                    _mr_dme_halt_reason,
                )
                await _send_protective_orders_halt_alert(
                    email_sender=email_sender,
                    halt_alert_deduper=halt_alert_deduper,
                    symbol=trader.symbol,
                    position_id=_mr_dme_position_id,
                    halt_reason=_mr_dme_halt_reason,
                    side=_mr_dme_side,
                    manual_intervention_required=True,
                    message=(
                        f"Middle runner protective SL failed: {sl_message}. "
                        f"Delayed market exit armed (30 min countdown). NO immediate market exit."
                    ),
                    extra={"sl_price": str(sl_price), "delayed_market_exit_armed": True},
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

    # ── Middle Bucket Split fast protection ──────────────────────────
    if middle_bucket_split_fast_protection_payload is not None:
        from src.position_management.middle_bucket_fast_protection import (
            build_fast_protection_decision,
        )
        side = middle_bucket_split_fast_protection_payload.get("side")
        avg_entry = float(middle_bucket_split_fast_protection_payload.get("avg_entry_price", 0.0) or 0.0)
        current_price = float(middle_bucket_split_fast_protection_payload.get("current_price", 0.0) or 0.0)
        fast_sl_price = middle_bucket_split_fast_protection_payload.get("fast_sl_price")
        invalid_action = str(middle_bucket_split_fast_protection_payload.get("invalid_action", "MARKET_EXIT"))
        enabled = bool(middle_bucket_split_fast_protection_payload.get("enabled", True))
        old_sl_order_id = middle_bucket_split_fast_protection_payload.get("old_sl_order_id")
        fee_buffer_pct = float(getattr(strategy.config, "middle_bucket_split_fast_sl_fee_buffer_pct", 0.001))

        decision = build_fast_protection_decision(
            side=side,
            avg_entry_price=avg_entry,
            current_price=current_price,
            fee_buffer_pct=fee_buffer_pct,
            invalid_action=invalid_action,
            enabled=enabled,
        )

        # ── No-loosen check: only replace existing fast SL if candidate is stronger ──
        _mbf_existing_sl: float | None = getattr(strategy.state, "middle_bucket_split_fast_sl_price", None)
        _mbf_candidate_sl: float | None = (
            float(decision.sl_price) if decision.sl_price is not None
            else float(fast_sl_price) if fast_sl_price is not None
            else None
        )
        _mbf_should_replace = should_replace_sl(
            side=side,
            existing_sl_price=_mbf_existing_sl,
            candidate_sl_price=_mbf_candidate_sl,
        )

        if decision.action == "PLACE_SL" and side is not None:
            # ── No-loosen: keep existing stronger SL ──
            if not _mbf_should_replace and _mbf_existing_sl is not None and old_sl_order_id:
                async with state_lock:
                    strategy.state.middle_bucket_split_fast_sl_price = _mbf_existing_sl
                    strategy.state.middle_bucket_split_fast_sl_order_id = old_sl_order_id
                    strategy.state.middle_bucket_split_fast_sl_protected = True
                    save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state),
                                          execution_state.cash_before_position)
                logger.warning(
                    "PROTECTIVE_SL_KEEP_EXISTING_STRONGER | target=middle_bucket_split_partial "
                    "side=%s existing_sl=%.4f candidate_sl=%s chosen_sl=%.4f "
                    "old_sl_order_id=%s reason=no_loosen",
                    side,
                    _mbf_existing_sl,
                    _mbf_candidate_sl,
                    _mbf_existing_sl,
                    old_sl_order_id,
                )
                if hasattr(journal, "append"):
                    journal.append(
                        "PROTECTIVE_SL_KEEP_EXISTING_STRONGER",
                        {
                            "position_id": middle_bucket_split_fast_protection_payload.get("position_id"),
                            "target": "middle_bucket_split_partial",
                            "side": side,
                            "existing_sl": _mbf_existing_sl,
                            "candidate_sl": _mbf_candidate_sl,
                            "chosen_sl": _mbf_existing_sl,
                            "old_sl_order_id": old_sl_order_id,
                            "reason": "no_loosen",
                        },
                        position_id=middle_bucket_split_fast_protection_payload.get("position_id"),
                    )
            else:
                # Cancel old fast SL if exists (only when placing new SL)
                if old_sl_order_id:
                    await trader.cancel_middle_bucket_fast_protective_stop(old_sl_order_id)
                net_contracts = middle_bucket_split_fast_protection_payload.get("net_contracts")
                sl_ok, sl_order_id, sl_message = (False, None, "side_missing")
                if net_contracts and float(net_contracts or 0) > 0:
                    try:
                        sl_ok, sl_order_id, sl_message = await trader.place_middle_bucket_fast_protective_stop_with_retries(
                            side,
                            net_contracts,
                            float(decision.sl_price or fast_sl_price or 0.0),
                            retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                            retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
                        )
                    except Exception as exc:
                        sl_ok = False
                        sl_order_id = None
                        sl_message = f"trader_exception: {type(exc).__name__}: {exc}"
                if sl_ok:
                    async with state_lock:
                        strategy.state.middle_bucket_split_fast_sl_order_id = sl_order_id
                        strategy.state.middle_bucket_split_fast_sl_price = float(decision.sl_price or fast_sl_price or 0.0)
                        strategy.state.middle_bucket_split_fast_sl_protected = True
                        save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state),
                                              execution_state.cash_before_position)
                    if hasattr(journal, "append"):
                        journal.append(
                            "MIDDLE_BUCKET_FAST_PROTECTIVE_SL_PLACED",
                            {
                                "position_id": middle_bucket_split_fast_protection_payload.get("position_id"),
                                "side": side,
                                "fast_sl_price": decision.sl_price,
                                "sl_order_id": sl_order_id,
                                "avg_entry_price": avg_entry,
                                "current_price": current_price,
                                "current_price_source": middle_bucket_split_fast_protection_payload.get(
                                    "current_price_source"),
                                "reason": middle_bucket_split_fast_protection_payload.get(
                                    "reason", "middle_bucket_fast_filled_protect"),
                            },
                            position_id=middle_bucket_split_fast_protection_payload.get("position_id"),
                        )
                    logger.warning(
                        "MIDDLE_BUCKET_FAST_PROTECTIVE_SL_PLACED | position_id=%s side=%s fast_sl_price=%.4f sl_order_id=%s avg_entry=%.4f current_price=%.4f",
                        middle_bucket_split_fast_protection_payload.get("position_id"),
                        side,
                        decision.sl_price,
                        sl_order_id,
                        avg_entry,
                        current_price,
                    )
                else:
                    # SL placement failed → delayed market exit (NO immediate exit)
                    _mbf_halt_reason = "middle_bucket_fast_sl_failed_delayed_market_exit_armed"
                    _mbf_position_id = middle_bucket_split_fast_protection_payload.get("position_id")

                    async with state_lock:
                        arm_payload = dme.arm_delayed_market_exit(
                            strategy_state=strategy.state,
                            execution_state=execution_state,
                            position_id=_mbf_position_id,
                            side=side or "UNKNOWN",
                            reason=_mbf_halt_reason,
                            context="middle_bucket_fast_sl_failed",
                            source_event="MIDDLE_BUCKET_FAST_PROTECTIVE_SL_FAILED",
                            now_ms=int(time.time() * 1000),
                            error=sl_message,
                        )
                        strategy.state.middle_bucket_split_fast_sl_invalid_action_taken = "DELAYED_MARKET_EXIT"
                    if hasattr(journal, "append"):
                        journal.append(
                            "MIDDLE_BUCKET_FAST_PROTECTIVE_SL_FAILED",
                            {
                                "position_id": _mbf_position_id,
                                "side": side,
                                "fast_sl_price": decision.sl_price,
                                "reason": sl_message,
                                "trading_halted": True,
                                "halt_reason": _mbf_halt_reason,
                                "market_exit_attempted": False,
                                "delayed_market_exit_armed": True,
                                "manual_intervention_required": True,
                                **arm_payload,
                            },
                            position_id=_mbf_position_id,
                        )
                    logger.error(
                        "MIDDLE_BUCKET_FAST_PROTECTIVE_SL_FAILED | position_id=%s side=%s sl_price=%s sl_message=%s delayed_market_exit_armed=true halt_reason=%s",
                        _mbf_position_id,
                        side,
                        decision.sl_price,
                        sl_message,
                        _mbf_halt_reason,
                    )
                    await _send_protective_orders_halt_alert(
                        email_sender=email_sender,
                        halt_alert_deduper=halt_alert_deduper,
                        symbol=trader.symbol,
                        position_id=_mbf_position_id,
                        halt_reason=_mbf_halt_reason,
                        side=side,
                        manual_intervention_required=True,
                        message=(
                            f"Middle bucket fast protective SL failed: {sl_message}. "
                            f"Delayed market exit armed (30 min countdown). NO immediate market exit."
                        ),
                        extra={"sl_price": str(decision.sl_price), "delayed_market_exit_armed": True},
                    )

        elif decision.action == "MARKET_EXIT" and side is not None:
            # ── Delayed market exit (NO immediate exit) ──────────────────
            halt_reason = "middle_bucket_fast_sl_invalid_delayed_market_exit_armed"
            position_id = middle_bucket_split_fast_protection_payload.get("position_id")

            async with state_lock:
                arm_payload = dme.arm_delayed_market_exit(
                    strategy_state=strategy.state,
                    execution_state=execution_state,
                    position_id=position_id,
                    side=side,
                    reason=halt_reason,
                    context="middle_bucket_fast_sl_invalid_market_exit",
                    source_event="MIDDLE_BUCKET_FAST_PROTECTIVE_SL_INVALID_MARKET_EXIT",
                    now_ms=int(time.time() * 1000),
                    error=decision.reason,
                )
                strategy.state.middle_bucket_split_fast_sl_invalid_action_taken = "DELAYED_MARKET_EXIT"
            if hasattr(journal, "append"):
                journal.append(
                    "MIDDLE_BUCKET_FAST_PROTECTIVE_SL_INVALID_MARKET_EXIT",
                    {
                        "position_id": position_id,
                        "side": side,
                        "fast_sl_price": decision.sl_price,
                        "current_price": current_price,
                        "avg_entry_price": avg_entry,
                        "reason": decision.reason,
                        "trading_halted": True,
                        "halt_reason": halt_reason,
                        "market_exit_attempted": False,
                        "delayed_market_exit_armed": True,
                        "manual_intervention_required": True,
                        **arm_payload,
                    },
                    position_id=position_id,
                )
            logger.error(
                "MIDDLE_BUCKET_FAST_PROTECTIVE_SL_INVALID_MARKET_EXIT | position_id=%s side=%s sl_price=%.4f current_price=%.4f delayed_market_exit_armed=true halt_reason=%s",
                position_id,
                side,
                decision.sl_price,
                current_price,
                halt_reason,
            )
            await _send_protective_orders_halt_alert(
                email_sender=email_sender,
                halt_alert_deduper=halt_alert_deduper,
                symbol=trader.symbol,
                position_id=position_id,
                halt_reason=halt_reason,
                side=side,
                manual_intervention_required=True,
                message=(
                    f"Middle bucket fast SL invalid → delayed market exit armed (30 min countdown). "
                    f"NO immediate market exit."
                ),
                extra={"sl_price": str(decision.sl_price), "current_price": str(current_price), "delayed_market_exit_armed": True},
            )

        elif decision.action == "HALT_ONLY":
            async with state_lock:
                execution_state.trading_halted = True
                execution_state.halt_reason = "middle_bucket_fast_sl_invalid_halt_only"
            logger.error(
                "MIDDLE_BUCKET_FAST_PROTECTIVE_SL_INVALID_HALT_ONLY | position_id=%s side=%s sl_price=%.4f current_price=%.4f manual_intervention_required=true",
                middle_bucket_split_fast_protection_payload.get("position_id"),
                side,
                decision.sl_price,
                current_price,
            )
            await _send_protective_orders_halt_alert(
                email_sender=email_sender,
                halt_alert_deduper=halt_alert_deduper,
                symbol=trader.symbol,
                position_id=middle_bucket_split_fast_protection_payload.get("position_id"),
                halt_reason="middle_bucket_fast_sl_invalid_halt_only",
                side=side,
                manual_intervention_required=True,
                message="Middle bucket fast SL invalid → HALT_ONLY. Manual intervention required.",
                extra={"sl_price": str(decision.sl_price), "current_price": str(current_price)},
            )

        elif decision.action == "KEEP_POSITION":
            logger.error(
                "MIDDLE_BUCKET_FAST_PROTECTIVE_SL_INVALID_KEEP_POSITION | position_id=%s side=%s sl_price=%.4f current_price=%.4f risk=naked_remaining_core manual_intervention_required=true",
                middle_bucket_split_fast_protection_payload.get("position_id"),
                side,
                decision.sl_price,
                current_price,
            )

    return AccountSyncProtectiveOrdersResult(
        save_state_payload=save_state_payload,
    )
