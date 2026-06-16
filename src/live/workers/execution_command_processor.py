from __future__ import annotations

import asyncio
import copy
import html
import os
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from src.execution.live_trader_protocol import LiveTraderProtocol
from src.execution.trader import LiveTradeResult, PositionSnapshot
from src.live import config_helpers as live_config_helpers
from src.live import runtime_types as live_runtime_types
from src.live.account_sync import flat_balance as live_flat_balance
from src.live.alerts.halt_alerts import (
    HaltAlertDeduper,
    HaltAlertPayload,
    send_halt_alert_once,
)
from src.live import delayed_market_exit as dme
from src.live.halt_modes import (
    FULL_HALT,
    allowed_intents_for_halt_mode,
    is_intent_allowed_during_halt,
    resolve_halt_mode,
)
from src.live.startup_recovery import basic_restore as startup_basic_restore
from src.position_management import core_position_view as core_position_view_helpers
from src.position_management import cost_runtime as position_cost_runtime
from src.position_management import runner_live_helpers
from src.position_management.middle_bucket_split_state import (
    clear_middle_bucket_split_state,
    degrade_middle_bucket_split_to_single_final,
)
from src.position_management import tp_progress as tp_progress_helpers
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.email_sender import EmailSender
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionCommandProcessor:
    state_lock: asyncio.Lock
    execution_state: live_runtime_types.ExecutionState
    account_snapshot: live_runtime_types.AccountSnapshot
    trader: LiveTraderProtocol
    strategy: BollCvdShockReclaimStrategy
    journal: LiveTradeJournal
    state_store: LiveStateStore
    email_sender: EmailSender
    halt_alert_deduper: HaltAlertDeduper = field(default_factory=HaltAlertDeduper)
    _background_tasks: set[asyncio.Task] = field(default_factory=set, init=False)

    async def _send_halt_alert(
        self,
        *,
        halt_reason: str,
        side: str | None = None,
        layer: int | None = None,
        has_position: bool = True,
        manual_intervention_required: bool = True,
        message: str = "",
        extra: dict | None = None,
    ) -> None:
        """Send a rate-limited halt alert email.  Never raises."""
        if self.email_sender is None:
            return
        try:
            await send_halt_alert_once(
                email_sender=self.email_sender,
                deduper=self.halt_alert_deduper,
                payload=HaltAlertPayload(
                    symbol=self.trader.symbol,
                    position_id=self.execution_state.current_position_id,
                    halt_reason=halt_reason,
                    halt_mode=resolve_halt_mode(halt_reason),
                    side=side,
                    layer=layer,
                    has_position=has_position,
                    manual_intervention_required=manual_intervention_required,
                    message=message,
                    extra=extra or {},
                ),
            )
        except Exception:
            logger.exception("PROCESSOR_HALT_ALERT_EXCEPTION | halt_reason=%s", halt_reason)

    async def process(self, command: live_runtime_types.TradeCommand) -> Any:
        """Process a single TradeCommand. Returns the LiveTradeResult on success, None if skipped."""

        # ── dirty post-TP1 SL guard ──────────────────────────────────────
        dirty_post_tp1_sl_blocked = False
        dirty_post_tp1_sl_should_record = False
        async with self.state_lock:
            if runner_live_helpers.three_stage_dirty_post_tp1_sl_after_tp2(self.strategy.state):
                dirty_post_tp1_sl_blocked = True
                dirty_post_tp1_sl_should_record = not (
                    self.execution_state.trading_halted
                    and self.execution_state.halt_reason == runner_live_helpers.THREE_STAGE_RUNTIME_DIRTY_HALT_REASON
                )
                self.execution_state.trading_halted = True
                self.execution_state.halt_reason = runner_live_helpers.THREE_STAGE_RUNTIME_DIRTY_HALT_REASON
                self.execution_state.halt_until_ts_ms = None
        if dirty_post_tp1_sl_blocked:
            if dirty_post_tp1_sl_should_record:
                runner_live_helpers.append_three_stage_dirty_post_tp1_event(
                    event_name="THREE_STAGE_DIRTY_POST_TP1_SL_BLOCKED_RUNNER_UPDATE",
                    strategy=self.strategy,
                    execution_state=self.execution_state,
                    journal=self.journal,
                    state_store=self.state_store,
                    trader_symbol=self.trader.symbol,
                    reason="dirty_post_tp1_sl_after_tp2_blocks_runner_update_manual_intervention_required",
                )
            logger.warning(
                "EXECUTION_SKIPPED | reason=three_stage_dirty_post_tp1_sl intent_type=%s side=%s tick_ts_ms=%s",
                command.intent.intent_type,
                command.intent.side,
                command.tick_ts_ms,
            )
            return None

        # ── trading halted guard (unified halt mode) ──────────────────────
        halt_mode: str | None = None
        async with self.state_lock:
            if self.execution_state.trading_halted:
                halt_mode = resolve_halt_mode(self.execution_state.halt_reason)
                allowed_intents = allowed_intents_for_halt_mode(halt_mode)
                if command.intent.intent_type not in allowed_intents:
                    logger.warning(
                        "EXECUTION_SKIPPED | reason=trading_halted halt_reason=%s halt_mode=%s intent_type=%s allowed_intents=%s side=%s tick_ts_ms=%s",
                        self.execution_state.halt_reason,
                        halt_mode,
                        command.intent.intent_type,
                        sorted(allowed_intents),
                        command.intent.side,
                        command.tick_ts_ms,
                    )
                    return None
            current_position_id = self.execution_state.current_position_id
            cash_before_position = self.execution_state.cash_before_position

        # ── entry cash before ────────────────────────────────────────────
        entry_cash_before = cash_before_position
        if command.intent.intent_type not in {"UPDATE_TP", "UPDATE_TREND_SL"} and current_position_id is None:
            entry_cash_before = await live_flat_balance.fetch_usdt_cash_balance(self.trader)

        # ── stale split add guard ────────────────────────────────────────
        if command.intent.intent_type in {"ADD_LONG", "ADD_SHORT"} and getattr(
            command.strategy_state_snapshot, "tp_plan", "SINGLE"
        ) in tp_progress_helpers.SPLIT_TP_PLANS:
            position = await self.trader.fetch_position_snapshot()
            if position.has_position and position.side == command.intent.side:
                consumed = False
                async with self.state_lock:
                    current_strategy_state = copy.deepcopy(self.strategy.state)
                    self.strategy.state = copy.deepcopy(command.strategy_state_snapshot)
                    consumed = tp_progress_helpers.mark_partial_tp_consumed_if_position_reduced(
                        self.strategy, position
                    )
                    if consumed:
                        position_cost_runtime.sync_strategy_cost_from_position(
                            self.strategy,
                            position,
                            restore_from_position=startup_basic_restore.restore_strategy_from_position,
                        )
                        current_position_id = self.execution_state.current_position_id
                        cash_before_position = self.execution_state.cash_before_position
                        strategy_state_for_save = copy.deepcopy(self.strategy.state)
                        self.account_snapshot.position = position
                        self.trader.position_contracts = position.contracts
                    else:
                        self.strategy.state = current_strategy_state
                if consumed:
                    self.state_store.save(
                        LiveStateStore.from_strategy_state(
                            position_id=current_position_id,
                            symbol=self.trader.symbol,
                            strategy_state=strategy_state_for_save,
                            cash_before_position=cash_before_position,
                        )
                    )
                    logger.warning(
                        "EXECUTION_SKIPPED | reason=partial_tp_consumed_before_add stale_add_command_skipped intent_type=%s side=%s layer=%s okx_eth_qty=%.8f strategy_eth_qty=%.8f tp_plan=%s",
                        command.intent.intent_type,
                        command.intent.side,
                        command.intent.layer_index,
                        position.eth_qty,
                        strategy_state_for_save.total_entry_qty,
                        strategy_state_for_save.tp_plan,
                    )
                    return None

        # ── execute intent ───────────────────────────────────────────────
        result = await self.trader.execute_intent(command.intent)
        if not result.ok:
            # ── UPDATE_TP / UPDATE_TREND_SL failure: arm delayed market exit ──
            if command.intent.intent_type in ("UPDATE_TP", "UPDATE_TREND_SL"):
                _failed_intent_type = command.intent.intent_type
                _failed_reason = (
                    "core_tp_place_failed_delayed_market_exit_armed"
                    if _failed_intent_type == "UPDATE_TP"
                    else "trend_sl_update_failed_delayed_market_exit_armed"
                )
                _failed_source = f"{_failed_intent_type}_FAILED"
                _failed_context = (
                    "update_tp_replace_take_profit_failed"
                    if _failed_intent_type == "UPDATE_TP"
                    else "update_trend_sl_failed"
                )
                # Only arm if not already armed for the same reason (avoid resetting countdown).
                already_armed_same = (
                    getattr(self.strategy.state, "delayed_market_exit_armed", False)
                    and getattr(self.strategy.state, "delayed_market_exit_reason", None) == _failed_reason
                )
                if not already_armed_same:
                    logger.error(
                        "%s_FAILED | position_id=%s side=%s message=%s delayed_market_exit_armed=true",
                        _failed_intent_type,
                        current_position_id,
                        command.intent.side,
                        getattr(result, "message", ""),
                    )
                    async with self.state_lock:
                        arm_payload = dme.arm_delayed_market_exit(
                            strategy_state=self.strategy.state,
                            execution_state=self.execution_state,
                            position_id=current_position_id,
                            side=command.intent.side,
                            reason=_failed_reason,
                            context=_failed_context,
                            source_event=_failed_source,
                            now_ms=command.intent.ts_ms,
                            error=getattr(result, "message", str(result)),
                        )
                    if hasattr(self.journal, "append"):
                        self.journal.append(
                            f"{_failed_intent_type}_FAILED_DELAYED_MARKET_EXIT_ARMED",
                            {
                                "position_id": current_position_id,
                                "side": command.intent.side,
                                "intent_type": _failed_intent_type,
                                "message": getattr(result, "message", ""),
                                "delayed_market_exit_armed": True,
                                "halt_reason": _failed_reason,
                                **arm_payload,
                            },
                            position_id=current_position_id,
                        )
                    self.state_store.save(
                        LiveStateStore.from_strategy_state(
                            position_id=current_position_id,
                            symbol=self.trader.symbol,
                            strategy_state=self.strategy.state,
                            cash_before_position=cash_before_position,
                        )
                    )
                    await self._send_halt_alert(
                        halt_reason=_failed_reason,
                        side=command.intent.side,
                        layer=command.intent.layer_index,
                        manual_intervention_required=True,
                        message=(
                            "UPDATE_TP / replace_take_profit failed. "
                            "Delayed market exit armed (30 min countdown). NO immediate market exit."
                            if _failed_intent_type == "UPDATE_TP"
                            else "UPDATE_TREND_SL / trend trailing SL update failed. "
                            "Delayed market exit armed (30 min countdown). NO immediate market exit."
                        ),
                        extra={
                            "intent_type": _failed_intent_type,
                            "delayed_market_exit_armed": True,
                            "error": getattr(result, "message", str(result)),
                        },
                    )
                else:
                    logger.warning(
                        "%s_FAILED_ALREADY_ARMED | position_id=%s side=%s — delayed exit already armed, skipping re-arm",
                        _failed_intent_type,
                        current_position_id,
                        command.intent.side,
                    )
                return result

            # ── Core entry filled but TP/SL placement failed ────────────
            entry_filled = getattr(result, "entry_filled", False)
            tp_ok = getattr(result, "tp_ok", True)
            if entry_filled and not tp_ok:
                logger.error(
                    "CORE_TP_PLACE_FAILED_AFTER_ENTRY | position_id=%s side=%s intent_type=%s message=%s delayed_market_exit_armed=true",
                    current_position_id,
                    command.intent.side,
                    command.intent.intent_type,
                    getattr(result, "message", ""),
                )
                async with self.state_lock:
                    arm_payload = dme.arm_delayed_market_exit(
                        strategy_state=self.strategy.state,
                        execution_state=self.execution_state,
                        position_id=current_position_id,
                        side=command.intent.side,
                        reason="core_tp_place_failed_delayed_market_exit_armed",
                        context="core_tp_place_failed_after_entry_filled",
                        source_event="CORE_TP_PLACE_FAILED",
                        now_ms=command.intent.ts_ms,
                        error=getattr(result, "message", str(result)),
                    )
                if hasattr(self.journal, "append"):
                    self.journal.append(
                        "CORE_TP_PLACE_FAILED",
                        {
                            "position_id": current_position_id,
                            "side": command.intent.side,
                            "intent_type": command.intent.intent_type,
                            "message": getattr(result, "message", ""),
                            "delayed_market_exit_armed": True,
                            "halt_reason": "core_tp_place_failed_delayed_market_exit_armed",
                            **arm_payload,
                        },
                        position_id=current_position_id,
                    )
                self.state_store.save(
                    LiveStateStore.from_strategy_state(
                        position_id=current_position_id,
                        symbol=self.trader.symbol,
                        strategy_state=self.strategy.state,
                        cash_before_position=cash_before_position,
                    )
                )
                # Send CRITICAL email
                await self._send_halt_alert(
                    halt_reason="core_tp_place_failed_delayed_market_exit_armed",
                    side=command.intent.side,
                    layer=command.intent.layer_index,
                    manual_intervention_required=True,
                    message=(
                        "Core TP placement failed after entry filled. "
                        "Delayed market exit armed (30 min countdown). NO immediate market exit."
                    ),
                    extra={
                        "entry_filled": True,
                        "tp_ok": False,
                        "delayed_market_exit_armed": True,
                        "error": getattr(result, "message", str(result)),
                    },
                )
            return result

        # ── apply result ─────────────────────────────────────────────────
        if command.intent.intent_type == "UPDATE_TP":
            await self._apply_update_tp_result(command, result)
        elif command.intent.intent_type == "UPDATE_TREND_SL":
            await self._apply_update_trend_sl_result(command, result)
        elif command.intent.intent_type == "MARKET_EXIT_RUNNER":
            await self._apply_market_exit_runner_result(command, result)
        else:
            await self._apply_entry_result(command, result, entry_cash_before)

        return result

    # ── result application helpers ───────────────────────────────────────

    def _maybe_apply_entry_protective_sl_state(
        self,
        command: live_runtime_types.TradeCommand,
        result: Any,
    ) -> bool:
        """Apply entry protective SL state from result only when the intent carries an entry protective SL price.

        IMPORTANT: Only entry intents (OPEN/ADD) carry ``entry_protective_sl_price``.
        Non-entry intents (UPDATE_TP, MARKET_EXIT_RUNNER) do NOT
        carry this field, so their ``protective_sl_order_id`` (which belongs to
        a middle-runner / three-stage runner protective SL) will never
        be written into ``entry_protective_sl_order_id``.

        Returns True if state was written, False otherwise.
        Must be called inside ``state_lock``.
        """
        entry_sl_price = getattr(command.intent, "entry_protective_sl_price", None)
        if entry_sl_price is None:
            return False

        order_id = getattr(result, "protective_sl_order_id", None)
        if not order_id:
            return False

        self.strategy.state.entry_protective_sl_order_id = order_id
        self.strategy.state.entry_protective_sl_price = entry_sl_price
        self.strategy.state.entry_protective_sl_protected = bool(
            getattr(result, "protective_sl_ok", False)
        )
        return True

    def _maybe_clear_middle_bucket_split_after_execution_result(
        self,
        *,
        result: Any,
        current_position_id: str | None,
    ) -> bool:
        """Clear or degrade strategy split state based on actual order mode.

        - FINAL_FULL_SIZE: degrade TP plan to SINGLE (full-size final TP).
        - UNSPLIT_MIDDLE_BUCKET: clear only split fields (keep plan).
        - None with backward-compat: infer from disabled_reason.

        Returns True if state was mutated, False otherwise.
        Must be called inside ``state_lock``.
        """
        split_executed = getattr(result, "middle_bucket_split_executed", None)
        if split_executed is not False:
            return False

        reason = getattr(result, "middle_bucket_split_disabled_reason", None) or "unknown"
        actual_order_mode = getattr(result, "middle_bucket_split_actual_order_mode", None)

        if actual_order_mode == "FINAL_FULL_SIZE":
            # ── Actual orders are full-size final TP ──────────────────
            previous_tp_plan = getattr(self.strategy.state, "tp_plan", "SINGLE")
            degrade_middle_bucket_split_to_single_final(
                self.strategy.state, reason=reason,
            )
            logger.warning(
                "MIDDLE_BUCKET_SPLIT_DEGRADED_TO_SINGLE_FINAL | "
                "reason=%s actual_order_mode=FINAL_FULL_SIZE "
                "previous_tp_plan=%s state_order_consistent=true",
                reason,
                previous_tp_plan,
            )
            if hasattr(self.journal, "append"):
                self.journal.append(
                    "MIDDLE_BUCKET_SPLIT_DEGRADED_TO_SINGLE_FINAL",
                    {
                        "reason": reason,
                        "actual_order_mode": "FINAL_FULL_SIZE",
                        "state_order_consistent": True,
                        "previous_tp_plan": previous_tp_plan,
                        "new_tp_plan": "SINGLE",
                        "tp_order_id": getattr(result, "tp_order_id", None),
                        "tp_order_ids": getattr(result, "tp_order_ids", ()),
                    },
                    position_id=current_position_id,
                )
            return True

        if actual_order_mode == "UNSPLIT_MIDDLE_BUCKET":
            # ── Actual orders are unsplit middle bucket ───────────────
            clear_middle_bucket_split_state(self.strategy.state, reason=reason)
            logger.warning(
                "MIDDLE_BUCKET_SPLIT_STATE_CLEARED_ON_ORDER_BUILD | "
                "reason=%s actual_orders_unsplit_or_final=true state_order_consistent=true",
                reason,
            )
            if hasattr(self.journal, "append"):
                self.journal.append(
                    "MIDDLE_BUCKET_SPLIT_DISABLED_ON_ORDER_BUILD",
                    {
                        "reason": reason,
                        "state_split_active_was": True,
                        "actual_orders_unsplit_or_final": True,
                        "state_order_consistent": True,
                    },
                    position_id=current_position_id,
                )
            return True

        # ── Backward-compat: actual_order_mode is None ────────────────
        logger.warning(
            "MIDDLE_BUCKET_SPLIT_ACTUAL_ORDER_MODE_MISSING | "
            "reason=%s actual_order_mode_missing_degraded_by_reason=true "
            "split_executed=False",
            reason,
        )
        if reason == "split_order_placement_failed_fallback_final":
            previous_tp_plan = getattr(self.strategy.state, "tp_plan", "SINGLE")
            degrade_middle_bucket_split_to_single_final(
                self.strategy.state, reason=reason,
            )
            if hasattr(self.journal, "append"):
                self.journal.append(
                    "MIDDLE_BUCKET_SPLIT_DEGRADED_TO_SINGLE_FINAL",
                    {
                        "reason": reason,
                        "actual_order_mode": "FINAL_FULL_SIZE",
                        "state_order_consistent": True,
                        "previous_tp_plan": previous_tp_plan,
                        "new_tp_plan": "SINGLE",
                        "tp_order_id": getattr(result, "tp_order_id", None),
                        "tp_order_ids": getattr(result, "tp_order_ids", ()),
                        "actual_order_mode_missing": True,
                    },
                    position_id=current_position_id,
                )
            return True

        # Default: clear split state only
        clear_middle_bucket_split_state(self.strategy.state, reason=reason)
        logger.warning(
            "MIDDLE_BUCKET_SPLIT_STATE_CLEARED_ON_ORDER_BUILD | "
            "reason=%s actual_orders_unsplit_or_final=true state_order_consistent=true",
            reason,
        )
        if hasattr(self.journal, "append"):
            self.journal.append(
                "MIDDLE_BUCKET_SPLIT_DISABLED_ON_ORDER_BUILD",
                {
                    "reason": reason,
                    "state_split_active_was": True,
                    "actual_orders_unsplit_or_final": True,
                    "state_order_consistent": True,
                },
                position_id=current_position_id,
            )
        return True

    async def _apply_update_tp_result(
        self,
        command: live_runtime_types.TradeCommand,
        result: Any,
    ) -> None:
        async with self.state_lock:
            current_position_id = self.execution_state.current_position_id
            cash_before_position = self.execution_state.cash_before_position
            self.execution_state.last_order_ts_ms = command.intent.ts_ms
            if getattr(command.intent, "middle_runner_active", False):
                if getattr(result, "protective_sl_order_id", None):
                    self.strategy.state.middle_runner_protective_sl_order_id = result.protective_sl_order_id
                if live_config_helpers._parse_optional_float(
                    getattr(result, "protective_sl_price", "")
                ) is not None:
                    self.strategy.state.middle_runner_protective_sl_price = (
                        live_config_helpers._parse_optional_float(result.protective_sl_price)
                    )
            if getattr(command.intent, "trend_runner_active", False):
                if getattr(result, "protective_sl_order_id", None):
                    self.strategy.state.trend_runner_sl_order_id = result.protective_sl_order_id
                if live_config_helpers._parse_optional_float(
                    getattr(result, "protective_sl_price", "")
                ) is not None:
                    self.strategy.state.trend_runner_sl_price = (
                        live_config_helpers._parse_optional_float(result.protective_sl_price)
                    )
                self.strategy.state.trend_runner_tp_order_id = result.tp_order_id
            # ── Middle Bucket Split state consistency ──────────────────
            # When the execution layer disabled split (subleg too small,
            # order placement failed fallback final, etc.), the strategy
            # state MUST be cleared to match the actual orders.
            self._maybe_clear_middle_bucket_split_after_execution_result(
                result=result,
                current_position_id=current_position_id,
            )
            self.strategy.state.tp_order_id = result.tp_order_id
            self.strategy.state.tp_order_ids = list(getattr(result, "tp_order_ids", ()) or [])
            self._maybe_apply_entry_protective_sl_state(command, result)
            if (
                getattr(command.intent, "three_stage_post_tp1_protective_sl_price", None) is not None
                and getattr(command.intent, "three_stage_tp1_consumed", False)
            ):
                if getattr(result, "protective_sl_order_id", None):
                    self.strategy.state.three_stage_post_tp1_protective_sl_order_id = result.protective_sl_order_id
                if live_config_helpers._parse_optional_float(
                    getattr(result, "protective_sl_price", "")
                ) is not None:
                    self.strategy.state.three_stage_post_tp1_protective_sl_price = (
                        live_config_helpers._parse_optional_float(result.protective_sl_price)
                    )
                self.strategy.state.three_stage_post_tp1_protected = bool(
                    getattr(result, "protective_sl_ok", False)
                )
            strategy_state_for_save = copy.deepcopy(self.strategy.state)
            equity = self.account_snapshot.equity
        self.journal.record_tp_update(
            position_id=current_position_id, intent=command.intent, result=result, equity=equity
        )
        if (
            (
                getattr(command.intent, "middle_runner_active", False)
                or getattr(command.intent, "middle_runner_pending", False)
            )
            and hasattr(self.journal, "append")
        ):
            self.journal.append(
                "MIDDLE_RUNNER_TP_UPDATED",
                {
                    "side": command.intent.side,
                    "first_tp_price": getattr(command.intent, "partial_tp_price", None),
                    "final_tp_price": command.intent.tp_price,
                    "protective_sl_price": getattr(result, "protective_sl_price", "")
                    or getattr(command.intent, "middle_runner_protective_sl_price", None),
                    "protective_sl_order_id": getattr(result, "protective_sl_order_id", None),
                    "boll_lower": command.intent.boll_lower,
                    "boll_middle": command.intent.boll_middle,
                    "boll_upper": command.intent.boll_upper,
                    "reason": command.intent.reason,
                },
                position_id=current_position_id,
            )
        if getattr(command.intent, "trend_runner_active", False) and hasattr(self.journal, "append"):
            self.journal.append(
                "TREND_RUNNER_UPDATE",
                {
                    "side": command.intent.side,
                    "tp_plan": "THREE_STAGE_RUNNER",
                    "runner_tp_price": getattr(command.intent, "trend_runner_tp_price", None)
                    or command.intent.tp_price,
                    "runner_sl_price": getattr(result, "protective_sl_price", "")
                    or getattr(command.intent, "trend_runner_sl_price", None),
                    "runner_sl_order_id": getattr(result, "protective_sl_order_id", None),
                    "trend_runner_active": True,
                    "trend_runner_adjust_count": getattr(command.intent, "trend_runner_adjust_count", 0),
                    "boll_lower": command.intent.boll_lower,
                    "boll_middle": command.intent.boll_middle,
                    "boll_upper": command.intent.boll_upper,
                    "reason": command.intent.reason,
                },
                position_id=current_position_id,
            )
        if (
            getattr(command.intent, "three_stage_post_tp1_protective_sl_price", None) is not None
            and getattr(command.intent, "three_stage_tp1_consumed", False)
            and not getattr(command.intent, "trend_runner_active", False)
            and hasattr(self.journal, "append")
        ):
            self.journal.append(
                "THREE_STAGE_TP1_PROTECTIVE_SL_UPDATED",
                {
                    "side": command.intent.side,
                    "contracts": result.contracts,
                    "protective_sl_price": getattr(result, "protective_sl_price", "")
                    or getattr(command.intent, "three_stage_post_tp1_protective_sl_price", None),
                    "protective_sl_order_id": getattr(result, "protective_sl_order_id", None),
                    "old_protective_sl_order_id": getattr(
                        command.intent, "three_stage_post_tp1_protective_sl_order_id", None
                    ),
                    "avg_entry_price": command.intent.avg_entry_price,
                    "tp1_price": getattr(command.intent, "three_stage_tp1_price", None),
                    "tp1_ratio": getattr(command.intent, "three_stage_tp1_ratio", 0.0),
                    "tp2_price": getattr(command.intent, "three_stage_tp2_price", None),
                    "tp2_ratio": getattr(command.intent, "three_stage_tp2_ratio", 0.0),
                    "runner_ratio": getattr(command.intent, "three_stage_runner_ratio", 0.0),
                    "reason": command.intent.reason,
                    "retry_config": "PROTECTIVE_SL_RETRY_COUNT/PROTECTIVE_SL_RETRY_INTERVAL_SECONDS",
                },
                position_id=current_position_id,
            )
        self.state_store.save(
            LiveStateStore.from_strategy_state(
                position_id=current_position_id,
                symbol=self.trader.symbol,
                strategy_state=strategy_state_for_save,
                cash_before_position=cash_before_position,
            )
        )
        logger.warning(
            "LIVE TP update success | side=%s layer=%s price=%.4f contracts=%s tp_price=%s tp_mode=%s tp_plan=%s partial_tp=%s avg_entry=%.4f breakeven=%.4f tp_order_id=%s",
            command.intent.side,
            command.intent.layer_index,
            command.intent.price,
            result.contracts,
            result.tp_price,
            command.intent.tp_mode,
            getattr(command.intent, "tp_plan", "SINGLE"),
            getattr(command.intent, "partial_tp_price", None),
            command.intent.avg_entry_price,
            command.intent.breakeven_price,
            result.tp_order_id,
        )

    async def _apply_update_trend_sl_result(
        self,
        command: live_runtime_types.TradeCommand,
        result: Any,
    ) -> None:
        """Apply UPDATE_TREND_SL result: update entry protective SL state.

        Only called when execution SUCCEEDED (result.ok is True).  State
        updates are gated on success so that a failed UPDATE_TREND_SL
        leaves the old SL price / old order ID intact.
        """
        async with self.state_lock:
            current_position_id = self.execution_state.current_position_id
            cash_before_position = self.execution_state.cash_before_position
            self.execution_state.last_order_ts_ms = command.intent.ts_ms

            # Update entry protective SL order ID and price from result
            sl_order_id = getattr(result, "protective_sl_order_id", None)
            if sl_order_id:
                self.strategy.state.entry_protective_sl_order_id = sl_order_id
            sl_price = getattr(command.intent, "entry_protective_sl_price", None)
            if sl_price is not None:
                self.strategy.state.entry_protective_sl_price = sl_price
                self.strategy.state.trend_trailing_sl_price = sl_price
                self.strategy.state.trend_last_sl_update_ts_ms = command.intent.ts_ms
                self.strategy.state.last_tp_update_ts_ms = command.intent.ts_ms

            strategy_state_for_save = copy.deepcopy(self.strategy.state)
            equity = self.account_snapshot.equity

        if hasattr(self.journal, "append"):
            self.journal.append(
                "TREND_TRAILING_SL_UPDATED",
                {
                    "side": command.intent.side,
                    "entry_protective_sl_price": getattr(
                        command.intent, "entry_protective_sl_price", None
                    ),
                    "protective_sl_order_id": sl_order_id,
                    "boll_middle": command.intent.boll_middle,
                    "reason": command.intent.reason,
                },
                position_id=current_position_id,
            )

        self.state_store.save(
            LiveStateStore.from_strategy_state(
                position_id=current_position_id,
                symbol=self.trader.symbol,
                strategy_state=strategy_state_for_save,
                cash_before_position=cash_before_position,
            )
        )
        logger.warning(
            "LIVE trend SL update success | side=%s layer=%s "
            "entry_protective_sl_price=%s sl_order_id=%s",
            command.intent.side,
            command.intent.layer_index,
            getattr(command.intent, "entry_protective_sl_price", None),
            sl_order_id,
        )

    async def _apply_market_exit_runner_result(
        self,
        command: live_runtime_types.TradeCommand,
        result: Any,
    ) -> None:
        async with self.state_lock:
            current_position_id = self.execution_state.current_position_id
            self.execution_state.last_order_ts_ms = command.intent.ts_ms
            self.execution_state.trading_halted = True
            self.execution_state.halt_reason = "trend_runner_market_exit_waiting_flat"
            self.strategy.state.trend_runner_exit_reason = getattr(
                command.intent, "trend_runner_exit_reason", None
            ) or command.intent.reason
            strategy_state_for_save = copy.deepcopy(self.strategy.state)
            cash_before_position = self.execution_state.cash_before_position
        self.journal.record_trend_runner_market_exit(
            position_id=current_position_id,
            symbol=self.trader.symbol,
            intent=command.intent,
            result=result,
        )
        self.state_store.save(
            LiveStateStore.from_strategy_state(
                position_id=current_position_id,
                symbol=self.trader.symbol,
                strategy_state=strategy_state_for_save,
                cash_before_position=cash_before_position,
            )
        )
        logger.warning(
            "LIVE Trend Runner market exit success | side=%s reason=%s contracts_before=%s contracts_after=%s message=%s",
            command.intent.side,
            command.intent.reason,
            result.contracts_before,
            result.contracts_after,
            result.message,
        )

    async def _apply_entry_result(
        self,
        command: live_runtime_types.TradeCommand,
        result: Any,
        entry_cash_before: float | None,
    ) -> None:
        new_position_id = None
        async with self.state_lock:
            if self.execution_state.current_position_id is None:
                new_position_id = self.journal.new_position_id(
                    self.trader.symbol, command.intent.side, command.intent.ts_ms
                )
                self.execution_state.current_position_id = new_position_id
                self.execution_state.cash_before_position = entry_cash_before
            current_position_id = self.execution_state.current_position_id
            cash_before_position = self.execution_state.cash_before_position
            self.execution_state.last_order_ts_ms = command.intent.ts_ms
            self.strategy.state.tp_order_id = result.tp_order_id
            self.strategy.state.tp_order_ids = list(getattr(result, "tp_order_ids", ()) or [])
            self._maybe_apply_entry_protective_sl_state(command, result)
            # ── Middle Bucket Split state consistency ──────────────────
            # Entry paths (OPEN/ADD) also carry split execution status from
            # replace_take_profit.  Clear state if split was disabled.
            self._maybe_clear_middle_bucket_split_after_execution_result(
                result=result,
                current_position_id=current_position_id,
            )
            strategy_state_for_save = copy.deepcopy(self.strategy.state)
            equity = self.account_snapshot.equity
        entry_status = "CORE_FILLED_OK"
        self.journal.record_entry(
            position_id=current_position_id or new_position_id or "",
            intent=command.intent,
            result=result,
            cash_before_position=cash_before_position,
            equity=equity,
            extra={
                "symbol": self.trader.symbol,
                "entry_status": entry_status,
            },
        )
        async with self.state_lock:
            strategy_state_for_save = copy.deepcopy(self.strategy.state)
        if (
            getattr(command.intent, "tp_plan", "SINGLE") == "MIDDLE_RUNNER"
            and hasattr(self.journal, "append")
            and getattr(result, "middle_bucket_split_actual_order_mode", None) != "FINAL_FULL_SIZE"
        ):
            self.journal.append(
                "MIDDLE_RUNNER_PLANNED",
                {
                    "side": command.intent.side,
                    "layers": command.intent.layer_index,
                    "avg_entry_price": command.intent.avg_entry_price,
                    "first_tp_price": getattr(command.intent, "partial_tp_price", None),
                    "final_tp_price": command.intent.tp_price,
                    "first_close_ratio": getattr(command.intent, "partial_tp_ratio", 0.0),
                    "keep_ratio": getattr(command.intent, "middle_runner_keep_ratio", 0.0),
                    "boll_lower": command.intent.boll_lower,
                    "boll_middle": command.intent.boll_middle,
                    "boll_upper": command.intent.boll_upper,
                },
                position_id=current_position_id or new_position_id or "",
            )
        if (
            getattr(command.intent, "tp_plan", "SINGLE") == "THREE_STAGE_RUNNER"
            and hasattr(self.journal, "append")
            and getattr(result, "middle_bucket_split_actual_order_mode", None) != "FINAL_FULL_SIZE"
        ):
            self.journal.append(
                "THREE_STAGE_RUNNER_PLANNED",
                {
                    "side": command.intent.side,
                    "layers": command.intent.layer_index,
                    "avg_entry_price": command.intent.avg_entry_price,
                    "tp_plan": "THREE_STAGE_RUNNER",
                    "tp1_price": getattr(command.intent, "three_stage_tp1_price", None),
                    "tp1_ratio": getattr(command.intent, "three_stage_tp1_ratio", 0.0),
                    "tp2_price": getattr(command.intent, "three_stage_tp2_price", None),
                    "tp2_ratio": getattr(command.intent, "three_stage_tp2_ratio", 0.0),
                    "runner_tp_price": getattr(command.intent, "three_stage_runner_tp_price", None),
                    "runner_sl_price": getattr(command.intent, "three_stage_runner_sl_price", None),
                    "runner_ratio": getattr(command.intent, "three_stage_runner_ratio", 0.0),
                    "runner_sl_order_id": getattr(result, "protective_sl_order_id", None),
                    "boll_lower": command.intent.boll_lower,
                    "boll_middle": command.intent.boll_middle,
                    "boll_upper": command.intent.boll_upper,
                },
                position_id=current_position_id or new_position_id or "",
            )
        self.state_store.save(
            LiveStateStore.from_strategy_state(
                position_id=current_position_id,
                symbol=self.trader.symbol,
                strategy_state=strategy_state_for_save,
                cash_before_position=cash_before_position,
            )
        )
        logger.warning(
            "LIVE entry success | intent_type=%s side=%s layer=%s price=%.4f contracts=%s tp_price=%s tp_mode=%s tp_plan=%s partial_tp=%s avg_entry=%.4f breakeven=%.4f order_id=%s tp_order_id=%s entry_sl_ok=%s entry_sl_order_id=%s entry_status=%s",
            command.intent.intent_type,
            command.intent.side,
            command.intent.layer_index,
            command.intent.price,
            result.contracts,
            result.tp_price,
            command.intent.tp_mode,
            getattr(command.intent, "tp_plan", "SINGLE"),
            getattr(command.intent, "partial_tp_price", None),
            command.intent.avg_entry_price,
            command.intent.breakeven_price,
            result.order_id,
            result.tp_order_id,
            getattr(result, "protective_sl_ok", False),
            getattr(result, "protective_sl_order_id", None),
            entry_status,
        )
