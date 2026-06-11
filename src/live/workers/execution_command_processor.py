from __future__ import annotations

import asyncio
import copy
import html
import os
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.execution.trader import LiveTradeResult, PositionSnapshot, Trader
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
    SIDECAR_DIRTY_HALT,
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
from src.position_management.sidecar import entry_runtime as sidecar_entry_runtime
from src.position_management.sidecar import runtime_state as sidecar_runtime_state
from src.position_management.sidecar.reconciler import mark_sidecar_leg_force_closed
from src.position_management.sidecar.core_exit_safety import (
    classify_sidecar_core_final_exit_risk,
    open_sidecar_legs,
    sidecar_core_exit_client_order_id,
)
from src.position_management.sidecar.planner import (
    SidecarExecutionPlan,
    build_combined_entry_intent,
)
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.email_sender import EmailSender
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.live.portfolio_allocator_shadow import PortfolioAllocatorShadowRunner
    from src.live.portfolio_allocator_enforcer import (
        PortfolioAllocatorEnforcer,
        PortfolioAllocatorPrecheckResult,
    )

logger = get_logger(__name__)


@dataclass
class ExecutionCommandProcessor:
    state_lock: asyncio.Lock
    execution_state: live_runtime_types.ExecutionState
    account_snapshot: live_runtime_types.AccountSnapshot
    trader: Trader
    strategy: BollCvdShockReclaimStrategy
    journal: LiveTradeJournal
    state_store: LiveStateStore
    email_sender: EmailSender
    halt_alert_deduper: HaltAlertDeduper = field(default_factory=HaltAlertDeduper)
    sidecar_skip_first_layer: bool = True
    portfolio_allocator_shadow_runner: "PortfolioAllocatorShadowRunner | None" = None
    portfolio_allocator_enforcer: "PortfolioAllocatorEnforcer | None" = None
    _background_tasks: set[asyncio.Task] = field(default_factory=set, init=False)

    async def _send_halt_alert(
        self,
        *,
        halt_reason: str,
        side: str | None = None,
        layer: int | None = None,
        has_position: bool = True,
        sidecar_dirty: bool = False,
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
                    sidecar_dirty=sidecar_dirty,
                    manual_intervention_required=manual_intervention_required,
                    message=message,
                    extra=extra or {},
                ),
            )
        except Exception:
            logger.exception("PROCESSOR_HALT_ALERT_EXCEPTION | halt_reason=%s", halt_reason)

    def _schedule_portfolio_allocator_shadow(
        self,
        *,
        command: live_runtime_types.TradeCommand,
        sidecar_plan: SidecarExecutionPlan | None,
        position_id: str | None,
    ) -> None:
        """Schedule a fire-and-forget shadow allocation check.

        Does **not** await the result.  The shadow runner is called in a
        background task whose done callback cleans up the internal task set.
        If the shadow runner raises, the exception is logged but never
        propagated — the real order path is never affected.
        """
        runner = self.portfolio_allocator_shadow_runner
        if runner is None:
            return

        async def _shadow() -> None:
            await runner.run_entry_shadow_check(
                command=command,
                trader=self.trader,
                strategy=self.strategy,
                journal=self.journal,
                position_id=position_id,
                sidecar_plan=sidecar_plan,
            )

        task = asyncio.create_task(_shadow())
        self._background_tasks.add(task)

        def _cleanup(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)
            try:
                exc = t.exception()
                if exc is not None:
                    logger.error(
                        "PORTFOLIO_ALLOCATOR_SHADOW_BG_TASK_FAILED | error=%s",
                        exc,
                    )
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("PORTFOLIO_ALLOCATOR_SHADOW_BG_CLEANUP_FAILED")

        task.add_done_callback(_cleanup)

    async def _precheck_portfolio_allocator_enforce(
        self,
        *,
        command: live_runtime_types.TradeCommand,
        trader: "Trader" = None,
        sidecar_plan: SidecarExecutionPlan | None,
        position_id: str | None,
    ) -> "PortfolioAllocatorPrecheckResult | None":
        """Run enforce precheck before order placement.

        Returns ``None`` when enforcer is not configured (no interception).
        Returns a ``PortfolioAllocatorPrecheckResult`` otherwise — the caller
        must check ``.allowed`` before proceeding.
        """
        enforcer = self.portfolio_allocator_enforcer
        if enforcer is None:
            return None

        result = await enforcer.precheck_entry_allocation(
            command=command,
            trader=trader if trader is not None else self.trader,
            strategy=self.strategy,
            journal=self.journal,
            position_id=position_id,
            sidecar_plan=sidecar_plan,
        )

        if not result.allowed:
            logger.warning(
                "PORTFOLIO_ALLOCATOR_ENFORCE_REJECTED | symbol=%s intent_type=%s reason=%s",
                getattr(self.trader, "symbol", ""),
                getattr(command.intent, "intent_type", ""),
                result.reason,
            )

        return result

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
        if command.intent.intent_type != "UPDATE_TP" and current_position_id is None:
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

        # ── managed core contracts ───────────────────────────────────────
        entry_intent = core_position_view_helpers.with_entry_add_managed_core_contracts(
            intent=command.intent,
            strategy_state=self.strategy.state,
            account_core_position=self.account_snapshot.position,
            trader=self.trader,
        )
        if entry_intent is not command.intent:
            command = replace(command, intent=entry_intent)

        # ── guard: sidecar blocks NEAR_TP_REDUCE ─────────────────────────
        if command.intent.intent_type == "NEAR_TP_REDUCE" and getattr(
            self.strategy.state, "sidecar_enabled_for_position", False
        ):
            logger.error(
                "SIDECAR_BLOCKS_NEAR_TP_REDUCE | sidecar_enabled_for_position=True; NEAR_TP_REDUCE would reduce sidecar portion of OKX net position trading_halted=true halt_reason=sidecar_blocks_near_tp_reduce",
            )
            async with self.state_lock:
                self.execution_state.trading_halted = True
                self.execution_state.halt_reason = "sidecar_blocks_near_tp_reduce"
                self.strategy.state.sidecar_dirty = True
                self.strategy.state.sidecar_halt_reason = "sidecar_blocks_near_tp_reduce"
                current_position_id = self.execution_state.current_position_id
                cash_before_position = self.execution_state.cash_before_position
            if hasattr(self.journal, "append"):
                self.journal.append(
                    "SIDECAR_BLOCKS_NEAR_TP_REDUCE",
                    {
                        "sidecar_enabled_for_position": True,
                        "trading_halted": True,
                        "halt_reason": "sidecar_blocks_near_tp_reduce",
                        "intent_type": command.intent.intent_type,
                        "side": command.intent.side,
                        "manual_intervention_required": True,
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
                halt_reason="sidecar_blocks_near_tp_reduce",
                side=command.intent.side,
                layer=command.intent.layer_index,
                sidecar_dirty=True,
                manual_intervention_required=True,
                message="Sidecar is enabled; NEAR_TP_REDUCE blocked to avoid sidecar portion reduction.",
            )
            return None

        # ── sidecar combined entry plan ──────────────────────────────────
        raw_entry_command = command
        sidecar_plan: SidecarExecutionPlan | None = None
        precheck_result: "PortfolioAllocatorPrecheckResult | None" = None
        created_position_id_for_this_command = False
        if command.intent.intent_type in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}:
            async with self.state_lock:
                if self.execution_state.current_position_id is None:
                    self.execution_state.current_position_id = self.journal.new_position_id(
                        self.trader.symbol,
                        command.intent.side,
                        command.intent.ts_ms,
                    )
                    self.execution_state.cash_before_position = entry_cash_before
                    created_position_id_for_this_command = True
                current_position_id = self.execution_state.current_position_id
            combined_plan = build_combined_entry_intent(
                intent=command.intent,
                sidecar_enabled=bool(getattr(self.strategy.state, "sidecar_enabled_for_position", False)),
                account_equity_usdt=float(self.trader.account_equity_usdt),
                leverage=float(
                    getattr(self.trader, "leverage", getattr(getattr(self.trader, "config", None), "leverage", 50)) or 50
                ),
                sidecar_margin_pct=float(getattr(self.strategy.state, "sidecar_margin_pct", 0.0) or 0.0),
                sidecar_tp_pct=float(getattr(self.strategy.state, "sidecar_tp_pct", 0.0) or 0.0),
                position_id=current_position_id,
                sidecar_skip_first_layer=self.sidecar_skip_first_layer,
                contract_multiplier=getattr(self.trader, "contract_multiplier", Decimal("0.1")),
                contract_precision=getattr(self.trader, "contract_precision", Decimal("0.01")),
            )
            sidecar_plan = combined_plan.sidecar_plan

            # ── G05: schedule fire-and-forget shadow allocator check ─────
            self._schedule_portfolio_allocator_shadow(
                command=raw_entry_command,
                sidecar_plan=sidecar_plan,
                position_id=current_position_id,
            )

            # ── G06a: enforce allocator precheck (awaited, may reject) ──
            precheck_result = await self._precheck_portfolio_allocator_enforce(
                command=raw_entry_command,
                sidecar_plan=sidecar_plan,
                position_id=current_position_id,
            )
            if precheck_result is not None and not precheck_result.allowed:
                if created_position_id_for_this_command:
                    async with self.state_lock:
                        if self.execution_state.current_position_id == current_position_id:
                            self.execution_state.current_position_id = None
                            self.execution_state.cash_before_position = None
                            logger.warning(
                                "PORTFOLIO_ALLOCATOR_ENFORCE_REJECTED_POSITION_ID_ROLLED_BACK | "
                                "symbol=%s intent_type=%s position_id=%s reason=%s",
                                getattr(self.trader, "symbol", ""),
                                getattr(command.intent, "intent_type", ""),
                                current_position_id,
                                precheck_result.reason,
                            )
                return None

            command = replace(command, intent=combined_plan.execution_intent)

        # ── sidecar core final exit safety guard ─────────────────────────
        if command.intent.intent_type == "UPDATE_TP":
            alignment_result = await self._align_sidecar_tp_with_unsafe_core_final_exit(command)
            if alignment_result is not None:
                return alignment_result

        # ── execute intent ───────────────────────────────────────────────
        result = await self.trader.execute_intent(command.intent)

        # ── G06a: commit projected snapshot after fill ──────────────────
        if precheck_result is not None and self.portfolio_allocator_enforcer is not None:
            await self.portfolio_allocator_enforcer.commit_projected_snapshot_after_fill(
                precheck_result=precheck_result,
                live_result=result,
                journal=self.journal,
                position_id=current_position_id,
            )

        if not result.ok:
            # ── UPDATE_TP failure: arm delayed market exit ──────────────
            if command.intent.intent_type == "UPDATE_TP":
                # Only arm if not already armed for the same reason (avoid resetting countdown).
                already_armed_same = (
                    getattr(self.strategy.state, "delayed_market_exit_armed", False)
                    and getattr(self.strategy.state, "delayed_market_exit_reason", None) == "core_tp_place_failed_delayed_market_exit_armed"
                )
                if not already_armed_same:
                    logger.error(
                        "UPDATE_TP_FAILED | position_id=%s side=%s message=%s delayed_market_exit_armed=true",
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
                            reason="core_tp_place_failed_delayed_market_exit_armed",
                            context="update_tp_replace_take_profit_failed",
                            source_event="UPDATE_TP_FAILED",
                            now_ms=command.intent.ts_ms,
                            error=getattr(result, "message", str(result)),
                        )
                    if hasattr(self.journal, "append"):
                        self.journal.append(
                            "UPDATE_TP_FAILED_DELAYED_MARKET_EXIT_ARMED",
                            {
                                "position_id": current_position_id,
                                "side": command.intent.side,
                                "intent_type": "UPDATE_TP",
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
                    await self._send_halt_alert(
                        halt_reason="core_tp_place_failed_delayed_market_exit_armed",
                        side=command.intent.side,
                        layer=command.intent.layer_index,
                        manual_intervention_required=True,
                        message=(
                            "UPDATE_TP / replace_take_profit failed. "
                            "Delayed market exit armed (30 min countdown). NO immediate market exit."
                        ),
                        extra={
                            "intent_type": "UPDATE_TP",
                            "delayed_market_exit_armed": True,
                            "error": getattr(result, "message", str(result)),
                        },
                    )
                else:
                    logger.warning(
                        "UPDATE_TP_FAILED_ALREADY_ARMED | position_id=%s side=%s — delayed exit already armed, skipping re-arm",
                        current_position_id,
                        command.intent.side,
                    )
                return result

            # ── Near-TP reduce: protective SL failed ────────────────────
            if (
                command.intent.intent_type == "NEAR_TP_REDUCE"
                and not getattr(result, "protective_sl_ok", True)
                and not getattr(result, "near_tp_exit_all", False)
            ):
                logger.error(
                    "NEAR_TP_PROTECTIVE_SL_FAILED | position_id=%s side=%s message=%s delayed_market_exit_armed=true",
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
                        reason="near_tp_protective_sl_failed_delayed_market_exit_armed",
                        context="near_tp_protective_sl_failed",
                        source_event="NEAR_TP_PROTECTIVE_SL_FAILED",
                        now_ms=command.intent.ts_ms,
                        error=getattr(result, "message", str(result)),
                    )
                if hasattr(self.journal, "append"):
                    self.journal.append(
                        "NEAR_TP_PROTECTIVE_SL_FAILED",
                        {
                            "position_id": current_position_id,
                            "side": command.intent.side,
                            "message": getattr(result, "message", ""),
                            "delayed_market_exit_armed": True,
                            "halt_reason": "near_tp_protective_sl_failed_delayed_market_exit_armed",
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
                    halt_reason="near_tp_protective_sl_failed_delayed_market_exit_armed",
                    side=command.intent.side,
                    layer=command.intent.layer_index,
                    manual_intervention_required=True,
                    message=(
                        "Near-TP protective SL failed. "
                        "Delayed market exit armed (30 min countdown). NO immediate market exit."
                    ),
                    extra={
                        "protective_sl_ok": False,
                        "delayed_market_exit_armed": True,
                        "error": getattr(result, "message", str(result)),
                    },
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
        elif command.intent.intent_type == "NEAR_TP_REDUCE":
            await self._apply_near_tp_reduce_result(command, result)
        elif command.intent.intent_type == "MARKET_EXIT_RUNNER":
            await self._apply_market_exit_runner_result(command, result)
        else:
            await self._apply_entry_result(command, result, entry_cash_before, sidecar_plan)

        return result

    # ── sidecar core final exit safety guard ────────────────────────────

    async def _align_sidecar_tp_with_unsafe_core_final_exit(
        self, command: live_runtime_types.TradeCommand,
    ) -> LiveTradeResult | None:
        """Realign sidecar TP orders when core final TP would leave sidecar exposed.

        Returns:
            None — alignment skipped or succeeded; core UPDATE_TP should proceed.
            LiveTradeResult — market exit succeeded; core UPDATE_TP must be skipped.
        Raises:
            RuntimeError — market exit failed; manual intervention required.
        """
        # 1. Guard: sidecar not enabled
        if not getattr(self.strategy.state, "sidecar_enabled_for_position", False):
            return None

        # 2. Guard: no open sidecar legs
        sidecar_legs: list[dict[str, Any]] = list(getattr(self.strategy.state, "sidecar_legs", []) or [])
        if not open_sidecar_legs(sidecar_legs):
            return None

        # 3. Read core TP params
        side = command.intent.side
        core_tp_price = command.intent.tp_price
        breakeven_price: float | None = getattr(command.intent, "breakeven_price", None)
        if breakeven_price is None or breakeven_price <= 0:
            breakeven_price = getattr(self.strategy.state, "breakeven_price", None)

        # 4. Classify risk
        risk = classify_sidecar_core_final_exit_risk(
            side=side,
            core_tp_price=core_tp_price,
            breakeven_price=breakeven_price,
            sidecar_legs=sidecar_legs,
        )
        if not risk.risky:
            return None

        logger.warning(
            "SIDECAR_CORE_FINAL_EXIT_RISK_DETECTED | risk=%s position_id=%s side=%s core_tp_price=%.4f breakeven_price=%s risky_leg_ids=%s",
            risk.reason,
            self.execution_state.current_position_id,
            side,
            core_tp_price,
            breakeven_price,
            list(risk.risky_leg_ids),
        )

        # 5. Realign all open sidecar legs (not just risky ones)
        async with self.state_lock:
            current_position_id = self.execution_state.current_position_id
            cash_before_position = self.execution_state.cash_before_position

        try:
            for leg in sidecar_legs:
                status = str(leg.get("status") or "")
                if status not in {"OPEN", "OPEN_UNPROTECTED"}:
                    continue

                old_tp_order_id = leg.get("tp_order_id")
                old_tp_price = leg.get("tp_price")
                contracts = leg["contracts"]
                leg_id = str(leg.get("leg_id") or "")

                # Cancel old sidecar TP if exists
                if old_tp_order_id:
                    ok = await self.trader.cancel_sidecar_take_profit(str(old_tp_order_id))
                    if not ok:
                        raise RuntimeError(
                            f"cancel_sidecar_tp_failed leg_id={leg_id} order_id={old_tp_order_id}"
                        )

                # Place new TP aligned to core final exit with unique clOrdId
                client_order_id = sidecar_core_exit_client_order_id(
                    position_id=current_position_id,
                    leg_id=leg_id,
                    old_tp_order_id=str(old_tp_order_id) if old_tp_order_id else None,
                    ts_ms=command.intent.ts_ms,
                )
                new_tp_order_id = await self.trader.place_sidecar_fixed_take_profit(
                    side=side,
                    contracts=contracts,
                    tp_price=core_tp_price,
                    client_order_id=client_order_id,
                )

                # Update leg state
                leg["tp_price"] = float(core_tp_price)
                leg["tp_order_id"] = new_tp_order_id
                leg["updated_ts_ms"] = command.intent.ts_ms
                leg["core_exit_aligned"] = True
                leg["core_exit_alignment_reason"] = risk.reason

                logger.warning(
                    "SIDECAR_TP_REALIGNED_TO_CORE_EXIT | position_id=%s side=%s core_tp_price=%.4f reason=%s leg_id=%s old_tp_price=%s old_tp_order_id=%s new_tp_order_id=%s client_order_id=%s",
                    current_position_id,
                    side,
                    core_tp_price,
                    risk.reason,
                    leg_id,
                    old_tp_price,
                    old_tp_order_id,
                    new_tp_order_id,
                    client_order_id,
                )

                self.journal.append(
                    "SIDECAR_TP_REALIGNED_TO_CORE_EXIT",
                    {
                        "side": side,
                        "core_tp_price": core_tp_price,
                        "reason": risk.reason,
                        "leg_id": leg_id,
                        "old_tp_price": old_tp_price,
                        "old_tp_order_id": old_tp_order_id,
                        "new_tp_order_id": new_tp_order_id,
                        "client_order_id": client_order_id,
                    },
                    position_id=current_position_id,
                )

            # Refresh sidecar state totals
            sidecar_runtime_state.refresh_sidecar_state_totals(
                self.strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10"))
            )
            self.state_store.save(
                LiveStateStore.from_strategy_state(
                    position_id=current_position_id,
                    symbol=self.trader.symbol,
                    strategy_state=self.strategy.state,
                    cash_before_position=cash_before_position,
                )
            )

        except Exception as exc:
            logger.error(
                "SIDECAR_CORE_EXIT_ALIGNMENT_FAILED | position_id=%s side=%s core_tp_price=%.4f reason=%s error=%s delayed_market_exit_armed=true",
                current_position_id,
                side,
                core_tp_price,
                risk.reason,
                exc,
            )

            # ── arm delayed emergency exit (do NOT market-exit immediately) ──
            delay_seconds = float(
                os.getenv("SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_AUTO_EXIT_DELAY_SECONDS", "1800")
            )

            async with self.state_lock:
                self.execution_state.trading_halted = True
                self.execution_state.halt_reason = (
                    "sidecar_core_exit_alignment_failed_delayed_market_exit_armed"
                )
                self.execution_state.halt_until_ts_ms = None
                self.strategy.state.sidecar_dirty = True
                self.strategy.state.sidecar_halt_reason = (
                    "sidecar_core_exit_alignment_failed_delayed_market_exit_armed"
                )
                # ── Persist via unified delayed market exit state ────────
                dme.arm_delayed_market_exit(
                    strategy_state=self.strategy.state,
                    execution_state=self.execution_state,
                    position_id=current_position_id,
                    side=side,
                    reason="sidecar_core_exit_alignment_failed_delayed_market_exit_armed",
                    context="sidecar_core_exit_alignment_failed",
                    source_event="SIDECAR_CORE_EXIT_ALIGNMENT_FAILED",
                    now_ms=command.intent.ts_ms,
                    delay_seconds=delay_seconds,
                    error=str(exc),
                )

            self.journal.append(
                "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED",
                {
                    "side": side,
                    "core_tp_price": core_tp_price,
                    "risk_reason": risk.reason,
                    "error": str(exc),
                    "delay_seconds": delay_seconds,
                    "manual_intervention_required": True,
                    **dme.delayed_market_exit_payload(self.strategy.state),
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

            logger.warning(
                "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED | position_id=%s side=%s core_tp_price=%.4f reason=%s delay_seconds=%.0f",
                current_position_id,
                side,
                core_tp_price,
                risk.reason,
                delay_seconds,
            )

            # Send CRITICAL arm email
            arm_subject = "CRITICAL: Sidecar core-exit alignment failed; delayed market exit armed"
            arm_content = (
                "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
                "<h2>Sidecar Core-Exit Alignment Failed</h2>"
                "<p>Sidecar TP realignment failed. Trading has been halted.</p>"
                "<p>A delayed market exit task has been scheduled.</p>"
                f"<p><b>position_id:</b> {html.escape(str(current_position_id))}</p>"
                f"<p><b>side:</b> {html.escape(str(side))}</p>"
                f"<p><b>core_tp_price:</b> {core_tp_price:.4f}</p>"
                f"<p><b>risk_reason:</b> {html.escape(risk.reason)}</p>"
                f"<p><b>error:</b> {html.escape(str(exc))}</p>"
                f"<p><b>delay_seconds:</b> {delay_seconds:.0f}</p>"
                f"<p><b>action:</b> manual window open</p>"
                "</div>"
            )
            ok = await self.email_sender.send_email_async(arm_subject, arm_content, content_type="html")
            if not ok:
                logger.error("Failed to send sidecar core-exit delayed arm email")

            # ── Delayed market exit is now managed by the unified DME phase ──
            # The account sync worker calls run_delayed_market_exit_phase()
            # which checks persisted delayed_market_exit_* state.  No background
            # task is needed — the persisted state survives restarts and the
            # account sync phase is the authoritative executor.
            # (Old _delayed_sidecar_core_exit_market_exit background task removed.)

            # Return synthetic ok result — skip core UPDATE_TP, avoid failure-handler rollback
            return LiveTradeResult(
                ok=True,
                action="SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED",
                order_id=None,
                tp_order_id=None,
                contracts="0",
                tp_price=str(core_tp_price),
                message="sidecar_core_exit_alignment_failed_delayed_market_exit_armed",
            )

    # ── DEPRECATED: old background task methods ────────────────────────────
    # These methods are no longer called.  The unified DME phase
    # (src/live/account_sync/delayed_market_exit_phase.py) now handles all
    # delayed market exit execution via persisted state.  Kept only for
    # reference; will be removed in a future cleanup.

    def _schedule_background_task(self, coro) -> None:
        """DEPRECATED: unused.  DME phase handles delayed exits now."""
        raise RuntimeError("_schedule_background_task is deprecated and must not be called")

    async def _delayed_sidecar_core_exit_market_exit(self, **kwargs) -> None:
        """DEPRECATED: unused.  DME phase handles delayed exits now."""
        raise RuntimeError("_delayed_sidecar_core_exit_market_exit is deprecated and must not be called")

    async def _delayed_sidecar_core_exit_market_exit_impl(self, **kwargs) -> None:
        """DEPRECATED: unused.  DME phase handles delayed exits now."""
        raise RuntimeError("_delayed_sidecar_core_exit_market_exit_impl is deprecated and must not be called")

    # ── result application helpers ───────────────────────────────────────

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
                    "retry_config": "NEAR_TP_PROTECTIVE_SL_RETRY_COUNT/NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS",
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

    async def _apply_near_tp_reduce_result(
        self,
        command: live_runtime_types.TradeCommand,
        result: Any,
    ) -> None:
        fail_action = None
        if not getattr(result, "protective_sl_ok", False) and getattr(result, "near_tp_exit_all", False):
            fail_action = "MARKET_EXIT"
        remaining_position: PositionSnapshot | None = None
        remaining_position_sync_error: str | None = None
        if getattr(result, "protective_sl_ok", False) and not getattr(result, "near_tp_exit_all", False):
            try:
                position = await self.trader.fetch_position_snapshot()
                if position.has_position and position.side == command.intent.side:
                    remaining_position = position
                else:
                    remaining_position_sync_error = (
                        f"position_absent_or_side_mismatch has_position={position.has_position} side={position.side}"
                    )
            except Exception:
                remaining_position_sync_error = "fetch_position_failed"
                logger.exception("NEAR_TP_STATE_PROTECTED | failed_to_sync_remaining_position_before_save")
        async with self.state_lock:
            current_position_id = self.execution_state.current_position_id
            cash_before_position = self.execution_state.cash_before_position
            self.execution_state.last_order_ts_ms = command.intent.ts_ms
            self.strategy.state.tp_order_id = result.tp_order_id
            self.strategy.state.tp_order_ids = list(getattr(result, "tp_order_ids", ()) or [])
            near_tp_state_synced = False
            if getattr(result, "near_tp_exit_all", False):
                self.execution_state.trading_halted = True
                self.execution_state.halt_reason = "near_tp_exit_all_waiting_flat"
            elif getattr(result, "protective_sl_ok", False):
                if remaining_position is not None:
                    position_cost_runtime.sync_strategy_cost_from_position(
                        self.strategy,
                        remaining_position,
                        restore_from_position=startup_basic_restore.restore_strategy_from_position,
                    )
                    self.account_snapshot.position = remaining_position
                    self.trader.position_contracts = remaining_position.contracts
                    near_tp_state_synced = True
                else:
                    self.execution_state.trading_halted = True
                    self.execution_state.halt_reason = "near_tp_protected_sync_failed"
                    logger.warning(
                        "NEAR_TP_STATE_PROTECTED_SYNC_FAILED | position_id=%s reason=%s trading_halted=true",
                        current_position_id,
                        remaining_position_sync_error or "unknown",
                    )
                self.strategy.state.near_tp_protected = True
                self.strategy.state.near_tp_reduce_pending = False
                strategy_config = getattr(self.strategy, "config", None)
                self.strategy.state.near_tp_add_disabled = bool(
                    getattr(strategy_config, "near_tp_disable_add_after_reduce", True)
                )
                self.strategy.state.near_tp_protective_sl_price = getattr(
                    command.intent, "near_tp_protective_sl_price", None
                ) or live_config_helpers._parse_optional_float(getattr(result, "protective_sl_price", ""))
                self.strategy.state.near_tp_protective_sl_order_id = getattr(
                    result, "protective_sl_order_id", None
                )
                self.strategy.state.tp_plan = "SINGLE"
                self.strategy.state.partial_tp_price = None
                self.strategy.state.partial_tp_ratio = 0.0
                self.strategy.state.partial_tp_consumed = True
                logger.warning(
                    "NEAR_TP_STATE_PROTECTED | position_id=%s protective_sl_order_id=%s protective_sl_price=%s",
                    current_position_id,
                    self.strategy.state.near_tp_protective_sl_order_id,
                    self.strategy.state.near_tp_protective_sl_price,
                )
            strategy_state_for_save = copy.deepcopy(self.strategy.state)
            equity = self.account_snapshot.equity
        self.journal.record_near_tp_reduce(
            position_id=current_position_id,
            symbol=self.trader.symbol,
            intent=command.intent,
            result=result,
            protective_sl_fail_action=fail_action,
        )
        if getattr(result, "protective_sl_ok", False) and not getattr(result, "near_tp_exit_all", False):
            self.state_store.save(
                LiveStateStore.from_strategy_state(
                    position_id=current_position_id,
                    symbol=self.trader.symbol,
                    strategy_state=strategy_state_for_save,
                    cash_before_position=cash_before_position,
                )
            )
            if not near_tp_state_synced:
                subject = "Near-TP protected but position sync failed"
                content = (
                    "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
                    "<h2>Near-TP protected but position sync failed</h2>"
                    "<p>Reduce succeeded and protective SL was placed. Protected state was saved.</p>"
                    "<p>Trading is temporarily halted until account sync refreshes the position.</p>"
                    f"<p><b>position_id:</b> {html.escape(str(current_position_id))}</p>"
                    f"<p><b>protective_sl_order_id:</b> {html.escape(str(getattr(result, 'protective_sl_order_id', None)))}</p>"
                    f"<p><b>reason:</b> {html.escape(str(remaining_position_sync_error or 'unknown'))}</p>"
                    "</div>"
                )
                ok = await self.email_sender.send_email_async(subject, content, content_type="html")
                if not ok:
                    logger.error("Failed to send Near-TP protected sync failure email")
        # ── Near-TP final TP placement failed but protective SL ok ─────
        # The position is protected by SL, but final TP is missing.
        # This is still a TP placement failure → arm delayed exit.
        near_tp_tp_failed = (
            getattr(result, "reduce_filled", False)
            and not getattr(result, "tp_ok", True)
            and getattr(result, "protective_sl_ok", False)
            and not getattr(result, "near_tp_exit_all", False)
        )
        if near_tp_tp_failed:
            async with self.state_lock:
                already_armed = getattr(self.strategy.state, "delayed_market_exit_armed", False)
                if not already_armed:
                    arm_payload = dme.arm_delayed_market_exit(
                        strategy_state=self.strategy.state,
                        execution_state=self.execution_state,
                        position_id=current_position_id,
                        side=command.intent.side,
                        reason="core_tp_place_failed_delayed_market_exit_armed",
                        context="near_tp_final_tp_replacement_failed",
                        source_event="NEAR_TP_FINAL_TP_REPLACE_FAILED",
                        now_ms=command.intent.ts_ms,
                        error=getattr(result, "message", ""),
                    )
                current_position_id = self.execution_state.current_position_id
                cash_before_position = self.execution_state.cash_before_position
            if not already_armed:
                if hasattr(self.journal, "append"):
                    self.journal.append(
                        "NEAR_TP_FINAL_TP_REPLACE_FAILED",
                        {
                            "position_id": current_position_id,
                            "side": command.intent.side,
                            "reduce_filled": True,
                            "tp_ok": False,
                            "protective_sl_ok": True,
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
                await self._send_halt_alert(
                    halt_reason="core_tp_place_failed_delayed_market_exit_armed",
                    side=command.intent.side,
                    layer=command.intent.layer_index,
                    manual_intervention_required=True,
                    message=(
                        "Near-TP final TP replacement failed (protective SL is OK). "
                        "Delayed market exit armed (30 min countdown). NO immediate market exit."
                    ),
                    extra={
                        "reduce_filled": True,
                        "tp_ok": False,
                        "protective_sl_ok": True,
                        "delayed_market_exit_armed": True,
                    },
                )

        if fail_action == "MARKET_EXIT":
            # ── Protective SL failed → arm delayed market exit ──────────
            async with self.state_lock:
                arm_payload = dme.arm_delayed_market_exit(
                    strategy_state=self.strategy.state,
                    execution_state=self.execution_state,
                    position_id=current_position_id,
                    side=command.intent.side,
                    reason="near_tp_protective_sl_failed_delayed_market_exit_armed",
                    context="near_tp_protective_sl_failed",
                    source_event="NEAR_TP_PROTECTIVE_SL_FAILED",
                    now_ms=command.intent.ts_ms,
                    error=result.message,
                )
            subject = "Near-TP protective SL failed; delayed market exit armed"
            content = (
                "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
                "<h2>Near-TP protective SL failed</h2>"
                "<p>Delayed market exit armed (30 min countdown). NO immediate market exit.</p>"
                "<p>Please check OKX position and decide whether to manually exit.</p>"
                f"<p><b>position_id:</b> {html.escape(str(current_position_id))}</p>"
                f"<p><b>message:</b> {html.escape(result.message)}</p>"
                "</div>"
            )
            ok = await self.email_sender.send_email_async(subject, content, content_type="html")
            if not ok:
                logger.error("Failed to send Near-TP delayed market exit arm email")
        logger.warning(
            "LIVE Near-TP reduce success | side=%s layer=%s price=%.4f contracts_before=%s contracts_reduced=%s contracts_after=%s tp_order_id=%s protective_sl_ok=%s protective_sl_order_id=%s near_tp_exit_all=%s equity=%.4f",
            command.intent.side,
            command.intent.layer_index,
            command.intent.price,
            result.contracts_before,
            result.contracts_reduced,
            result.contracts_after,
            result.tp_order_id,
            result.protective_sl_ok,
            result.protective_sl_order_id,
            result.near_tp_exit_all,
            equity or 0.0,
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
        sidecar_plan: SidecarExecutionPlan | None,
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
            # ── Middle Bucket Split state consistency ──────────────────
            # Entry paths (OPEN/ADD) also carry split execution status from
            # replace_take_profit.  Clear state if split was disabled.
            self._maybe_clear_middle_bucket_split_after_execution_result(
                result=result,
                current_position_id=current_position_id,
            )
            strategy_state_for_save = copy.deepcopy(self.strategy.state)
            equity = self.account_snapshot.equity
        sidecar_ok = True
        sidecar_halt_reason: str | None = None
        if sidecar_plan is not None:
            sidecar_ok = await sidecar_entry_runtime.attach_sidecar_after_combined_entry(
                trader=self.trader,
                strategy_state=self.strategy.state,
                execution_state=self.execution_state,
                intent=command.intent,
                sidecar_plan=sidecar_plan,
                journal=self.journal,
                state_store=self.state_store,
                trader_symbol=self.trader.symbol,
                fee_buffer_pct=self.strategy.config.breakeven_fee_buffer_pct,
                email_sender=self.email_sender,
                halt_alert_deduper=self.halt_alert_deduper,
            )
            if not sidecar_ok:
                sidecar_halt_reason = self.execution_state.halt_reason

        entry_status = "CORE_FILLED_SIDECAR_OK" if sidecar_ok else "CORE_FILLED_SIDECAR_FAILED"
        self.journal.record_entry(
            position_id=current_position_id or new_position_id or "",
            intent=command.intent,
            result=result,
            cash_before_position=cash_before_position,
            equity=equity,
            extra={
                "symbol": self.trader.symbol,
                "sidecar_ok": sidecar_ok,
                "sidecar_halt_reason": sidecar_halt_reason,
                "entry_status": entry_status,
            },
        )
        async with self.state_lock:
            strategy_state_for_save = copy.deepcopy(self.strategy.state)
        if not sidecar_ok:
            logger.error(
                "LIVE core entry success but sidecar failed | position_id=%s intent_type=%s side=%s layer=%s trading_halted=true halt_reason=%s entry_status=%s",
                current_position_id or new_position_id,
                command.intent.intent_type,
                command.intent.side,
                command.intent.layer_index,
                sidecar_halt_reason,
                entry_status,
            )
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
        if sidecar_ok:
            logger.warning(
                "LIVE entry success | intent_type=%s side=%s layer=%s price=%.4f contracts=%s tp_price=%s tp_mode=%s tp_plan=%s partial_tp=%s avg_entry=%.4f breakeven=%.4f order_id=%s tp_order_id=%s entry_status=%s",
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
                entry_status,
            )
        else:
            logger.error(
                "LIVE core entry success but sidecar failed | intent_type=%s side=%s layer=%s price=%.4f contracts=%s tp_price=%s tp_mode=%s tp_plan=%s partial_tp=%s avg_entry=%.4f breakeven=%.4f order_id=%s tp_order_id=%s entry_status=%s trading_halted=true halt_reason=%s",
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
                entry_status,
                sidecar_halt_reason,
            )
