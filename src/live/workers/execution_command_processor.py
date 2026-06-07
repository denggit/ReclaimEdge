from __future__ import annotations

import asyncio
import copy
import html
import os
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from src.execution.trader import LiveTradeResult, PositionSnapshot, Trader
from src.live import config_helpers as live_config_helpers
from src.live import runtime_types as live_runtime_types
from src.live.account_sync import flat_balance as live_flat_balance
from src.live.startup_recovery import basic_restore as startup_basic_restore
from src.live.workers import strategy_tick_worker as strategy_tick_worker_module
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
from src.risk.rolling_loss_guard import ROLLING_LOSS_HALT_REASONS
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.email_sender import EmailSender
from src.utils.log import get_logger

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
    sidecar_skip_first_layer: bool = True
    _background_tasks: set[asyncio.Task] = field(default_factory=set, init=False)

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

        # ── trading halted guard ─────────────────────────────────────────
        async with self.state_lock:
            rolling_management_allowed = (
                self.execution_state.trading_halted
                and self.execution_state.halt_reason in ROLLING_LOSS_HALT_REASONS
                and command.intent.intent_type in strategy_tick_worker_module.POSITION_MANAGEMENT_INTENTS
            )
            if self.execution_state.trading_halted and not rolling_management_allowed:
                logger.warning(
                    "EXECUTION_SKIPPED | reason=trading_halted intent_type=%s side=%s tick_ts_ms=%s",
                    command.intent.intent_type,
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
            return None

        # ── sidecar combined entry plan ──────────────────────────────────
        sidecar_plan: SidecarExecutionPlan | None = None
        if command.intent.intent_type in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}:
            async with self.state_lock:
                if self.execution_state.current_position_id is None:
                    self.execution_state.current_position_id = self.journal.new_position_id(
                        self.trader.symbol,
                        command.intent.side,
                        command.intent.ts_ms,
                    )
                    self.execution_state.cash_before_position = entry_cash_before
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
            command = replace(command, intent=combined_plan.execution_intent)
            sidecar_plan = combined_plan.sidecar_plan

        # ── sidecar core final exit safety guard ─────────────────────────
        if command.intent.intent_type == "UPDATE_TP":
            alignment_result = await self._align_sidecar_tp_with_unsafe_core_final_exit(command)
            if alignment_result is not None:
                return alignment_result

        # ── execute intent ───────────────────────────────────────────────
        result = await self.trader.execute_intent(command.intent)
        if not result.ok:
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
                os.getenv("SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_AUTO_EXIT_DELAY_SECONDS", "900")
            )
            arm_payload = {
                "side": side,
                "core_tp_price": core_tp_price,
                "risk_reason": risk.reason,
                "error": str(exc),
                "delay_seconds": delay_seconds,
                "manual_intervention_required": delay_seconds < 0,
            }

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

            self.journal.append(
                "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED",
                arm_payload,
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

            # Schedule background delayed task (non-blocking)
            if delay_seconds >= 0:
                self._schedule_background_task(
                    self._delayed_sidecar_core_exit_market_exit(
                        delay_seconds=delay_seconds,
                        side=side,
                        position_id=current_position_id,
                        cash_before_position=cash_before_position,
                        core_tp_price=core_tp_price,
                        risk_reason=risk.reason,
                        original_error=str(exc),
                        ts_ms=command.intent.ts_ms,
                    )
                )

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

    # ── delayed sidecar core exit helpers ────────────────────────────────

    def _schedule_background_task(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _delayed_sidecar_core_exit_market_exit(
        self,
        *,
        delay_seconds: float,
        side: str,
        position_id: str | None,
        cash_before_position: float | None,
        core_tp_price: float,
        risk_reason: str,
        original_error: str,
        ts_ms: int,
    ) -> None:
        """Background task: wait delay_seconds, then check position and market-exit if needed."""
        try:
            await self._delayed_sidecar_core_exit_market_exit_impl(
                delay_seconds=delay_seconds,
                side=side,
                position_id=position_id,
                cash_before_position=cash_before_position,
                core_tp_price=core_tp_price,
                risk_reason=risk_reason,
                original_error=original_error,
                ts_ms=ts_ms,
            )
        except Exception as exc:
            logger.exception(
                "SIDECAR_CORE_EXIT_DELAYED_MARKET_EXIT_TASK_FAILED | position_id=%s side=%s error=%s",
                position_id,
                side,
                exc,
            )
            async with self.state_lock:
                self.execution_state.trading_halted = True
                self.execution_state.halt_reason = "sidecar_core_exit_delayed_market_exit_task_failed"
                self.execution_state.halt_until_ts_ms = None
                self.strategy.state.sidecar_dirty = True
                self.strategy.state.sidecar_halt_reason = "sidecar_core_exit_delayed_market_exit_task_failed"

            self.journal.append(
                "SIDECAR_CORE_EXIT_DELAYED_MARKET_EXIT_TASK_FAILED",
                {
                    "side": side,
                    "core_tp_price": core_tp_price,
                    "risk_reason": risk_reason,
                    "original_error": original_error,
                    "delay_seconds": delay_seconds,
                    "error": str(exc),
                    "trading_halted": True,
                    "manual_intervention_required": True,
                },
                position_id=position_id,
            )

            self.state_store.save(
                LiveStateStore.from_strategy_state(
                    position_id=position_id,
                    symbol=self.trader.symbol,
                    strategy_state=self.strategy.state,
                    cash_before_position=cash_before_position,
                )
            )

            task_fail_subject = "CRITICAL: Sidecar core-exit delayed market exit TASK FAILED"
            task_fail_content = (
                "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
                "<h2>Delayed Market Exit Task Crashed — Manual Intervention Required</h2>"
                f"<p><b>position_id:</b> {html.escape(str(position_id))}</p>"
                f"<p><b>side:</b> {html.escape(str(side))}</p>"
                f"<p><b>core_tp_price:</b> {core_tp_price:.4f}</p>"
                f"<p><b>risk_reason:</b> {html.escape(risk_reason)}</p>"
                f"<p><b>error:</b> {html.escape(str(exc))}</p>"
                f"<p><b>action:</b> task crashed</p>"
                "</div>"
            )
            ok = await self.email_sender.send_email_async(task_fail_subject, task_fail_content, content_type="html")
            if not ok:
                logger.error("Failed to send sidecar core-exit delayed task failed email")

    async def _delayed_sidecar_core_exit_market_exit_impl(
        self,
        *,
        delay_seconds: float,
        side: str,
        position_id: str | None,
        cash_before_position: float | None,
        core_tp_price: float,
        risk_reason: str,
        original_error: str,
        ts_ms: int,
    ) -> None:
        """Implementation: wait delay, check position, and market-exit if needed."""
        # 1. Wait for the configured delay
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        # 2. Re-check OKX position — fetch failure does NOT mean flat
        position = None
        position_fetch_error: Exception | None = None
        try:
            position = await self.trader.fetch_position_snapshot()
        except Exception as exc:
            position_fetch_error = exc
            logger.warning(
                "SIDECAR_CORE_EXIT_DELAYED_MARKET_EXIT_POSITION_FETCH_FAILED | position_id=%s side=%s error=%s will_attempt_market_exit=true",
                position_id,
                side,
                exc,
            )

        # 3. Only skip market exit when fetch succeeded AND position is truly flat/wrong-side
        if position_fetch_error is None and position is not None and (
            not position.has_position or position.side != side
        ):
            async with self.state_lock:
                self.execution_state.trading_halted = True
                self.execution_state.halt_reason = (
                    "sidecar_core_exit_alignment_failed_waiting_flat"
                )
                self.strategy.state.sidecar_halt_reason = (
                    "sidecar_core_exit_alignment_failed_waiting_flat"
                )
                self.strategy.state.sidecar_legs = [
                    mark_sidecar_leg_force_closed(leg, ts_ms)
                    if str(leg.get("status") or "") in {"OPEN", "OPEN_UNPROTECTED"}
                    else leg
                    for leg in self.strategy.state.sidecar_legs
                ]
                for leg in self.strategy.state.sidecar_legs:
                    if leg.get("status") == "FORCE_CLOSED" and leg.get("updated_ts_ms") == ts_ms:
                        leg["core_exit_alignment_reason"] = risk_reason
                        leg["core_exit_market_exit"] = False
                        leg["core_exit_already_flat"] = True
                sidecar_runtime_state.refresh_sidecar_state_totals(
                    self.strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10"))
                )

            self.journal.append(
                "SIDECAR_CORE_EXIT_DELAYED_MARKET_EXIT_SKIPPED_ALREADY_FLAT",
                {
                    "side": side,
                    "core_tp_price": core_tp_price,
                    "risk_reason": risk_reason,
                    "original_error": original_error,
                    "delay_seconds": delay_seconds,
                    "reason": "okx_already_flat_or_wrong_side",
                    "position_fetch_error": None,
                },
                position_id=position_id,
            )

            self.state_store.save(
                LiveStateStore.from_strategy_state(
                    position_id=position_id,
                    symbol=self.trader.symbol,
                    strategy_state=self.strategy.state,
                    cash_before_position=cash_before_position,
                )
            )

            logger.warning(
                "SIDECAR_CORE_EXIT_DELAYED_MARKET_EXIT_SKIPPED_ALREADY_FLAT | position_id=%s side=%s delay_seconds=%.0f",
                position_id,
                side,
                delay_seconds,
            )

            skip_subject = "Sidecar core-exit delayed market exit: already flat"
            skip_content = (
                "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
                "<h2>Delayed Market Exit Skipped — Already Flat</h2>"
                f"<p><b>position_id:</b> {html.escape(str(position_id))}</p>"
                f"<p><b>side:</b> {html.escape(str(side))}</p>"
                f"<p><b>core_tp_price:</b> {core_tp_price:.4f}</p>"
                f"<p><b>risk_reason:</b> {html.escape(risk_reason)}</p>"
                f"<p><b>action:</b> market exit skipped (already flat)</p>"
                "</div>"
            )
            ok = await self.email_sender.send_email_async(skip_subject, skip_content, content_type="html")
            if not ok:
                logger.error("Failed to send sidecar core-exit delayed skip email")
            return

        # 4. Position fetch failed OR position still exists: execute market exit
        retry_count = int(
            os.getenv(
                "SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_MARKET_EXIT_RETRY_COUNT",
                os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3"),
            )
        )
        exit_ok, exit_message = await self.trader.market_exit_remaining_position_with_retries(
            side,
            retry_count=retry_count,
        )

        position_fetch_error_str = str(position_fetch_error) if position_fetch_error else None

        if exit_ok:
            async with self.state_lock:
                self.execution_state.trading_halted = True
                self.execution_state.halt_reason = (
                    "sidecar_core_exit_delayed_market_exit_waiting_flat"
                )
                self.strategy.state.sidecar_halt_reason = (
                    "sidecar_core_exit_delayed_market_exit_waiting_flat"
                )
                self.strategy.state.sidecar_legs = [
                    mark_sidecar_leg_force_closed(leg, ts_ms)
                    if str(leg.get("status") or "") in {"OPEN", "OPEN_UNPROTECTED"}
                    else leg
                    for leg in self.strategy.state.sidecar_legs
                ]
                for leg in self.strategy.state.sidecar_legs:
                    if leg.get("status") == "FORCE_CLOSED" and leg.get("updated_ts_ms") == ts_ms:
                        leg["core_exit_alignment_reason"] = risk_reason
                        leg["core_exit_market_exit"] = True
                sidecar_runtime_state.refresh_sidecar_state_totals(
                    self.strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10"))
                )

            self.journal.append(
                "SIDECAR_CORE_EXIT_DELAYED_MARKET_EXIT_EXECUTED",
                {
                    "side": side,
                    "core_tp_price": core_tp_price,
                    "risk_reason": risk_reason,
                    "original_error": original_error,
                    "delay_seconds": delay_seconds,
                    "exit_message": exit_message,
                    "trading_halted": True,
                    "position_fetch_error": position_fetch_error_str,
                },
                position_id=position_id,
            )

            self.state_store.save(
                LiveStateStore.from_strategy_state(
                    position_id=position_id,
                    symbol=self.trader.symbol,
                    strategy_state=self.strategy.state,
                    cash_before_position=cash_before_position,
                )
            )

            logger.warning(
                "SIDECAR_CORE_EXIT_DELAYED_MARKET_EXIT_EXECUTED | position_id=%s side=%s delay_seconds=%.0f exit_message=%s position_fetch_error=%s",
                position_id,
                side,
                delay_seconds,
                exit_message,
                position_fetch_error_str,
            )

            exec_subject = "Sidecar core-exit delayed market exit: executed"
            exec_content = (
                "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
                "<h2>Delayed Market Exit Executed</h2>"
                f"<p><b>position_id:</b> {html.escape(str(position_id))}</p>"
                f"<p><b>side:</b> {html.escape(str(side))}</p>"
                f"<p><b>core_tp_price:</b> {core_tp_price:.4f}</p>"
                f"<p><b>risk_reason:</b> {html.escape(risk_reason)}</p>"
                f"<p><b>exit_message:</b> {html.escape(exit_message)}</p>"
                + (f"<p><b>position_fetch_error:</b> {html.escape(position_fetch_error_str)}</p>" if position_fetch_error_str else "")
                + f"<p><b>action:</b> market exit executed</p>"
                "</div>"
            )
            ok = await self.email_sender.send_email_async(exec_subject, exec_content, content_type="html")
            if not ok:
                logger.error("Failed to send sidecar core-exit delayed executed email")
        else:
            async with self.state_lock:
                self.execution_state.trading_halted = True
                self.execution_state.halt_reason = "sidecar_core_exit_delayed_market_exit_failed"
                self.strategy.state.sidecar_dirty = True
                self.strategy.state.sidecar_halt_reason = (
                    "sidecar_core_exit_delayed_market_exit_failed"
                )

            self.journal.append(
                "SIDECAR_CORE_EXIT_DELAYED_MARKET_EXIT_FAILED",
                {
                    "side": side,
                    "core_tp_price": core_tp_price,
                    "risk_reason": risk_reason,
                    "original_error": original_error,
                    "delay_seconds": delay_seconds,
                    "exit_message": exit_message,
                    "trading_halted": True,
                    "manual_intervention_required": True,
                    "position_fetch_error": position_fetch_error_str,
                },
                position_id=position_id,
            )

            self.state_store.save(
                LiveStateStore.from_strategy_state(
                    position_id=position_id,
                    symbol=self.trader.symbol,
                    strategy_state=self.strategy.state,
                    cash_before_position=cash_before_position,
                )
            )

            logger.error(
                "SIDECAR_CORE_EXIT_DELAYED_MARKET_EXIT_FAILED | position_id=%s side=%s delay_seconds=%.0f exit_message=%s position_fetch_error=%s manual_intervention_required=true",
                position_id,
                side,
                delay_seconds,
                exit_message,
                position_fetch_error_str,
            )

            fail_subject = "CRITICAL: Sidecar core-exit delayed market exit FAILED"
            fail_content = (
                "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
                "<h2>Delayed Market Exit FAILED — Manual Intervention Required</h2>"
                f"<p><b>position_id:</b> {html.escape(str(position_id))}</p>"
                f"<p><b>side:</b> {html.escape(str(side))}</p>"
                f"<p><b>core_tp_price:</b> {core_tp_price:.4f}</p>"
                f"<p><b>risk_reason:</b> {html.escape(risk_reason)}</p>"
                f"<p><b>exit_message:</b> {html.escape(exit_message)}</p>"
                + (f"<p><b>position_fetch_error:</b> {html.escape(position_fetch_error_str)}</p>" if position_fetch_error_str else "")
                + f"<p><b>action:</b> market exit failed</p>"
                "</div>"
            )
            ok = await self.email_sender.send_email_async(fail_subject, fail_content, content_type="html")
            if not ok:
                logger.error("Failed to send sidecar core-exit delayed failed email")

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
        if fail_action == "MARKET_EXIT":
            subject = "Near-TP protective SL failed; market-exited remaining position"
            content = (
                "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
                "<h2>Near-TP protective SL failed</h2>"
                f"<p>Remaining position was market-exited successfully. Trading is temporarily halted until account sync records FLAT.</p>"
                f"<p><b>position_id:</b> {html.escape(str(current_position_id))}</p>"
                f"<p><b>message:</b> {html.escape(result.message)}</p>"
                "</div>"
            )
            ok = await self.email_sender.send_email_async(subject, content, content_type="html")
            if not ok:
                logger.error("Failed to send Near-TP market-exit success email")
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
        self.journal.record_entry(
            position_id=current_position_id or new_position_id or "",
            intent=command.intent,
            result=result,
            cash_before_position=cash_before_position,
            equity=equity,
            extra={"symbol": self.trader.symbol},
        )
        sidecar_ok = True
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
            )
        async with self.state_lock:
            strategy_state_for_save = copy.deepcopy(self.strategy.state)
        if not sidecar_ok:
            logger.error(
                "LIVE sidecar failed after core entry | position_id=%s intent_type=%s side=%s layer=%s trading_halted=true halt_reason=%s",
                current_position_id or new_position_id,
                command.intent.intent_type,
                command.intent.side,
                command.intent.layer_index,
                self.execution_state.halt_reason,
            )
        if getattr(command.intent, "tp_plan", "SINGLE") == "MIDDLE_RUNNER" and hasattr(self.journal, "append"):
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
        if getattr(command.intent, "tp_plan", "SINGLE") == "THREE_STAGE_RUNNER" and hasattr(self.journal, "append"):
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
            "LIVE entry success | intent_type=%s side=%s layer=%s price=%.4f contracts=%s tp_price=%s tp_mode=%s tp_plan=%s partial_tp=%s avg_entry=%.4f breakeven=%.4f order_id=%s tp_order_id=%s",
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
        )
