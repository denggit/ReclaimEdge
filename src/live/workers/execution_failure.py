from __future__ import annotations

import asyncio
import copy
import html
import time
from typing import Any

from src.execution.trader import Trader
from src.live import runtime_types as live_runtime_types
from src.reporting import live_report_helpers as report_helpers
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.email_sender import EmailSender
from src.utils.log import get_logger

logger = get_logger(__name__)


async def handle_execution_failure(
    *,
    command: live_runtime_types.TradeCommand,
    result: Any | None,
    error: Exception,
    state_lock: asyncio.Lock,
    execution_state: live_runtime_types.ExecutionState,
    trader: Trader,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    email_sender: EmailSender,
) -> live_runtime_types.ExecutionReport:
    contracts = trader.position_contracts
    if result is None or getattr(result, "entry_filled", False) or getattr(result, "reduce_filled", False):
        try:
            position = await trader.fetch_position_snapshot()
            contracts = position.contracts
        except Exception:
            contracts = trader.position_contracts

    entry_may_be_live = bool(getattr(result, "entry_filled", False)) or bool(getattr(result, "reduce_filled", False)) or contracts > 0
    rolled_back = False
    async with state_lock:
        current_position_id = execution_state.current_position_id
        if entry_may_be_live:
            execution_state.trading_halted = True
            if str(error) == "reduce_only_order_identity_unknown":
                execution_state.halt_reason = "reduce_only_order_identity_unknown"
            elif command.intent.intent_type == "NEAR_TP_REDUCE" and getattr(result, "reduce_filled", False):
                execution_state.halt_reason = "near_tp_reduce_failure"
            else:
                execution_state.halt_reason = "execution_failure_live_position"
            trader.position_contracts = contracts
        else:
            strategy.state = copy.deepcopy(command.strategy_state_snapshot)
            rolled_back = True
        halted = execution_state.trading_halted

    if entry_may_be_live:
        logger.exception("LIVE execution failed after/possibly after entry. Trading halted; strategy state NOT rolled back.")
    else:
        logger.exception("LIVE execution failed before entry; strategy state has been rolled back")

    try:
        journal.record_error(position_id=current_position_id, intent=command.intent, error=error, rolled_back=rolled_back, halted=halted)
        if str(error) == "reduce_only_order_identity_unknown" and hasattr(journal, "append"):
            journal.append(
                "REDUCE_ONLY_ORDER_IDENTITY_UNKNOWN",
                {
                    "intent_type": command.intent.intent_type,
                    "side": command.intent.side,
                    "trading_halted": halted,
                    "manual_intervention_required": True,
                },
                position_id=current_position_id,
            )
    except Exception:
        logger.exception("Failed to write trade journal error event")

    if command.intent.intent_type == "NEAR_TP_REDUCE" and getattr(result, "reduce_filled", False) and halted:
        subject = "CRITICAL: Near-TP protective SL and market exit failed"
        content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <h2>CRITICAL: Near-TP protective SL and market exit failed</h2>
  <p><strong>Trading has been halted. Manual OKX intervention is required.</strong></p>
  <p><strong>position_id:</strong> {html.escape(str(current_position_id))}</p>
  <p><strong>side:</strong> {html.escape(command.intent.side)}</p>
  <p><strong>contracts_after:</strong> {html.escape(str(getattr(result, 'contracts_after', contracts)))}</p>
  <p><strong>protective_sl_price:</strong> {html.escape(str(getattr(result, 'protective_sl_price', '-')))}</p>
  <p><strong>Error:</strong> {html.escape(str(error))}</p>
</div>
""".strip()
    else:
        subject, content = report_helpers.build_live_failure_email(command.intent, error, rolled_back=rolled_back, halted=halted)
    ok = await email_sender.send_email_async(subject, content, content_type="html")
    if not ok:
        logger.error("Failed to send live execution failure email")

    return live_runtime_types.ExecutionReport(
        command=command,
        result=result,
        ok=False,
        error=error,
        entry_may_be_live=entry_may_be_live,
        created_monotonic=command.created_monotonic,
        finished_monotonic=time.monotonic(),
    )
