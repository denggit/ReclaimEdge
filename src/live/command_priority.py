from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from src.live.runtime_types import TradeCommand


RISK_REDUCING_INTENTS = {
    "MARKET_EXIT",
    "MARKET_EXIT_RUNNER",
    "CLOSE",
    "CLOSE_LONG",
    "CLOSE_SHORT",
    "EMERGENCY_EXIT",
}


def is_risk_reducing_intent(intent_type: str) -> bool:
    return str(intent_type or "").upper() in RISK_REDUCING_INTENTS


def trade_command_priority(command: TradeCommand) -> tuple[int, int]:
    intent_type = str(getattr(command.intent, "intent_type", "") or "").upper()
    if is_risk_reducing_intent(intent_type):
        return (0, 0)
    if intent_type == "NEAR_TP_REDUCE":
        return (1, 0)
    if intent_type == "UPDATE_TP":
        return (2, 0)
    return (3, 0)


@dataclass(frozen=True)
class TradeCommandReorderResult:
    command: TradeCommand
    reordered: bool
    before_intents: tuple[str, ...]
    after_intents: tuple[str, ...]


def prioritize_dequeued_command(
    command: TradeCommand,
    queued_commands: Iterable[TradeCommand],
) -> tuple[TradeCommand, list[TradeCommand], bool, tuple[str, ...], tuple[str, ...]]:
    commands = [command, *list(queued_commands)]
    before = tuple(str(getattr(item.intent, "intent_type", "") or "") for item in commands)
    sorted_commands = sorted(enumerate(commands), key=lambda item: (trade_command_priority(item[1]), item[0]))
    ordered = [item for _index, item in sorted_commands]
    after = tuple(str(getattr(item.intent, "intent_type", "") or "") for item in ordered)
    return ordered[0], ordered[1:], before != after, before, after


def pop_next_priority_command(command: TradeCommand, execution_queue) -> TradeCommandReorderResult:
    try:
        queued = list(execution_queue._queue)  # type: ignore[attr-defined]
    except Exception:
        return TradeCommandReorderResult(
            command=command,
            reordered=False,
            before_intents=(str(getattr(command.intent, "intent_type", "") or ""),),
            after_intents=(str(getattr(command.intent, "intent_type", "") or ""),),
        )
    if not queued:
        intent = str(getattr(command.intent, "intent_type", "") or "")
        return TradeCommandReorderResult(command, False, (intent,), (intent,))
    selected, remaining, reordered, before, after = prioritize_dequeued_command(command, queued)
    if reordered:
        execution_queue._queue = deque(remaining)  # type: ignore[attr-defined]
    return TradeCommandReorderResult(selected, reordered, before, after)
