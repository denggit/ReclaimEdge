from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass
from typing import Iterable

from src.live.supervisor.reclaim_supervisor import ReclaimSupervisor
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SignalHandlerInstallResult:
    registered: tuple[str, ...]
    unsupported: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.registered)


def install_supervisor_signal_handlers(
    supervisor: ReclaimSupervisor,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    signals: Iterable[signal.Signals] | None = None,
) -> SignalHandlerInstallResult:
    event_loop = loop or asyncio.get_running_loop()
    target_signals = tuple(signals or (signal.SIGINT, signal.SIGTERM))
    registered: list[str] = []
    unsupported: list[str] = []

    for sig in target_signals:
        try:
            event_loop.add_signal_handler(sig, supervisor.request_stop)
        except (NotImplementedError, RuntimeError, ValueError):
            unsupported.append(sig.name)
            continue
        registered.append(sig.name)

    if registered:
        logger.info("RECLAIM_SUPERVISOR_SIGNAL_HANDLERS_REGISTERED | signals=%s", ",".join(registered))
    if unsupported:
        logger.info("RECLAIM_SUPERVISOR_SIGNAL_HANDLERS_UNSUPPORTED | signals=%s", ",".join(unsupported))

    return SignalHandlerInstallResult(
        registered=tuple(registered),
        unsupported=tuple(unsupported),
    )
