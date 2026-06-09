from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass
from typing import Iterable

from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WorkerShutdownInstallResult:
    """Result of installing signal handlers for a :class:`WorkerShutdownController`."""

    registered: tuple[str, ...]
    unsupported: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.registered)


class WorkerShutdownController:
    """Lightweight shutdown controller driven by an :class:`asyncio.Event`.

    The controller is signal-agnostic — it does not import ``signal`` or
    install handlers itself.  Use :func:`install_symbol_worker_signal_handlers`
    to wire OS signals to this controller.

    D06b: This controller is used by the symbol worker app to enter drain
    mode without forcibly closing positions, removing TP/SL/Algo/Sidecar
    protections, or calling any OKX private write APIs.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    @property
    def event(self) -> asyncio.Event:
        return self._event

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def request_shutdown(self, reason: str = "signal") -> None:
        """Request a graceful shutdown.

        Idempotent — the first call sets the reason; subsequent calls are
        ignored.
        """
        if not self._event.is_set():
            self._reason = reason
            logger.warning("SYMBOL_WORKER_SHUTDOWN_REQUESTED | reason=%s", reason)
            self._event.set()


def install_symbol_worker_signal_handlers(
    controller: WorkerShutdownController,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    signals: Iterable[signal.Signals] | None = None,
) -> WorkerShutdownInstallResult:
    """Install OS signal handlers that trigger ``controller.request_shutdown``.

    On platforms where :meth:`asyncio.AbstractEventLoop.add_signal_handler`
    is unsupported (e.g. some Windows configurations), the unsupported signals
    are recorded in the result and the function does **not** raise.

    Parameters
    ----------
    controller:
        The shutdown controller whose ``request_shutdown`` method will be
        called when one of the target signals is received.
    loop:
        Event loop to register handlers on.  Defaults to the running loop.
    signals:
        Signals to register.  Defaults to ``(SIGINT, SIGTERM)``.
    """
    event_loop = loop or asyncio.get_running_loop()
    target_signals = tuple(signals or (signal.SIGINT, signal.SIGTERM))
    registered: list[str] = []
    unsupported: list[str] = []

    for sig in target_signals:
        try:
            event_loop.add_signal_handler(
                sig,
                controller.request_shutdown,
                sig.name,
            )
        except (NotImplementedError, RuntimeError, ValueError):
            unsupported.append(sig.name)
            continue
        registered.append(sig.name)

    if registered:
        logger.info(
            "SYMBOL_WORKER_SIGNAL_HANDLERS_REGISTERED | signals=%s",
            ",".join(registered),
        )
    if unsupported:
        logger.info(
            "SYMBOL_WORKER_SIGNAL_HANDLERS_UNSUPPORTED | signals=%s",
            ",".join(unsupported),
        )

    return WorkerShutdownInstallResult(
        registered=tuple(registered),
        unsupported=tuple(unsupported),
    )
