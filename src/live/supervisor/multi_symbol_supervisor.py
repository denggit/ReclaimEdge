from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

from src.live.supervisor.reclaim_supervisor import ReclaimSupervisor
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SupervisorTaskResult:
    child_name: str
    return_code: int
    error: str | None = None
    unexpected_return: bool = False


class MultiSymbolSupervisor:
    """Thin orchestrator for multiple single-child ReclaimSupervisor objects."""

    def __init__(self, supervisors: Sequence[ReclaimSupervisor]) -> None:
        if not supervisors:
            raise ValueError("supervisors must not be empty")
        self._supervisors = tuple(supervisors)
        self._stop_requested = False
        self._shutdown_started = False
        self._task_results: list[SupervisorTaskResult] = []

    @property
    def supervisors(self) -> tuple[ReclaimSupervisor, ...]:
        return self._supervisors

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def task_results(self) -> tuple[SupervisorTaskResult, ...]:
        return tuple(self._task_results)

    def request_stop(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        logger.info("RECLAIM_MULTI_SUPERVISOR_STOP_REQUESTED")
        for supervisor in self._supervisors:
            supervisor.request_stop()

    async def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self.request_stop()
        await asyncio.gather(
            *(supervisor.shutdown() for supervisor in self._supervisors),
            return_exceptions=True,
        )

    async def _run_one(self, supervisor: ReclaimSupervisor) -> SupervisorTaskResult:
        child_name = supervisor.config.child_name
        try:
            run_method = getattr(supervisor, "run", None)
            if run_method is None:
                await supervisor.run_forever()
            else:
                await run_method()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "RECLAIM_MULTI_SUPERVISOR_CHILD_SUPERVISOR_FAILED | child=%s error=%s",
                child_name,
                error,
            )
            return SupervisorTaskResult(
                child_name=child_name,
                return_code=1,
                error=error,
            )

        unexpected = not self._stop_requested
        if unexpected:
            logger.error(
                "RECLAIM_MULTI_SUPERVISOR_CHILD_SUPERVISOR_RETURNED | child=%s",
                child_name,
            )
        return SupervisorTaskResult(
            child_name=child_name,
            return_code=1 if unexpected else 0,
            unexpected_return=unexpected,
        )

    async def run(self) -> int:
        logger.info(
            "RECLAIM_MULTI_SUPERVISOR_STARTED | workers=%s",
            ",".join(supervisor.config.child_name for supervisor in self._supervisors),
        )
        tasks = [
            asyncio.create_task(self._run_one(supervisor), name=f"supervisor:{supervisor.config.child_name}")
            for supervisor in self._supervisors
        ]
        pending = set(tasks)

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    result = task.result()
                    self._task_results.append(result)
                if self._stop_requested and pending:
                    await self.shutdown()
            return 1 if any(result.return_code != 0 for result in self._task_results) else 0
        except asyncio.CancelledError:
            await self.shutdown()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
        finally:
            if self._stop_requested:
                await self.shutdown()
            logger.info("RECLAIM_MULTI_SUPERVISOR_STOPPED")
