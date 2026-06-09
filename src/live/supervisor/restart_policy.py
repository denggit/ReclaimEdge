from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass(frozen=True)
class RestartPolicyConfig:
    """Immutable configuration for the controlled child restart policy.

    Attributes:
        enabled: When ``False``, :meth:`RestartPolicy.evaluate` always returns
            ``allowed=False`` with reason ``"disabled"``.
        cooldown_seconds: Minimum wall-clock time (monotonic) that must elapse
            between two consecutive restarts.
        max_restarts: Maximum number of restarts allowed within
            ``window_seconds``.  Set to ``0`` to forbid all restarts.
        window_seconds: Sliding window over which ``max_restarts`` is enforced.
            Entries older than ``now - window_seconds`` are pruned on each
            evaluation or record.
    """

    enabled: bool = True
    cooldown_seconds: float = 10.0
    max_restarts: int = 3
    window_seconds: float = 600.0

    def __post_init__(self) -> None:
        if self.cooldown_seconds < 0:
            raise ValueError("restart cooldown_seconds must be >= 0")
        if self.max_restarts < 0:
            raise ValueError("restart max_restarts must be >= 0")
        if self.window_seconds <= 0:
            raise ValueError("restart window_seconds must be > 0")


@dataclass(frozen=True)
class RestartDecision:
    """Immutable result of a restart evaluation.

    Attributes:
        allowed: Whether a restart is permitted at this moment.
        reason: Short machine-readable reason string.
        restart_count_in_window: Number of restarts currently inside the
            sliding window (after pruning).
        next_allowed_monotonic: Earliest monotonic timestamp at which the next
            restart would be allowed, or ``None`` when no cooldown is in effect.
    """

    allowed: bool
    reason: str
    restart_count_in_window: int
    next_allowed_monotonic: float | None = None


class RestartPolicy:
    """Controls how often a child process may be restarted.

    This is a **pure logic** module — it performs no file I/O, no network
    calls, and imports none of the trading or process-management modules.
    """

    def __init__(self, config: RestartPolicyConfig) -> None:
        self._config = config
        self._restart_timestamps: Deque[float] = deque()
        self._last_restart_monotonic: float | None = None

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> RestartPolicyConfig:
        return self._config

    @property
    def restart_count_in_window(self) -> int:
        """Number of recorded restarts currently inside the sliding window."""
        return len(self._restart_timestamps)

    @property
    def last_restart_monotonic(self) -> float | None:
        """Monotonic timestamp of the most recent restart, or ``None``."""
        return self._last_restart_monotonic

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, *, now_monotonic: float) -> RestartDecision:
        """Evaluate whether a restart is allowed at *now_monotonic*.

        Checks (in order):
        1. Disabled.
        2. ``max_restarts == 0``.
        3. Cooldown not yet elapsed.
        4. Window exhausted.
        """
        if not self._config.enabled:
            return RestartDecision(False, "disabled", self.restart_count_in_window)

        if self._config.max_restarts <= 0:
            return RestartDecision(False, "max_restarts_zero", self.restart_count_in_window)

        self._prune(now_monotonic)

        if self._last_restart_monotonic is not None:
            next_allowed = self._last_restart_monotonic + self._config.cooldown_seconds
            if now_monotonic < next_allowed:
                return RestartDecision(
                    False,
                    "cooldown",
                    self.restart_count_in_window,
                    next_allowed_monotonic=next_allowed,
                )

        if len(self._restart_timestamps) >= self._config.max_restarts:
            return RestartDecision(False, "max_restarts_exceeded", self.restart_count_in_window)

        return RestartDecision(True, "allowed", self.restart_count_in_window)

    # ------------------------------------------------------------------
    # record_restart
    # ------------------------------------------------------------------

    def record_restart(self, *, now_monotonic: float) -> int:
        """Record a restart that just occurred.

        Returns the number of restarts now inside the window (including this
        one).
        """
        self._prune(now_monotonic)
        self._restart_timestamps.append(now_monotonic)
        self._last_restart_monotonic = now_monotonic
        return len(self._restart_timestamps)

    # ------------------------------------------------------------------
    # _prune
    # ------------------------------------------------------------------

    def _prune(self, now_monotonic: float) -> None:
        cutoff = now_monotonic - self._config.window_seconds
        while self._restart_timestamps and self._restart_timestamps[0] < cutoff:
            self._restart_timestamps.popleft()
