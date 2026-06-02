from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.reporting.trade_journal import JournalEvent
from src.utils.log import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ROLLING_LOSS_HALT_REASONS = {"rolling_loss_soft_halt", "rolling_loss_hard_halt"}
MS_PER_HOUR = 60 * 60 * 1000


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RollingLossGuardConfig:
    enabled: bool = True
    window_hours: float = 24.0
    warn_pct: float = 0.10
    soft_halt_pct: float = 0.15
    soft_halt_hours: float = 6.0
    hard_halt_pct: float = 0.20
    hard_halt_hours: float = 12.0
    email_enabled: bool = True

    @classmethod
    def from_env(cls) -> "RollingLossGuardConfig":
        return cls(
            enabled=_env_bool("ROLLING_LOSS_GUARD_ENABLED", True),
            window_hours=float(os.getenv("ROLLING_LOSS_WINDOW_HOURS", "24")),
            warn_pct=float(os.getenv("ROLLING_LOSS_WARN_PCT", "0.10")),
            soft_halt_pct=float(os.getenv("ROLLING_LOSS_SOFT_HALT_PCT", "0.15")),
            soft_halt_hours=float(os.getenv("ROLLING_LOSS_SOFT_HALT_HOURS", "6")),
            hard_halt_pct=float(os.getenv("ROLLING_LOSS_HARD_HALT_PCT", "0.20")),
            hard_halt_hours=float(os.getenv("ROLLING_LOSS_HARD_HALT_HOURS", "12")),
            email_enabled=_env_bool("ROLLING_LOSS_EMAIL_ENABLED", True),
        )


@dataclass
class RollingLossGuardState:
    enabled: bool
    window_start_ts_ms: int
    window_end_ts_ms: int
    baseline_equity: float
    warn_triggered: bool = False
    soft_halt_triggered: bool = False
    hard_halt_triggered: bool = False
    halt_active: bool = False
    halt_level: str | None = None
    halt_until_ts_ms: int | None = None
    last_event_ts_ms: int | None = None
    last_window_realized_pnl: float = 0.0
    last_loss_pct: float = 0.0


@dataclass(frozen=True)
class RollingLossGuardDecision:
    action: str | None
    state: RollingLossGuardState
    rolling_realized_pnl: float
    loss_usdt: float
    loss_pct: float
    threshold_pct: float | None = None
    halt_hours: float | None = None
    halt_until_ts_ms: int | None = None
    reason: str | None = None

    @property
    def should_halt(self) -> bool:
        return self.action in {"SOFT_HALT", "HARD_HALT"}


class RollingLossGuard:
    def __init__(self, state_path: Path, config: RollingLossGuardConfig) -> None:
        self.state_path = state_path
        if not self.state_path.is_absolute():
            self.state_path = ROOT / self.state_path
        self.config = config
        self.state: RollingLossGuardState | None = None

    @classmethod
    def from_env(cls) -> "RollingLossGuard":
        raw_path = os.getenv(
            "ROLLING_LOSS_STATE_PATH",
            "data/trade_journal/rolling_loss_guard_state.json",
        )
        return cls(Path(raw_path), RollingLossGuardConfig.from_env())

    def load_or_initialize(self, now_ms: int, equity: float) -> RollingLossGuardState:
        state = self._load_state()
        if state is None or state.baseline_equity <= 0:
            if state is not None and state.baseline_equity <= 0:
                logger.warning("ROLLING_LOSS_GUARD_STATE_INVALID | baseline_equity=%s reinitializing=true", state.baseline_equity)
            self.state = self._new_state(now_ms, equity)
            self.save()
            return self.state
        state.enabled = self.config.enabled
        self.state = state
        return state

    def save(self) -> None:
        if self.state is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(asdict(self.state), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.state_path)

    def reset_window(self, now_ms: int, equity: float) -> None:
        self.state = self._new_state(now_ms, equity)

    def should_resume(self, now_ms: int, has_position: bool) -> bool:
        state = self._require_state()
        return bool(
            state.enabled
            and state.halt_active
            and state.halt_until_ts_ms is not None
            and now_ms >= state.halt_until_ts_ms
            and not has_position
        )

    def mark_resumed(self, now_ms: int, equity: float) -> None:
        self.reset_window(now_ms, equity)

    def should_reset_expired_window(self, now_ms: int, has_position: bool) -> bool:
        state = self._require_state()
        return bool(state.enabled and not state.halt_active and not has_position and now_ms >= state.window_end_ts_ms)

    def evaluate_after_flat(
        self,
        *,
        now_ms: int,
        journal_events: Iterable[JournalEvent],
        has_position: bool = False,
    ) -> RollingLossGuardDecision:
        state = self._require_state()
        if not state.enabled:
            return self._decision(None, 0.0, 0.0, 0.0, reason="disabled")
        if has_position:
            return self._decision(None, state.last_window_realized_pnl, 0.0, state.last_loss_pct, reason="position_open")
        if state.baseline_equity <= 0:
            logger.warning("ROLLING_LOSS_GUARD_BASELINE_INVALID | baseline_equity=%s skip=true", state.baseline_equity)
            return self._decision(None, 0.0, 0.0, 0.0, reason="invalid_baseline")

        rolling_realized_pnl = self.window_realized_pnl(journal_events, now_ms=now_ms)
        loss_usdt = max(0.0, -rolling_realized_pnl)
        loss_pct = loss_usdt / state.baseline_equity
        state.last_window_realized_pnl = rolling_realized_pnl
        state.last_loss_pct = loss_pct
        state.last_event_ts_ms = now_ms

        if loss_pct >= self.config.hard_halt_pct and not state.hard_halt_triggered:
            halt_until = now_ms + int(self.config.hard_halt_hours * MS_PER_HOUR)
            state.warn_triggered = True
            state.soft_halt_triggered = True
            state.hard_halt_triggered = True
            state.halt_active = True
            state.halt_level = "HARD"
            state.halt_until_ts_ms = halt_until
            self.save()
            return self._decision(
                "HARD_HALT",
                rolling_realized_pnl,
                loss_usdt,
                loss_pct,
                threshold_pct=self.config.hard_halt_pct,
                halt_hours=self.config.hard_halt_hours,
                halt_until_ts_ms=halt_until,
                reason="rolling_realized_loss_reached_hard_threshold",
            )

        if loss_pct >= self.config.soft_halt_pct and not state.soft_halt_triggered:
            halt_until = now_ms + int(self.config.soft_halt_hours * MS_PER_HOUR)
            state.warn_triggered = True
            state.soft_halt_triggered = True
            state.halt_active = True
            state.halt_level = "SOFT"
            state.halt_until_ts_ms = halt_until
            self.save()
            return self._decision(
                "SOFT_HALT",
                rolling_realized_pnl,
                loss_usdt,
                loss_pct,
                threshold_pct=self.config.soft_halt_pct,
                halt_hours=self.config.soft_halt_hours,
                halt_until_ts_ms=halt_until,
                reason="rolling_realized_loss_reached_soft_threshold",
            )

        if loss_pct >= self.config.warn_pct and not state.warn_triggered:
            state.warn_triggered = True
            if state.halt_level is None:
                state.halt_level = "WARN"
            self.save()
            return self._decision(
                "WARN",
                rolling_realized_pnl,
                loss_usdt,
                loss_pct,
                threshold_pct=self.config.warn_pct,
                reason="rolling_realized_loss_reached_warn_threshold",
            )

        self.save()
        return self._decision(None, rolling_realized_pnl, loss_usdt, loss_pct, reason="threshold_not_reached")

    def window_realized_pnl(self, journal_events: Iterable[JournalEvent], *, now_ms: int) -> float:
        state = self._require_state()
        total = 0.0
        for event in journal_events:
            if event.event_type != "FLAT":
                continue
            event_ts_ms = self._event_ts_ms(event)
            if event_ts_ms is None:
                continue
            if event_ts_ms < state.window_start_ts_ms or event_ts_ms >= now_ms:
                continue
            value = event.payload.get("realized_pnl_usdt_est")
            try:
                pnl = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(pnl):
                continue
            total += pnl
        return total

    def _new_state(self, now_ms: int, equity: float) -> RollingLossGuardState:
        return RollingLossGuardState(
            enabled=self.config.enabled,
            window_start_ts_ms=now_ms,
            window_end_ts_ms=now_ms + int(self.config.window_hours * MS_PER_HOUR),
            baseline_equity=float(equity),
        )

    def _load_state(self) -> RollingLossGuardState | None:
        if not self.state_path.exists():
            return None
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            if not raw.strip():
                logger.warning("ROLLING_LOSS_GUARD_STATE_EMPTY | path=%s reinitializing=true", self.state_path)
                return None
            data = json.loads(raw)
            return RollingLossGuardState(**data)
        except Exception as exc:
            logger.warning("ROLLING_LOSS_GUARD_STATE_LOAD_FAILED | path=%s error=%s reinitializing=true", self.state_path, exc)
            return None

    def _require_state(self) -> RollingLossGuardState:
        if self.state is None:
            raise RuntimeError("RollingLossGuard state is not loaded")
        return self.state

    def _decision(
        self,
        action: str | None,
        rolling_realized_pnl: float,
        loss_usdt: float,
        loss_pct: float,
        *,
        threshold_pct: float | None = None,
        halt_hours: float | None = None,
        halt_until_ts_ms: int | None = None,
        reason: str | None = None,
    ) -> RollingLossGuardDecision:
        return RollingLossGuardDecision(
            action=action,
            state=self._require_state(),
            rolling_realized_pnl=rolling_realized_pnl,
            loss_usdt=loss_usdt,
            loss_pct=loss_pct,
            threshold_pct=threshold_pct,
            halt_hours=halt_hours,
            halt_until_ts_ms=halt_until_ts_ms,
            reason=reason,
        )

    @staticmethod
    def _event_ts_ms(event: JournalEvent) -> int | None:
        try:
            ts = datetime.fromisoformat(event.ts_iso)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return int(ts.timestamp() * 1000)
        except Exception:
            return None
