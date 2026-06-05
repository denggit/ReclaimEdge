from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


@dataclass(frozen=True)
class RollingLossGuardConfig:
    enabled: bool = True
    warn_pct: float = 0.50
    soft_halt_pct: float = 0.10
    soft_halt_hours: float = 12.0
    hard_halt_pct: float = 0.20
    hard_halt_hours: float = 24.0
    email_enabled: bool = True
    event_time_tolerance_ms: int = 5000

    @classmethod
    def from_env(cls) -> "RollingLossGuardConfig":
        return cls(
            enabled=_env_bool("ROLLING_LOSS_GUARD_ENABLED", True),
            warn_pct=float(os.getenv("ROLLING_LOSS_WARN_PCT", "0.50")),
            soft_halt_pct=float(os.getenv("ROLLING_LOSS_SOFT_HALT_PCT", "0.10")),
            soft_halt_hours=float(os.getenv("ROLLING_LOSS_SOFT_HALT_HOURS", "12")),
            hard_halt_pct=float(os.getenv("ROLLING_LOSS_HARD_HALT_PCT", "0.20")),
            hard_halt_hours=float(os.getenv("ROLLING_LOSS_HARD_HALT_HOURS", "24")),
            email_enabled=_env_bool("ROLLING_LOSS_EMAIL_ENABLED", True),
            event_time_tolerance_ms=int(os.getenv("ROLLING_LOSS_EVENT_TIME_TOLERANCE_MS", "5000")),
        )


@dataclass
class RollingLossGuardState:
    enabled: bool
    reference_flat_equity: float
    cumulative_retention: float = 1.0
    drawdown_pct: float = 0.0
    last_flat_equity: float = 0.0
    last_flat_ts_ms: int | None = None
    last_flat_event_id: str | None = None
    last_segment_retention: float = 1.0
    last_segment_return_pct: float = 0.0
    last_segment_drawdown_delta_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    warn_triggered: bool = False
    soft_halt_triggered: bool = False
    hard_halt_triggered: bool = False
    halt_active: bool = False
    halt_level: str | None = None
    halt_until_ts_ms: int | None = None
    last_event_ts_ms: int | None = None


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
    segment_retention: float = 1.0
    segment_return_pct: float = 0.0
    cumulative_retention: float = 1.0
    drawdown_pct: float = 0.0
    reference_flat_equity: float = 0.0
    flat_equity: float = 0.0
    max_drawdown_pct: float = 0.0

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
        state = self._load_state(current_equity=equity)
        if state is None or state.reference_flat_equity <= 0:
            if state is not None and state.reference_flat_equity <= 0:
                logger.warning(
                    "ROLLING_DRAWDOWN_GUARD_STATE_INVALID | reference_flat_equity=%s reinitializing=true",
                    state.reference_flat_equity,
                )
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
        logger.warning("ROLLING_DRAWDOWN_GUARD_RESET_WINDOW_NOOP | reason=time_window_removed")
        state = self._require_state()
        if state.reference_flat_equity <= 0 < equity:
            state.reference_flat_equity = float(equity)
            state.last_flat_equity = float(equity)
            state.last_event_ts_ms = now_ms
            self.save()

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
        state = self._require_state()
        state.halt_active = False
        state.halt_level = None
        state.halt_until_ts_ms = None
        state.last_event_ts_ms = now_ms
        if state.reference_flat_equity <= 0 < equity:
            state.reference_flat_equity = float(equity)
            state.last_flat_equity = float(equity)
        self._clear_trigger_flags_if_recovered(state)
        self.save()

    def should_reset_expired_window(self, now_ms: int, has_position: bool) -> bool:
        return False

    def evaluate_after_flat(
        self,
        *,
        now_ms: int,
        flat_equity: float | None = None,
        flat_event_id: str | None = None,
        journal_events: Iterable[JournalEvent] | None = None,
        has_position: bool = False,
    ) -> RollingLossGuardDecision:
        del journal_events
        state = self._require_state()
        if not state.enabled:
            return self._decision(None, 0.0, 0.0, state.drawdown_pct, reason="disabled")
        if has_position:
            return self._decision(None, 0.0, 0.0, state.drawdown_pct, reason="position_open")
        if flat_event_id is not None and flat_event_id == state.last_flat_event_id:
            return self._decision(None, 0.0, 0.0, state.drawdown_pct, reason="duplicate_flat_event")

        equity = _safe_float(flat_equity, 0.0)
        if equity <= 0:
            logger.warning("ROLLING_DRAWDOWN_GUARD_FLAT_EQUITY_INVALID | flat_equity=%s skip=true", flat_equity)
            return self._decision(None, 0.0, 0.0, state.drawdown_pct, reason="invalid_flat_equity")

        if state.reference_flat_equity <= 0:
            state.reference_flat_equity = equity
            state.last_flat_equity = equity
            state.last_flat_ts_ms = now_ms
            state.last_flat_event_id = flat_event_id
            state.last_event_ts_ms = now_ms
            self.save()
            logger.warning(
                "ROLLING_DRAWDOWN_GUARD_REFERENCE_INITIALIZED | reference_flat_equity=%.4f flat_event_id=%s",
                equity,
                flat_event_id,
            )
            return self._decision(
                None,
                0.0,
                0.0,
                state.drawdown_pct,
                reason="initialized_flat_reference",
                flat_equity=equity,
                reference_flat_equity=equity,
            )

        reference_before = state.reference_flat_equity
        segment_retention = equity / reference_before
        if segment_retention <= 0 or not math.isfinite(segment_retention):
            logger.warning(
                "ROLLING_DRAWDOWN_GUARD_SEGMENT_RETENTION_INVALID | flat_equity=%s reference_flat_equity=%s skip=true",
                equity,
                reference_before,
            )
            return self._decision(None, 0.0, 0.0, state.drawdown_pct, reason="invalid_segment_retention")

        old_drawdown_pct = state.drawdown_pct
        old_max_drawdown_pct = state.max_drawdown_pct
        old_retention = state.cumulative_retention if state.cumulative_retention > 0 else 1.0
        new_retention = old_retention * segment_retention
        if new_retention >= 1.0:
            new_retention = 1.0
            drawdown_pct = 0.0
            self._clear_trigger_flags(state)
        else:
            drawdown_pct = 1.0 - new_retention

        state.reference_flat_equity = equity
        state.last_flat_equity = equity
        state.last_flat_event_id = flat_event_id
        state.last_flat_ts_ms = now_ms
        state.last_segment_retention = segment_retention
        state.last_segment_return_pct = segment_retention - 1.0
        state.last_segment_drawdown_delta_pct = drawdown_pct - old_drawdown_pct
        state.cumulative_retention = new_retention
        state.drawdown_pct = drawdown_pct
        state.max_drawdown_pct = max(old_max_drawdown_pct, drawdown_pct)
        state.last_event_ts_ms = now_ms
        self._clear_trigger_flags_if_recovered(state)

        loss_usdt = max(0.0, reference_before - equity)
        action: str | None = None
        threshold_pct: float | None = None
        halt_hours: float | None = None
        halt_until: int | None = None
        reason = "threshold_not_reached"
        drawdown_worsened = drawdown_pct > old_drawdown_pct + 1e-12

        if drawdown_worsened and self._threshold_reached(drawdown_pct, self.config.hard_halt_pct) and not state.hard_halt_triggered:
            action = "HARD_HALT"
            threshold_pct = self.config.hard_halt_pct
            halt_hours = self.config.hard_halt_hours
            halt_until = now_ms + int(self.config.hard_halt_hours * MS_PER_HOUR)
            state.warn_triggered = True
            state.soft_halt_triggered = True
            state.hard_halt_triggered = True
            state.halt_active = True
            state.halt_level = "HARD"
            state.halt_until_ts_ms = halt_until
            reason = "flat_to_flat_drawdown_reached_hard_threshold"
        elif drawdown_worsened and self._threshold_reached(drawdown_pct, self.config.soft_halt_pct) and not state.soft_halt_triggered:
            action = "SOFT_HALT"
            threshold_pct = self.config.soft_halt_pct
            halt_hours = self.config.soft_halt_hours
            halt_until = now_ms + int(self.config.soft_halt_hours * MS_PER_HOUR)
            state.warn_triggered = True
            state.soft_halt_triggered = True
            state.halt_active = True
            state.halt_level = "SOFT"
            state.halt_until_ts_ms = halt_until
            reason = "flat_to_flat_drawdown_reached_soft_threshold"
        elif drawdown_worsened and self._threshold_reached(drawdown_pct, self.config.warn_pct) and not state.warn_triggered:
            action = "WARN"
            threshold_pct = self.config.warn_pct
            state.warn_triggered = True
            if state.halt_level is None:
                state.halt_level = "WARN"
            reason = "flat_to_flat_drawdown_reached_warn_threshold"
        elif not drawdown_worsened:
            reason = "drawdown_not_worsened"

        self.save()
        if action is not None:
            logger.warning(
                "ROLLING_DRAWDOWN_GUARD_TRIGGERED | action=%s reference_flat_equity=%.4f flat_equity=%.4f segment_retention=%.8f cumulative_retention=%.8f drawdown_pct=%.6f",
                action,
                reference_before,
                equity,
                segment_retention,
                new_retention,
                drawdown_pct,
            )
        return self._decision(
            action,
            0.0,
            loss_usdt,
            drawdown_pct,
            threshold_pct=threshold_pct,
            halt_hours=halt_hours,
            halt_until_ts_ms=halt_until,
            reason=reason,
            segment_retention=segment_retention,
            segment_return_pct=segment_retention - 1.0,
            cumulative_retention=new_retention,
            drawdown_pct=drawdown_pct,
            reference_flat_equity=reference_before,
            flat_equity=equity,
            max_drawdown_pct=state.max_drawdown_pct,
        )

    def adjust_flat_reference_for_cash_transfer(
        self,
        *,
        now_ms: int,
        new_flat_equity: float,
        reason: str,
    ) -> None:
        state = self._require_state()
        equity = _safe_float(new_flat_equity, 0.0)
        if not state.enabled or equity <= 0:
            return
        old_reference = state.reference_flat_equity
        state.reference_flat_equity = equity
        state.last_flat_equity = equity
        state.last_event_ts_ms = now_ms
        self.save()
        logger.warning(
            "ROLLING_DRAWDOWN_GUARD_FLAT_REFERENCE_ADJUSTED | reason=%s old_reference=%.4f new_reference=%.4f cumulative_retention=%.8f drawdown_pct=%.6f",
            reason,
            old_reference,
            equity,
            state.cumulative_retention,
            state.drawdown_pct,
        )

    def _new_state(self, now_ms: int, equity: float) -> RollingLossGuardState:
        reference = max(_safe_float(equity, 0.0), 0.0)
        return RollingLossGuardState(
            enabled=self.config.enabled,
            reference_flat_equity=reference,
            last_flat_equity=reference,
            last_flat_ts_ms=now_ms if reference > 0 else None,
            last_event_ts_ms=now_ms,
        )

    def _load_state(self, *, current_equity: float) -> RollingLossGuardState | None:
        if not self.state_path.exists():
            return None
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            if not raw.strip():
                logger.warning("ROLLING_DRAWDOWN_GUARD_STATE_EMPTY | path=%s reinitializing=true", self.state_path)
                return None
            data = json.loads(raw)
            if not isinstance(data, dict):
                logger.warning("ROLLING_DRAWDOWN_GUARD_STATE_INVALID_JSON | path=%s reinitializing=true", self.state_path)
                return None
            return self._state_from_dict(data, current_equity=current_equity)
        except Exception as exc:
            logger.warning("ROLLING_DRAWDOWN_GUARD_STATE_LOAD_FAILED | path=%s error=%s reinitializing=true", self.state_path, exc)
            return None

    def _state_from_dict(self, data: dict[str, object], *, current_equity: float) -> RollingLossGuardState:
        migrated_from_window = "reference_flat_equity" not in data and (
            "baseline_equity" in data or "last_loss_pct" in data or "window_start_ts_ms" in data
        )
        old_loss_pct = max(_safe_float(data.get("last_loss_pct"), 0.0), 0.0)
        reference = _safe_float(data.get("reference_flat_equity"), 0.0)
        if reference <= 0:
            reference = _safe_float(data.get("baseline_equity"), 0.0)
        if reference <= 0:
            reference = _safe_float(current_equity, 0.0)

        cumulative_retention = _safe_float(data.get("cumulative_retention"), 0.0)
        drawdown_pct = max(_safe_float(data.get("drawdown_pct"), 0.0), 0.0)
        if cumulative_retention <= 0:
            if old_loss_pct > 0:
                cumulative_retention = max(0.0, 1.0 - old_loss_pct)
                drawdown_pct = old_loss_pct
            else:
                cumulative_retention = 1.0
        if drawdown_pct <= 0:
            drawdown_pct = max(0.0, 1.0 - cumulative_retention)

        state = RollingLossGuardState(
            enabled=bool(data.get("enabled", self.config.enabled)),
            reference_flat_equity=reference,
            cumulative_retention=min(max(cumulative_retention, 0.0), 1.0),
            drawdown_pct=drawdown_pct,
            last_flat_equity=_safe_float(data.get("last_flat_equity"), reference),
            last_flat_ts_ms=self._optional_int(data.get("last_flat_ts_ms")),
            last_flat_event_id=self._optional_str(data.get("last_flat_event_id")),
            last_segment_retention=_safe_float(data.get("last_segment_retention"), 1.0),
            last_segment_return_pct=_safe_float(data.get("last_segment_return_pct"), 0.0),
            last_segment_drawdown_delta_pct=_safe_float(data.get("last_segment_drawdown_delta_pct"), 0.0),
            max_drawdown_pct=max(_safe_float(data.get("max_drawdown_pct"), drawdown_pct), drawdown_pct),
            warn_triggered=bool(data.get("warn_triggered", False)),
            soft_halt_triggered=bool(data.get("soft_halt_triggered", False)),
            hard_halt_triggered=bool(data.get("hard_halt_triggered", False)),
            halt_active=bool(data.get("halt_active", False)),
            halt_level=self._optional_str(data.get("halt_level")),
            halt_until_ts_ms=self._optional_int(data.get("halt_until_ts_ms")),
            last_event_ts_ms=self._optional_int(data.get("last_event_ts_ms")),
        )
        if migrated_from_window:
            logger.warning(
                "ROLLING_DRAWDOWN_GUARD_STATE_MIGRATED | old_last_loss_pct=%.6f reference_flat_equity=%.4f cumulative_retention=%.8f drawdown_pct=%.6f",
                old_loss_pct,
                state.reference_flat_equity,
                state.cumulative_retention,
                state.drawdown_pct,
            )
        return state

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
        segment_retention: float | None = None,
        segment_return_pct: float | None = None,
        cumulative_retention: float | None = None,
        drawdown_pct: float | None = None,
        reference_flat_equity: float | None = None,
        flat_equity: float | None = None,
        max_drawdown_pct: float | None = None,
    ) -> RollingLossGuardDecision:
        state = self._require_state()
        return RollingLossGuardDecision(
            action=action,
            state=state,
            rolling_realized_pnl=rolling_realized_pnl,
            loss_usdt=loss_usdt,
            loss_pct=loss_pct,
            threshold_pct=threshold_pct,
            halt_hours=halt_hours,
            halt_until_ts_ms=halt_until_ts_ms,
            reason=reason,
            segment_retention=state.last_segment_retention if segment_retention is None else segment_retention,
            segment_return_pct=state.last_segment_return_pct if segment_return_pct is None else segment_return_pct,
            cumulative_retention=state.cumulative_retention if cumulative_retention is None else cumulative_retention,
            drawdown_pct=state.drawdown_pct if drawdown_pct is None else drawdown_pct,
            reference_flat_equity=state.reference_flat_equity if reference_flat_equity is None else reference_flat_equity,
            flat_equity=state.last_flat_equity if flat_equity is None else flat_equity,
            max_drawdown_pct=state.max_drawdown_pct if max_drawdown_pct is None else max_drawdown_pct,
        )

    def _clear_trigger_flags_if_recovered(self, state: RollingLossGuardState) -> None:
        if state.cumulative_retention >= 1.0 or state.drawdown_pct + 1e-12 < self.config.warn_pct:
            self._clear_trigger_flags(state)

    @staticmethod
    def _clear_trigger_flags(state: RollingLossGuardState) -> None:
        state.warn_triggered = False
        state.soft_halt_triggered = False
        state.hard_halt_triggered = False
        if not state.halt_active:
            state.halt_level = None

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_str(value: object) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _threshold_reached(value: float, threshold: float) -> bool:
        return value + 1e-12 >= threshold
