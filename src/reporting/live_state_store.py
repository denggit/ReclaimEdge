from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = ROOT / "data" / "trade_journal" / "live_state.json"


@dataclass
class LivePositionState:
    position_id: str | None = None
    symbol: str = "ETH-USDT-SWAP"
    side: str | None = None
    layers: int = 0
    last_entry_price: float | None = None
    tp_price: float | None = None
    partial_tp_price: float | None = None
    partial_tp_ratio: float = 0.0
    tp_plan: str = "SINGLE"
    partial_tp_consumed: bool = False
    total_entry_qty: float = 0.0
    total_entry_notional: float = 0.0
    avg_entry_price: float = 0.0
    breakeven_price: float = 0.0
    tp_mode: str = "MIDDLE"
    last_order_ts_ms: int = 0
    last_tp_update_ts_ms: int = 0
    last_tp_update_candle_ts_ms: int = 0
    near_tp_armed: bool = False
    near_tp_reduce_pending: bool = False
    near_tp_protected: bool = False
    near_tp_best_price: float | None = None
    near_tp_armed_ts_ms: int = 0
    near_tp_pending_ts_ms: int = 0
    near_tp_trigger_ts_ms: int = 0
    near_tp_protective_sl_price: float | None = None
    near_tp_protective_sl_order_id: str | None = None
    near_tp_add_disabled: bool = False
    middle_runner_enabled_for_position: bool = False
    middle_runner_pending: bool = False
    middle_runner_active: bool = False
    middle_runner_first_close_ratio: float = 0.0
    middle_runner_keep_ratio: float = 0.0
    middle_runner_first_tp_price: float | None = None
    middle_runner_final_tp_price: float | None = None
    middle_runner_protective_sl_price: float | None = None
    middle_runner_protective_sl_order_id: str | None = None
    middle_runner_extension_triggered: bool = False
    middle_runner_add_disabled: bool = False
    three_stage_runner_enabled_for_position: bool = False
    three_stage_tp1_price: float | None = None
    three_stage_tp2_price: float | None = None
    three_stage_runner_initial_tp_price: float | None = None
    three_stage_tp1_ratio: float = 0.0
    three_stage_tp2_ratio: float = 0.0
    three_stage_runner_ratio: float = 0.0
    three_stage_tp1_consumed: bool = False
    three_stage_tp2_consumed: bool = False
    trend_runner_active: bool = False
    trend_runner_trend_start_ts_ms: int = 0
    trend_runner_adjust_count: int = 0
    trend_runner_last_update_candle_ts_ms: int = 0
    trend_runner_tp_price: float | None = None
    trend_runner_sl_price: float | None = None
    trend_runner_tp_order_id: str | None = None
    trend_runner_sl_order_id: str | None = None
    trend_runner_exit_reason: str | None = None
    trend_runner_reverse_candidate: bool = False
    trend_runner_reverse_start_ts_ms: int = 0
    trend_runner_reverse_start_price: float | None = None
    trend_runner_reverse_extreme_price: float | None = None
    trend_runner_reverse_fast_cvd_start: float = 0.0
    trend_runner_reverse_samples: list | None = None
    cash_before_position: float | None = None
    updated_at: str = ""


class LiveStateStore:
    """Small JSON state store for restart recovery.

    OKX can tell us the current net position, but not how many strategy layers
    created it or what the strategy thought its TP mode was. This state store
    fills that gap for restarts and daily reporting.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        raw_path = path or os.getenv("LIVE_STATE_PATH") or DEFAULT_STATE_PATH
        self.path = Path(raw_path)
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> LivePositionState | None:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            fields = set(LivePositionState.__dataclass_fields__.keys())
            return LivePositionState(**{key: value for key, value in raw.items() if key in fields})
        except Exception:
            return None

    def save(self, state: LivePositionState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    @staticmethod
    def from_strategy_state(*, position_id: str | None, symbol: str, strategy_state: Any, cash_before_position: float | None) -> LivePositionState:
        return LivePositionState(
            position_id=position_id,
            symbol=symbol,
            side=strategy_state.side,
            layers=int(strategy_state.layers or 0),
            last_entry_price=strategy_state.last_entry_price,
            tp_price=strategy_state.tp_price,
            partial_tp_price=getattr(strategy_state, "partial_tp_price", None),
            partial_tp_ratio=float(getattr(strategy_state, "partial_tp_ratio", 0.0) or 0.0),
            tp_plan=_normalize_tp_plan(str(getattr(strategy_state, "tp_plan", "SINGLE") or "SINGLE")),
            partial_tp_consumed=bool(getattr(strategy_state, "partial_tp_consumed", False)),
            total_entry_qty=float(strategy_state.total_entry_qty or 0.0),
            total_entry_notional=float(strategy_state.total_entry_notional or 0.0),
            avg_entry_price=float(strategy_state.avg_entry_price or 0.0),
            breakeven_price=float(strategy_state.breakeven_price or 0.0),
            tp_mode=strategy_state.tp_mode,
            last_order_ts_ms=int(strategy_state.last_order_ts_ms or 0),
            last_tp_update_ts_ms=int(strategy_state.last_tp_update_ts_ms or 0),
            last_tp_update_candle_ts_ms=int(getattr(strategy_state, "last_tp_update_candle_ts_ms", 0) or 0),
            near_tp_armed=bool(getattr(strategy_state, "near_tp_armed", False)),
            near_tp_reduce_pending=bool(getattr(strategy_state, "near_tp_reduce_pending", False)),
            near_tp_protected=bool(getattr(strategy_state, "near_tp_protected", False)),
            near_tp_best_price=getattr(strategy_state, "near_tp_best_price", None),
            near_tp_armed_ts_ms=int(getattr(strategy_state, "near_tp_armed_ts_ms", 0) or 0),
            near_tp_pending_ts_ms=int(getattr(strategy_state, "near_tp_pending_ts_ms", 0) or 0),
            near_tp_trigger_ts_ms=int(getattr(strategy_state, "near_tp_trigger_ts_ms", 0) or 0),
            near_tp_protective_sl_price=getattr(strategy_state, "near_tp_protective_sl_price", None),
            near_tp_protective_sl_order_id=getattr(strategy_state, "near_tp_protective_sl_order_id", None),
            near_tp_add_disabled=bool(getattr(strategy_state, "near_tp_add_disabled", False)),
            middle_runner_enabled_for_position=bool(getattr(strategy_state, "middle_runner_enabled_for_position", False)),
            middle_runner_pending=bool(getattr(strategy_state, "middle_runner_pending", False)),
            middle_runner_active=bool(getattr(strategy_state, "middle_runner_active", False)),
            middle_runner_first_close_ratio=float(getattr(strategy_state, "middle_runner_first_close_ratio", 0.0) or 0.0),
            middle_runner_keep_ratio=float(getattr(strategy_state, "middle_runner_keep_ratio", 0.0) or 0.0),
            middle_runner_first_tp_price=getattr(strategy_state, "middle_runner_first_tp_price", None),
            middle_runner_final_tp_price=getattr(strategy_state, "middle_runner_final_tp_price", None),
            middle_runner_protective_sl_price=getattr(strategy_state, "middle_runner_protective_sl_price", None),
            middle_runner_protective_sl_order_id=getattr(strategy_state, "middle_runner_protective_sl_order_id", None),
            middle_runner_extension_triggered=bool(getattr(strategy_state, "middle_runner_extension_triggered", False)),
            middle_runner_add_disabled=bool(getattr(strategy_state, "middle_runner_add_disabled", False)),
            three_stage_runner_enabled_for_position=bool(getattr(strategy_state, "three_stage_runner_enabled_for_position", False)),
            three_stage_tp1_price=getattr(strategy_state, "three_stage_tp1_price", None),
            three_stage_tp2_price=getattr(strategy_state, "three_stage_tp2_price", None),
            three_stage_runner_initial_tp_price=getattr(strategy_state, "three_stage_runner_initial_tp_price", None),
            three_stage_tp1_ratio=float(getattr(strategy_state, "three_stage_tp1_ratio", 0.0) or 0.0),
            three_stage_tp2_ratio=float(getattr(strategy_state, "three_stage_tp2_ratio", 0.0) or 0.0),
            three_stage_runner_ratio=float(getattr(strategy_state, "three_stage_runner_ratio", 0.0) or 0.0),
            three_stage_tp1_consumed=bool(getattr(strategy_state, "three_stage_tp1_consumed", False)),
            three_stage_tp2_consumed=bool(getattr(strategy_state, "three_stage_tp2_consumed", False)),
            trend_runner_active=bool(getattr(strategy_state, "trend_runner_active", False)),
            trend_runner_trend_start_ts_ms=int(getattr(strategy_state, "trend_runner_trend_start_ts_ms", 0) or 0),
            trend_runner_adjust_count=int(getattr(strategy_state, "trend_runner_adjust_count", 0) or 0),
            trend_runner_last_update_candle_ts_ms=int(getattr(strategy_state, "trend_runner_last_update_candle_ts_ms", 0) or 0),
            trend_runner_tp_price=getattr(strategy_state, "trend_runner_tp_price", None),
            trend_runner_sl_price=getattr(strategy_state, "trend_runner_sl_price", None),
            trend_runner_tp_order_id=getattr(strategy_state, "trend_runner_tp_order_id", None),
            trend_runner_sl_order_id=getattr(strategy_state, "trend_runner_sl_order_id", None),
            trend_runner_exit_reason=getattr(strategy_state, "trend_runner_exit_reason", None),
            trend_runner_reverse_candidate=bool(getattr(strategy_state, "trend_runner_reverse_candidate", False)),
            trend_runner_reverse_start_ts_ms=int(getattr(strategy_state, "trend_runner_reverse_start_ts_ms", 0) or 0),
            trend_runner_reverse_start_price=getattr(strategy_state, "trend_runner_reverse_start_price", None),
            trend_runner_reverse_extreme_price=getattr(strategy_state, "trend_runner_reverse_extreme_price", None),
            trend_runner_reverse_fast_cvd_start=float(getattr(strategy_state, "trend_runner_reverse_fast_cvd_start", 0.0) or 0.0),
            trend_runner_reverse_samples=list(getattr(strategy_state, "trend_runner_reverse_samples", []) or []),
            cash_before_position=cash_before_position,
        )


def _normalize_tp_plan(tp_plan: str) -> str:
    if tp_plan == "SPLIT_50_50":
        return "SPLIT_PARTIAL_FINAL"
    if tp_plan == "SPLIT_PARTIAL_FINAL":
        return tp_plan
    if tp_plan == "MIDDLE_RUNNER":
        return tp_plan
    if tp_plan == "THREE_STAGE_RUNNER":
        return tp_plan
    return "SINGLE"
