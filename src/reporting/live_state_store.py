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
    tp_order_id: str | None = None
    tp_order_ids: list[str] | None = None
    partial_tp_price: float | None = None
    partial_tp_ratio: float = 0.0
    tp_plan: str = "SINGLE"
    partial_tp_consumed: bool = False
    total_entry_qty: float = 0.0
    total_entry_notional: float = 0.0
    avg_entry_price: float = 0.0
    breakeven_price: float = 0.0
    position_cost_entry_notional: float = 0.0
    position_cost_exit_notional: float = 0.0
    position_cost_remaining_qty: float = 0.0
    net_remaining_breakeven_price: float = 0.0
    tp_mode: str = "MIDDLE"
    last_order_ts_ms: int = 0
    first_entry_ts_ms: int = 0
    add_freeze_until_ts_ms: int = 0
    add_freeze_penalty_count: int = 0
    last_tp_update_ts_ms: int = 0
    last_tp_update_candle_ts_ms: int = 0
    entry_protective_sl_price: float | None = None
    entry_protective_sl_order_id: str | None = None
    entry_protective_sl_protected: bool = False
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
    middle_runner_size_mismatch_protected: bool = False
    middle_runner_size_mismatch_warning_ts_ms: int = 0
    middle_runner_sl_diag_last_signature: str | None = None
    middle_runner_sl_time_tighten_candle_count: int = 0
    middle_runner_sl_time_tighten_last_candle_ts_ms: int = 0
    middle_runner_sl_time_tighten_log_candle_ts_ms: int = 0
    three_stage_runner_enabled_for_position: bool = False
    three_stage_tp1_price: float | None = None
    three_stage_tp2_price: float | None = None
    three_stage_runner_initial_tp_price: float | None = None
    three_stage_tp1_ratio: float = 0.0
    three_stage_tp2_ratio: float = 0.0
    three_stage_runner_ratio: float = 0.0
    three_stage_tp1_consumed: bool = False
    three_stage_tp2_consumed: bool = False
    three_stage_post_tp1_protective_sl_price: float | None = None
    three_stage_post_tp1_protective_sl_order_id: str | None = None
    three_stage_post_tp1_sl_extension_triggered: bool = False
    three_stage_post_tp1_protected: bool = False
    three_stage_post_tp1_sl_diag_last_signature: str | None = None
    three_stage_post_tp1_sl_time_tighten_candle_count: int = 0
    three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms: int = 0
    three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms: int = 0
    three_stage_pre_tp1_degrade_stage: str | None = None
    three_stage_pre_tp1_degraded_ts_ms: int = 0
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
    last_add_skip_log_reason: str | None = None
    last_add_skip_log_ts_ms: int = 0
    core_contracts: str | None = None
    core_eth_qty: float = 0.0
    startup_force_tp_reconcile: bool = False
    cash_before_position: float | None = None
    updated_at: str = ""
    # ── Middle Bucket Split fields ────────────────────────────────────
    middle_bucket_split_active: bool = False
    middle_bucket_split_fast_consumed: bool = False
    middle_bucket_split_slow_consumed: bool = False
    middle_bucket_split_fast_price: float | None = None
    middle_bucket_split_slow_price: float | None = None
    middle_bucket_split_effective_price: float | None = None
    middle_bucket_split_middle_bucket_ratio: float = 0.0
    middle_bucket_split_fast_ratio_of_bucket: float = 0.0
    middle_bucket_split_slow_ratio_of_bucket: float = 0.0
    middle_bucket_split_fast_total_ratio: float = 0.0
    middle_bucket_split_slow_total_ratio: float = 0.0
    middle_bucket_split_reason: str | None = None
    middle_bucket_split_fast_sl_price: float | None = None
    middle_bucket_split_fast_sl_order_id: str | None = None
    middle_bucket_split_fast_sl_protected: bool = False
    middle_bucket_split_fast_sl_invalid_action_taken: str | None = None
    middle_bucket_split_add_disabled: bool = False

    # ── Post-Entry SL Cooldown ────────────────────────────────────────
    post_entry_sl_cooldown_until_ts_ms: int = 0
    post_entry_sl_cooldown_side: str | None = None
    post_entry_sl_cooldown_reason: str | None = None


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
    def from_strategy_state(*, position_id: str | None, symbol: str, strategy_state: Any,
                            cash_before_position: float | None) -> LivePositionState:
        return LivePositionState(
            position_id=position_id,
            symbol=symbol,
            side=strategy_state.side,
            layers=int(strategy_state.layers or 0),
            last_entry_price=strategy_state.last_entry_price,
            tp_price=strategy_state.tp_price,
            tp_order_id=getattr(strategy_state, "tp_order_id", None),
            tp_order_ids=list(getattr(strategy_state, "tp_order_ids", []) or []),
            partial_tp_price=getattr(strategy_state, "partial_tp_price", None),
            partial_tp_ratio=float(getattr(strategy_state, "partial_tp_ratio", 0.0) or 0.0),
            tp_plan=_normalize_tp_plan(str(getattr(strategy_state, "tp_plan", "SINGLE") or "SINGLE")),
            partial_tp_consumed=bool(getattr(strategy_state, "partial_tp_consumed", False)),
            total_entry_qty=float(strategy_state.total_entry_qty or 0.0),
            total_entry_notional=float(strategy_state.total_entry_notional or 0.0),
            avg_entry_price=float(strategy_state.avg_entry_price or 0.0),
            breakeven_price=float(strategy_state.breakeven_price or 0.0),
            position_cost_entry_notional=float(getattr(strategy_state, "position_cost_entry_notional", 0.0) or 0.0),
            position_cost_exit_notional=float(getattr(strategy_state, "position_cost_exit_notional", 0.0) or 0.0),
            position_cost_remaining_qty=float(getattr(strategy_state, "position_cost_remaining_qty", 0.0) or 0.0),
            net_remaining_breakeven_price=float(getattr(strategy_state, "net_remaining_breakeven_price", 0.0) or 0.0),
            tp_mode=strategy_state.tp_mode,
            last_order_ts_ms=int(strategy_state.last_order_ts_ms or 0),
            first_entry_ts_ms=int(getattr(strategy_state, "first_entry_ts_ms", 0) or 0),
            add_freeze_until_ts_ms=int(getattr(strategy_state, "add_freeze_until_ts_ms", 0) or 0),
            add_freeze_penalty_count=int(getattr(strategy_state, "add_freeze_penalty_count", 0) or 0),
            last_tp_update_ts_ms=int(strategy_state.last_tp_update_ts_ms or 0),
            last_tp_update_candle_ts_ms=int(getattr(strategy_state, "last_tp_update_candle_ts_ms", 0) or 0),
            entry_protective_sl_price=getattr(strategy_state, "entry_protective_sl_price", None),
            entry_protective_sl_order_id=getattr(strategy_state, "entry_protective_sl_order_id", None),
            entry_protective_sl_protected=bool(getattr(strategy_state, "entry_protective_sl_protected", False)),
            middle_runner_enabled_for_position=bool(
                getattr(strategy_state, "middle_runner_enabled_for_position", False)),
            middle_runner_pending=bool(getattr(strategy_state, "middle_runner_pending", False)),
            middle_runner_active=bool(getattr(strategy_state, "middle_runner_active", False)),
            middle_runner_first_close_ratio=float(
                getattr(strategy_state, "middle_runner_first_close_ratio", 0.0) or 0.0),
            middle_runner_keep_ratio=float(getattr(strategy_state, "middle_runner_keep_ratio", 0.0) or 0.0),
            middle_runner_first_tp_price=getattr(strategy_state, "middle_runner_first_tp_price", None),
            middle_runner_final_tp_price=getattr(strategy_state, "middle_runner_final_tp_price", None),
            middle_runner_protective_sl_price=getattr(strategy_state, "middle_runner_protective_sl_price", None),
            middle_runner_protective_sl_order_id=getattr(strategy_state, "middle_runner_protective_sl_order_id", None),
            middle_runner_extension_triggered=bool(getattr(strategy_state, "middle_runner_extension_triggered", False)),
            middle_runner_add_disabled=bool(getattr(strategy_state, "middle_runner_add_disabled", False)),
            middle_runner_size_mismatch_protected=bool(
                getattr(strategy_state, "middle_runner_size_mismatch_protected", False)),
            middle_runner_size_mismatch_warning_ts_ms=int(
                getattr(strategy_state, "middle_runner_size_mismatch_warning_ts_ms", 0) or 0),
            middle_runner_sl_diag_last_signature=getattr(strategy_state, "middle_runner_sl_diag_last_signature", None),
            middle_runner_sl_time_tighten_candle_count=int(
                getattr(strategy_state, "middle_runner_sl_time_tighten_candle_count", 0) or 0),
            middle_runner_sl_time_tighten_last_candle_ts_ms=int(
                getattr(strategy_state, "middle_runner_sl_time_tighten_last_candle_ts_ms", 0) or 0),
            middle_runner_sl_time_tighten_log_candle_ts_ms=int(
                getattr(strategy_state, "middle_runner_sl_time_tighten_log_candle_ts_ms", 0) or 0),
            three_stage_runner_enabled_for_position=bool(
                getattr(strategy_state, "three_stage_runner_enabled_for_position", False)),
            three_stage_tp1_price=getattr(strategy_state, "three_stage_tp1_price", None),
            three_stage_tp2_price=getattr(strategy_state, "three_stage_tp2_price", None),
            three_stage_runner_initial_tp_price=getattr(strategy_state, "three_stage_runner_initial_tp_price", None),
            three_stage_tp1_ratio=float(getattr(strategy_state, "three_stage_tp1_ratio", 0.0) or 0.0),
            three_stage_tp2_ratio=float(getattr(strategy_state, "three_stage_tp2_ratio", 0.0) or 0.0),
            three_stage_runner_ratio=float(getattr(strategy_state, "three_stage_runner_ratio", 0.0) or 0.0),
            three_stage_tp1_consumed=bool(getattr(strategy_state, "three_stage_tp1_consumed", False)),
            three_stage_tp2_consumed=bool(getattr(strategy_state, "three_stage_tp2_consumed", False)),
            three_stage_post_tp1_protective_sl_price=getattr(strategy_state, "three_stage_post_tp1_protective_sl_price",
                                                             None),
            three_stage_post_tp1_protective_sl_order_id=getattr(strategy_state,
                                                                "three_stage_post_tp1_protective_sl_order_id", None),
            three_stage_post_tp1_sl_extension_triggered=bool(
                getattr(strategy_state, "three_stage_post_tp1_sl_extension_triggered", False)),
            three_stage_post_tp1_protected=bool(getattr(strategy_state, "three_stage_post_tp1_protected", False)),
            three_stage_post_tp1_sl_diag_last_signature=getattr(strategy_state,
                                                                "three_stage_post_tp1_sl_diag_last_signature", None),
            three_stage_post_tp1_sl_time_tighten_candle_count=int(
                getattr(strategy_state, "three_stage_post_tp1_sl_time_tighten_candle_count", 0) or 0),
            three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms=int(
                getattr(strategy_state, "three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms", 0) or 0),
            three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms=int(
                getattr(strategy_state, "three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms", 0) or 0),
            three_stage_pre_tp1_degrade_stage=getattr(strategy_state, "three_stage_pre_tp1_degrade_stage", None),
            three_stage_pre_tp1_degraded_ts_ms=int(
                getattr(strategy_state, "three_stage_pre_tp1_degraded_ts_ms", 0) or 0),
            trend_runner_active=bool(getattr(strategy_state, "trend_runner_active", False)),
            trend_runner_trend_start_ts_ms=int(getattr(strategy_state, "trend_runner_trend_start_ts_ms", 0) or 0),
            trend_runner_adjust_count=int(getattr(strategy_state, "trend_runner_adjust_count", 0) or 0),
            trend_runner_last_update_candle_ts_ms=int(
                getattr(strategy_state, "trend_runner_last_update_candle_ts_ms", 0) or 0),
            trend_runner_tp_price=getattr(strategy_state, "trend_runner_tp_price", None),
            trend_runner_sl_price=getattr(strategy_state, "trend_runner_sl_price", None),
            trend_runner_tp_order_id=getattr(strategy_state, "trend_runner_tp_order_id", None),
            trend_runner_sl_order_id=getattr(strategy_state, "trend_runner_sl_order_id", None),
            trend_runner_exit_reason=getattr(strategy_state, "trend_runner_exit_reason", None),
            trend_runner_reverse_candidate=bool(getattr(strategy_state, "trend_runner_reverse_candidate", False)),
            trend_runner_reverse_start_ts_ms=int(getattr(strategy_state, "trend_runner_reverse_start_ts_ms", 0) or 0),
            trend_runner_reverse_start_price=getattr(strategy_state, "trend_runner_reverse_start_price", None),
            trend_runner_reverse_extreme_price=getattr(strategy_state, "trend_runner_reverse_extreme_price", None),
            trend_runner_reverse_fast_cvd_start=float(
                getattr(strategy_state, "trend_runner_reverse_fast_cvd_start", 0.0) or 0.0),
            trend_runner_reverse_samples=list(getattr(strategy_state, "trend_runner_reverse_samples", []) or []),
            last_add_skip_log_reason=getattr(strategy_state, "last_add_skip_log_reason", None),
            last_add_skip_log_ts_ms=int(getattr(strategy_state, "last_add_skip_log_ts_ms", 0) or 0),
            core_contracts=getattr(strategy_state, "core_contracts", None),
            core_eth_qty=float(getattr(strategy_state, "core_eth_qty", 0.0) or 0.0),
            startup_force_tp_reconcile=bool(getattr(strategy_state, "startup_force_tp_reconcile", False)),
            cash_before_position=cash_before_position,
            # ── Middle Bucket Split fields ────────────────────────────
            middle_bucket_split_active=bool(getattr(strategy_state, "middle_bucket_split_active", False)),
            middle_bucket_split_fast_consumed=bool(getattr(strategy_state, "middle_bucket_split_fast_consumed", False)),
            middle_bucket_split_slow_consumed=bool(getattr(strategy_state, "middle_bucket_split_slow_consumed", False)),
            middle_bucket_split_fast_price=getattr(strategy_state, "middle_bucket_split_fast_price", None),
            middle_bucket_split_slow_price=getattr(strategy_state, "middle_bucket_split_slow_price", None),
            middle_bucket_split_effective_price=getattr(strategy_state, "middle_bucket_split_effective_price", None),
            middle_bucket_split_middle_bucket_ratio=float(
                getattr(strategy_state, "middle_bucket_split_middle_bucket_ratio", 0.0) or 0.0),
            middle_bucket_split_fast_ratio_of_bucket=float(
                getattr(strategy_state, "middle_bucket_split_fast_ratio_of_bucket", 0.0) or 0.0),
            middle_bucket_split_slow_ratio_of_bucket=float(
                getattr(strategy_state, "middle_bucket_split_slow_ratio_of_bucket", 0.0) or 0.0),
            middle_bucket_split_fast_total_ratio=float(
                getattr(strategy_state, "middle_bucket_split_fast_total_ratio", 0.0) or 0.0),
            middle_bucket_split_slow_total_ratio=float(
                getattr(strategy_state, "middle_bucket_split_slow_total_ratio", 0.0) or 0.0),
            middle_bucket_split_reason=getattr(strategy_state, "middle_bucket_split_reason", None),
            middle_bucket_split_fast_sl_price=getattr(strategy_state, "middle_bucket_split_fast_sl_price", None),
            middle_bucket_split_fast_sl_order_id=getattr(strategy_state, "middle_bucket_split_fast_sl_order_id", None),
            middle_bucket_split_fast_sl_protected=bool(
                getattr(strategy_state, "middle_bucket_split_fast_sl_protected", False)),
            middle_bucket_split_fast_sl_invalid_action_taken=getattr(
                strategy_state, "middle_bucket_split_fast_sl_invalid_action_taken", None),
            middle_bucket_split_add_disabled=bool(
                getattr(strategy_state, "middle_bucket_split_add_disabled", False)),
            # ── Post-Entry SL Cooldown ────────────────────────────────
            post_entry_sl_cooldown_until_ts_ms=int(
                getattr(strategy_state, "post_entry_sl_cooldown_until_ts_ms", 0) or 0),
            post_entry_sl_cooldown_side=getattr(strategy_state, "post_entry_sl_cooldown_side", None),
            post_entry_sl_cooldown_reason=getattr(strategy_state, "post_entry_sl_cooldown_reason", None),
        )


def _normalize_tp_plan(tp_plan: str) -> str:
    if tp_plan == "MIDDLE_RUNNER":
        return tp_plan
    if tp_plan == "THREE_STAGE_RUNNER":
        return tp_plan
    return "SINGLE"
