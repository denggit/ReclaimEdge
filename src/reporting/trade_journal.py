from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from src.utils import to_json_safe

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOURNAL_PATH = ROOT / "data" / "trade_journal" / "live_trade_events.jsonl"
DEFAULT_SUMMARY_PATH = ROOT / "data" / "trade_journal" / "live_trade_summary.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class JournalEvent:
    event_id: str
    event_type: str
    ts_iso: str
    position_id: str | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class UnclosedPositionMatch:
    position_id: str
    cash_before_position: float | None
    first_event_ts_iso: str
    last_event_ts_iso: str


class JournalPathProvider(Protocol):
    """Structural interface for any object that exposes symbol‑scoped journal paths.

    This protocol lets :class:`LiveTradeJournal` accept a :class:`RuntimePaths`
    instance **without** importing ``src.live.runtime_paths``, keeping the
    reporting layer decoupled from the live layer.
    """

    @property
    def journal_file(self) -> Path:
        ...

    @property
    def trade_summary_file(self) -> Path:
        ...


class LiveTradeJournal:
    """Append-only JSONL journal for live trading review and daily reports."""

    def __init__(self, path: str | Path | None = None, summary_path: str | Path | None = None) -> None:
        raw_path = path or os.getenv("TRADE_JOURNAL_PATH") or DEFAULT_JOURNAL_PATH
        self.path = Path(raw_path)
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw_summary_path = (
            summary_path
            or os.getenv("TRADE_SUMMARY_PATH")
            or self.path.with_name("live_trade_summary.jsonl")
        )
        self.summary_path = Path(raw_summary_path)
        if not self.summary_path.is_absolute():
            self.summary_path = ROOT / self.summary_path
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_runtime_paths(cls, runtime_paths: JournalPathProvider) -> "LiveTradeJournal":
        """Create a journal using symbol‑scoped RuntimePaths journal files.

        This does **not** read env and does **not** migrate legacy journal files.
        It delegates entirely to the supplied path provider's ``journal_file``
        and ``trade_summary_file`` properties.
        """
        return cls(
            path=runtime_paths.journal_file,
            summary_path=runtime_paths.trade_summary_file,
        )

    def new_position_id(self, symbol: str, side: str, ts_ms: int | None = None) -> str:
        if ts_ms is None:
            # Startup recovery calls this without ts_ms. If the local live_state.json
            # was lost but the journal still has an unclosed matching position,
            # reuse that position_id so reports do not create an orphan incomplete
            # record for the same real OKX position.
            unclosed = self.find_latest_unclosed_position(symbol, side)
            if unclosed is not None:
                return unclosed.position_id
        seed = ts_ms if ts_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
        return f"{symbol}:{side}:{seed}:{uuid.uuid4().hex[:8]}"

    def find_latest_unclosed_position(self, symbol: str, side: str) -> UnclosedPositionMatch | None:
        grouped = group_position_events(self.load_events())
        desired_symbol = str(symbol or "")
        desired_side = str(side or "").upper()
        best: UnclosedPositionMatch | None = None
        best_ts: datetime | None = None

        for position_id, items in grouped.items():
            if position_id == "UNKNOWN":
                continue
            if any(event.event_type == "FLAT" for event in items):
                continue

            lifecycle_events = [event for event in items if event.event_type in {"ENTRY", "STARTUP_RECOVERY"}]
            if not lifecycle_events:
                continue
            if not any(self._event_matches_position(event, desired_symbol, desired_side) for event in lifecycle_events):
                continue

            last_event = items[-1]
            last_ts = self._parse_event_ts(last_event.ts_iso)
            if best is not None and last_ts is not None and best_ts is not None and last_ts <= best_ts:
                continue
            if best is not None and (
                    last_ts is None or best_ts is None) and last_event.ts_iso <= best.last_event_ts_iso:
                continue

            best = UnclosedPositionMatch(
                position_id=position_id,
                cash_before_position=self._unclosed_position_cash_before(lifecycle_events),
                first_event_ts_iso=items[0].ts_iso,
                last_event_ts_iso=last_event.ts_iso,
            )
            best_ts = last_ts

        return best

    @classmethod
    def _event_matches_position(cls, event: JournalEvent, symbol: str, side: str) -> bool:
        payload = event.payload
        event_symbol = str(payload.get("symbol") or "ETH-USDT-SWAP")
        event_side = str(payload.get("side") or "").upper()
        return event_symbol == symbol and event_side == side

    @classmethod
    def _unclosed_position_cash_before(cls, lifecycle_events: list[JournalEvent]) -> float | None:
        for event in lifecycle_events:
            if event.event_type == "ENTRY":
                value = cls._payload_float(event.payload.get("cash_before_position"))
                if value is not None:
                    return value
        for event in lifecycle_events:
            if event.event_type == "STARTUP_RECOVERY":
                value = cls._payload_float(event.payload.get("cash"))
                if value is not None:
                    return value
        return None

    @staticmethod
    def _payload_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_event_ts(ts_iso: str) -> datetime | None:
        try:
            return datetime.fromisoformat(ts_iso)
        except Exception:
            return None

    def append(self, event_type: str, payload: dict[str, Any], position_id: str | None = None) -> None:
        event = JournalEvent(
            event_id=uuid.uuid4().hex,
            event_type=event_type,
            ts_iso=utc_now_iso(),
            position_id=position_id,
            payload=to_json_safe(dict(payload)),
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")) + "\n")

    def append_event(self, event: JournalEvent, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")) + "\n")

    def load_events(self, start: datetime | None = None, end: datetime | None = None) -> list[JournalEvent]:
        if not self.path.exists():
            return []
        return self._load_events_from_path(self.path, start=start, end=end)

    def load_summary_events(self, start: datetime | None = None, end: datetime | None = None) -> list[JournalEvent]:
        if not self.summary_path.exists():
            return []
        return self._load_events_from_path(self.summary_path, start=start, end=end)

    def _load_events_from_path(self, path: Path, start: datetime | None = None, end: datetime | None = None) -> list[
        JournalEvent]:
        events: list[JournalEvent] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    ts = datetime.fromisoformat(raw["ts_iso"])
                    if start is not None and ts < start:
                        continue
                    if end is not None and ts >= end:
                        continue
                    events.append(JournalEvent(**raw))
                except Exception:
                    continue
        return events

    def has_event_type(self, event_type: str) -> bool:
        return any(event.event_type == event_type for event in self.load_events())

    def record_cash_baseline(
            self,
            *,
            source: str,
            cash: float | None,
            equity: float | None,
            note: str | None = None,
    ) -> None:
        if source != "manual" and self.has_event_type("CASH_BASELINE"):
            return
        self.append(
            "CASH_BASELINE",
            {
                "source": source,
                "cash": cash,
                "equity": equity,
                "note": note,
            },
        )

    def record_cash_transfer(
            self,
            *,
            direction: str,
            amount: float,
            cash_before: float,
            cash_after: float,
            equity_before: float | None,
            equity_after: float | None,
            reason: str,
    ) -> None:
        if direction not in {"DEPOSIT", "WITHDRAWAL"}:
            raise ValueError(f"Invalid cash transfer direction={direction}")
        if direction == "DEPOSIT" and amount <= 0:
            raise ValueError("DEPOSIT amount must be positive")
        if direction == "WITHDRAWAL" and amount >= 0:
            raise ValueError("WITHDRAWAL amount must be negative")
        self.append(
            "CASH_TRANSFER",
            {
                "direction": direction,
                "amount": amount,
                "cash_before": cash_before,
                "cash_after": cash_after,
                "equity_before": equity_before,
                "equity_after": equity_after,
                "reason": reason,
            },
        )

    def record_account_cash_drift(
            self,
            *,
            amount: float,
            cash_before: float,
            cash_after: float,
            equity_before: float | None,
            equity_after: float | None,
            reason: str,
    ) -> None:
        self.append(
            "ACCOUNT_CASH_DRIFT",
            {
                "amount": amount,
                "cash_before": cash_before,
                "cash_after": cash_after,
                "equity_before": equity_before,
                "equity_after": equity_after,
                "reason": reason,
            },
        )

    def record_summary_snapshot(self, payload: dict[str, Any]) -> None:
        event = JournalEvent(
            event_id=uuid.uuid4().hex,
            event_type="SUMMARY_SNAPSHOT",
            ts_iso=utc_now_iso(),
            position_id=None,
            payload=payload,
        )
        self.append_event(event, self.summary_path)

    def record_journal_compacted(
            self,
            *,
            archived_event_count: int,
            retained_event_count: int,
            archive_path: str | None,
            summary_path: str | None,
            snapshot_until: str,
    ) -> None:
        self.append(
            "JOURNAL_COMPACTED",
            {
                "archived_event_count": archived_event_count,
                "retained_event_count": retained_event_count,
                "archive_path": archive_path,
                "summary_path": summary_path,
                "snapshot_until": snapshot_until,
            },
        )

    def record_startup_recovery(
            self,
            *,
            position_id: str,
            symbol: str,
            side: str,
            contracts: str,
            eth_qty: float,
            avg_entry: float,
            cash: float | None,
            equity: float | None,
            extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "symbol": symbol,
            "side": side,
            "contracts": contracts,
            "eth_qty": eth_qty,
            "avg_entry": avg_entry,
            "cash": cash,
            "equity": equity,
            "note": "Recovered existing OKX position. Earlier entry details may be incomplete.",
        }
        if extra:
            payload.update(extra)
        self.append(
            "STARTUP_RECOVERY",
            payload,
            position_id=position_id,
        )

    def record_entry(self, *, position_id: str, intent: Any, result: Any, cash_before_position: float | None,
                     equity: float | None, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "intent_type": intent.intent_type,
            "side": intent.side,
            "layer_index": intent.layer_index,
            "price": intent.price,
            "contracts": result.contracts,
            "tp_price": intent.tp_price,
            "partial_tp_price": getattr(intent, "partial_tp_price", None),
            "partial_tp_ratio": getattr(intent, "partial_tp_ratio", 0.0),
            "tp_plan": getattr(intent, "tp_plan", "SINGLE"),
            "partial_tp_consumed": getattr(intent, "partial_tp_consumed", False),
            "tp_mode": intent.tp_mode,
            "avg_entry_price": intent.avg_entry_price,
            "breakeven_price": intent.breakeven_price,
            "reason": intent.reason,
            "size_margin_usdt": intent.size.margin_usdt,
            "size_notional_usdt": intent.size.notional_usdt,
            "size_eth_qty": intent.size.eth_qty,
            "layer_multiplier": getattr(intent.size, "layer_multiplier", 1.0),
            "fast_cvd": intent.fast_cvd,
            "previous_fast_cvd": intent.previous_fast_cvd,
            "buy_ratio": intent.buy_ratio,
            "sell_ratio": intent.sell_ratio,
            "boll_upper": intent.boll_upper,
            "boll_middle": intent.boll_middle,
            "boll_lower": intent.boll_lower,
            "order_id": result.order_id,
            "tp_order_id": result.tp_order_id,
            "tp_order_ids": list(getattr(result, "tp_order_ids", ()) or ()),
            "three_stage_tp1_price": getattr(intent, "three_stage_tp1_price", None),
            "three_stage_tp1_ratio": getattr(intent, "three_stage_tp1_ratio", 0.0),
            "three_stage_tp2_price": getattr(intent, "three_stage_tp2_price", None),
            "three_stage_tp2_ratio": getattr(intent, "three_stage_tp2_ratio", 0.0),
            "three_stage_runner_tp_price": getattr(intent, "three_stage_runner_tp_price", None),
            "three_stage_runner_sl_price": getattr(intent, "three_stage_runner_sl_price", None),
            "three_stage_runner_ratio": getattr(intent, "three_stage_runner_ratio", 0.0),
            "trend_runner_active": getattr(intent, "trend_runner_active", False),
            "trend_runner_adjust_count": getattr(intent, "trend_runner_adjust_count", 0),
            "cash_before_position": cash_before_position,
            "equity": equity,
        }
        if extra:
            payload.update(extra)
        self.append("ENTRY", payload, position_id=position_id)

    def record_tp_update(self, *, position_id: str | None, intent: Any, result: Any, equity: float | None) -> None:
        self.append(
            "TP_UPDATE",
            {
                "intent_type": intent.intent_type,
                "side": intent.side,
                "layer_index": intent.layer_index,
                "price": intent.price,
                "contracts": result.contracts,
                "tp_price": intent.tp_price,
                "partial_tp_price": getattr(intent, "partial_tp_price", None),
                "partial_tp_ratio": getattr(intent, "partial_tp_ratio", 0.0),
                "tp_plan": getattr(intent, "tp_plan", "SINGLE"),
                "partial_tp_consumed": getattr(intent, "partial_tp_consumed", False),
                "tp_mode": intent.tp_mode,
                "avg_entry_price": intent.avg_entry_price,
                "breakeven_price": intent.breakeven_price,
                "reason": intent.reason,
                "boll_upper": intent.boll_upper,
                "boll_middle": intent.boll_middle,
                "boll_lower": intent.boll_lower,
                "tp_order_id": result.tp_order_id,
                "tp_order_ids": list(getattr(result, "tp_order_ids", ()) or ()),
                "three_stage_tp1_price": getattr(intent, "three_stage_tp1_price", None),
                "three_stage_tp1_ratio": getattr(intent, "three_stage_tp1_ratio", 0.0),
                "three_stage_tp2_price": getattr(intent, "three_stage_tp2_price", None),
                "three_stage_tp2_ratio": getattr(intent, "three_stage_tp2_ratio", 0.0),
                "three_stage_runner_tp_price": getattr(intent, "three_stage_runner_tp_price", None),
                "three_stage_runner_sl_price": getattr(intent, "three_stage_runner_sl_price", None),
                "three_stage_runner_ratio": getattr(intent, "three_stage_runner_ratio", 0.0),
                "trend_runner_active": getattr(intent, "trend_runner_active", False),
                "trend_runner_adjust_count": getattr(intent, "trend_runner_adjust_count", 0),
                "trend_runner_exit_reason": getattr(intent, "trend_runner_exit_reason", None),
                "equity": equity,
            },
            position_id=position_id,
        )

    def record_near_tp_reduce(
            self,
            *,
            position_id: str | None,
            symbol: str,
            intent: Any,
            result: Any,
            protective_sl_fail_action: str | None = None,
    ) -> None:
        self.append(
            "NEAR_TP_REDUCE",
            {
                "symbol": symbol,
                "side": getattr(intent, "side", None),
                "reduce_ratio": getattr(intent, "near_tp_reduce_ratio", 0.0),
                "contracts_before": getattr(result, "contracts_before", ""),
                "contracts_reduced": getattr(result, "contracts_reduced", ""),
                "contracts_after": getattr(result, "contracts_after", ""),
                "avg_entry_price": getattr(intent, "avg_entry_price", None),
                "tp_price": getattr(intent, "tp_price", None),
                "near_tp_best_price": getattr(intent, "near_tp_best_price", None),
                "near_tp_progress_ratio": getattr(intent, "near_tp_progress_ratio", 0.0),
                "near_tp_giveback": getattr(intent, "near_tp_giveback", 0.0),
                "near_tp_giveback_threshold": getattr(intent, "near_tp_giveback_threshold", 0.0),
                "protective_sl_price": getattr(result, "protective_sl_price", "") or getattr(intent,
                                                                                             "near_tp_protective_sl_price",
                                                                                             None),
                "protective_sl_order_id": getattr(result, "protective_sl_order_id", None),
                "protective_sl_ok": bool(getattr(result, "protective_sl_ok", False)),
                "protective_sl_fail_action": protective_sl_fail_action,
                "near_tp_exit_all": bool(getattr(result, "near_tp_exit_all", False)),
                "reason": "near_tp_giveback_protection",
            },
            position_id=position_id,
        )

    def record_trend_runner_market_exit(self, *, position_id: str | None, symbol: str, intent: Any,
                                        result: Any) -> None:
        self.append(
            "TREND_RUNNER_MARKET_EXIT_SIGNAL",
            {
                "symbol": symbol,
                "side": getattr(intent, "side", None),
                "contracts_before": getattr(result, "contracts_before", ""),
                "contracts_reduced": getattr(result, "contracts_reduced", ""),
                "contracts_after": getattr(result, "contracts_after", ""),
                "tp_plan": getattr(intent, "tp_plan", "SINGLE"),
                "runner_tp_price": getattr(intent, "trend_runner_tp_price", None),
                "runner_sl_price": getattr(intent, "trend_runner_sl_price", None),
                "runner_ratio": getattr(intent, "three_stage_runner_ratio", 0.0),
                "trend_runner_active": getattr(intent, "trend_runner_active", False),
                "trend_runner_adjust_count": getattr(intent, "trend_runner_adjust_count", 0),
                "trend_runner_exit_reason": getattr(intent, "trend_runner_exit_reason", None) or getattr(intent,
                                                                                                         "reason",
                                                                                                         None),
                "reason": getattr(intent, "reason", None),
                "price": getattr(intent, "price", None),
            },
            position_id=position_id,
        )

    def record_flat(
            self,
            *,
            position_id: str | None,
            symbol: str,
            side: str | None,
            cash_before_position: float | None,
            cash_after: float | None,
            equity_after: float | None,
            reason: str,
            layers: int,
            avg_entry_price: float,
            last_tp_price: float | None,
            last_partial_tp_price: float | None = None,
            last_tp_plan: str = "SINGLE",
            partial_tp_consumed: bool = False,
            trend_runner_exit_reason: str | None = None,
            **extra: Any,
    ) -> None:
        pnl = None
        pnl_pct = None
        if cash_before_position is not None and cash_after is not None:
            pnl = cash_after - cash_before_position
            pnl_pct = pnl / cash_before_position * 100 if cash_before_position else None
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "cash_before_position": cash_before_position,
            "cash_after": cash_after,
            "equity_after": equity_after,
            "realized_pnl_usdt_est": pnl,
            "realized_pnl_pct_est": pnl_pct,
            "flat_reason": reason,
            "layers": layers,
            "avg_entry_price": avg_entry_price,
            "last_tp_price": last_tp_price,
            "last_partial_tp_price": last_partial_tp_price,
            "last_tp_plan": last_tp_plan,
            "partial_tp_consumed": partial_tp_consumed,
            "trend_runner_exit_reason": trend_runner_exit_reason,
        }
        if extra:
            payload.update(extra)
        self.append("FLAT", payload, position_id=position_id)

    def record_rolling_loss_guard(
            self,
            *,
            action: str,
            window_start_ts_ms: int | None = None,
            window_end_ts_ms: int | None = None,
            baseline_equity: float | None = None,
            rolling_realized_pnl: float = 0.0,
            loss_usdt: float = 0.0,
            loss_pct: float = 0.0,
            mode: str = "flat_to_flat_drawdown",
            reference_flat_equity: float | None = None,
            flat_equity: float | None = None,
            segment_retention: float | None = None,
            segment_return_pct: float | None = None,
            cumulative_retention: float | None = None,
            drawdown_pct: float | None = None,
            max_drawdown_pct: float | None = None,
            threshold_pct: float | None = None,
            halt_hours: float | None = None,
            halt_until_ts_ms: int | None = None,
            reason: str | None = None,
    ) -> None:
        self.append(
            "ROLLING_LOSS_GUARD",
            {
                "action": action,
                "mode": mode,
                "window_start_ts_ms": window_start_ts_ms,
                "window_end_ts_ms": window_end_ts_ms,
                "baseline_equity": baseline_equity,
                "reference_flat_equity": reference_flat_equity,
                "flat_equity": flat_equity,
                "segment_retention": segment_retention,
                "segment_return_pct": segment_return_pct,
                "cumulative_retention": cumulative_retention,
                "drawdown_pct": drawdown_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "rolling_realized_pnl": rolling_realized_pnl,
                "loss_usdt": loss_usdt,
                "loss_pct": loss_pct,
                "threshold_pct": threshold_pct,
                "halt_hours": halt_hours,
                "halt_until_ts_ms": halt_until_ts_ms,
                "reason": reason,
            },
        )

    def record_error(self, *, position_id: str | None, intent: Any, error: Exception, rolled_back: bool,
                     halted: bool) -> None:
        self.append(
            "ERROR",
            {
                "intent_type": getattr(intent, "intent_type", None),
                "side": getattr(intent, "side", None),
                "layer_index": getattr(intent, "layer_index", None),
                "price": getattr(intent, "price", None),
                "tp_price": getattr(intent, "tp_price", None),
                "partial_tp_price": getattr(intent, "partial_tp_price", None),
                "tp_plan": getattr(intent, "tp_plan", "SINGLE"),
                "partial_tp_consumed": getattr(intent, "partial_tp_consumed", False),
                "error": str(error),
                "rolled_back": rolled_back,
                "halted": halted,
            },
            position_id=position_id,
        )


def group_position_events(events: Iterable[JournalEvent]) -> dict[str, list[JournalEvent]]:
    grouped: dict[str, list[JournalEvent]] = {}
    for event in events:
        key = event.position_id or "UNKNOWN"
        grouped.setdefault(key, []).append(event)
    for items in grouped.values():
        items.sort(key=lambda item: item.ts_iso)
    return grouped
