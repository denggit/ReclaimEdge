from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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


class LiveTradeJournal:
    """Append-only JSONL journal for live trading review and daily reports."""

    def __init__(self, path: str | Path | None = None) -> None:
        raw_path = path or os.getenv("TRADE_JOURNAL_PATH") or DEFAULT_JOURNAL_PATH
        self.path = Path(raw_path)
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw_summary_path = os.getenv("TRADE_SUMMARY_PATH") or self.path.with_name("live_trade_summary.jsonl")
        self.summary_path = Path(raw_summary_path)
        if not self.summary_path.is_absolute():
            self.summary_path = ROOT / self.summary_path
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    def new_position_id(self, symbol: str, side: str, ts_ms: int | None = None) -> str:
        seed = ts_ms if ts_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
        return f"{symbol}:{side}:{seed}:{uuid.uuid4().hex[:8]}"

    def append(self, event_type: str, payload: dict[str, Any], position_id: str | None = None) -> None:
        event = JournalEvent(
            event_id=uuid.uuid4().hex,
            event_type=event_type,
            ts_iso=utc_now_iso(),
            position_id=position_id,
            payload=payload,
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

    def _load_events_from_path(self, path: Path, start: datetime | None = None, end: datetime | None = None) -> list[JournalEvent]:
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

    def record_startup_recovery(self, *, position_id: str, symbol: str, side: str, contracts: str, eth_qty: float, avg_entry: float, cash: float | None, equity: float | None) -> None:
        self.append(
            "STARTUP_RECOVERY",
            {
                "symbol": symbol,
                "side": side,
                "contracts": contracts,
                "eth_qty": eth_qty,
                "avg_entry": avg_entry,
                "cash": cash,
                "equity": equity,
                "note": "Recovered existing OKX position. Earlier entry details may be incomplete.",
            },
            position_id=position_id,
        )

    def record_entry(self, *, position_id: str, intent: Any, result: Any, cash_before_position: float | None, equity: float | None, extra: dict[str, Any] | None = None) -> None:
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
                "equity": equity,
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
    ) -> None:
        pnl = None
        pnl_pct = None
        if cash_before_position is not None and cash_after is not None:
            pnl = cash_after - cash_before_position
            pnl_pct = pnl / cash_before_position * 100 if cash_before_position else None
        self.append(
            "FLAT",
            {
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
            },
            position_id=position_id,
        )

    def record_error(self, *, position_id: str | None, intent: Any, error: Exception, rolled_back: bool, halted: bool) -> None:
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
