from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.live.runtime_paths import RuntimePaths
from src.reporting.trade_journal import (
    DEFAULT_JOURNAL_PATH,
    DEFAULT_SUMMARY_PATH,
    JournalEvent,
    LiveTradeJournal,
    ROOT,
)


# ===========================================================================
# Default behaviour – unchanged
# ===========================================================================


def test_default_journal_paths_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``LiveTradeJournal()`` keeps DEFAULT_JOURNAL_PATH and derives
    summary from ``self.path.with_name("live_trade_summary.jsonl")``
    when no explicit summary_path or env is given — old behaviour unchanged."""
    import src.reporting.trade_journal as tj_module

    default_journal = tmp_path / "default_events.jsonl"
    monkeypatch.setattr(tj_module, "DEFAULT_JOURNAL_PATH", default_journal)
    monkeypatch.delenv("TRADE_JOURNAL_PATH", raising=False)
    monkeypatch.delenv("TRADE_SUMMARY_PATH", raising=False)

    journal = LiveTradeJournal()
    assert journal.path == default_journal
    # summary defaults to path.with_name("live_trade_summary.jsonl")
    assert journal.summary_path == default_journal.with_name("live_trade_summary.jsonl")


def test_env_journal_paths_still_override_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TRADE_JOURNAL_PATH`` and ``TRADE_SUMMARY_PATH`` env vars still override defaults."""
    env_events = tmp_path / "env_events.jsonl"
    env_summary = tmp_path / "env_summary.jsonl"
    monkeypatch.setenv("TRADE_JOURNAL_PATH", str(env_events))
    monkeypatch.setenv("TRADE_SUMMARY_PATH", str(env_summary))

    journal = LiveTradeJournal()
    assert journal.path == env_events
    assert journal.summary_path == env_summary


def test_explicit_journal_paths_win_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``path=`` and ``summary_path=`` win over env vars."""
    monkeypatch.setenv("TRADE_JOURNAL_PATH", str(tmp_path / "env_events.jsonl"))
    monkeypatch.setenv("TRADE_SUMMARY_PATH", str(tmp_path / "env_summary.jsonl"))

    explicit_events = tmp_path / "explicit_events.jsonl"
    explicit_summary = tmp_path / "explicit_summary.jsonl"

    journal = LiveTradeJournal(path=explicit_events, summary_path=explicit_summary)
    assert journal.path == explicit_events
    assert journal.summary_path == explicit_summary


def test_explicit_path_without_summary_keeps_old_summary_default_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``summary_path`` is not passed and env is not set, summary defaults
    to ``self.path.with_name("live_trade_summary.jsonl")`` — old behaviour unchanged."""
    monkeypatch.delenv("TRADE_JOURNAL_PATH", raising=False)
    monkeypatch.delenv("TRADE_SUMMARY_PATH", raising=False)

    custom_events = tmp_path / "custom_events.jsonl"
    journal = LiveTradeJournal(path=custom_events)
    assert journal.path == custom_events
    assert journal.summary_path == tmp_path / "live_trade_summary.jsonl"


# ===========================================================================
# from_runtime_paths
# ===========================================================================


def test_from_runtime_paths_uses_symbol_scoped_journal_files(tmp_path: Path) -> None:
    """``from_runtime_paths`` sets both journal_file and trade_summary_file."""
    runtime_paths = RuntimePaths(
        runtime_dir=tmp_path / "runtime", inst_id="ETH-USDT-SWAP",
    )
    journal = LiveTradeJournal.from_runtime_paths(runtime_paths)

    assert journal.path == tmp_path / "runtime" / "journal" / "live_trades_ETH-USDT-SWAP.jsonl"
    assert journal.summary_path == tmp_path / "runtime" / "journal" / "live_trade_summary_ETH-USDT-SWAP.jsonl"


def test_from_runtime_paths_accepts_btc_for_path_only(tmp_path: Path) -> None:
    """Path builder is generic — BTC symbol works for path generation only.

    This does NOT enable BTC live trading, create a TOML, or modify the
    validator.  It only proves that the path builder is symbol‑agnostic.
    """
    runtime_paths = RuntimePaths(
        runtime_dir=tmp_path / "runtime", inst_id="BTC-USDT-SWAP",
    )
    journal = LiveTradeJournal.from_runtime_paths(runtime_paths)

    assert journal.path == tmp_path / "runtime" / "journal" / "live_trades_BTC-USDT-SWAP.jsonl"
    assert journal.summary_path == tmp_path / "runtime" / "journal" / "live_trade_summary_BTC-USDT-SWAP.jsonl"


# ===========================================================================
# Symbol‑scoped journal append / load roundtrip
# ===========================================================================


def test_symbol_scoped_journal_append_load_roundtrip(tmp_path: Path) -> None:
    """Append → load roundtrip via ``from_runtime_paths`` journal."""
    runtime_paths = RuntimePaths(
        runtime_dir=tmp_path / "runtime", inst_id="ETH-USDT-SWAP",
    )
    journal = LiveTradeJournal.from_runtime_paths(runtime_paths)

    journal.append("ENTRY", {"symbol": "ETH-USDT-SWAP", "side": "LONG"}, position_id="pos-1")
    events = journal.load_events()

    assert len(events) == 1
    assert events[0].event_type == "ENTRY"
    assert events[0].payload["symbol"] == "ETH-USDT-SWAP"
    assert journal.path.exists()


def test_symbol_scoped_summary_append_load_roundtrip(tmp_path: Path) -> None:
    """Append → load roundtrip for summary journal via ``from_runtime_paths``."""
    runtime_paths = RuntimePaths(
        runtime_dir=tmp_path / "runtime", inst_id="ETH-USDT-SWAP",
    )
    journal = LiveTradeJournal.from_runtime_paths(runtime_paths)

    event = JournalEvent(
        event_id="summary-1",
        event_type="SUMMARY_SNAPSHOT",
        ts_iso="2026-06-09T00:00:00+00:00",
        position_id=None,
        payload={"equity": 1234.56},
    )
    journal.append_event(event, path=journal.summary_path)
    summary_events = journal.load_summary_events()

    assert len(summary_events) == 1
    assert summary_events[0].event_type == "SUMMARY_SNAPSHOT"
    assert summary_events[0].payload["equity"] == 1234.56


# ===========================================================================
# from_runtime_paths does not read env
# ===========================================================================


def test_from_runtime_paths_does_not_read_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_runtime_paths`` ignores ``TRADE_JOURNAL_PATH`` and
    ``TRADE_SUMMARY_PATH`` env vars."""
    monkeypatch.setenv("TRADE_JOURNAL_PATH", str(tmp_path / "env_events.jsonl"))
    monkeypatch.setenv("TRADE_SUMMARY_PATH", str(tmp_path / "env_summary.jsonl"))

    runtime_paths = RuntimePaths(
        runtime_dir=tmp_path / "runtime", inst_id="ETH-USDT-SWAP",
    )
    journal = LiveTradeJournal.from_runtime_paths(runtime_paths)

    # Must use symbol‑scoped paths, NOT env paths
    assert journal.path == tmp_path / "runtime" / "journal" / "live_trades_ETH-USDT-SWAP.jsonl"
    assert journal.summary_path == tmp_path / "runtime" / "journal" / "live_trade_summary_ETH-USDT-SWAP.jsonl"
