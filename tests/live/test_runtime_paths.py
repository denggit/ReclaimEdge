from __future__ import annotations

from pathlib import Path

import pytest

from src.live.runtime_paths import (
    RuntimePaths,
    build_runtime_paths,
    sanitize_inst_id,
)


# ============================================================================
# sanitize_inst_id
# ============================================================================

class TestSanitizeInstId:
    """Tests for :func:`sanitize_inst_id`."""

    def test_keeps_okx_swap_symbol(self) -> None:
        """``ETH-USDT-SWAP`` must pass through unchanged."""
        assert sanitize_inst_id("ETH-USDT-SWAP") == "ETH-USDT-SWAP"

    def test_rejects_empty_string(self) -> None:
        """Empty string must raise ValueError."""
        with pytest.raises(ValueError):
            sanitize_inst_id("")

    def test_rejects_whitespace_only(self) -> None:
        """Whitespace-only string must raise ValueError."""
        with pytest.raises(ValueError):
            sanitize_inst_id("   ")

    def test_rejects_forward_slash(self) -> None:
        """``ETH/USDT`` contains a path separator and must be rejected."""
        with pytest.raises(ValueError):
            sanitize_inst_id("ETH/USDT")

    def test_rejects_backslash(self) -> None:
        r"""``ETH\\USDT`` contains a path separator and must be rejected."""
        with pytest.raises(ValueError):
            sanitize_inst_id(r"ETH\USDT")

    def test_allows_single_dot_inside_symbol(self) -> None:
        """``ETH.USDT-SWAP`` must be allowed — `.` is a safe character."""
        assert sanitize_inst_id("ETH.USDT-SWAP") == "ETH.USDT-SWAP"

    def test_rejects_asterisk(self) -> None:
        """``ETH*USDT`` must be rejected — `*` is a glob/wildcard risk."""
        with pytest.raises(ValueError, match=r"characters outside"):
            sanitize_inst_id("ETH*USDT")

    def test_rejects_dot_segment(self) -> None:
        """``.`` must be rejected as a path‑traversal risk."""
        with pytest.raises(ValueError):
            sanitize_inst_id(".")

    def test_rejects_double_dot_segment(self) -> None:
        """``..`` must be rejected as a path‑traversal risk."""
        with pytest.raises(ValueError):
            sanitize_inst_id("..")

    def test_rejects_path_traversal_prefix(self) -> None:
        """``../ETH-USDT-SWAP`` must be rejected."""
        with pytest.raises(ValueError):
            sanitize_inst_id("../ETH-USDT-SWAP")

    def test_rejects_embedded_double_dots(self) -> None:
        """``ETH..USDT`` contains ``..`` and must be rejected."""
        with pytest.raises(ValueError):
            sanitize_inst_id("ETH..USDT")

    def test_rejects_unsafe_characters(self) -> None:
        """Symbols with ``!``, ``@``, ``#``, spaces, etc. must be rejected."""
        for bad in ("ETH USDT", "ETH!USDT", "ETH@USDT", "ETH#USDT", "ÉTH-USDT"):
            with pytest.raises(ValueError, match=r"characters outside"):
                sanitize_inst_id(bad)


# ============================================================================
# RuntimePaths – path generation
# ============================================================================

class TestRuntimePathsForETH:
    """Verify path layout for the current production symbol."""

    @pytest.fixture
    def paths(self) -> RuntimePaths:
        return RuntimePaths(runtime_dir=Path("runtime"), inst_id="ETH-USDT-SWAP")

    def test_symbol_slug(self, paths: RuntimePaths) -> None:
        assert paths.symbol_slug == "ETH-USDT-SWAP"

    def test_state_file(self, paths: RuntimePaths) -> None:
        assert paths.state_file == Path(
            "runtime/state/live_state_ETH-USDT-SWAP.json"
        )

    def test_journal_file(self, paths: RuntimePaths) -> None:
        assert paths.journal_file == Path(
            "runtime/journal/live_trades_ETH-USDT-SWAP.jsonl"
        )

    def test_trade_summary_file(self, paths: RuntimePaths) -> None:
        assert paths.trade_summary_file == Path(
            "runtime/journal/live_trade_summary_ETH-USDT-SWAP.jsonl"
        )

    def test_heartbeat_file(self, paths: RuntimePaths) -> None:
        assert paths.heartbeat_file == Path(
            "runtime/heartbeats/ETH-USDT-SWAP.heartbeat.json"
        )

    def test_events_file(self, paths: RuntimePaths) -> None:
        assert paths.events_file == Path(
            "runtime/events/ETH-USDT-SWAP.events.jsonl"
        )

    def test_log_file(self, paths: RuntimePaths) -> None:
        assert paths.log_file == Path("runtime/logs/ETH-USDT-SWAP.log")

    def test_daily_reports_dir(self, paths: RuntimePaths) -> None:
        assert paths.daily_reports_dir == Path(
            "runtime/reports/ETH-USDT-SWAP/daily"
        )

    def test_weekly_reports_dir(self, paths: RuntimePaths) -> None:
        assert paths.weekly_reports_dir == Path(
            "runtime/reports/ETH-USDT-SWAP/weekly"
        )

    def test_summary_reports_dir(self, paths: RuntimePaths) -> None:
        assert paths.summary_reports_dir == Path(
            "runtime/reports/ETH-USDT-SWAP/summary"
        )

    def test_latest_daily_report_file(self, paths: RuntimePaths) -> None:
        assert paths.latest_daily_report_file == Path(
            "runtime/reports/ETH-USDT-SWAP/daily/latest.html"
        )

    def test_latest_weekly_report_file(self, paths: RuntimePaths) -> None:
        assert paths.latest_weekly_report_file == Path(
            "runtime/reports/ETH-USDT-SWAP/weekly/latest.html"
        )

    def test_latest_summary_report_file(self, paths: RuntimePaths) -> None:
        assert paths.latest_summary_report_file == Path(
            "runtime/reports/ETH-USDT-SWAP/summary/latest.html"
        )

    def test_report_index_file(self, paths: RuntimePaths) -> None:
        assert paths.report_index_file == Path(
            "runtime/reports/ETH-USDT-SWAP/index.json"
        )


class TestRuntimePathsDoesNotCreateDirectories:
    """RuntimePaths is a pure path-builder — no filesystem side effects."""

    def test_no_directory_created(self, tmp_path: Path) -> None:
        runtime_dir = tmp_path / "runtime"
        paths = RuntimePaths(runtime_dir=runtime_dir, inst_id="ETH-USDT-SWAP")

        # Access every path property to ensure none of them trigger IO.
        _ = paths.state_dir
        _ = paths.journal_dir
        _ = paths.reports_dir
        _ = paths.heartbeats_dir
        _ = paths.events_dir
        _ = paths.logs_dir
        _ = paths.state_file
        _ = paths.journal_file
        _ = paths.heartbeat_file
        _ = paths.events_file
        _ = paths.log_file
        _ = paths.daily_reports_dir
        _ = paths.weekly_reports_dir
        _ = paths.summary_reports_dir
        _ = paths.latest_daily_report_file
        _ = paths.latest_weekly_report_file
        _ = paths.latest_summary_report_file
        _ = paths.report_index_file

        assert not runtime_dir.exists(), (
            "RuntimePaths must not create directories"
        )


class TestRuntimePathsAcceptsOtherSafeSymbols:
    """The path builder is a generic infrastructure utility — it must accept
    any safe symbol (e.g. BTC, SOL) for path generation only.  This does NOT
    mean those symbols are enabled for live trading."""

    def test_btc_usdt_swap(self) -> None:
        paths = RuntimePaths(runtime_dir=Path("runtime"), inst_id="BTC-USDT-SWAP")
        assert paths.symbol_slug == "BTC-USDT-SWAP"
        assert paths.state_file == Path(
            "runtime/state/live_state_BTC-USDT-SWAP.json"
        )
        assert paths.journal_file == Path(
            "runtime/journal/live_trades_BTC-USDT-SWAP.jsonl"
        )
        assert paths.trade_summary_file == Path(
            "runtime/journal/live_trade_summary_BTC-USDT-SWAP.jsonl"
        )
        assert paths.heartbeat_file == Path(
            "runtime/heartbeats/BTC-USDT-SWAP.heartbeat.json"
        )
        assert paths.report_index_file == Path(
            "runtime/reports/BTC-USDT-SWAP/index.json"
        )

    def test_sol_usdt_swap(self) -> None:
        paths = RuntimePaths(runtime_dir=Path("runtime"), inst_id="SOL-USDT-SWAP")
        assert paths.symbol_slug == "SOL-USDT-SWAP"
        assert paths.log_file == Path("runtime/logs/SOL-USDT-SWAP.log")
        assert paths.events_file == Path(
            "runtime/events/SOL-USDT-SWAP.events.jsonl"
        )


class TestBuildRuntimePathsHelper:
    """``build_runtime_paths`` is a convenience that wraps the dataclass."""

    def test_string_runtime_dir_is_coerced(self) -> None:
        paths = build_runtime_paths("runtime", "ETH-USDT-SWAP")
        assert paths.runtime_dir == Path("runtime")
        assert paths.state_file == Path(
            "runtime/state/live_state_ETH-USDT-SWAP.json"
        )

    def test_path_runtime_dir_is_preserved(self) -> None:
        paths = build_runtime_paths(Path("custom"), "ETH-USDT-SWAP")
        assert paths.runtime_dir == Path("custom")


# ============================================================================
# Legacy path references (for future B06 migration)
# ============================================================================

class TestLegacyPaths:
    """Legacy paths reference the current single‑coin filenames so that B06
    can detect and migrate them without guessing."""

    def test_legacy_state_file_matches_current_naming(self) -> None:
        paths = RuntimePaths(runtime_dir=Path("runtime"), inst_id="ETH-USDT-SWAP")
        assert paths.legacy_state_file == Path("runtime/live_state.json")

    def test_legacy_journal_file_matches_current_naming(self) -> None:
        paths = RuntimePaths(runtime_dir=Path("runtime"), inst_id="ETH-USDT-SWAP")
        assert paths.legacy_journal_file == Path("runtime/live_trade_events.jsonl")


# ============================================================================
# Immutability
# ============================================================================

class TestFrozen:
    """RuntimePaths must be frozen — no mutable fields allowed."""

    def test_cannot_set_attribute(self) -> None:
        paths = RuntimePaths(runtime_dir=Path("runtime"), inst_id="ETH-USDT-SWAP")
        with pytest.raises(Exception):  # FrozenInstanceError is internal
            paths.state_file = Path("hacked")  # type: ignore[misc]


# ============================================================================
# B05 guard – RuntimePaths allowed in live entry, banned from tick path
# ============================================================================


def test_b05_runtime_paths_wired_in_live_entry() -> None:
    """B05 wires RuntimePaths into the live entry point.

    This test confirms the wiring is present — it replaces the old B02/B03
    guards that *prevented* RuntimePaths from appearing in the live script.
    """
    source = Path("scripts/run_boll_cvd_live.py").read_text(encoding="utf-8")
    assert "RuntimePaths(" in source, (
        "B05 must wire RuntimePaths into run_boll_cvd_live.py"
    )
    assert "from_runtime_paths(" in source, (
        "B05 must wire from_runtime_paths into run_boll_cvd_live.py"
    )


def test_b05_runtime_paths_not_in_tick_workers() -> None:
    """RuntimePaths must NOT appear in tick / execution / sync worker files.

    RuntimePaths is a startup-only concern — tick-path workers receive
    already-constructed journal and state_store objects.
    """
    worker_files = [
        "src/live/workers/strategy_tick_worker.py",
        "src/live/workers/execution_worker.py",
        "src/live/workers/execution_command_processor.py",
        "src/live/workers/account_position_sync_worker.py",
    ]
    for rel_path in worker_files:
        full_path = Path(rel_path)
        if not full_path.exists():
            continue
        source = full_path.read_text(encoding="utf-8")
        assert "RuntimePaths" not in source, (
            f"{rel_path} must not reference RuntimePaths"
        )
        assert "handoff_legacy_runtime_files" not in source, (
            f"{rel_path} must not reference handoff_legacy_runtime_files"
        )


# ============================================================================
# B04 guard – prevent accidental live‑entry / report‑builder wiring
# ============================================================================


def test_b04_does_not_touch_live_entry() -> None:
    """B04 only adds report path data structures.

    B05 has wired RuntimePaths into the live entry for state and journal,
    but SymbolReportPaths and report artifact paths are still NOT wired.
    This test fails if anyone accidentally adds report artifact paths to
    ``scripts/run_boll_cvd_live.py`` before the report-artifact B0X task.
    """
    source = Path("scripts/run_boll_cvd_live.py").read_text(encoding="utf-8")
    assert "SymbolReportPaths" not in source, (
        "B04 must not wire SymbolReportPaths into run_boll_cvd_live.py"
    )
    assert "build_symbol_report_paths" not in source, (
        "B04 must not wire build_symbol_report_paths into run_boll_cvd_live.py"
    )
    assert "latest_daily_report_file" not in source, (
        "B04 must not wire latest_daily_report_file into run_boll_cvd_live.py"
    )
    assert "report_index_file" not in source, (
        "B04 must not wire report_index_file into run_boll_cvd_live.py"
    )


def test_b04_does_not_make_daily_reporter_write_files() -> None:
    """B04 does not change DailyTradeReporter to write files.

    DailyTradeReporter continues to send HTML emails — it does NOT write
    report artifacts to disk.  This test fails if anyone accidentally adds
    file‑write calls to the reporter before the report-artifact B0X task
    is ready.
    """
    source = Path("src/reporting/daily_trade_reporter.py").read_text(encoding="utf-8")
    assert "SymbolReportPaths" not in source, (
        "B04 must not reference SymbolReportPaths in DailyTradeReporter"
    )
    assert "build_symbol_report_paths" not in source, (
        "B04 must not reference build_symbol_report_paths in DailyTradeReporter"
    )
    assert ".write_text(" not in source, (
        "B04 must not make DailyTradeReporter write files to disk"
    )
    assert ".open(" not in source, (
        "B04 must not make DailyTradeReporter open files"
    )
