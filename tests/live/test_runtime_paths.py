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
        assert paths.heartbeat_file == Path(
            "runtime/heartbeats/BTC-USDT-SWAP.heartbeat.json"
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
