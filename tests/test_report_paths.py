from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.live.runtime_paths import RuntimePaths
from src.reporting.report_paths import (
    SymbolReportPaths,
    build_symbol_report_paths,
)


# ============================================================================
# SymbolReportPaths — date‑scoped file helpers
# ============================================================================


class TestSymbolReportPathsDailyFile:
    """Verify daily report artifact path generation."""

    def test_daily_report_file(self) -> None:
        runtime_paths = RuntimePaths(
            runtime_dir=Path("runtime"), inst_id="ETH-USDT-SWAP"
        )
        report_paths = SymbolReportPaths(runtime_paths)

        assert report_paths.daily_report_file(date(2026, 6, 9)) == Path(
            "runtime/reports/ETH-USDT-SWAP/daily/2026-06-09.html"
        )


class TestSymbolReportPathsWeeklyFile:
    """Verify weekly report artifact path generation."""

    def test_weekly_report_file(self) -> None:
        runtime_paths = RuntimePaths(
            runtime_dir=Path("runtime"), inst_id="ETH-USDT-SWAP"
        )
        report_paths = SymbolReportPaths(runtime_paths)

        assert report_paths.weekly_report_file(date(2026, 6, 8)) == Path(
            "runtime/reports/ETH-USDT-SWAP/weekly/week_2026-06-08.html"
        )


class TestSymbolReportPathsSummaryFile:
    """Verify summary report artifact path generation."""

    def test_summary_report_file(self) -> None:
        runtime_paths = RuntimePaths(
            runtime_dir=Path("runtime"), inst_id="ETH-USDT-SWAP"
        )
        report_paths = SymbolReportPaths(runtime_paths)

        assert report_paths.summary_report_file(date(2026, 6, 9)) == Path(
            "runtime/reports/ETH-USDT-SWAP/summary/2026-06-09.html"
        )


class TestSymbolReportPathsLatestFiles:
    """Verify that latest‑file targets are delegated to the provider."""

    def test_latest_files(self) -> None:
        runtime_paths = RuntimePaths(
            runtime_dir=Path("runtime"), inst_id="ETH-USDT-SWAP"
        )
        report_paths = SymbolReportPaths(runtime_paths)

        assert report_paths.latest_daily_report_file == Path(
            "runtime/reports/ETH-USDT-SWAP/daily/latest.html"
        )
        assert report_paths.latest_weekly_report_file == Path(
            "runtime/reports/ETH-USDT-SWAP/weekly/latest.html"
        )
        assert report_paths.latest_summary_report_file == Path(
            "runtime/reports/ETH-USDT-SWAP/summary/latest.html"
        )
        assert report_paths.report_index_file == Path(
            "runtime/reports/ETH-USDT-SWAP/index.json"
        )


class TestSymbolReportPathsAcceptsBTCForPathOnly:
    """BTC is only a path builder test — no BTC TOML, no live trading."""

    def test_btc_daily_report_file(self) -> None:
        runtime_paths = RuntimePaths(
            runtime_dir=Path("runtime"), inst_id="BTC-USDT-SWAP"
        )
        report_paths = SymbolReportPaths(runtime_paths)

        assert report_paths.daily_report_file(date(2026, 6, 9)) == Path(
            "runtime/reports/BTC-USDT-SWAP/daily/2026-06-09.html"
        )


class TestSymbolReportPathsDoesNotCreateDirectories:
    """SymbolReportPaths is a pure path-builder — no filesystem side effects."""

    def test_no_directory_created(self, tmp_path: Path) -> None:
        runtime_dir = tmp_path / "runtime"
        runtime_paths = RuntimePaths(
            runtime_dir=runtime_dir, inst_id="ETH-USDT-SWAP"
        )
        report_paths = SymbolReportPaths(runtime_paths)

        # Access all properties and date methods
        _ = report_paths.daily_report_file(date(2026, 6, 9))
        _ = report_paths.weekly_report_file(date(2026, 6, 8))
        _ = report_paths.summary_report_file(date(2026, 6, 9))
        _ = report_paths.latest_daily_report_file
        _ = report_paths.latest_weekly_report_file
        _ = report_paths.latest_summary_report_file
        _ = report_paths.report_index_file

        assert not runtime_dir.exists(), (
            "SymbolReportPaths must not create directories"
        )


# ============================================================================
# build_symbol_report_paths convenience constructor
# ============================================================================


class TestBuildSymbolReportPaths:
    """``build_symbol_report_paths`` is a convenience wrapper."""

    def test_build_from_runtime_paths(self) -> None:
        runtime_paths = RuntimePaths(
            runtime_dir=Path("runtime"), inst_id="ETH-USDT-SWAP"
        )
        report_paths = build_symbol_report_paths(runtime_paths)

        assert isinstance(report_paths, SymbolReportPaths)
        assert report_paths.daily_report_file(date(2026, 6, 9)) == Path(
            "runtime/reports/ETH-USDT-SWAP/daily/2026-06-09.html"
        )
