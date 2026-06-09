from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol


# ---------------------------------------------------------------------------
# ReportPathProvider Protocol
# ---------------------------------------------------------------------------

class ReportPathProvider(Protocol):
    """Protocol for any object that provides symbol‑scoped report directory
    and artifact file paths.

    This protocol is intentionally defined in the reporting layer so that
    ``RuntimePaths`` (in ``src.live``) can satisfy it without the reporting
    layer importing from ``live``.  Any class that exposes these six
    properties is a valid provider.
    """

    @property
    def daily_reports_dir(self) -> Path: ...

    @property
    def weekly_reports_dir(self) -> Path: ...

    @property
    def summary_reports_dir(self) -> Path: ...

    @property
    def latest_daily_report_file(self) -> Path: ...

    @property
    def latest_weekly_report_file(self) -> Path: ...

    @property
    def latest_summary_report_file(self) -> Path: ...

    @property
    def report_index_file(self) -> Path: ...


# ---------------------------------------------------------------------------
# SymbolReportPaths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolReportPaths:
    """Pure path-builder for date‑scoped report artifact files.

    This dataclass generates **deterministic paths only**.  It delegates
    directory roots and latest‑file targets to a :class:`ReportPathProvider`
    and adds date‑based filename helpers on top.

    It never:
    * reads ``.env``
    * creates directories
    * opens or writes files
    * talks to the network
    * imports from ``src.live``

    Parameters
    ----------
    provider : ReportPathProvider
        Any object that satisfies the :class:`ReportPathProvider` protocol
        (e.g. ``RuntimePaths``).
    """

    provider: ReportPathProvider

    # -- date‑scoped artifact helpers -------------------------------------

    def daily_report_file(self, report_date: date) -> Path:
        """Return the path for a daily report artifact on *report_date*.

        Example: ``daily/2026-06-09.html``
        """
        return self.provider.daily_reports_dir / f"{report_date.isoformat()}.html"

    def weekly_report_file(self, week_start: date) -> Path:
        """Return the path for a weekly report artifact starting on *week_start*.

        Example: ``weekly/week_2026-06-08.html``
        """
        return self.provider.weekly_reports_dir / f"week_{week_start.isoformat()}.html"

    def summary_report_file(self, report_date: date) -> Path:
        """Return the path for a summary report artifact on *report_date*.

        Example: ``summary/2026-06-09.html``
        """
        return self.provider.summary_reports_dir / f"{report_date.isoformat()}.html"

    # -- latest‑file targets (delegated) ----------------------------------

    @property
    def latest_daily_report_file(self) -> Path:
        """Stable target path for the latest daily report artifact."""
        return self.provider.latest_daily_report_file

    @property
    def latest_weekly_report_file(self) -> Path:
        """Stable target path for the latest weekly report artifact."""
        return self.provider.latest_weekly_report_file

    @property
    def latest_summary_report_file(self) -> Path:
        """Stable target path for the latest summary report artifact."""
        return self.provider.latest_summary_report_file

    @property
    def report_index_file(self) -> Path:
        """Stable target path for the symbol report index."""
        return self.provider.report_index_file


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def build_symbol_report_paths(provider: ReportPathProvider) -> SymbolReportPaths:
    """Construct :class:`SymbolReportPaths` from any report path provider.

    Parameters
    ----------
    provider : ReportPathProvider
        Any object satisfying the :class:`ReportPathProvider` protocol
        (e.g. a :class:`~src.live.runtime_paths.RuntimePaths` instance).

    Returns
    -------
    SymbolReportPaths
    """
    return SymbolReportPaths(provider=provider)
