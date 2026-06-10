from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

_SAFE_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def sanitize_inst_id(inst_id: str) -> str:
    """Return a filesystem-safe slug for an OKX instrument ID.

    Rules
    -----
    * Input must be a non-empty ``str`` whose stripped form is non-empty.
    * Slashes (``/``, ``\\``), dots used as path segments (``"."``, ``".."``),
      and any form of ``..`` path-traversal are **rejected unconditionally**.
    * The slug may only contain ``A-Z a-z 0-9 _ - .``.

    Returns
    -------
    str
        The input unchanged when it passes all checks (e.g. ``"ETH-USDT-SWAP"``).

    Raises
    ------
    ValueError
        If *inst_id* is empty / whitespace-only, contains path separators or
        ``..`` segments, or includes characters outside the safe set.
    """
    if not isinstance(inst_id, str):
        raise ValueError(f"inst_id must be str, got {type(inst_id).__name__}")
    stripped = inst_id.strip()
    if not stripped:
        raise ValueError(f"inst_id must not be empty or whitespace-only: {inst_id!r}")

    # Block path separators
    if "/" in inst_id or "\\" in inst_id:
        raise ValueError(
            f"inst_id contains path separators (forbidden): {inst_id!r}"
        )

    # Block dot-segments and any ".." path‑traversal
    if inst_id == "." or inst_id == ".." or ".." in inst_id:
        raise ValueError(
            f"inst_id contains dot‑segment or path‑traversal (forbidden): {inst_id!r}"
        )

    if not _SAFE_SLUG_RE.match(inst_id):
        raise ValueError(
            f"inst_id contains characters outside the safe set [A-Za-z0-9_.-]: {inst_id!r}"
        )

    return inst_id


# ---------------------------------------------------------------------------
# RuntimePaths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimePaths:
    """Pure path-builder for symbol‑scoped runtime files.

    This dataclass generates **deterministic paths only**.  It never:
    * reads ``.env``
    * creates directories
    * opens or writes files
    * talks to the network
    * depends on any live runtime state

    Symbol‑scoped files provided:

    * ``journal_file`` — main trade event journal
    * ``trade_summary_file`` — trade summary journal

    Report artifact paths (B04 — deterministic only, no IO):

    * ``latest_daily_report_file`` — stable target for latest daily report
    * ``latest_weekly_report_file`` — stable target for latest weekly report
    * ``latest_summary_report_file`` — stable target for latest summary report
    * ``report_index_file`` — symbol report index for parent/supervisor discovery

    Parameters
    ----------
    runtime_dir : Path
        Root directory under which all runtime artefacts live (e.g.
        ``Path("runtime")`` or ``Path("data/trade_journal")``).
    inst_id : str
        OKX instrument ID such as ``"ETH-USDT-SWAP"``.  It is sanitised via
        :func:`sanitize_inst_id` during ``__post_init__``.
    """

    runtime_dir: Path
    inst_id: str

    def __post_init__(self) -> None:
        # Coerce runtime_dir to Path so callers can pass strings safely.
        object.__setattr__(self, "runtime_dir", Path(self.runtime_dir))

        # Sanitise and freeze the instrument id as the symbol slug.
        slug = sanitize_inst_id(self.inst_id)
        object.__setattr__(self, "inst_id", slug)

    # -- read-only helpers -------------------------------------------------

    @property
    def symbol_slug(self) -> str:
        """Return the sanitised instrument id used in filenames."""
        return self.inst_id

    # Directories ----------------------------------------------------------

    @property
    def state_dir(self) -> Path:
        return self.runtime_dir / "state"

    @property
    def journal_dir(self) -> Path:
        return self.runtime_dir / "journal"

    @property
    def reports_dir(self) -> Path:
        return self.runtime_dir / "reports" / self.symbol_slug

    @property
    def heartbeats_dir(self) -> Path:
        return self.runtime_dir / "heartbeats"

    @property
    def events_dir(self) -> Path:
        return self.runtime_dir / "events"

    @property
    def logs_dir(self) -> Path:
        return self.runtime_dir / "logs"

    @property
    def risk_dir(self) -> Path:
        """Account-level runtime risk directory (NOT symbol-scoped)."""
        return self.runtime_dir / "risk"

    @property
    def rolling_loss_guard_state_file(self) -> Path:
        """Account-level rolling loss guard state file (NOT symbol-scoped)."""
        return self.risk_dir / "rolling_loss_guard_state.json"

    # Symbol‑scoped files --------------------------------------------------

    @property
    def state_file(self) -> Path:
        return self.state_dir / f"live_state_{self.symbol_slug}.json"

    @property
    def journal_file(self) -> Path:
        return self.journal_dir / f"live_trades_{self.symbol_slug}.jsonl"

    @property
    def trade_summary_file(self) -> Path:
        return self.journal_dir / f"live_trade_summary_{self.symbol_slug}.jsonl"

    @property
    def heartbeat_file(self) -> Path:
        return self.heartbeats_dir / f"{self.symbol_slug}.heartbeat.json"

    @property
    def events_file(self) -> Path:
        return self.events_dir / f"{self.symbol_slug}.events.jsonl"

    @property
    def worker_event_outbox_file(self) -> Path:
        """Worker event outbox JSONL path (E02).

        Example: ``runtime/events/worker_events_ETH-USDT-SWAP.jsonl``.
        """
        return self.events_dir / f"worker_events_{self.symbol_slug}.jsonl"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / f"{self.symbol_slug}.log"

    # Report sub‑directories -----------------------------------------------

    @property
    def daily_reports_dir(self) -> Path:
        return self.reports_dir / "daily"

    @property
    def weekly_reports_dir(self) -> Path:
        return self.reports_dir / "weekly"

    @property
    def summary_reports_dir(self) -> Path:
        return self.reports_dir / "summary"

    # Report artifact file paths ------------------------------------------

    @property
    def latest_daily_report_file(self) -> Path:
        """Stable target path for the latest daily report artifact.

        This is a **deterministic path only** — B04 does not write this file.
        Future B05/B06 may use it as the output destination for daily HTML
        reports.
        """
        return self.daily_reports_dir / "latest.html"

    @property
    def latest_weekly_report_file(self) -> Path:
        """Stable target path for the latest weekly report artifact.

        This is a **deterministic path only** — B04 does not write this file.
        """
        return self.weekly_reports_dir / "latest.html"

    @property
    def latest_summary_report_file(self) -> Path:
        """Stable target path for the latest overall summary report artifact.

        This is a **deterministic path only** — B04 does not write this file.
        """
        return self.summary_reports_dir / "latest.html"

    @property
    def report_index_file(self) -> Path:
        """Stable target path for the symbol report index.

        Future parent/supervisor processes can read ``index.json`` to
        discover available report artifacts for this symbol.  B04 does
        **not** write this file.
        """
        return self.reports_dir / "index.json"

    # Legacy (single‑coin) paths for future B06 migration ------------------
    # These match the *current* filenames used in the reporting layer so that
    # B06 can detect / migrate them without guessing.

    @property
    def legacy_state_file(self) -> Path:
        """Path to the pre‑symbol‑scope state file (``live_state.json``)."""
        return self.runtime_dir / "live_state.json"

    @property
    def legacy_journal_file(self) -> Path:
        """Path to the pre‑symbol‑scope journal file (``live_trade_events.jsonl``)."""
        return self.runtime_dir / "live_trade_events.jsonl"


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def build_runtime_paths(runtime_dir: str | Path, inst_id: str) -> RuntimePaths:
    """Construct :class:`RuntimePaths` without reading any environment.

    Parameters
    ----------
    runtime_dir : str | Path
        Root runtime directory.
    inst_id : str
        OKX instrument id (e.g. ``"ETH-USDT-SWAP"``).

    Returns
    -------
    RuntimePaths
    """
    return RuntimePaths(runtime_dir=Path(runtime_dir), inst_id=inst_id)
