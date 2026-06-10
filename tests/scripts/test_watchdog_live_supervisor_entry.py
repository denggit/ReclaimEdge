# -*- coding: utf-8 -*-
"""Source-guard tests for scripts/watchdog_live.py supervisor entry (D08).

These tests verify that the watchdog script points at the supervisor entry
by default, keeps legacy env-var fallbacks, and does not import trading modules.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "watchdog_live.py"


def _source() -> str:
    return SOURCE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. default target
# ---------------------------------------------------------------------------


def test_watchdog_defaults_to_reclaim_supervisor() -> None:
    source = _source()

    assert 'ROOT / "scripts" / "run_reclaim_supervisor.py"' in source
    assert 'ROOT / "scripts" / "run_boll_cvd_live.py"' not in source

    assert "DEFAULT_SUPERVISOR_SCRIPT" in source

    # After the DEFAULT_SUPERVISOR_SCRIPT definition, the old live script
    # must not appear as a default constant.
    _, after = source.split("DEFAULT_SUPERVISOR_SCRIPT", 1)
    assert '"run_boll_cvd_live.py"' not in after


# ---------------------------------------------------------------------------
# 2. legacy env fallback order
# ---------------------------------------------------------------------------


def test_watchdog_keeps_legacy_live_script_override_only_as_env_fallback() -> None:
    source = _source()

    assert "WATCHDOG_CHILD_SCRIPT" in source
    assert "WATCHDOG_SUPERVISOR_SCRIPT" in source
    assert "LIVE_SCRIPT" in source

    # LIVE_SCRIPT must appear AFTER the new env names (it is a fallback, not
    # the preferred override).
    idx_new = source.index("WATCHDOG_CHILD_SCRIPT")
    idx_legacy = source.index("LIVE_SCRIPT")
    assert idx_new < idx_legacy, "LIVE_SCRIPT should appear after WATCHDOG_CHILD_SCRIPT (fallback only)"


# ---------------------------------------------------------------------------
# 3. default log and pid file names
# ---------------------------------------------------------------------------


def test_watchdog_default_log_and_pid_are_supervisor_named() -> None:
    source = _source()

    assert "reclaim_supervisor.out" in source
    assert "reclaim_supervisor.pid" in source
    assert "boll_cvd_live.out" not in source
    assert "boll_cvd_live.pid" not in source


# ---------------------------------------------------------------------------
# 4. log messages
# ---------------------------------------------------------------------------


def test_watchdog_log_messages_use_supervisor_child() -> None:
    source = _source()

    assert "supervisor child" in source.lower()
    assert "Supervisor child" in source
    assert "Live child" not in source
    assert "live child" not in source


# ---------------------------------------------------------------------------
# 5. no trading-module imports
# ---------------------------------------------------------------------------

TRADING_MODULE_MARKERS = [
    "from src.live.supervisor",
    "import src.live.supervisor",
    "from src.execution",
    "import src.execution",
    "from src.strategies",
    "import src.strategies",
    "from src.risk",
    "import src.risk",
    "from src.position_management",
    "import src.position_management",
    "SymbolWorkerApp",
    "ReclaimSupervisor",
    "Trader",
    "BollCvd",
    "import okx",
    "from okx",
    "import requests",
    "import httpx",
    "import websocket",
]


def test_watchdog_does_not_import_trading_modules() -> None:
    source = _source()

    for marker in TRADING_MODULE_MARKERS:
        assert marker not in source, f"watchdog must not import trading module: {marker!r}"


# ---------------------------------------------------------------------------
# 6. remains outer process keeper only
# ---------------------------------------------------------------------------


def test_watchdog_remains_outer_process_keeper_only() -> None:
    source = _source()

    assert "subprocess.Popen" in source
    assert "time.sleep" in source
    assert "HeartbeatMonitor" not in source
    assert "fetch_position_snapshot" not in source
    assert "state_store" not in source


# ---------------------------------------------------------------------------
# 7. pid file env overrides
# ---------------------------------------------------------------------------


def test_watchdog_pid_file_can_be_overridden() -> None:
    source = _source()

    assert "WATCHDOG_CHILD_PID_FILE" in source
    assert "WATCHDOG_SUPERVISOR_PID_FILE" in source
