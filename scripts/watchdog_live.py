#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple watchdog for ReclaimEdge supervisor.

This process starts scripts/run_reclaim_supervisor.py as a child process by default.
If the supervisor process exits unexpectedly, the watchdog restarts it after a short delay.

The supervisor owns SymbolWorker child lifecycle and heartbeat checks.
The watchdog only keeps the supervisor process alive.

Stop behavior:
- kill the watchdog process with SIGTERM or Ctrl+C
- watchdog will terminate the supervisor child process before exiting

This is intentionally simpler than systemd and is suitable for early live tests.
"""
from __future__ import annotations

import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPERVISOR_SCRIPT = ROOT / "scripts" / "run_reclaim_supervisor.py"
DEFAULT_SUPERVISOR_LOG = ROOT / "reclaim_supervisor.out"
DEFAULT_CHILD_PID_FILE = ROOT / "reclaim_supervisor.pid"

_running = True
_child: Optional[subprocess.Popen] = None


def ts() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{ts()} | WATCHDOG | {message}", flush=True)


def child_pid_file() -> Path:
    value = (
        os.getenv("WATCHDOG_CHILD_PID_FILE")
        or os.getenv("WATCHDOG_SUPERVISOR_PID_FILE")
        or str(DEFAULT_CHILD_PID_FILE)
    )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def terminate_child(reason: str) -> None:
    global _child
    child = _child
    if child is None or child.poll() is not None:
        return

    log(f"Stopping supervisor child pid={child.pid} reason={reason}")
    try:
        child.terminate()
        child.wait(timeout=20)
        log(f"Supervisor child stopped pid={child.pid} returncode={child.returncode}")
    except subprocess.TimeoutExpired:
        log(f"Supervisor child did not stop in time, killing pid={child.pid}")
        child.kill()
        child.wait(timeout=10)
    finally:
        _child = None
        try:
            child_pid_file().unlink(missing_ok=True)
        except Exception:
            pass


def handle_stop_signal(signum: int, _frame) -> None:  # type: ignore[no-untyped-def]
    global _running
    _running = False
    log(f"Received signal={signum}. Watchdog is shutting down.")
    terminate_child(f"watchdog_signal_{signum}")


def build_command() -> list[str]:
    python_bin = os.getenv("WATCHDOG_PYTHON_BIN") or os.getenv("LIVE_PYTHON_BIN") or sys.executable
    child_script_text = (
        os.getenv("WATCHDOG_CHILD_SCRIPT")
        or os.getenv("WATCHDOG_SUPERVISOR_SCRIPT")
        or os.getenv("LIVE_SCRIPT")
        or str(DEFAULT_SUPERVISOR_SCRIPT)
    )
    child_script = Path(child_script_text).expanduser()
    if not child_script.is_absolute():
        child_script = ROOT / child_script
    return [python_bin, "-u", str(child_script)]


def start_child() -> subprocess.Popen:
    command = build_command()
    live_log_path = Path(
        os.getenv("WATCHDOG_LOG_FILE")
        or os.getenv("WATCHDOG_SUPERVISOR_LOG_FILE")
        or os.getenv("LIVE_LOG_FILE")
        or str(DEFAULT_SUPERVISOR_LOG)
    ).expanduser()
    if not live_log_path.is_absolute():
        live_log_path = ROOT / live_log_path
    live_log_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Starting supervisor child: {' '.join(command)}")
    log(f"Supervisor child log file: {live_log_path}")
    log_file = open(live_log_path, "a", buffering=1, encoding="utf-8")
    child = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=os.environ.copy(),
        text=True,
    )
    child_pid_file().write_text(str(child.pid), encoding="utf-8")
    log(f"Supervisor child started pid={child.pid}")
    return child


def main() -> int:
    global _child
    signal.signal(signal.SIGTERM, handle_stop_signal)
    signal.signal(signal.SIGINT, handle_stop_signal)

    restart_seconds = float(os.getenv("WATCHDOG_RESTART_SECONDS", "5"))
    max_restarts = int(os.getenv("WATCHDOG_MAX_RESTARTS", "0"))  # 0 means unlimited
    restart_count = 0

    log("Watchdog started")
    log(f"Project root: {ROOT}")
    log(f"Restart delay: {restart_seconds}s, max_restarts={max_restarts or 'unlimited'}")

    while _running:
        _child = start_child()
        returncode = _child.wait()
        try:
            child_pid_file().unlink(missing_ok=True)
        except Exception:
            pass

        if not _running:
            break

        restart_count += 1
        log(f"Supervisor child exited unexpectedly returncode={returncode}. restart_count={restart_count}")
        if max_restarts > 0 and restart_count >= max_restarts:
            log("Max restart count reached. Watchdog exits.")
            return returncode if returncode != 0 else 1

        time.sleep(restart_seconds)

    log("Watchdog stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
