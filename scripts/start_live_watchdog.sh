#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCHDOG_SCRIPT="$ROOT_DIR/scripts/watchdog_live.py"
WATCHDOG_LOG="$ROOT_DIR/watchdog_live.out"
WATCHDOG_PID="$ROOT_DIR/watchdog_live.pid"
LIVE_PID="$ROOT_DIR/boll_cvd_live.pid"

cd "$ROOT_DIR"

start() {
  if [[ -f "$WATCHDOG_PID" ]]; then
    old_pid="$(cat "$WATCHDOG_PID" || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "watchdog already running, pid=$old_pid"
      exit 0
    fi
    rm -f "$WATCHDOG_PID"
  fi

  echo "starting watchdog..."
  nohup "${LIVE_PYTHON_BIN:-python}" -u "$WATCHDOG_SCRIPT" >> "$WATCHDOG_LOG" 2>&1 &
  watchdog_pid=$!
  echo "$watchdog_pid" > "$WATCHDOG_PID"
  echo "watchdog started, pid=$watchdog_pid"
  echo "watchdog log: $WATCHDOG_LOG"
  echo "live log: $ROOT_DIR/boll_cvd_live.out"
}

stop() {
  if [[ ! -f "$WATCHDOG_PID" ]]; then
    echo "watchdog pid file not found; nothing to stop"
    exit 0
  fi

  pid="$(cat "$WATCHDOG_PID" || true)"
  if [[ -z "${pid:-}" ]]; then
    rm -f "$WATCHDOG_PID"
    echo "empty watchdog pid file removed"
    exit 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping watchdog pid=$pid ..."
    kill "$pid"
    for _ in {1..20}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "watchdog did not stop, killing pid=$pid ..."
      kill -9 "$pid" || true
    fi
  else
    echo "watchdog pid=$pid is not running"
  fi

  rm -f "$WATCHDOG_PID"
  echo "watchdog stopped"
}

status() {
  if [[ -f "$WATCHDOG_PID" ]]; then
    pid="$(cat "$WATCHDOG_PID" || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "watchdog: running pid=$pid"
    else
      echo "watchdog: pid file exists but process not running"
    fi
  else
    echo "watchdog: not running"
  fi

  if [[ -f "$LIVE_PID" ]]; then
    live_pid="$(cat "$LIVE_PID" || true)"
    if [[ -n "${live_pid:-}" ]] && kill -0 "$live_pid" 2>/dev/null; then
      echo "live child: running pid=$live_pid"
    else
      echo "live child: pid file exists but process not running"
    fi
  else
    echo "live child: no pid file"
  fi
}

logs() {
  tail -f "$WATCHDOG_LOG" "$ROOT_DIR/boll_cvd_live.out"
}

case "${1:-start}" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    stop || true
    start
    ;;
  status)
    status
    ;;
  logs)
    logs
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}"
    exit 1
    ;;
esac
