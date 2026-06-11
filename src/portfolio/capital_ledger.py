# -*- coding: utf-8 -*-
"""
G01: 账户资金账本 CapitalLedger

纯基础设施层 —— JSON 文件 + 文件锁。

职责:
  - 读账本 / 写账本 / 初始化账本
  - POSIX fcntl.flock 保护下的 read/modify/write
  - schema version 管理
  - SymbolCapitalState DTO

不负责:
  - 是否允许开仓 / 加仓
  - leader/follower 判断
  - position plan 生成
  - 订单下单 / OKX 请求 / 邮件发送 / 策略信号判断
"""

from __future__ import annotations

import fcntl
import functools
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from src.live.outbox.atomic_json import read_json_or_none, write_json_atomic

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CapitalLedgerError(RuntimeError):
    """CapitalLedger 模块基础异常。"""


class CapitalLedgerLockTimeout(CapitalLedgerError):
    """获取文件锁超时。"""


class CapitalLedgerSchemaError(CapitalLedgerError):
    """账本 JSON schema 不符合预期。"""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEDGER_VERSION = 1
DEFAULT_SYMBOLS: tuple[str, ...] = ("ETH-USDT-SWAP", "BTC-USDT-SWAP")
DEFAULT_LEDGER_RELATIVE_PATH: Path = Path("runtime/portfolio/capital_ledger.json")
DEFAULT_LOCK_RELATIVE_PATH: Path = Path("runtime/portfolio/capital_ledger.lock")

# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolCapitalState:
    """单个 symbol 的资金 / 仓位状态快照。

    All monetary / contract / multiplier values are stored as *string*
    representations so that the JSON on-disk never contains float.
    """

    state: str = "FLAT"
    side: str | None = None
    used_layers: int = 0
    position_plan_id: str | None = None
    planned_main_contracts: tuple[str, ...] = ()
    base_main_contracts: str = "0"
    plan_max_layers: int = 8
    permission_max_layers: int = 8
    add_gap_multiplier: str = "1.0"
    add_freeze_multiplier: str = "1.0"
    main_used_margin_usdt: str = "0"
    sidecar_enabled: bool = False
    sidecar_used_margin_usdt: str = "0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "side": self.side,
            "used_layers": self.used_layers,
            "position_plan_id": self.position_plan_id,
            "planned_main_contracts": list(self.planned_main_contracts),
            "base_main_contracts": self.base_main_contracts,
            "plan_max_layers": self.plan_max_layers,
            "permission_max_layers": self.permission_max_layers,
            "add_gap_multiplier": self.add_gap_multiplier,
            "add_freeze_multiplier": self.add_freeze_multiplier,
            "main_used_margin_usdt": self.main_used_margin_usdt,
            "sidecar_enabled": self.sidecar_enabled,
            "sidecar_used_margin_usdt": self.sidecar_used_margin_usdt,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SymbolCapitalState:
        return cls(
            state=_require_str(d, "state"),
            side=_require_optional_str(d, "side"),
            used_layers=_require_int(d, "used_layers"),
            position_plan_id=_require_optional_str(d, "position_plan_id"),
            planned_main_contracts=_require_str_tuple(d, "planned_main_contracts"),
            base_main_contracts=_require_str(d, "base_main_contracts"),
            plan_max_layers=_require_int(d, "plan_max_layers"),
            permission_max_layers=_require_int(d, "permission_max_layers"),
            add_gap_multiplier=_require_str(d, "add_gap_multiplier"),
            add_freeze_multiplier=_require_str(d, "add_freeze_multiplier"),
            main_used_margin_usdt=_require_str(d, "main_used_margin_usdt"),
            sidecar_enabled=_require_bool(d, "sidecar_enabled"),
            sidecar_used_margin_usdt=_require_str(d, "sidecar_used_margin_usdt"),
        )


def default_symbol_state(inst_id: str) -> SymbolCapitalState:
    """Return a FLAT default state for *inst_id*.

    ETH defaults sidecar_enabled=True, BTC defaults sidecar_enabled=False.
    """
    sidecar = inst_id == "ETH-USDT-SWAP"
    return SymbolCapitalState(sidecar_enabled=sidecar)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapitalLedgerSnapshot:
    """一次完整的账本快照。"""

    version: int
    updated_ms: int
    leader_symbol: str | None
    global_no_new_entry: bool
    symbols: Mapping[str, SymbolCapitalState]

    _KNOWN_FIELDS: tuple[str, ...] = (
        "version",
        "updated_ms",
        "leader_symbol",
        "global_no_new_entry",
        "symbols",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_ms": self.updated_ms,
            "leader_symbol": self.leader_symbol,
            "global_no_new_entry": self.global_no_new_entry,
            "symbols": {
                inst_id: state.to_dict()
                for inst_id, state in self.symbols.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CapitalLedgerSnapshot:
        _validate_version(d)
        if "symbols" not in d:
            raise CapitalLedgerSchemaError("ledger missing 'symbols' key")
        if not isinstance(d["symbols"], dict):
            raise CapitalLedgerSchemaError(
                f"'symbols' must be a dict, got {type(d['symbols']).__name__}"
            )

        raw_symbols: dict[str, Any] = d["symbols"]
        symbols: dict[str, SymbolCapitalState] = {}
        for inst_id, raw_state in raw_symbols.items():
            if not isinstance(raw_state, dict):
                raise CapitalLedgerSchemaError(
                    f"symbol state for '{inst_id}' must be a dict, "
                    f"got {type(raw_state).__name__}"
                )
            symbols[inst_id] = SymbolCapitalState.from_dict(raw_state)

        return cls(
            version=d["version"],
            updated_ms=_require_int(d, "updated_ms"),
            leader_symbol=_require_optional_str(d, "leader_symbol"),
            global_no_new_entry=_require_bool(d, "global_no_new_entry"),
            symbols=symbols,
        )


def default_snapshot(*, updated_ms: int | None = None) -> CapitalLedgerSnapshot:
    """Return the default (all-FLAT) snapshot for both ETH and BTC."""
    _updated_ms = updated_ms if updated_ms is not None else 0
    symbols: dict[str, SymbolCapitalState] = {}
    for inst_id in DEFAULT_SYMBOLS:
        symbols[inst_id] = default_symbol_state(inst_id)
    return CapitalLedgerSnapshot(
        version=LEDGER_VERSION,
        updated_ms=_updated_ms,
        leader_symbol=None,
        global_no_new_entry=False,
        symbols=symbols,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_str(d: Mapping[str, Any], key: str) -> str:
    val = d.get(key)
    if not isinstance(val, str):
        raise CapitalLedgerSchemaError(
            f"'{key}' must be a string, got {type(val).__name__}: {val!r}"
        )
    return val


def _require_optional_str(d: Mapping[str, Any], key: str) -> str | None:
    val = d.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise CapitalLedgerSchemaError(
            f"'{key}' must be a string or null, "
            f"got {type(val).__name__}: {val!r}"
        )
    return val


def _require_bool(d: Mapping[str, Any], key: str) -> bool:
    val = d.get(key)
    if not isinstance(val, bool):
        raise CapitalLedgerSchemaError(
            f"'{key}' must be a bool, got {type(val).__name__}: {val!r}"
        )
    return val


def _require_int(d: Mapping[str, Any], key: str) -> int:
    val = d.get(key)
    if isinstance(val, bool) or not isinstance(val, int):
        raise CapitalLedgerSchemaError(
            f"'{key}' must be an int, got {type(val).__name__}: {val!r}"
        )
    return val


def _require_str_tuple(d: Mapping[str, Any], key: str) -> tuple[str, ...]:
    val = d.get(key)
    if not isinstance(val, (list, tuple)):
        raise CapitalLedgerSchemaError(
            f"'{key}' must be a list or tuple, "
            f"got {type(val).__name__}: {val!r}"
        )
    for i, item in enumerate(val):
        if not isinstance(item, str):
            raise CapitalLedgerSchemaError(
                f"'{key}' item at index {i} must be a string, "
                f"got {type(item).__name__}: {item!r}"
            )
    return tuple(val)


def _validate_version(d: Mapping[str, Any]) -> None:
    if "version" not in d:
        raise CapitalLedgerSchemaError("ledger missing 'version' key")
    version = d["version"]
    if version != LEDGER_VERSION:
        raise CapitalLedgerSchemaError(
            f"unsupported ledger version: {version!r} (expected {LEDGER_VERSION})"
        )


def _now_ms() -> int:
    """Return current monotonic-ish wall-clock in milliseconds.

    Uses time.time() for cross-platform simplicity.  We only need ordering /
    freshness, not an absolute timestamp.
    """
    return int(math.floor(time.time() * 1000))


# ---------------------------------------------------------------------------
# File lock (POSIX fcntl.flock)
# ---------------------------------------------------------------------------


class _FileLock:
    """Exclusive file lock using fcntl.flock with non-blocking poll + timeout."""

    def __init__(
        self,
        path: Path,
        timeout_seconds: float = 5.0,
        poll_seconds: float = 0.05,
    ) -> None:
        self._path = path
        self._timeout = timeout_seconds
        self._poll = poll_seconds
        self._fd: int | None = None

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> _FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        self.release()

    # -- acquire / release ---------------------------------------------------

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR, 0o644)
        self._fd = fd

        deadline = time.monotonic() + self._timeout
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return  # acquired
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        self._close_fd_without_unlock()
                        raise CapitalLedgerLockTimeout(
                            f"could not acquire lock on {self._path} "
                            f"within {self._timeout:.1f}s"
                        ) from None
                    time.sleep(self._poll)
        except Exception:
            if self._fd is not None:
                self._close_fd_without_unlock()
            raise

    def _close_fd_without_unlock(self) -> None:
        """Close the underlying fd without releasing the flock.

        Used when we haven't acquired the lock yet (timeout / error path)
        so there is no lock to release.
        """
        fd = self._fd
        self._fd = None
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        finally:
            try:
                os.close(fd)
            except Exception:
                pass
        self._fd = None


# ---------------------------------------------------------------------------
# CapitalLedger
# ---------------------------------------------------------------------------


class CapitalLedger:
    """账户资金账本 —— JSON 文件 + 文件锁。

    Usage::

        ledger = CapitalLedger()

        # One-shot init (idempotent)
        ledger.initialize_if_missing()

        # Read
        snap = ledger.read_locked()

        # Update under lock
        def toggle_no_new_entry(snap: CapitalLedgerSnapshot) -> CapitalLedgerSnapshot:
            return CapitalLedgerSnapshot(
                version=snap.version,
                updated_ms=_now_ms(),
                leader_symbol=snap.leader_symbol,
                global_no_new_entry=not snap.global_no_new_entry,
                symbols=snap.symbols,
            )

        new_snap = ledger.update_locked(toggle_no_new_entry)
    """

    def __init__(
        self,
        ledger_path: str | Path = DEFAULT_LEDGER_RELATIVE_PATH,
        lock_path: str | Path = DEFAULT_LOCK_RELATIVE_PATH,
        *,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        self._ledger_path = Path(ledger_path)
        self._lock_path = Path(lock_path)
        self._lock_timeout = lock_timeout_seconds

    # -- public read / write primitives --------------------------------------

    def read_unlocked(self) -> CapitalLedgerSnapshot:
        """Read ledger without acquiring a lock.

        Returns ``default_snapshot(updated_ms=0)`` if the file does not exist
        (without creating the file).
        """
        raw = read_json_or_none(self._ledger_path)
        if raw is None:
            return default_snapshot(updated_ms=0)
        return CapitalLedgerSnapshot.from_dict(raw)

    def write_unlocked(self, snapshot: CapitalLedgerSnapshot) -> None:
        """Write *snapshot* to the ledger file atomically, no lock."""
        write_json_atomic(self._ledger_path, snapshot.to_dict())

    def initialize_if_missing(
        self,
        *,
        updated_ms: int | None = None,
    ) -> CapitalLedgerSnapshot:
        """Acquire lock, create default snapshot if file missing, return current.

        Idempotent: if the file already exists it is read and returned without
        modification.
        """
        with _FileLock(self._lock_path, timeout_seconds=self._lock_timeout):
            existing = read_json_or_none(self._ledger_path)
            if existing is not None:
                return CapitalLedgerSnapshot.from_dict(existing)
            snap = default_snapshot(updated_ms=updated_ms or 0)
            write_json_atomic(self._ledger_path, snap.to_dict())
            return snap

    def read_locked(self) -> CapitalLedgerSnapshot:
        """Acquire lock, read ledger, return snapshot."""
        with _FileLock(self._lock_path, timeout_seconds=self._lock_timeout):
            raw = read_json_or_none(self._ledger_path)
            if raw is None:
                return default_snapshot(updated_ms=0)
            return CapitalLedgerSnapshot.from_dict(raw)

    def update_locked(
        self,
        mutator: Callable[[CapitalLedgerSnapshot], CapitalLedgerSnapshot],
    ) -> CapitalLedgerSnapshot:
        """Acquire lock, read → mutate → validate → write, return new snapshot.

        *mutator* receives the current snapshot (or default if file missing)
        and **must** return a ``CapitalLedgerSnapshot`` instance —— returning a
        plain ``dict`` raises ``CapitalLedgerSchemaError``.
        """
        with _FileLock(self._lock_path, timeout_seconds=self._lock_timeout):
            raw = read_json_or_none(self._ledger_path)
            current = (
                CapitalLedgerSnapshot.from_dict(raw)
                if raw is not None
                else default_snapshot(updated_ms=0)
            )
            new_snap = mutator(current)
            if not isinstance(new_snap, CapitalLedgerSnapshot):
                raise CapitalLedgerSchemaError(
                    f"mutator must return CapitalLedgerSnapshot, "
                    f"got {type(new_snap).__name__}"
                )
            write_json_atomic(self._ledger_path, new_snap.to_dict())
            return new_snap

    def get_symbol(self, inst_id: str) -> SymbolCapitalState:
        """Convenience: read-locked single symbol state lookup."""
        snap = self.read_locked()
        state = snap.symbols.get(inst_id)
        if state is None:
            raise CapitalLedgerSchemaError(
                f"symbol '{inst_id}' not found in ledger"
            )
        return state

    def with_symbol_state(
        self,
        inst_id: str,
        updater: Callable[[SymbolCapitalState], SymbolCapitalState],
        *,
        updated_ms: int | None = None,
    ) -> CapitalLedgerSnapshot:
        """Convenience: read-locked, update a single symbol's state, write back.

        *updater* receives the current ``SymbolCapitalState`` for *inst_id*
        and must return the new state.

        Returns the full new ``CapitalLedgerSnapshot``.
        """
        with _FileLock(self._lock_path, timeout_seconds=self._lock_timeout):
            raw = read_json_or_none(self._ledger_path)
            current = (
                CapitalLedgerSnapshot.from_dict(raw)
                if raw is not None
                else default_snapshot(updated_ms=0)
            )
            if inst_id not in current.symbols:
                raise CapitalLedgerSchemaError(
                    f"symbol '{inst_id}' not found in ledger"
                )
            old_state = current.symbols[inst_id]
            new_state = updater(old_state)
            if not isinstance(new_state, SymbolCapitalState):
                raise CapitalLedgerSchemaError(
                    f"updater must return SymbolCapitalState, "
                    f"got {type(new_state).__name__}"
                )
            new_symbols = dict(current.symbols)
            new_symbols[inst_id] = new_state
            new_snap = CapitalLedgerSnapshot(
                version=current.version,
                updated_ms=updated_ms
                if updated_ms is not None
                else _now_ms(),
                leader_symbol=current.leader_symbol,
                global_no_new_entry=current.global_no_new_entry,
                symbols=new_symbols,
            )
            write_json_atomic(self._ledger_path, new_snap.to_dict())
            return new_snap
