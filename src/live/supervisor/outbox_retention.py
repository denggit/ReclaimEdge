from __future__ import annotations

import fcntl
import os
import time
from dataclasses import dataclass
from pathlib import Path

from src.live.outbox.atomic_json import write_json_atomic
from src.utils.log import get_logger

logger = get_logger(__name__)


def _lock_path_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


@dataclass(frozen=True)
class OutboxRetentionResult:
    rotated: bool
    reason: str
    outbox_path: str
    archive_path: str | None = None
    size_bytes: int = 0
    cursor_offset: int = 0
    archives_deleted: int = 0


class WorkerEventOutboxRetention:
    def __init__(
        self,
        *,
        outbox_path: Path,
        cursor_path: Path,
        archive_dir: Path | None = None,
        max_bytes: int = 10 * 1024 * 1024,
        keep_archives: int = 20,
    ) -> None:
        # -- validate max_bytes ------------------------------------------------
        if type(max_bytes) is not int:
            raise ValueError(
                f"max_bytes must be int, got {type(max_bytes).__name__}={max_bytes!r}"
            )
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")

        # -- validate keep_archives --------------------------------------------
        if type(keep_archives) is not int:
            raise ValueError(
                f"keep_archives must be int, got {type(keep_archives).__name__}={keep_archives!r}"
            )
        if keep_archives < 0:
            raise ValueError(f"keep_archives must be >= 0, got {keep_archives}")

        # -- validate/coerce paths ---------------------------------------------
        self._outbox_path = Path(outbox_path)
        self._cursor_path = Path(cursor_path)
        if archive_dir is None:
            self._archive_dir = self._outbox_path.parent / "archive"
        else:
            self._archive_dir = Path(archive_dir)
        self._max_bytes = max_bytes
        self._keep_archives = keep_archives

    def rotate_if_processed(
        self,
        *,
        cursor_offset: int,
        reached_eof: bool,
        now_ms: int | None = None,
    ) -> OutboxRetentionResult:
        # -- validate params ---------------------------------------------------
        if type(reached_eof) is not bool:
            raise ValueError(
                f"reached_eof must be bool, got {type(reached_eof).__name__}={reached_eof!r}"
            )
        if type(cursor_offset) is not int or cursor_offset < 0:
            raise ValueError(
                f"cursor_offset must be int >= 0, got {type(cursor_offset).__name__}={cursor_offset!r}"
            )
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        # -- guard: not EOF ----------------------------------------------------
        if not reached_eof:
            return OutboxRetentionResult(
                rotated=False,
                reason="not_eof",
                outbox_path=str(self._outbox_path),
                cursor_offset=cursor_offset,
            )

        # -- guard: outbox missing ---------------------------------------------
        if not self._outbox_path.exists():
            return OutboxRetentionResult(
                rotated=False,
                reason="outbox_missing",
                outbox_path=str(self._outbox_path),
                cursor_offset=cursor_offset,
            )

        # -- stat size ---------------------------------------------------------
        try:
            size_before = self._outbox_path.stat().st_size
        except OSError:
            return OutboxRetentionResult(
                rotated=False,
                reason="outbox_missing",
                outbox_path=str(self._outbox_path),
                cursor_offset=cursor_offset,
            )

        # -- guard: below threshold --------------------------------------------
        if size_before < self._max_bytes:
            return OutboxRetentionResult(
                rotated=False,
                reason="below_threshold",
                outbox_path=str(self._outbox_path),
                size_bytes=size_before,
                cursor_offset=cursor_offset,
            )

        # -- guard: cursor before EOF ------------------------------------------
        if cursor_offset < size_before:
            return OutboxRetentionResult(
                rotated=False,
                reason="cursor_before_eof",
                outbox_path=str(self._outbox_path),
                size_bytes=size_before,
                cursor_offset=cursor_offset,
            )

        # -- acquire lock (same lock as JsonlOutbox.append_event) --------------
        lock_path = _lock_path_for(self._outbox_path)
        try:
            with lock_path.open("a", encoding="utf-8") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                try:
                    return self._rotate_under_lock(
                        cursor_offset=cursor_offset,
                        now_ms=now_ms,
                    )
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            logger.exception("SUPERVISOR_OUTBOX_RETENTION_FAILED")
            return OutboxRetentionResult(
                rotated=False,
                reason=f"error:{type(exc).__name__}",
                outbox_path=str(self._outbox_path),
                cursor_offset=cursor_offset,
            )

    def _rotate_under_lock(
        self,
        *,
        cursor_offset: int,
        now_ms: int,
    ) -> OutboxRetentionResult:
        # -- re-stat under lock ------------------------------------------------
        if not self._outbox_path.exists():
            return OutboxRetentionResult(
                rotated=False,
                reason="outbox_missing_after_lock",
                outbox_path=str(self._outbox_path),
                cursor_offset=cursor_offset,
            )

        try:
            size_after_lock = self._outbox_path.stat().st_size
        except OSError:
            return OutboxRetentionResult(
                rotated=False,
                reason="outbox_missing_after_lock",
                outbox_path=str(self._outbox_path),
                cursor_offset=cursor_offset,
            )

        # -- re-check thresholds under lock ------------------------------------
        if size_after_lock < self._max_bytes:
            return OutboxRetentionResult(
                rotated=False,
                reason="below_threshold_after_lock",
                outbox_path=str(self._outbox_path),
                size_bytes=size_after_lock,
                cursor_offset=cursor_offset,
            )

        if cursor_offset < size_after_lock:
            return OutboxRetentionResult(
                rotated=False,
                reason="new_data_after_read",
                outbox_path=str(self._outbox_path),
                size_bytes=size_after_lock,
                cursor_offset=cursor_offset,
            )

        # -- archive -----------------------------------------------------------
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        stem = self._outbox_path.stem
        archive_name = f"{stem}.{now_ms}.{size_after_lock}.jsonl"
        archive_path = self._archive_dir / archive_name

        self._outbox_path.rename(archive_path)

        # -- reset cursor ------------------------------------------------------
        write_json_atomic(
            self._cursor_path,
            {
                "path": str(self._outbox_path),
                "offset": 0,
                "updated_ts_ms": now_ms,
                "rotated_from_size_bytes": size_after_lock,
                "archive_path": str(archive_path),
            },
        )

        # -- cleanup old archives ----------------------------------------------
        archives_deleted = self._cleanup_archives()
        if archives_deleted > 0:
            logger.info(
                "OUTBOX_RETENTION_ARCHIVE_CLEANUP | deleted=%d keep=%d",
                archives_deleted,
                self._keep_archives,
            )

        return OutboxRetentionResult(
            rotated=True,
            reason="rotated",
            outbox_path=str(self._outbox_path),
            archive_path=str(archive_path),
            size_bytes=size_after_lock,
            cursor_offset=0,
            archives_deleted=archives_deleted,
        )

    def _cleanup_archives(self) -> int:
        """Delete oldest archives, keeping only the most recent ``keep_archives``.

        Archives are matched by the outbox stem prefix and sorted by mtime
        (newest first, falling back to name ordering).

        Returns the number of files deleted.
        """
        if not self._archive_dir.exists():
            return 0

        stem = self._outbox_path.stem
        # Match files like: worker_events_ETH-USDT-SWAP.<ts>.<size>.jsonl
        archives: list[Path] = []
        try:
            for entry in self._archive_dir.iterdir():
                if not entry.is_file():
                    continue
                if entry.suffix != ".jsonl":
                    continue
                if not entry.name.startswith(stem + "."):
                    continue
                archives.append(entry)
        except OSError:
            return 0

        if len(archives) <= self._keep_archives:
            return 0

        # Sort newest first by mtime, falling back to name.
        def _sort_key(p: Path) -> tuple[float, str]:
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                mtime = 0.0
            return (-mtime, p.name)

        archives.sort(key=_sort_key)

        to_delete = archives[self._keep_archives :]
        deleted = 0
        for path in to_delete:
            try:
                path.unlink()
                deleted += 1
            except OSError:
                logger.warning(
                    "OUTBOX_RETENTION_ARCHIVE_DELETE_FAILED | path=%s",
                    path,
                )

        return deleted
