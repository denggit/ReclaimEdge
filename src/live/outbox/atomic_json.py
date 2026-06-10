from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def write_json_atomic(
    path: str | Path,
    data: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> None:
    """
    Atomically write JSON to path.

    Implementation:
    1. create parent directory
    2. write to temp file in same directory
    3. flush + fsync temp file
    4. os.replace(temp, target)
    5. best-effort fsync parent directory

    No network.
    No env reads.
    No trading imports.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_name = fh.name
            json.dump(data, fh, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_name, target)
        tmp_name = None
        _fsync_parent_dir(target.parent)
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass


def read_json_or_none(path: str | Path) -> Any | None:
    target = Path(path)
    if not target.exists():
        return None
    with target.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _fsync_parent_dir(path: Path) -> None:
    """
    Best-effort directory fsync.

    On some platforms this may fail. It must never make JSON write fail
    after the target has already been replaced.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except Exception:
        return
    try:
        os.fsync(fd)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass
