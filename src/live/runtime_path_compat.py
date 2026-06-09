from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Iterable

from src.live.runtime_paths import RuntimePaths
from src.reporting.live_state_store import DEFAULT_STATE_PATH
from src.reporting.trade_journal import DEFAULT_JOURNAL_PATH, DEFAULT_SUMMARY_PATH

LEGACY_RUNTIME_SYMBOL = "ETH-USDT-SWAP"
"""Only this symbol is allowed to receive legacy file handoff.

Future symbols (e.g. BTC-USDT-SWAP) must **not** inherit ETH legacy
state / journal / summary files, because those files contain
ETH-specific position history that would corrupt a fresh BTC session.
"""


@dataclass(frozen=True)
class LegacyRuntimeFileHandoff:
    """Result for a single file in a legacy-to-symbol handoff operation.

    Attributes
    ----------
    label : str
        Human-readable file kind (``"state"``, ``"journal"``, ``"summary"``).
    legacy_path : Path
        Absolute path to the legacy (single-coin) file.
    symbol_path : Path
        Absolute path to the new symbol-scoped file.
    action : str
        What happened: ``"copied"`` or ``"skipped"``.
    reason : str
        Why the action was taken (e.g. ``"target exists"``,
        ``"legacy missing"``, ``"copied legacy file to symbol-scoped path"``).
    """

    label: str
    legacy_path: Path
    symbol_path: Path
    action: str
    reason: str


@dataclass(frozen=True)
class LegacyRuntimeHandoffResult:
    """Aggregate result of a :func:`handoff_legacy_runtime_files` call.

    Attributes
    ----------
    inst_id : str
        The instrument ID that was passed in.
    items : tuple of LegacyRuntimeFileHandoff
        One entry per file kind (state, journal, summary).
    """

    inst_id: str
    items: tuple[LegacyRuntimeFileHandoff, ...]


def handoff_legacy_runtime_files(
    *,
    runtime_paths: RuntimePaths,
    inst_id: str,
) -> LegacyRuntimeHandoffResult:
    """Copy legacy single-coin runtime files to symbol-scoped paths.

    This is a **startup-only** helper.  It must never be called from a
    tick path, worker loop, or strategy callback.

    Rules (per file kind — state, journal, summary):

    * Only ``inst_id == "ETH-USDT-SWAP"`` is eligible for handoff.
      All other symbols receive ``action="skipped"`` with a reason
      explaining the restriction.
    * If the symbol-scoped target already exists → skipped (never overwrite).
    * If the legacy source does not exist → skipped.
    * If the legacy source exists but is not a regular file → skipped.
    * If legacy path equals symbol path → skipped (same file).
    * Otherwise the legacy file is copied via :func:`shutil.copy2` to the
      symbol-scoped path (parent directories are created as needed).

    Legacy files are **never** deleted, moved, or renamed.  Journal /
    state contents are **never** read or modified — this is a pure
    filesystem copy.

    Parameters
    ----------
    runtime_paths : RuntimePaths
        Symbol-scoped path builder for the target symbol.
    inst_id : str
        OKX instrument ID (must match *runtime_paths.inst_id*).

    Returns
    -------
    LegacyRuntimeHandoffResult
        Per-file results describing what was done and why.
    """
    file_specs: list[tuple[str, Path, Path]] = [
        ("state", DEFAULT_STATE_PATH, runtime_paths.state_file),
        ("journal", DEFAULT_JOURNAL_PATH, runtime_paths.journal_file),
        ("summary", DEFAULT_SUMMARY_PATH, runtime_paths.trade_summary_file),
    ]

    items: list[LegacyRuntimeFileHandoff] = []

    for label, legacy_path, symbol_path in file_specs:
        if inst_id != LEGACY_RUNTIME_SYMBOL:
            items.append(
                LegacyRuntimeFileHandoff(
                    label=label,
                    legacy_path=legacy_path,
                    symbol_path=symbol_path,
                    action="skipped",
                    reason=(
                        f"legacy handoff only allowed for {LEGACY_RUNTIME_SYMBOL}, "
                        f"not {inst_id}"
                    ),
                )
            )
            continue

        if legacy_path.resolve() == symbol_path.resolve():
            items.append(
                LegacyRuntimeFileHandoff(
                    label=label,
                    legacy_path=legacy_path,
                    symbol_path=symbol_path,
                    action="skipped",
                    reason="same path",
                )
            )
            continue

        if symbol_path.exists():
            items.append(
                LegacyRuntimeFileHandoff(
                    label=label,
                    legacy_path=legacy_path,
                    symbol_path=symbol_path,
                    action="skipped",
                    reason="target exists",
                )
            )
            continue

        if not legacy_path.exists():
            items.append(
                LegacyRuntimeFileHandoff(
                    label=label,
                    legacy_path=legacy_path,
                    symbol_path=symbol_path,
                    action="skipped",
                    reason="legacy missing",
                )
            )
            continue

        if not legacy_path.is_file():
            items.append(
                LegacyRuntimeFileHandoff(
                    label=label,
                    legacy_path=legacy_path,
                    symbol_path=symbol_path,
                    action="skipped",
                    reason="legacy not file",
                )
            )
            continue

        # All guards passed — perform the copy.
        symbol_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_path, symbol_path)

        items.append(
            LegacyRuntimeFileHandoff(
                label=label,
                legacy_path=legacy_path,
                symbol_path=symbol_path,
                action="copied",
                reason="copied legacy file to symbol-scoped path",
            )
        )

    return LegacyRuntimeHandoffResult(
        inst_id=inst_id,
        items=tuple(items),
    )
