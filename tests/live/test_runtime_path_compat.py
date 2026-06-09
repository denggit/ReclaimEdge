from __future__ import annotations

from pathlib import Path

import pytest

from src.live.runtime_path_compat import (
    LegacyRuntimeFileHandoff,
    LegacyRuntimeHandoffResult,
    handoff_legacy_runtime_files,
)
from src.live.runtime_paths import RuntimePaths


# ============================================================================
# Helper
# ============================================================================


def _eth_paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(runtime_dir=tmp_path / "runtime", inst_id="ETH-USDT-SWAP")


def _btc_paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(runtime_dir=tmp_path / "runtime", inst_id="BTC-USDT-SWAP")


# ============================================================================
# 1. test_handoff_copies_missing_eth_legacy_files
# ============================================================================


def test_handoff_copies_missing_eth_legacy_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When legacy files exist and symbol-scoped targets are missing,
    handoff must copy all three files and leave old files in place."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir(parents=True)

    legacy_state = legacy_dir / "live_state.json"
    legacy_journal = legacy_dir / "live_trade_events.jsonl"
    legacy_summary = legacy_dir / "live_trade_summary.jsonl"

    legacy_state.write_text('{"symbol":"ETH-USDT-SWAP"}')
    legacy_journal.write_text('{"event_type":"ENTRY"}\n')
    legacy_summary.write_text('{"event_type":"SUMMARY"}\n')

    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_STATE_PATH",
        legacy_state,
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_JOURNAL_PATH",
        legacy_journal,
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_SUMMARY_PATH",
        legacy_summary,
    )

    runtime_paths = _eth_paths(tmp_path)
    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="ETH-USDT-SWAP",
    )

    # -- All three items must be "copied" -----------------------------------
    for item in result.items:
        assert item.action == "copied", (
            f"Expected copied for {item.label}, got {item.action}: {item.reason}"
        )

    # -- Symbol-scoped targets must exist with correct content --------------
    assert runtime_paths.state_file.read_text() == '{"symbol":"ETH-USDT-SWAP"}'
    assert runtime_paths.journal_file.read_text() == '{"event_type":"ENTRY"}\n'
    assert runtime_paths.trade_summary_file.read_text() == '{"event_type":"SUMMARY"}\n'

    # -- Legacy files must still exist (not deleted / moved) ----------------
    assert legacy_state.exists()
    assert legacy_journal.exists()
    assert legacy_summary.exists()


# ============================================================================
# 2. test_handoff_does_not_overwrite_existing_symbol_files
# ============================================================================


def test_handoff_does_not_overwrite_existing_symbol_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both legacy and symbol-scoped targets exist, handoff must
    **not** overwrite the targets — their content must be preserved."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir(parents=True)

    legacy_state = legacy_dir / "live_state.json"
    legacy_journal = legacy_dir / "live_trade_events.jsonl"
    legacy_summary = legacy_dir / "live_trade_summary.jsonl"

    legacy_state.write_text("LEGACY_STATE")
    legacy_journal.write_text("LEGACY_JOURNAL\n")
    legacy_summary.write_text("LEGACY_SUMMARY\n")

    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_STATE_PATH",
        legacy_state,
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_JOURNAL_PATH",
        legacy_journal,
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_SUMMARY_PATH",
        legacy_summary,
    )

    runtime_paths = _eth_paths(tmp_path)
    # Pre-create symbol-scoped targets with different content
    runtime_paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_paths.journal_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_paths.state_file.write_text("EXISTING_STATE")
    runtime_paths.journal_file.write_text("EXISTING_JOURNAL\n")
    runtime_paths.trade_summary_file.write_text("EXISTING_SUMMARY\n")

    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="ETH-USDT-SWAP",
    )

    # -- All items must be "skipped" with reason "target exists" ------------
    for item in result.items:
        assert item.action == "skipped", (
            f"Expected skipped for {item.label}, got {item.action}"
        )
        assert item.reason == "target exists", (
            f"Expected 'target exists' for {item.label}, got {item.reason}"
        )

    # -- Existing content must be preserved --------------------------------
    assert runtime_paths.state_file.read_text() == "EXISTING_STATE"
    assert runtime_paths.journal_file.read_text() == "EXISTING_JOURNAL\n"
    assert runtime_paths.trade_summary_file.read_text() == "EXISTING_SUMMARY\n"


# ============================================================================
# 3. test_handoff_skips_when_legacy_missing
# ============================================================================


def test_handoff_skips_when_legacy_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When legacy files don't exist, handoff must skip without creating
    empty symbol-scoped files."""
    missing = tmp_path / "nonexistent"
    # Do NOT create the file.

    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_STATE_PATH",
        missing / "live_state.json",
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_JOURNAL_PATH",
        missing / "live_trade_events.jsonl",
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_SUMMARY_PATH",
        missing / "live_trade_summary.jsonl",
    )

    runtime_paths = _eth_paths(tmp_path)
    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="ETH-USDT-SWAP",
    )

    for item in result.items:
        assert item.action == "skipped", (
            f"Expected skipped for {item.label}, got {item.action}"
        )
        assert item.reason == "legacy missing", (
            f"Expected 'legacy missing' for {item.label}, got {item.reason}"
        )

    # -- Symbol-scoped targets must NOT exist -------------------------------
    assert not runtime_paths.state_file.exists()
    assert not runtime_paths.journal_file.exists()
    assert not runtime_paths.trade_summary_file.exists()


# ============================================================================
# 4. test_handoff_skips_non_eth_symbol
# ============================================================================


def test_handoff_skips_non_eth_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BTC-USDT-SWAP must NOT receive ETH legacy files.  All items must be
    skipped with a reason mentioning the ETH-only restriction."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir(parents=True)

    legacy_state = legacy_dir / "live_state.json"
    legacy_journal = legacy_dir / "live_trade_events.jsonl"
    legacy_summary = legacy_dir / "live_trade_summary.jsonl"

    legacy_state.write_text("ETH_STATE")
    legacy_journal.write_text("ETH_JOURNAL\n")
    legacy_summary.write_text("ETH_SUMMARY\n")

    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_STATE_PATH",
        legacy_state,
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_JOURNAL_PATH",
        legacy_journal,
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_SUMMARY_PATH",
        legacy_summary,
    )

    runtime_paths = _btc_paths(tmp_path)
    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="BTC-USDT-SWAP",
    )

    for item in result.items:
        assert item.action == "skipped", (
            f"BTC {item.label} must be skipped, got {item.action}"
        )
        assert "ETH-USDT-SWAP" in item.reason, (
            f"Reason must mention ETH-only restriction: {item.reason}"
        )

    # -- BTC symbol-scoped targets must NOT exist ---------------------------
    assert not runtime_paths.state_file.exists()
    assert not runtime_paths.journal_file.exists()
    assert not runtime_paths.trade_summary_file.exists()

    # -- Legacy files must still exist (not deleted / moved) ----------------
    assert legacy_state.exists()
    assert legacy_journal.exists()
    assert legacy_summary.exists()


# ============================================================================
# 5. test_handoff_skips_when_legacy_not_file
# ============================================================================


def test_handoff_skips_when_legacy_not_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the legacy path exists but is a directory (not a regular file),
    handoff must skip it."""
    legacy_dir = tmp_path / "legacy_dir"
    legacy_dir.mkdir(parents=True)

    # Make the "legacy" paths point to directories.
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_STATE_PATH",
        legacy_dir,
    )

    runtime_paths = _eth_paths(tmp_path)
    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="ETH-USDT-SWAP",
    )

    # The state item must be skipped because legacy is a dir.
    state_item = result.items[0]
    assert state_item.label == "state"
    assert state_item.action == "skipped"
    assert state_item.reason == "legacy not file"

    # The other two (journal, summary) use the real DEFAULT paths which may
    # or may not exist — we only care that none were copied.
    for item in result.items:
        assert item.action != "copied", (
            f"{item.label} must not be copied when legacy is a directory"
        )


# ============================================================================
# 6. test_handoff_skips_same_path
# ============================================================================


def test_handoff_skips_same_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the legacy path equals the symbol-scoped path, handoff must
    skip it (same file, nothing to copy)."""
    runtime_paths = _eth_paths(tmp_path)

    # Point the DEFAULT_STATE_PATH to the same location as the
    # symbol-scoped state file so they are the same file.
    runtime_paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_paths.state_file.write_text("shared content")

    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_STATE_PATH",
        runtime_paths.state_file,
    )

    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="ETH-USDT-SWAP",
    )

    state_item = result.items[0]
    assert state_item.label == "state"
    assert state_item.action == "skipped"
    assert state_item.reason == "same path"

    # Content must be unchanged.
    assert runtime_paths.state_file.read_text() == "shared content"


# ============================================================================
# Dataclass immutability
# ============================================================================


def test_legacy_runtime_file_handoff_is_frozen() -> None:
    """LegacyRuntimeFileHandoff must be frozen."""
    item = LegacyRuntimeFileHandoff(
        label="state",
        legacy_path=Path("/a"),
        symbol_path=Path("/b"),
        action="copied",
        reason="test",
    )
    with pytest.raises(Exception):
        item.action = "hacked"  # type: ignore[misc]


def test_legacy_runtime_handoff_result_is_frozen() -> None:
    """LegacyRuntimeHandoffResult must be frozen."""
    result = LegacyRuntimeHandoffResult(
        inst_id="ETH-USDT-SWAP",
        items=(),
    )
    with pytest.raises(Exception):
        result.inst_id = "hacked"  # type: ignore[misc]
