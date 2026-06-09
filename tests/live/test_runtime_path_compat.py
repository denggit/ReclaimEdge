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
    handoff must copy all four files and leave old files in place."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir(parents=True)

    legacy_state = legacy_dir / "live_state.json"
    legacy_journal = legacy_dir / "live_trade_events.jsonl"
    legacy_summary = legacy_dir / "live_trade_summary.jsonl"
    legacy_rolling_loss = legacy_dir / "rolling_loss_guard_state.json"

    legacy_state.write_text('{"symbol":"ETH-USDT-SWAP"}')
    legacy_journal.write_text('{"event_type":"ENTRY"}\n')
    legacy_summary.write_text('{"event_type":"SUMMARY"}\n')
    legacy_rolling_loss.write_text('{"enabled":true}')

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
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_ROLLING_LOSS_STATE_PATH",
        legacy_rolling_loss,
    )

    runtime_paths = _eth_paths(tmp_path)
    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="ETH-USDT-SWAP",
    )

    # -- All four items must be "copied" -----------------------------------
    for item in result.items:
        assert item.action == "copied", (
            f"Expected copied for {item.label}, got {item.action}: {item.reason}"
        )

    # -- Symbol-scoped targets must exist with correct content --------------
    assert runtime_paths.state_file.read_text() == '{"symbol":"ETH-USDT-SWAP"}'
    assert runtime_paths.journal_file.read_text() == '{"event_type":"ENTRY"}\n'
    assert runtime_paths.trade_summary_file.read_text() == '{"event_type":"SUMMARY"}\n'
    assert runtime_paths.rolling_loss_guard_state_file.read_text() == '{"enabled":true}'

    # -- Legacy files must still exist (not deleted / moved) ----------------
    assert legacy_state.exists()
    assert legacy_journal.exists()
    assert legacy_summary.exists()
    assert legacy_rolling_loss.exists()


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
    legacy_rolling_loss = legacy_dir / "rolling_loss_guard_state.json"

    legacy_state.write_text("LEGACY_STATE")
    legacy_journal.write_text("LEGACY_JOURNAL\n")
    legacy_summary.write_text("LEGACY_SUMMARY\n")
    legacy_rolling_loss.write_text("LEGACY_ROLLING_LOSS")

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
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_ROLLING_LOSS_STATE_PATH",
        legacy_rolling_loss,
    )

    runtime_paths = _eth_paths(tmp_path)
    # Pre-create symbol-scoped targets with different content
    runtime_paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_paths.journal_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_paths.rolling_loss_guard_state_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_paths.state_file.write_text("EXISTING_STATE")
    runtime_paths.journal_file.write_text("EXISTING_JOURNAL\n")
    runtime_paths.trade_summary_file.write_text("EXISTING_SUMMARY\n")
    runtime_paths.rolling_loss_guard_state_file.write_text("EXISTING_ROLLING_LOSS")

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
    assert runtime_paths.rolling_loss_guard_state_file.read_text() == "EXISTING_ROLLING_LOSS"


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
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_ROLLING_LOSS_STATE_PATH",
        missing / "rolling_loss_guard_state.json",
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
    assert not runtime_paths.rolling_loss_guard_state_file.exists()


# ============================================================================
# 4. test_handoff_skips_non_eth_symbol
# ============================================================================


def test_handoff_skips_non_eth_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both *inst_id* and *runtime_paths* are non-ETH (e.g. BTC),
    handoff must skip all items.

    All items must be skipped with a reason mentioning the dual-guard
    restriction (both inst_id and runtime_paths symbol must be ETH-USDT-SWAP).
    """
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir(parents=True)

    legacy_state = legacy_dir / "live_state.json"
    legacy_journal = legacy_dir / "live_trade_events.jsonl"
    legacy_summary = legacy_dir / "live_trade_summary.jsonl"
    legacy_rolling_loss = legacy_dir / "rolling_loss_guard_state.json"

    legacy_state.write_text("ETH_STATE")
    legacy_journal.write_text("ETH_JOURNAL\n")
    legacy_summary.write_text("ETH_SUMMARY\n")
    legacy_rolling_loss.write_text("ETH_ROLLING_LOSS")

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
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_ROLLING_LOSS_STATE_PATH",
        legacy_rolling_loss,
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
        assert "both inst_id and runtime_paths symbol" in item.reason, (
            f"Reason must mention dual-guard restriction: {item.reason}"
        )

    # -- BTC symbol-scoped targets must NOT exist ---------------------------
    assert not runtime_paths.state_file.exists()
    assert not runtime_paths.journal_file.exists()
    assert not runtime_paths.trade_summary_file.exists()
    assert not runtime_paths.rolling_loss_guard_state_file.exists()

    # -- Legacy files must still exist (not deleted / moved) ----------------
    assert legacy_state.exists()
    assert legacy_journal.exists()
    assert legacy_summary.exists()
    assert legacy_rolling_loss.exists()


# ============================================================================
# 5. test_handoff_skips_when_runtime_paths_symbol_mismatches_inst_id
# ============================================================================


def test_handoff_skips_when_runtime_paths_symbol_mismatches_inst_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When runtime_paths targets BTC but inst_id claims ETH, handoff
    must skip everything — ETH legacy data must NOT land in BTC paths."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir(parents=True)

    legacy_state = legacy_dir / "live_state.json"
    legacy_journal = legacy_dir / "live_trade_events.jsonl"
    legacy_summary = legacy_dir / "live_trade_summary.jsonl"
    legacy_rolling_loss = legacy_dir / "rolling_loss_guard_state.json"

    legacy_state.write_text("ETH_STATE")
    legacy_journal.write_text("ETH_JOURNAL\n")
    legacy_summary.write_text("ETH_SUMMARY\n")
    legacy_rolling_loss.write_text("ETH_ROLLING_LOSS")

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
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_ROLLING_LOSS_STATE_PATH",
        legacy_rolling_loss,
    )

    runtime_paths = _btc_paths(tmp_path)
    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="ETH-USDT-SWAP",
    )

    for item in result.items:
        assert item.action == "skipped", (
            f"mismatched {item.label} must be skipped, got {item.action}"
        )
        assert "both inst_id and runtime_paths symbol" in item.reason, (
            f"Reason must mention dual-guard restriction: {item.reason}"
        )

    # -- BTC symbol-scoped targets must NOT exist ---------------------------
    assert not runtime_paths.state_file.exists()
    assert not runtime_paths.journal_file.exists()
    assert not runtime_paths.trade_summary_file.exists()
    assert not runtime_paths.rolling_loss_guard_state_file.exists()

    # -- Legacy files must still exist (not deleted / moved) ----------------
    assert legacy_state.exists()
    assert legacy_journal.exists()
    assert legacy_summary.exists()
    assert legacy_rolling_loss.exists()


# ============================================================================
# 6. test_handoff_skips_when_inst_id_non_eth_even_if_runtime_paths_eth
# ============================================================================


def test_handoff_skips_when_inst_id_non_eth_even_if_runtime_paths_eth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When runtime_paths targets ETH but inst_id claims BTC, handoff
    must skip everything — the caller is inconsistent and must not copy."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir(parents=True)

    legacy_state = legacy_dir / "live_state.json"
    legacy_journal = legacy_dir / "live_trade_events.jsonl"
    legacy_summary = legacy_dir / "live_trade_summary.jsonl"
    legacy_rolling_loss = legacy_dir / "rolling_loss_guard_state.json"

    legacy_state.write_text("ETH_STATE")
    legacy_journal.write_text("ETH_JOURNAL\n")
    legacy_summary.write_text("ETH_SUMMARY\n")
    legacy_rolling_loss.write_text("ETH_ROLLING_LOSS")

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
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_ROLLING_LOSS_STATE_PATH",
        legacy_rolling_loss,
    )

    runtime_paths = _eth_paths(tmp_path)
    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="BTC-USDT-SWAP",
    )

    for item in result.items:
        assert item.action == "skipped", (
            f"mismatched {item.label} must be skipped, got {item.action}"
        )
        assert "both inst_id and runtime_paths symbol" in item.reason, (
            f"Reason must mention dual-guard restriction: {item.reason}"
        )

    # -- ETH symbol-scoped targets must NOT exist (mismatch blocked copy) ---
    assert not runtime_paths.state_file.exists()
    assert not runtime_paths.journal_file.exists()
    assert not runtime_paths.trade_summary_file.exists()
    assert not runtime_paths.rolling_loss_guard_state_file.exists()

    # -- Legacy files must still exist (not deleted / moved) ----------------
    assert legacy_state.exists()
    assert legacy_journal.exists()
    assert legacy_summary.exists()
    assert legacy_rolling_loss.exists()


# ============================================================================
# 7. test_handoff_skips_when_legacy_not_file
# ============================================================================


def test_handoff_skips_when_legacy_not_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When every legacy path exists but is a directory (not a regular file),
    handoff must skip all of them.

    All four DEFAULT_* paths are monkeypatched to separate directories
    inside *tmp_path* so the test is fully isolated and does not depend on
    whether the real ``data/trade_journal/`` directory happens to exist.
    """
    state_dir = tmp_path / "legacy_state_dir"
    journal_dir = tmp_path / "legacy_journal_dir"
    summary_dir = tmp_path / "legacy_summary_dir"
    rolling_loss_dir = tmp_path / "legacy_rolling_loss_dir"
    state_dir.mkdir(parents=True)
    journal_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    rolling_loss_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_STATE_PATH",
        state_dir,
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_JOURNAL_PATH",
        journal_dir,
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_SUMMARY_PATH",
        summary_dir,
    )
    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_ROLLING_LOSS_STATE_PATH",
        rolling_loss_dir,
    )

    runtime_paths = _eth_paths(tmp_path)
    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="ETH-USDT-SWAP",
    )

    # All four items must be skipped with reason "legacy not file".
    for item in result.items:
        assert item.action == "skipped", (
            f"Expected skipped for {item.label}, got {item.action}"
        )
        assert item.reason == "legacy not file", (
            f"Expected 'legacy not file' for {item.label}, got {item.reason}"
        )

    # No target files must be created.
    assert not runtime_paths.state_file.exists()
    assert not runtime_paths.journal_file.exists()
    assert not runtime_paths.trade_summary_file.exists()
    assert not runtime_paths.rolling_loss_guard_state_file.exists()


# ============================================================================
# 8. test_handoff_skips_same_path
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


def test_handoff_rolling_loss_same_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the legacy rolling loss path equals the runtime risk path,
    handoff must skip it (same file, nothing to copy)."""
    runtime_paths = _eth_paths(tmp_path)

    runtime_paths.rolling_loss_guard_state_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_paths.rolling_loss_guard_state_file.write_text("shared rolling loss")

    monkeypatch.setattr(
        "src.live.runtime_path_compat.DEFAULT_ROLLING_LOSS_STATE_PATH",
        runtime_paths.rolling_loss_guard_state_file,
    )

    result = handoff_legacy_runtime_files(
        runtime_paths=runtime_paths,
        inst_id="ETH-USDT-SWAP",
    )

    rl_item = [i for i in result.items if i.label == "rolling_loss_guard"][0]
    assert rl_item.action == "skipped"
    assert rl_item.reason == "same path"

    assert runtime_paths.rolling_loss_guard_state_file.read_text() == "shared rolling loss"


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
