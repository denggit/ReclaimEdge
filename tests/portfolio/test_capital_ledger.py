# -*- coding: utf-8 -*-
"""Unit tests for src/portfolio/capital_ledger.py (G01)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from src.portfolio.capital_ledger import (
    LEDGER_VERSION,
    CapitalLedger,
    CapitalLedgerError,
    CapitalLedgerLockTimeout,
    CapitalLedgerSchemaError,
    CapitalLedgerSnapshot,
    SymbolCapitalState,
    _FileLock,
    _now_ms,
    default_snapshot,
    default_symbol_state,
)
from src.live.outbox.atomic_json import read_json_or_none


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _temp_paths(tmp_path: Path):
    """Return (ledger_path, lock_path) rooted in *tmp_path*."""
    return tmp_path / "ledger.json", tmp_path / "ledger.lock"


# ===================================================================
# 1. default snapshot
# ===================================================================


class TestDefaultSnapshot:
    def test_contains_eth_and_btc(self):
        snap = default_snapshot()
        assert "ETH-USDT-SWAP" in snap.symbols
        assert "BTC-USDT-SWAP" in snap.symbols

    def test_eth_sidecar_enabled(self):
        snap = default_snapshot()
        assert snap.symbols["ETH-USDT-SWAP"].sidecar_enabled is True

    def test_btc_sidecar_disabled(self):
        snap = default_snapshot()
        assert snap.symbols["BTC-USDT-SWAP"].sidecar_enabled is False

    def test_version_is_1(self):
        snap = default_snapshot()
        assert snap.version == LEDGER_VERSION
        assert snap.version == 1

    def test_leader_symbol_is_none(self):
        snap = default_snapshot()
        assert snap.leader_symbol is None

    def test_global_no_new_entry_is_false(self):
        snap = default_snapshot()
        assert snap.global_no_new_entry is False

    def test_default_max_layers(self):
        snap = default_snapshot()
        for st in snap.symbols.values():
            assert st.plan_max_layers == 8
            assert st.permission_max_layers == 8

    def test_default_updated_ms_zero_when_not_passed(self):
        snap = default_snapshot()
        assert snap.updated_ms == 0

    def test_default_updated_ms_custom(self):
        snap = default_snapshot(updated_ms=1728000000000)
        assert snap.updated_ms == 1728000000000


# ===================================================================
# 2. default_symbol_state
# ===================================================================


class TestDefaultSymbolState:
    def test_eth_state(self):
        st = default_symbol_state("ETH-USDT-SWAP")
        assert st.state == "FLAT"
        assert st.sidecar_enabled is True

    def test_btc_state(self):
        st = default_symbol_state("BTC-USDT-SWAP")
        assert st.state == "FLAT"
        assert st.sidecar_enabled is False

    def test_unknown_symbol_still_flat(self):
        st = default_symbol_state("SOL-USDT-SWAP")
        assert st.state == "FLAT"
        assert st.sidecar_enabled is False


# ===================================================================
# 3. to_dict / from_dict round trip
# ===================================================================


class TestRoundTrip:
    def test_snapshot_round_trip(self):
        snap = default_snapshot(updated_ms=100)
        d = snap.to_dict()
        restored = CapitalLedgerSnapshot.from_dict(d)
        assert restored.version == snap.version
        assert restored.updated_ms == snap.updated_ms
        assert restored.leader_symbol == snap.leader_symbol
        assert restored.global_no_new_entry == snap.global_no_new_entry
        assert set(restored.symbols.keys()) == set(snap.symbols.keys())
        for inst_id in snap.symbols:
            assert restored.symbols[inst_id] == snap.symbols[inst_id]

    def test_planned_main_contracts_list_converts_to_tuple(self):
        d = default_snapshot().to_dict()
        # Simulate JSON round-trip where tuples become lists
        d["symbols"]["ETH-USDT-SWAP"]["planned_main_contracts"] = ["100", "200"]
        restored = CapitalLedgerSnapshot.from_dict(d)
        eth = restored.symbols["ETH-USDT-SWAP"]
        assert isinstance(eth.planned_main_contracts, tuple)
        assert eth.planned_main_contracts == ("100", "200")

    def test_symbol_state_round_trip(self):
        original = SymbolCapitalState(
            state="OPEN",
            side="LONG",
            used_layers=3,
            position_plan_id="plan-abc",
            planned_main_contracts=("50", "100", "150"),
            base_main_contracts="50",
            plan_max_layers=8,
            permission_max_layers=6,
            add_gap_multiplier="1.5",
            add_freeze_multiplier="2.0",
            main_used_margin_usdt="1000.5",
            sidecar_enabled=True,
            sidecar_used_margin_usdt="200.3",
        )
        restored = SymbolCapitalState.from_dict(original.to_dict())
        assert restored == original


# ===================================================================
# 4. initialize_if_missing
# ===================================================================


class TestInitializeIfMissing:
    def test_creates_file_when_missing(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        snap = ledger.initialize_if_missing(updated_ms=123)
        assert snap.updated_ms == 123
        assert lp.exists()

    def test_idempotent_does_not_overwrite(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        snap1 = ledger.initialize_if_missing(updated_ms=111)
        assert snap1.updated_ms == 111

        # second call must NOT overwrite
        snap2 = ledger.initialize_if_missing(updated_ms=999)
        assert snap2.updated_ms == 111

    def test_returns_existing_when_present(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.initialize_if_missing(updated_ms=5)
        snap = ledger.initialize_if_missing(updated_ms=10)
        assert snap.updated_ms == 5


# ===================================================================
# 5. read_unlocked missing file
# ===================================================================


class TestReadUnlocked:
    def test_missing_file_returns_default_does_not_create(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        snap = ledger.read_unlocked()
        assert snap.version == 1
        assert snap.updated_ms == 0
        assert not lp.exists()


# ===================================================================
# 6. write_unlocked atomic output
# ===================================================================


class TestWriteUnlocked:
    def test_write_unlocked_creates_readable_json(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        snap = default_snapshot(updated_ms=500)
        ledger.write_unlocked(snap)
        assert lp.exists()
        raw = read_json_or_none(lp)
        assert raw is not None
        assert raw["version"] == 1
        assert "symbols" in raw

    def test_write_unlocked_overwrites(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.write_unlocked(default_snapshot(updated_ms=1))
        ledger.write_unlocked(default_snapshot(updated_ms=2))
        snap = ledger.read_unlocked()
        assert snap.updated_ms == 2


# ===================================================================
# 7. update_locked
# ===================================================================


class TestUpdateLocked:
    def test_modify_leader_symbol(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.initialize_if_missing()

        def set_leader(snap: CapitalLedgerSnapshot) -> CapitalLedgerSnapshot:
            return CapitalLedgerSnapshot(
                version=snap.version,
                updated_ms=_now_ms(),
                leader_symbol="ETH-USDT-SWAP",
                global_no_new_entry=snap.global_no_new_entry,
                symbols=snap.symbols,
            )

        new_snap = ledger.update_locked(set_leader)
        assert new_snap.leader_symbol == "ETH-USDT-SWAP"

        # re-read
        snap2 = ledger.read_locked()
        assert snap2.leader_symbol == "ETH-USDT-SWAP"

    def test_modify_global_no_new_entry(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.initialize_if_missing()

        def toggle(snap: CapitalLedgerSnapshot) -> CapitalLedgerSnapshot:
            return CapitalLedgerSnapshot(
                version=snap.version,
                updated_ms=_now_ms(),
                leader_symbol=snap.leader_symbol,
                global_no_new_entry=True,
                symbols=snap.symbols,
            )

        new_snap = ledger.update_locked(toggle)
        assert new_snap.global_no_new_entry is True

    def test_mutator_returns_dict_raises(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.initialize_if_missing()

        def bad_mutator(snap: CapitalLedgerSnapshot) -> dict:  # type: ignore[return]
            return snap.to_dict()

        with pytest.raises(CapitalLedgerSchemaError, match="mutator must return"):
            ledger.update_locked(bad_mutator)  # type: ignore[arg-type]


# ===================================================================
# 8. with_symbol_state
# ===================================================================


class TestWithSymbolState:
    def test_update_eth_used_layers(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.initialize_if_missing()

        def set_layers(st: SymbolCapitalState) -> SymbolCapitalState:
            return SymbolCapitalState(
                state="OPEN",
                side="LONG",
                used_layers=3,
                base_main_contracts=st.base_main_contracts,
                main_used_margin_usdt=st.main_used_margin_usdt,
                sidecar_enabled=st.sidecar_enabled,
                sidecar_used_margin_usdt=st.sidecar_used_margin_usdt,
            )

        new_snap = ledger.with_symbol_state("ETH-USDT-SWAP", set_layers)
        assert new_snap.symbols["ETH-USDT-SWAP"].used_layers == 3
        assert new_snap.symbols["ETH-USDT-SWAP"].state == "OPEN"
        # BTC unchanged
        assert new_snap.symbols["BTC-USDT-SWAP"].used_layers == 0

        # re-read
        eth = ledger.get_symbol("ETH-USDT-SWAP")
        assert eth.used_layers == 3

    def test_unknown_symbol_raises(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.initialize_if_missing()

        with pytest.raises(CapitalLedgerSchemaError, match="not found"):
            ledger.with_symbol_state(
                "SOL-USDT-SWAP",
                lambda s: s,
            )

    def test_updater_returns_wrong_type_raises(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.initialize_if_missing()

        with pytest.raises(CapitalLedgerSchemaError, match="must return SymbolCapitalState"):
            ledger.with_symbol_state(
                "ETH-USDT-SWAP",
                lambda s: {"state": "OPEN"},  # type: ignore[return-value]
            )

    def test_updated_ms_passed(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.initialize_if_missing()

        new_snap = ledger.with_symbol_state(
            "ETH-USDT-SWAP",
            lambda s: s,
            updated_ms=1730000000000,
        )
        assert new_snap.updated_ms == 1730000000000


# ===================================================================
# 9. get_symbol
# ===================================================================


class TestGetSymbol:
    def test_gets_eth_symbol(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.initialize_if_missing()
        eth = ledger.get_symbol("ETH-USDT-SWAP")
        assert eth.state == "FLAT"
        assert eth.sidecar_enabled is True

    def test_unknown_symbol_raises(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)
        ledger.initialize_if_missing()
        with pytest.raises(CapitalLedgerSchemaError, match="not found"):
            ledger.get_symbol("SOL-USDT-SWAP")


# ===================================================================
# 10. invalid schema
# ===================================================================


class TestInvalidSchema:
    def test_version_not_1_raises(self):
        with pytest.raises(CapitalLedgerSchemaError, match="unsupported ledger version"):
            CapitalLedgerSnapshot.from_dict({"version": 2, "symbols": {}})

    def test_version_missing_raises(self):
        with pytest.raises(CapitalLedgerSchemaError, match="missing 'version'"):
            CapitalLedgerSnapshot.from_dict({"symbols": {}})

    def test_symbols_missing_raises(self):
        with pytest.raises(CapitalLedgerSchemaError, match="missing 'symbols'"):
            CapitalLedgerSnapshot.from_dict({"version": 1})

    def test_symbols_not_dict_raises(self):
        with pytest.raises(CapitalLedgerSchemaError, match="'symbols' must be a dict"):
            CapitalLedgerSnapshot.from_dict({"version": 1, "symbols": []})

    def test_symbol_state_not_dict_raises(self):
        with pytest.raises(CapitalLedgerSchemaError, match="must be a dict"):
            CapitalLedgerSnapshot.from_dict({
                "version": 1,
                "updated_ms": 0,
                "leader_symbol": None,
                "global_no_new_entry": False,
                "symbols": {"ETH-USDT-SWAP": "not_a_dict"},
            })

    def test_symbol_state_missing_required_str_raises(self):
        d = default_snapshot().to_dict()
        del d["symbols"]["ETH-USDT-SWAP"]["state"]
        with pytest.raises(CapitalLedgerSchemaError, match="'state' must be a string"):
            CapitalLedgerSnapshot.from_dict(d)

    def test_symbol_state_bad_used_layers_raises(self):
        d = default_snapshot().to_dict()
        d["symbols"]["ETH-USDT-SWAP"]["used_layers"] = "three"
        with pytest.raises(CapitalLedgerSchemaError, match="'used_layers' must be an int"):
            CapitalLedgerSnapshot.from_dict(d)

    # -- strict bool checks ---------------------------------------------------

    def test_sidecar_enabled_must_be_bool(self):
        d = default_snapshot().to_dict()
        d["symbols"]["ETH-USDT-SWAP"]["sidecar_enabled"] = "false"
        with pytest.raises(CapitalLedgerSchemaError, match="sidecar_enabled.*bool"):
            CapitalLedgerSnapshot.from_dict(d)

    def test_global_no_new_entry_must_be_bool(self):
        d = default_snapshot().to_dict()
        d["global_no_new_entry"] = "false"
        with pytest.raises(CapitalLedgerSchemaError, match="global_no_new_entry.*bool"):
            CapitalLedgerSnapshot.from_dict(d)

    # -- strict nullable str checks -------------------------------------------

    def test_leader_symbol_must_be_str_or_none(self):
        d = default_snapshot().to_dict()
        d["leader_symbol"] = 123
        with pytest.raises(CapitalLedgerSchemaError, match="leader_symbol"):
            CapitalLedgerSnapshot.from_dict(d)

    def test_side_must_be_str_or_none(self):
        d = default_snapshot().to_dict()
        d["symbols"]["ETH-USDT-SWAP"]["side"] = 123
        with pytest.raises(CapitalLedgerSchemaError, match="side"):
            CapitalLedgerSnapshot.from_dict(d)

    def test_position_plan_id_must_be_str_or_none(self):
        d = default_snapshot().to_dict()
        d["symbols"]["ETH-USDT-SWAP"]["position_plan_id"] = True
        with pytest.raises(CapitalLedgerSchemaError, match="position_plan_id"):
            CapitalLedgerSnapshot.from_dict(d)

    # -- strict planned_main_contracts checks ---------------------------------

    def test_planned_main_contracts_must_not_be_string(self):
        d = default_snapshot().to_dict()
        d["symbols"]["ETH-USDT-SWAP"]["planned_main_contracts"] = "123"
        with pytest.raises(CapitalLedgerSchemaError, match="planned_main_contracts"):
            CapitalLedgerSnapshot.from_dict(d)

    def test_planned_main_contracts_items_must_be_strings(self):
        d = default_snapshot().to_dict()
        d["symbols"]["ETH-USDT-SWAP"]["planned_main_contracts"] = ["1", 2]
        with pytest.raises(CapitalLedgerSchemaError, match="planned_main_contracts"):
            CapitalLedgerSnapshot.from_dict(d)


# ===================================================================
# 11. lock timeout
# ===================================================================


class TestLockTimeout:
    def test_lock_timeout_raises(self, tmp_path: Path):
        lock_path = tmp_path / "timeout.lock"
        lock1 = _FileLock(lock_path, timeout_seconds=0.3, poll_seconds=0.02)
        lock2 = _FileLock(lock_path, timeout_seconds=0.05, poll_seconds=0.01)

        lock1.acquire()
        try:
            with pytest.raises(CapitalLedgerLockTimeout):
                lock2.acquire()
        finally:
            lock1.release()

    def test_lock_then_release_then_acquire(self, tmp_path: Path):
        """After release, another lock can acquire."""
        lock_path = tmp_path / "sequential.lock"
        lock1 = _FileLock(lock_path, timeout_seconds=0.5, poll_seconds=0.02)
        lock2 = _FileLock(lock_path, timeout_seconds=0.5, poll_seconds=0.02)

        lock1.acquire()
        lock1.release()

        # second lock should succeed immediately
        lock2.acquire()
        lock2.release()

    def test_lock_timeout_allows_reacquire(self, tmp_path: Path):
        """After a lock times out, a new lock on the same path can acquire.

        This guards against leaked fd / lock state from the timeout path.
        """
        lock_path = tmp_path / "timeout_reacquire.lock"
        lock1 = _FileLock(lock_path, timeout_seconds=0.3, poll_seconds=0.02)
        lock2 = _FileLock(lock_path, timeout_seconds=0.05, poll_seconds=0.01)

        lock1.acquire()
        try:
            with pytest.raises(CapitalLedgerLockTimeout):
                lock2.acquire()
        finally:
            lock1.release()

        # lock2 timed out; a fresh lock should still be able to acquire
        lock3 = _FileLock(lock_path, timeout_seconds=0.5, poll_seconds=0.01)
        lock3.acquire()
        lock3.release()


# ===================================================================
# 12. _now_ms helper
# ===================================================================


class TestNowMs:
    def test_now_ms_returns_int_and_is_recent(self):
        ms = _now_ms()
        assert isinstance(ms, int)
        # Must be within last 5 seconds
        assert ms > 0
        now_s = int(time.time())
        assert abs(ms / 1000.0 - now_s) < 5

    def test_now_ms_is_monotonic_enough(self):
        a = _now_ms()
        b = _now_ms()
        assert b >= a


# ===================================================================
# 13. SymbolCapitalState immutability
# ===================================================================


class TestSymbolCapitalStateImmutability:
    def test_frozen_dataclass(self):
        st = default_symbol_state("ETH-USDT-SWAP")
        with pytest.raises(Exception):
            st.state = "OPEN"  # type: ignore[misc]

    def test_hashable(self):
        st1 = default_symbol_state("ETH-USDT-SWAP")
        st2 = default_symbol_state("ETH-USDT-SWAP")
        assert hash(st1) == hash(st2)
        s = {st1, st2}
        assert len(s) == 1


# ===================================================================
# 14. CapitalLedgerSnapshot immutability
# ===================================================================


class TestSnapshotImmutability:
    def test_frozen_dataclass(self):
        snap = default_snapshot()
        with pytest.raises(Exception):
            snap.version = 2  # type: ignore[misc]

    def test_symbols_not_hashable_due_to_dict(self):
        """Snapshot contains a mutable Mapping (dict), so it is NOT hashable."""
        snap = default_snapshot(updated_ms=0)
        with pytest.raises(TypeError):
            hash(snap)


# ===================================================================
# 15. integration-ish: full flow
# ===================================================================


class TestFullFlow:
    def test_init_read_update_read_cycle(self, tmp_path: Path):
        lp, lock = _temp_paths(tmp_path)
        ledger = CapitalLedger(lp, lock)

        # 1. init
        snap = ledger.initialize_if_missing(updated_ms=100)
        assert snap.version == 1
        assert snap.symbols["ETH-USDT-SWAP"].state == "FLAT"

        # 2. update ETH to OPEN
        def open_eth(st: SymbolCapitalState) -> SymbolCapitalState:
            return SymbolCapitalState(
                state="OPEN",
                side="LONG",
                used_layers=1,
                position_plan_id="plan-1",
                planned_main_contracts=("50",),
                base_main_contracts="50",
                plan_max_layers=st.plan_max_layers,
                permission_max_layers=st.permission_max_layers,
                add_gap_multiplier=st.add_gap_multiplier,
                add_freeze_multiplier=st.add_freeze_multiplier,
                main_used_margin_usdt="500",
                sidecar_enabled=st.sidecar_enabled,
                sidecar_used_margin_usdt=st.sidecar_used_margin_usdt,
            )

        new_snap = ledger.with_symbol_state("ETH-USDT-SWAP", open_eth)
        assert new_snap.symbols["ETH-USDT-SWAP"].state == "OPEN"
        assert new_snap.symbols["ETH-USDT-SWAP"].used_layers == 1

        # 3. global flag toggle via update_locked
        def set_no_entry(snap: CapitalLedgerSnapshot) -> CapitalLedgerSnapshot:
            return CapitalLedgerSnapshot(
                version=snap.version,
                updated_ms=_now_ms(),
                leader_symbol=snap.leader_symbol,
                global_no_new_entry=True,
                symbols=snap.symbols,
            )

        snap3 = ledger.update_locked(set_no_entry)
        assert snap3.global_no_new_entry is True
        assert snap3.symbols["ETH-USDT-SWAP"].state == "OPEN"

        # 4. read back
        snap4 = ledger.read_locked()
        assert snap4.global_no_new_entry is True
        assert snap4.symbols["ETH-USDT-SWAP"].used_layers == 1
        assert snap4.symbols["BTC-USDT-SWAP"].used_layers == 0


# ===================================================================
# 16. verify no forbidden imports / side effects
# ===================================================================


class TestSourcePurity:
    def test_capital_ledger_source_has_no_runtime_side_effect_imports(self) -> None:
        source_path = (
            Path(__file__).parents[2]
            / "src" / "portfolio" / "capital_ledger.py"
        )
        source = source_path.read_text()

        forbidden = [
            "Trader",
            "Strategy",
            "requests",
            "httpx",
            "websocket",
            "okx",
            "EmailSender",
            "os.getenv",
            "load_dotenv",
            "redis",
            "sqlite",
            "pydantic",
            "portalocker",
            "filelock",
        ]
        for token in forbidden:
            assert token not in source, (
                f"capital_ledger.py must not import/use {token}"
            )


# ===================================================================
# 17. CapitalLedger.__init__ accepts Path objects
# ===================================================================


class TestInit:
    def test_accepts_path_objects(self, tmp_path: Path):
        lp = tmp_path / "sub" / "ledger.json"
        lock = tmp_path / "sub" / "ledger.lock"
        ledger = CapitalLedger(lp, lock, lock_timeout_seconds=1.0)
        snap = ledger.initialize_if_missing()
        assert lp.exists()
        assert snap.version == 1


# ===================================================================
# 18. Base exception hierarchy
# ===================================================================


class TestExceptionHierarchy:
    def test_lock_timeout_is_ledger_error(self):
        assert issubclass(CapitalLedgerLockTimeout, CapitalLedgerError)

    def test_schema_error_is_ledger_error(self):
        assert issubclass(CapitalLedgerSchemaError, CapitalLedgerError)

    def test_ledger_error_is_runtime_error(self):
        assert issubclass(CapitalLedgerError, RuntimeError)
