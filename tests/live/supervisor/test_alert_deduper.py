from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live.outbox.atomic_json import read_json_or_none, write_json_atomic
from src.live.supervisor.alert_deduper import AlertDedupeDecision, AlertDeduper

# ============================================================================
# Source path for guards
# ============================================================================

_ALERT_DEDUPER_SOURCE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "src"
    / "live"
    / "supervisor"
    / "alert_deduper.py"
)

# ============================================================================
# 1. first alert is allowed
# ============================================================================


class TestFirstAlertAllowed:
    def test_first_alert_is_allowed(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_STARTED",
            now_ms=1000,
        )

        assert decision.allowed is True
        assert decision.dedupe_key == "ETH-USDT-SWAP|WORKER_STARTED|INFO|-"
        assert decision.last_sent_ts_ms is None
        assert decision.now_ts_ms == 1000

        # State file must exist.
        state_path = tmp_path / "state.json"
        assert state_path.exists()
        state = read_json_or_none(state_path)
        assert state is not None
        assert isinstance(state, dict)
        entries = state.get("entries", {})
        assert len(entries) == 1
        key = next(iter(entries))
        assert entries[key]["last_sent_ts_ms"] == 1000


# ============================================================================
# 2. duplicate within cooldown is suppressed
# ============================================================================


class TestDuplicateWithinCooldownSuppressed:
    def test_duplicate_suppressed(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(
            state_path=tmp_path / "state.json",
            cooldown_seconds=60,
        )

        # First — allowed.
        d1 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="test halt",
            now_ms=1000,
        )
        assert d1.allowed is True

        # Second within cooldown — suppressed.
        d2 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="test halt",
            now_ms=2000,
        )
        assert d2.allowed is False
        assert d2.last_sent_ts_ms == 1000
        assert d2.next_allowed_ts_ms == 61000

        # State last_sent_ts_ms must still be 1000 (not updated on suppress).
        state = read_json_or_none(tmp_path / "state.json")
        key = next(iter(state["entries"]))
        assert state["entries"][key]["last_sent_ts_ms"] == 1000


# ============================================================================
# 3. after cooldown allowed again
# ============================================================================


class TestAfterCooldownAllowedAgain:
    def test_after_cooldown_allowed(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(
            state_path=tmp_path / "state.json",
            cooldown_seconds=60,
        )

        # First.
        d1 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="test halt",
            now_ms=1000,
        )
        assert d1.allowed is True

        # After cooldown.
        d2 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="test halt",
            now_ms=61000,
        )
        assert d2.allowed is True
        assert d2.last_sent_ts_ms == 1000

        # State updated.
        state = read_json_or_none(tmp_path / "state.json")
        key = next(iter(state["entries"]))
        assert state["entries"][key]["last_sent_ts_ms"] == 61000


# ============================================================================
# 4. different symbol does not dedupe each other
# ============================================================================


class TestDifferentSymbolNoDedupe:
    def test_different_symbol_both_allowed(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(
            state_path=tmp_path / "state.json",
            cooldown_seconds=60,
        )

        d_eth = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="halt",
            now_ms=1000,
        )
        d_btc = deduper.should_send(
            symbol="BTC-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="halt",
            now_ms=1000,
        )

        assert d_eth.allowed is True
        assert d_btc.allowed is True
        assert d_eth.dedupe_key != d_btc.dedupe_key

        state = read_json_or_none(tmp_path / "state.json")
        assert len(state["entries"]) == 2


# ============================================================================
# 5. different event_type does not dedupe each other
# ============================================================================


class TestDifferentEventTypeNoDedupe:
    def test_different_event_type_both_allowed(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(
            state_path=tmp_path / "state.json",
            cooldown_seconds=60,
        )

        d1 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_STARTED",
            now_ms=1000,
        )
        d2 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_STOPPED",
            now_ms=1000,
        )

        assert d1.allowed is True
        assert d2.allowed is True
        assert d1.dedupe_key != d2.dedupe_key

        state = read_json_or_none(tmp_path / "state.json")
        assert len(state["entries"]) == 2


# ============================================================================
# 6. different severity does not dedupe each other
# ============================================================================


class TestDifferentSeverityNoDedupe:
    def test_different_severity_both_allowed(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(
            state_path=tmp_path / "state.json",
            cooldown_seconds=60,
        )

        d1 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="INFO",
            reason="halt",
            now_ms=1000,
        )
        d2 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="ERROR",
            reason="halt",
            now_ms=1000,
        )

        assert d1.allowed is True
        assert d2.allowed is True
        assert d1.dedupe_key != d2.dedupe_key

        state = read_json_or_none(tmp_path / "state.json")
        assert len(state["entries"]) == 2


# ============================================================================
# 7. different reason does not dedupe each other
# ============================================================================


class TestDifferentReasonNoDedupe:
    def test_different_reason_both_allowed(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(
            state_path=tmp_path / "state.json",
            cooldown_seconds=60,
        )

        d1 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="reason_a",
            now_ms=1000,
        )
        d2 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="reason_b",
            now_ms=1000,
        )

        assert d1.allowed is True
        assert d2.allowed is True
        assert d1.dedupe_key != d2.dedupe_key

        state = read_json_or_none(tmp_path / "state.json")
        assert len(state["entries"]) == 2


# ============================================================================
# 8. payload reason extraction
# ============================================================================


class TestPayloadReasonExtraction:
    def test_payload_reason_extracted(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            payload={"reason": "halted by guard"},
            now_ms=1000,
        )

        assert decision.allowed is True
        assert decision.reason == "halted by guard"
        # Key should contain hash of the reason, not the raw text.
        assert "halted by guard" not in decision.dedupe_key


# ============================================================================
# 9. payload halt_reason extraction
# ============================================================================


class TestPayloadHaltReasonExtraction:
    def test_payload_halt_reason_extracted(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            payload={"halt_reason": "rolling_loss"},
            now_ms=1000,
        )

        assert decision.allowed is True
        assert decision.reason == "rolling_loss"


# ============================================================================
# 10. payload error_type extraction
# ============================================================================


class TestPayloadErrorTypeExtraction:
    def test_payload_error_type_extracted(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_HEARTBEAT_WRITE_FAILED",
            severity="ERROR",
            payload={"error_type": "BAD_JSON"},
            now_ms=1000,
        )

        assert decision.allowed is True
        assert decision.reason == "BAD_JSON"


# ============================================================================
# 11. explicit reason takes precedence over payload reason
# ============================================================================


class TestExplicitReasonPrecedence:
    def test_explicit_reason_wins(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="explicit",
            payload={"reason": "payload"},
            now_ms=1000,
        )

        assert decision.reason == "explicit"


# ============================================================================
# 12. reason preview is truncated
# ============================================================================


class TestReasonPreviewTruncated:
    def test_reason_preview_truncated(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(
            state_path=tmp_path / "state.json",
            max_reason_chars=8,
        )

        deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="abcdefghijklmnop",
            now_ms=1000,
        )

        state = read_json_or_none(tmp_path / "state.json")
        key = next(iter(state["entries"]))
        assert state["entries"][key]["reason_preview"] == "abcdefgh"


# ============================================================================
# 13. dedupe key does not contain full long reason
# ============================================================================


class TestDedupeKeyNoFullReason:
    def test_dedupe_key_no_full_reason(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        long_reason = "this is a very long reason that should not appear in full in the dedupe key " * 10
        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason=long_reason,
            now_ms=1000,
        )

        assert long_reason not in decision.dedupe_key
        # Key should be: symbol|event_type|severity|16_char_hex_hash
        parts = decision.dedupe_key.split("|")
        assert len(parts) == 4
        assert len(parts[3]) == 16  # hex hash


# ============================================================================
# 14. cooldown_seconds zero always allows
# ============================================================================


class TestCooldownZeroAlwaysAllows:
    def test_cooldown_zero_always_allows(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(
            state_path=tmp_path / "state.json",
            cooldown_seconds=0,
        )

        d1 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="halt",
            now_ms=1000,
        )
        assert d1.allowed is True
        assert d1.next_allowed_ts_ms is None

        d2 = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="halt",
            now_ms=2000,
        )
        assert d2.allowed is True
        assert d2.next_allowed_ts_ms is None

        # send_count should have increased.
        state = read_json_or_none(tmp_path / "state.json")
        key = next(iter(state["entries"]))
        assert state["entries"][key]["send_count"] == 2


# ============================================================================
# 15. invalid state file treated as empty
# ============================================================================


class TestInvalidStateFileTreatedAsEmpty:
    def test_non_dict_state_treated_as_empty(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        # Write an array as state (not a dict).
        write_json_atomic(state_path, [1, 2, 3])

        deduper = AlertDeduper(state_path=state_path)
        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_STARTED",
            now_ms=1000,
        )

        # Must not raise and must allow.
        assert decision.allowed is True

        # State should be overwritten with valid format.
        state = read_json_or_none(state_path)
        assert isinstance(state, dict)
        assert "entries" in state

    def test_missing_state_file_ok(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "nonexistent" / "state.json")
        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_STARTED",
            now_ms=1000,
        )
        assert decision.allowed is True


# ============================================================================
# 16. invalid entries pruned
# ============================================================================


class TestInvalidEntriesPruned:
    def test_invalid_entries_pruned(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"

        # Pre-populate state with invalid entries.
        bad_state = {
            "version": 1,
            "updated_ts_ms": 1000,
            "entries": {
                "GOOD|KEY|INFO|abcd1234": {
                    "last_sent_ts_ms": 1000,
                    "symbol": "ETH-USDT-SWAP",
                    "event_type": "GOOD",
                    "severity": "INFO",
                    "reason_preview": None,
                    "send_count": 1,
                },
                "NOT_DICT|KEY|INFO|0000": "not_a_dict",
                "BAD_LAST_STR|KEY|INFO|1111": {
                    "last_sent_ts_ms": "bad",
                    "symbol": "X",
                    "event_type": "Y",
                    "severity": "INFO",
                    "reason_preview": None,
                    "send_count": 1,
                },
                "BAD_LAST_NEG|KEY|INFO|2222": {
                    "last_sent_ts_ms": -1,
                    "symbol": "X",
                    "event_type": "Y",
                    "severity": "INFO",
                    "reason_preview": None,
                    "send_count": 1,
                },
                "MAYBE_VALID|KEY|INFO|3333": {
                    "last_sent_ts_ms": 2000,
                    "symbol": "ETH-USDT-SWAP",
                    "event_type": "MAYBE_VALID",
                    "severity": "INFO",
                    "reason_preview": None,
                    "send_count": 1,
                },
            },
        }
        write_json_atomic(state_path, bad_state)

        deduper = AlertDeduper(state_path=state_path)
        deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_STARTED",
            now_ms=5000,
        )

        state = read_json_or_none(state_path)
        entries = state["entries"]
        keys = list(entries.keys())

        # Invalid entries must be gone.
        assert "NOT_DICT|KEY|INFO|0000" not in entries
        assert "BAD_LAST_STR|KEY|INFO|1111" not in entries
        assert "BAD_LAST_NEG|KEY|INFO|2222" not in entries

        # Valid entry should still exist.
        assert "MAYBE_VALID|KEY|INFO|3333" in entries


# ============================================================================
# 17. old entries pruned
# ============================================================================


class TestOldEntriesPruned:
    def test_old_entries_pruned(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"

        # Pre-populate with a very old entry.
        bad_state = {
            "version": 1,
            "updated_ts_ms": 0,
            "entries": {
                "OLD|KEY|INFO|aaaa": {
                    "last_sent_ts_ms": 1,
                    "symbol": "ETH-USDT-SWAP",
                    "event_type": "OLD",
                    "severity": "INFO",
                    "reason_preview": None,
                    "send_count": 1,
                },
            },
        }
        write_json_atomic(state_path, bad_state)

        deduper = AlertDeduper(
            state_path=state_path,
            cooldown_seconds=60,
        )
        # now_ms is far in the future relative to the old entry.
        # cutoff = now_ms - cooldown_seconds * 1000 * 4
        # = 1_000_000 - 60*1000*4 = 1_000_000 - 240_000 = 760_000
        # Old entry last_sent_ts_ms=1 is way below 760_000 → pruned.
        deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_STARTED",
            now_ms=1_000_000,
        )

        state = read_json_or_none(state_path)
        entries = state["entries"]

        # The old entry should be gone.
        assert "OLD|KEY|INFO|aaaa" not in entries
        # Only the current key should remain.
        assert len(entries) == 1


# ============================================================================
# 18. max_entries enforced
# ============================================================================


class TestMaxEntriesEnforced:
    def test_max_entries_enforced(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"

        # Pre-populate with 5 entries.
        entries = {}
        for i in range(5):
            key = f"ETH-USDT-SWAP|EVENT_{i}|INFO|hash{i:016d}"
            entries[key] = {
                "last_sent_ts_ms": 1000 + i,
                "symbol": "ETH-USDT-SWAP",
                "event_type": f"EVENT_{i}",
                "severity": "INFO",
                "reason_preview": None,
                "send_count": 1,
            }
        write_json_atomic(
            state_path,
            {"version": 1, "updated_ts_ms": 1000, "entries": entries},
        )

        deduper = AlertDeduper(
            state_path=state_path,
            max_entries=3,
        )
        deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="NEW_EVENT",
            severity="INFO",
            now_ms=5000,
        )

        state = read_json_or_none(state_path)
        assert len(state["entries"]) <= 3


# ============================================================================
# 19. current key is preserved during prune
# ============================================================================


class TestCurrentKeyPreserved:
    def test_current_key_preserved(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"

        # Pre-populate with many old entries.
        entries = {}
        for i in range(5):
            key = f"ETH-USDT-SWAP|EVENT_{i}|INFO|hash{i:016d}"
            entries[key] = {
                "last_sent_ts_ms": i,
                "symbol": "ETH-USDT-SWAP",
                "event_type": f"EVENT_{i}",
                "severity": "INFO",
                "reason_preview": None,
                "send_count": 1,
            }
        write_json_atomic(
            state_path,
            {"version": 1, "updated_ts_ms": 0, "entries": entries},
        )

        deduper = AlertDeduper(
            state_path=state_path,
            max_entries=1,
            cooldown_seconds=60,
        )
        deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="CURRENT_EVENT",
            severity="CRITICAL",
            reason="important",
            now_ms=1_000_000,
        )

        state = read_json_or_none(state_path)
        entries = state["entries"]
        assert len(entries) == 1
        # The only remaining key must be the current one.
        key = next(iter(entries))
        assert "CURRENT_EVENT" in key


# ============================================================================
# 20. now_ms before last_sent suppresses
# ============================================================================


class TestNowMsBeforeLastSentSuppresses:
    def test_clock_backwards_suppresses(self, tmp_path: Path) -> None:
        import hashlib

        state_path = tmp_path / "state.json"

        # Compute the dedupe key that matches reason="halt".
        reason_hash = hashlib.sha256(b"halt").hexdigest()[:16]
        dedupe_key = f"ETH-USDT-SWAP|WORKER_TRADING_HALTED|CRITICAL|{reason_hash}"

        # Pre-populate state with a key at ts 10000.
        entries = {
            dedupe_key: {
                "last_sent_ts_ms": 10000,
                "symbol": "ETH-USDT-SWAP",
                "event_type": "WORKER_TRADING_HALTED",
                "severity": "CRITICAL",
                "reason_preview": "halt",
                "send_count": 1,
            },
        }
        write_json_atomic(
            state_path,
            {"version": 1, "updated_ts_ms": 10000, "entries": entries},
        )

        deduper = AlertDeduper(
            state_path=state_path,
            cooldown_seconds=60,
        )
        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="halt",
            now_ms=5000,  # before last_sent_ts_ms
        )

        assert decision.allowed is False
        assert decision.last_sent_ts_ms == 10000


# ============================================================================
# 21. invalid constructor args
# ============================================================================


class TestInvalidConstructorArgs:
    def test_cooldown_seconds_negative(self) -> None:
        with pytest.raises(ValueError):
            AlertDeduper(state_path=Path("/tmp/x.json"), cooldown_seconds=-1)

    def test_cooldown_seconds_bool(self) -> None:
        with pytest.raises(ValueError):
            AlertDeduper(state_path=Path("/tmp/x.json"), cooldown_seconds=True)  # type: ignore[arg-type]

    def test_max_entries_zero(self) -> None:
        with pytest.raises(ValueError):
            AlertDeduper(state_path=Path("/tmp/x.json"), max_entries=0)

    def test_max_entries_negative(self) -> None:
        with pytest.raises(ValueError):
            AlertDeduper(state_path=Path("/tmp/x.json"), max_entries=-1)

    def test_max_entries_bool(self) -> None:
        with pytest.raises(ValueError):
            AlertDeduper(state_path=Path("/tmp/x.json"), max_entries=True)  # type: ignore[arg-type]

    def test_max_reason_chars_zero(self) -> None:
        with pytest.raises(ValueError):
            AlertDeduper(state_path=Path("/tmp/x.json"), max_reason_chars=0)

    def test_max_reason_chars_negative(self) -> None:
        with pytest.raises(ValueError):
            AlertDeduper(state_path=Path("/tmp/x.json"), max_reason_chars=-1)

    def test_max_reason_chars_bool(self) -> None:
        with pytest.raises(ValueError):
            AlertDeduper(state_path=Path("/tmp/x.json"), max_reason_chars=True)  # type: ignore[arg-type]

    def test_state_path_not_path(self) -> None:
        with pytest.raises(ValueError):
            AlertDeduper(state_path="not a path")  # type: ignore[arg-type]


# ============================================================================
# 22. invalid should_send args
# ============================================================================


class TestInvalidShouldSendArgs:
    def test_symbol_empty(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="", event_type="X", now_ms=1000)

    def test_symbol_whitespace(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="   ", event_type="X", now_ms=1000)

    def test_symbol_none(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol=None, event_type="X", now_ms=1000)  # type: ignore[arg-type]

    def test_event_type_empty(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="ETH", event_type="", now_ms=1000)

    def test_event_type_whitespace(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="ETH", event_type="   ", now_ms=1000)

    def test_event_type_none(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="ETH", event_type=None, now_ms=1000)  # type: ignore[arg-type]

    def test_severity_empty(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="ETH", event_type="X", severity="", now_ms=1000)

    def test_severity_whitespace(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="ETH", event_type="X", severity="   ", now_ms=1000)

    def test_reason_not_str(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="ETH", event_type="X", reason=123, now_ms=1000)  # type: ignore[arg-type]

    def test_payload_not_dict(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="ETH", event_type="X", payload=[], now_ms=1000)  # type: ignore[arg-type]

    def test_now_ms_negative(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="ETH", event_type="X", now_ms=-1)

    def test_now_ms_bool(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")
        with pytest.raises(ValueError):
            deduper.should_send(symbol="ETH", event_type="X", now_ms=True)  # type: ignore[arg-type]


# ============================================================================
# 23. severity normalized uppercase
# ============================================================================


class TestSeverityNormalizedUppercase:
    def test_severity_normalized_uppercase(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="warning",
            reason="test",
            now_ms=1000,
        )

        state = read_json_or_none(tmp_path / "state.json")
        key = next(iter(state["entries"]))
        assert state["entries"][key]["severity"] == "WARNING"
        # The key should also use uppercase severity.
        assert "WARNING" in key


# ============================================================================
# 24. symbol and event_type stripped
# ============================================================================


class TestSymbolAndEventTypeStripped:
    def test_symbol_and_event_type_stripped(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        deduper.should_send(
            symbol=" ETH-USDT-SWAP ",
            event_type=" WORKER_STARTED ",
            now_ms=1000,
        )

        state = read_json_or_none(tmp_path / "state.json")
        key = next(iter(state["entries"]))
        entry = state["entries"][key]
        assert entry["symbol"] == "ETH-USDT-SWAP"
        assert entry["event_type"] == "WORKER_STARTED"


# ============================================================================
# 25. state written atomically
# ============================================================================


class TestStateWrittenAtomically:
    def test_uses_write_json_atomic(self) -> None:
        source = _ALERT_DEDUPER_SOURCE.read_text(encoding="utf-8")
        assert "write_json_atomic" in source, (
            "alert_deduper.py must use write_json_atomic for state writes"
        )

    def test_uses_read_json_or_none(self) -> None:
        source = _ALERT_DEDUPER_SOURCE.read_text(encoding="utf-8")
        assert "read_json_or_none" in source, (
            "alert_deduper.py must use read_json_or_none for state reads"
        )


# ============================================================================
# 26. source guard
# ============================================================================


class TestAlertDeduperSourceGuard:
    def test_no_forbidden_imports(self) -> None:
        source = _ALERT_DEDUPER_SOURCE.read_text(encoding="utf-8")

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
            "asyncio",
            "ChildEventReader",
            "WorkerEventEmitter",
            "JsonlOutbox",
            "src.live.workers",
            "src.trader",
            "src.strategies",
            "src.live.symbol_worker_app",
            "src.live.symbol_worker_factory",
        ]
        for token in forbidden:
            assert token not in source, (
                f"alert_deduper.py must not import/use {token}"
            )


# ============================================================================
# 27. no append / JSONL / unbounded history guard
# ============================================================================


class TestNoAppendOrUnboundedHistory:
    def test_no_open_append(self) -> None:
        source = _ALERT_DEDUPER_SOURCE.read_text(encoding="utf-8")
        assert '.open("a"' not in source, (
            "alert_deduper.py must not use append-mode file open"
        )

    def test_no_jsonl_outbox(self) -> None:
        source = _ALERT_DEDUPER_SOURCE.read_text(encoding="utf-8")
        assert "JsonlOutbox" not in source, (
            "alert_deduper.py must not use JsonlOutbox"
        )

    def test_no_suppression_history(self) -> None:
        source = _ALERT_DEDUPER_SOURCE.read_text(encoding="utf-8")
        assert "suppression_history" not in source, (
            "alert_deduper.py must not maintain suppression_history"
        )

    def test_no_history_field(self) -> None:
        source = _ALERT_DEDUPER_SOURCE.read_text(encoding="utf-8")
        assert "history" not in source, (
            "alert_deduper.py must not maintain a history field"
        )


# ============================================================================
# Additional: send_count increments correctly
# ============================================================================


class TestSendCountIncrements:
    def test_send_count_increments_on_allowed(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(
            state_path=tmp_path / "state.json",
            cooldown_seconds=1,
        )

        # First.
        deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="halt",
            now_ms=1000,
        )

        # After cooldown.
        deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            reason="halt",
            now_ms=2000,
        )

        state = read_json_or_none(tmp_path / "state.json")
        key = next(iter(state["entries"]))
        assert state["entries"][key]["send_count"] == 2


# ============================================================================
# Additional: reason_preview is None when no reason
# ============================================================================


class TestReasonPreviewNone:
    def test_reason_preview_none_when_no_reason(self, tmp_path: Path) -> None:
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_STARTED",
            now_ms=1000,
        )

        assert decision.reason is None

        state = read_json_or_none(tmp_path / "state.json")
        key = next(iter(state["entries"]))
        assert state["entries"][key]["reason_preview"] is None


# ============================================================================
# Additional: multiple reason source extraction order
# ============================================================================


class TestReasonSourcePriority:
    def test_halt_reason_over_error_type(self, tmp_path: Path) -> None:
        """halt_reason should take priority over error_type."""
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            payload={"halt_reason": "halted", "error_type": "SOME_ERROR"},
            now_ms=1000,
        )

        assert decision.reason == "halted"

    def test_reason_over_halt_reason(self, tmp_path: Path) -> None:
        """payload['reason'] should take priority over payload['halt_reason']."""
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            payload={"reason": "primary", "halt_reason": "secondary"},
            now_ms=1000,
        )

        assert decision.reason == "primary"

    def test_reason_source_whitespace_treated_as_none(self, tmp_path: Path) -> None:
        """Whitespace-only reason in payload should be treated as None."""
        deduper = AlertDeduper(state_path=tmp_path / "state.json")

        decision = deduper.should_send(
            symbol="ETH-USDT-SWAP",
            event_type="WORKER_TRADING_HALTED",
            severity="CRITICAL",
            payload={"reason": "   "},
            now_ms=1000,
        )

        assert decision.reason is None
        assert decision.dedupe_key.endswith("|-")
