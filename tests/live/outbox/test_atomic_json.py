from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live.outbox.atomic_json import read_json_or_none, write_json_atomic


class TestWriteJsonAtomicCreatesParentAndFile:
    def test_write_json_atomic_creates_parent_and_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "state.json"
        data = {"a": 1, "中文": "测试"}
        write_json_atomic(path, data)

        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == data


class TestWriteJsonAtomicReplacesExistingFile:
    def test_write_json_atomic_replaces_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        write_json_atomic(path, {"old": True})
        write_json_atomic(path, {"new": True})

        loaded = read_json_or_none(path)
        assert loaded == {"new": True}


class TestReadJsonOrNoneMissingReturnsNone:
    def test_read_json_or_none_missing_returns_none(self, tmp_path: Path) -> None:
        assert read_json_or_none(tmp_path / "missing.json") is None


class TestReadJsonOrNoneReadsJson:
    def test_read_json_or_none_reads_json(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        write_json_atomic(path, {"key": "value"})
        assert read_json_or_none(path) == {"key": "value"}


class TestWriteJsonAtomicCleansTempFileOnDumpError:
    def test_write_json_atomic_cleans_temp_file_on_dump_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "bad.json"

        with pytest.raises(TypeError):
            write_json_atomic(path, {"bad": object()})

        # Target file should not exist since write never succeeded.
        assert not path.exists()

        # No .tmp residue in parent directory.
        parent = path.parent
        tmp_files = list(parent.glob(f".{path.name}.*.tmp"))
        assert len(tmp_files) == 0, f"temp residue: {tmp_files}"


class TestAtomicJsonSourceHasNoRuntimeSideEffectImports:
    def test_atomic_json_source_has_no_runtime_side_effect_imports(self) -> None:
        source_path = Path(__file__).parents[3] / "src" / "live" / "outbox" / "atomic_json.py"
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
        ]
        for token in forbidden:
            assert token not in source, f"atomic_json.py must not import/use {token}"
