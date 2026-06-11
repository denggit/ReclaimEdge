from __future__ import annotations

from pathlib import Path

import pytest

from src.live.supervisor.symbol_worker_plan import (
    SUPPORTED_SUPERVISOR_SYMBOLS,
    build_symbol_worker_plans,
    parse_worker_modes,
    validate_supported_supervisor_symbol,
    worker_mode_for_symbol,
)


def test_supported_supervisor_symbols_are_eth_and_btc_only() -> None:
    assert SUPPORTED_SUPERVISOR_SYMBOLS == frozenset({
        "ETH-USDT-SWAP",
        "BTC-USDT-SWAP",
    })


def test_validate_supported_supervisor_symbol_allows_eth() -> None:
    assert validate_supported_supervisor_symbol("ETH-USDT-SWAP") == "ETH-USDT-SWAP"


def test_validate_supported_supervisor_symbol_allows_btc() -> None:
    assert validate_supported_supervisor_symbol("BTC-USDT-SWAP") == "BTC-USDT-SWAP"


def test_validate_supported_supervisor_symbol_trims_whitespace() -> None:
    assert validate_supported_supervisor_symbol(" BTC-USDT-SWAP ") == "BTC-USDT-SWAP"


def test_validate_supported_supervisor_symbol_rejects_sol() -> None:
    with pytest.raises(ValueError, match="unsupported supervisor symbol"):
        validate_supported_supervisor_symbol("SOL-USDT-SWAP")


def test_validate_supported_supervisor_symbol_rejects_doge() -> None:
    with pytest.raises(ValueError, match="unsupported supervisor symbol"):
        validate_supported_supervisor_symbol("DOGE-USDT-SWAP")


def test_unsupported_supervisor_symbol_error_contains_symbol_and_supported() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_supported_supervisor_symbol("SOL-USDT-SWAP")

    message = str(exc_info.value)
    assert "SOL-USDT-SWAP" in message
    assert "ETH-USDT-SWAP" in message
    assert "BTC-USDT-SWAP" in message


def test_parse_worker_modes_live_for_eth_and_btc() -> None:
    modes = parse_worker_modes(
        {"RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:live,BTC-USDT-SWAP:live"}
    )
    assert modes == {
        "ETH-USDT-SWAP": "live",
        "BTC-USDT-SWAP": "live",
    }


def test_parse_worker_modes_trims_whitespace_and_lowercases_mode() -> None:
    modes = parse_worker_modes(
        {"RECLAIM_WORKER_MODES": " ETH-USDT-SWAP : LIVE , BTC-USDT-SWAP : paper "}
    )
    assert modes == {
        "ETH-USDT-SWAP": "live",
        "BTC-USDT-SWAP": "paper",
    }


def test_worker_mode_defaults_to_live() -> None:
    assert worker_mode_for_symbol("ETH-USDT-SWAP", {}) == "live"


def test_worker_mode_uses_reclaim_worker_mode_fallback() -> None:
    assert worker_mode_for_symbol(
        "BTC-USDT-SWAP",
        {"RECLAIM_WORKER_MODE": " paper "},
    ) == "paper"


def test_parse_worker_modes_duplicate_symbol_raises() -> None:
    with pytest.raises(ValueError, match="duplicate symbol"):
        parse_worker_modes(
            {"RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:live,ETH-USDT-SWAP:paper"}
        )


def test_parse_worker_modes_empty_symbol_raises() -> None:
    with pytest.raises(ValueError, match="symbol must not be empty"):
        parse_worker_modes({"RECLAIM_WORKER_MODES": ":live"})


def test_parse_worker_modes_empty_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must not be empty"):
        parse_worker_modes({"RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:"})


def test_parse_worker_modes_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        parse_worker_modes({"RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:dry-run"})


def test_build_symbol_worker_plans_isolates_child_env_and_paths(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    plans = build_symbol_worker_plans(
        ["ETH-USDT-SWAP", "BTC-USDT-SWAP"],
        base_env={
            "RECLAIM_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "RECLAIM_WORKER_MODES": "ETH-USDT-SWAP:live,BTC-USDT-SWAP:live",
        },
        runtime_dir=runtime_dir,
        heartbeat_dir=runtime_dir / "heartbeats",
        event_dir=runtime_dir / "events",
    )

    eth, btc = plans

    assert eth.symbol == "ETH-USDT-SWAP"
    assert eth.child_env["OKX_INST_ID"] == "ETH-USDT-SWAP"
    assert eth.child_env["RECLAIM_SYMBOL"] == "ETH-USDT-SWAP"
    assert eth.child_env["RECLAIM_SYMBOLS"] == "ETH-USDT-SWAP"
    assert eth.child_env["RECLAIM_WORKER_MODE"] == "live"

    assert btc.symbol == "BTC-USDT-SWAP"
    assert btc.child_env["OKX_INST_ID"] == "BTC-USDT-SWAP"
    assert btc.child_env["RECLAIM_SYMBOL"] == "BTC-USDT-SWAP"
    assert btc.child_env["RECLAIM_SYMBOLS"] == "BTC-USDT-SWAP"
    assert btc.child_env["RECLAIM_WORKER_MODE"] == "live"

    assert "ETH-USDT-SWAP,BTC-USDT-SWAP" not in {
        eth.child_env["RECLAIM_SYMBOLS"],
        btc.child_env["RECLAIM_SYMBOLS"],
    }
    assert eth.child_name != btc.child_name
    assert eth.heartbeat_path != btc.heartbeat_path
    assert eth.event_outbox_path != btc.event_outbox_path


def test_build_symbol_worker_plans_rejects_unsupported_symbol(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"

    with pytest.raises(ValueError, match="unsupported supervisor symbol"):
        build_symbol_worker_plans(
            ["ETH-USDT-SWAP", "SOL-USDT-SWAP"],
            base_env={},
            runtime_dir=runtime_dir,
            heartbeat_dir=runtime_dir / "heartbeats",
            event_dir=runtime_dir / "events",
        )


def test_btc_worker_mode_can_remain_live(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"

    plan = build_symbol_worker_plans(
        ["BTC-USDT-SWAP"],
        base_env={"RECLAIM_WORKER_MODES": "BTC-USDT-SWAP:live"},
        runtime_dir=runtime_dir,
        heartbeat_dir=runtime_dir / "heartbeats",
        event_dir=runtime_dir / "events",
    )[0]

    assert plan.symbol == "BTC-USDT-SWAP"
    assert plan.worker_mode == "live"
    assert plan.child_env["RECLAIM_WORKER_MODE"] == "live"


def test_btc_worker_mode_can_be_configured_to_paper(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"

    plan = build_symbol_worker_plans(
        ["BTC-USDT-SWAP"],
        base_env={"RECLAIM_WORKER_MODES": "BTC-USDT-SWAP:paper"},
        runtime_dir=runtime_dir,
        heartbeat_dir=runtime_dir / "heartbeats",
        event_dir=runtime_dir / "events",
    )[0]

    assert plan.symbol == "BTC-USDT-SWAP"
    assert plan.worker_mode == "paper"
    assert plan.child_env["RECLAIM_WORKER_MODE"] == "paper"
