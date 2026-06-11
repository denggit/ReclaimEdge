#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G08 tests for PaperTrader — dry-run BTC simulator.

These tests verify:
1. PaperTrader starts/initializes without OKX env or LIVE_TRADING=true.
2. fetch_usdt_equity returns PAPER_ACCOUNT_EQUITY_USDT.
3. fetch_position_snapshot returns flat initially.
4. execute_intent OPEN_LONG → entry_filled=True, ok=True, position_contracts > 0.
5. replace_take_profit returns fake tp id, never calls OKX.
6. execute_market_exit_runner → mark flat.
7. eth_qty_to_contracts uses BTC metadata (multiplier=0.01, precision=0.01).
8. Below min_contracts → RuntimeError.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.execution.paper_trader import PaperTrader, PaperTraderConfig, _parse_paper_symbols
from src.execution.trader import LiveTradeResult, PositionSnapshot
from src.strategies.boll_cvd_reclaim_strategy import PositionSide


# ── helpers ────────────────────────────────────────────────────────────────


class _FakeIntent:
    """Minimal fake TradeIntent for paper trader tests."""

    def __init__(
        self,
        intent_type: str = "OPEN_LONG",
        side: PositionSide = "LONG",
        eth_qty: float = 0.1,
        tp_price: float = 50000.0,
        layer_index: int = 1,
    ) -> None:
        self.intent_type = intent_type
        self.side = side
        self.size = _FakeSize(eth_qty)
        self.tp_price = tp_price
        self.layer_index = layer_index
        self.middle_bucket_split_order_input = None
        self.three_stage_runner_active = False
        self.three_stage_tp1_price = None
        self.three_stage_tp2_price = None


class _FakeSize:
    def __init__(self, eth_qty: float) -> None:
        self.eth_qty = eth_qty


# ── env helper ─────────────────────────────────────────────────────────────


def _paper_env(
    *,
    inst_id: str = "BTC-USDT-SWAP",
    paper_symbols: str = "BTC-USDT-SWAP",
    equity: str = "1000",
) -> dict[str, str]:
    return {
        "OKX_INST_ID": inst_id,
        "RECLAIM_PAPER_SYMBOLS": paper_symbols,
        "PAPER_ACCOUNT_EQUITY_USDT": equity,
        "LIVE_TRADING": "false",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. PaperTraderConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderConfig:
    def test_config_defaults(self) -> None:
        config = PaperTraderConfig(symbol="BTC-USDT-SWAP", account_equity_usdt=1000.0)
        assert config.symbol == "BTC-USDT-SWAP"
        assert config.account_equity_usdt == 1000.0
        assert config.contract_multiplier == Decimal("0.01")
        assert config.contract_precision == Decimal("0.01")
        assert config.min_contracts == Decimal("0.01")

    def test_config_is_frozen(self) -> None:
        config = PaperTraderConfig(symbol="BTC-USDT-SWAP", account_equity_usdt=500.0)
        with pytest.raises(Exception):
            config.symbol = "ETH-USDT-SWAP"  # type: ignore[misc]

    def test_config_custom_values(self) -> None:
        config = PaperTraderConfig(
            symbol="BTC-USDT-SWAP",
            account_equity_usdt=5000.0,
            td_mode="cross",
            leverage="10",
        )
        assert config.td_mode == "cross"
        assert config.leverage == "10"


# ═══════════════════════════════════════════════════════════════════════════
# 2. parse_paper_symbols helper
# ═══════════════════════════════════════════════════════════════════════════


class TestParsePaperSymbols:
    def test_default(self) -> None:
        assert _parse_paper_symbols(None) == ("BTC-USDT-SWAP",)

    def test_empty_string(self) -> None:
        assert _parse_paper_symbols("") == ("BTC-USDT-SWAP",)

    def test_single(self) -> None:
        assert _parse_paper_symbols("BTC-USDT-SWAP") == ("BTC-USDT-SWAP",)

    def test_multiple_deduplicates(self) -> None:
        result = _parse_paper_symbols("BTC-USDT-SWAP,ETH-USDT-SWAP,BTC-USDT-SWAP")
        assert result == ("BTC-USDT-SWAP", "ETH-USDT-SWAP")

    def test_strips_whitespace(self) -> None:
        result = _parse_paper_symbols(" BTC-USDT-SWAP , ETH-USDT-SWAP ")
        assert result == ("BTC-USDT-SWAP", "ETH-USDT-SWAP")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PaperTrader construction — no OKX requirements
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderConstruction:
    def test_constructs_with_live_trading_false(self) -> None:
        """PaperTrader must construct even with LIVE_TRADING=false."""
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            assert pt.live_trading is False
            assert pt.paper_trading is True
            assert pt.symbol == "BTC-USDT-SWAP"

    def test_constructs_without_okx_api_key(self) -> None:
        """PaperTrader must construct without OKX_API_KEY set."""
        env = _paper_env()
        env.pop("OKX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            pt = PaperTrader()
            assert pt.symbol == "BTC-USDT-SWAP"

    def test_rejects_non_btc_symbol(self) -> None:
        """PaperTrader must reject ETH-USDT-SWAP."""
        with patch.dict(os.environ, _paper_env(inst_id="ETH-USDT-SWAP"), clear=True):
            with pytest.raises(RuntimeError, match="only supports BTC-USDT-SWAP"):
                PaperTrader()

    def test_rejects_symbol_not_in_paper_list(self) -> None:
        """PaperTrader must reject when symbol not in RECLAIM_PAPER_SYMBOLS."""
        env = _paper_env(paper_symbols="SOL-USDT-SWAP")
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="not in RECLAIM_PAPER_SYMBOLS"):
                PaperTrader()

    def test_btc_metadata_is_correct(self) -> None:
        """PaperTrader must use BTC contract metadata (0.01 / 0.01 / 0.01)."""
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            assert pt.contract_multiplier == Decimal("0.01")
            assert pt.contract_precision == Decimal("0.01")
            assert pt.min_contracts == Decimal("0.01")


# ═══════════════════════════════════════════════════════════════════════════
# 4. lifecycle (start / initialize / close)
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_close_noop(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            await pt.start()
            await pt.close()
            # Must not raise.

    @pytest.mark.asyncio
    async def test_initialize_sets_equity(self) -> None:
        with patch.dict(os.environ, _paper_env(equity="5000"), clear=True):
            pt = PaperTrader()
            await pt.initialize()
            assert pt.account_equity_usdt == 5000.0
            assert pt.position_contracts == Decimal("0")


# ═══════════════════════════════════════════════════════════════════════════
# 5. fetch_usdt_equity
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderFetchEquity:
    @pytest.mark.asyncio
    async def test_returns_paper_equity(self) -> None:
        with patch.dict(os.environ, _paper_env(equity="2500"), clear=True):
            pt = PaperTrader()
            equity = await pt.fetch_usdt_equity()
            assert equity == 2500.0

    @pytest.mark.asyncio
    async def test_default_equity(self) -> None:
        env = _paper_env()
        env.pop("PAPER_ACCOUNT_EQUITY_USDT", None)
        with patch.dict(os.environ, env, clear=True):
            pt = PaperTrader()
            equity = await pt.fetch_usdt_equity()
            assert equity == 1000.0  # default


# ═══════════════════════════════════════════════════════════════════════════
# 6. fetch_position_snapshot
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderPositionSnapshot:
    @pytest.mark.asyncio
    async def test_initial_flat(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            snap = await pt.fetch_position_snapshot()
            assert snap.side is None
            assert snap.contracts == Decimal("0")
            assert snap.has_position is False

    @pytest.mark.asyncio
    async def test_after_open_long(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            intent = _FakeIntent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.1)
            await pt.execute_intent(intent)
            snap = await pt.fetch_position_snapshot()
            assert snap.side == "LONG"
            assert snap.contracts > 0
            assert snap.has_position is True
            # eth_qty = contracts * 0.01
            assert snap.eth_qty > 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. execute_intent
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderExecuteIntent:
    @pytest.mark.asyncio
    async def test_open_long_entry_filled(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            intent = _FakeIntent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.1)
            result = await pt.execute_intent(intent)
            assert result.ok is True
            assert result.entry_filled is True
            assert result.tp_ok is True
            assert result.order_id is not None
            assert result.order_id.startswith("paper-entry-")
            assert pt.position_contracts > 0

    @pytest.mark.asyncio
    async def test_open_short_entry_filled(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            intent = _FakeIntent(intent_type="OPEN_SHORT", side="SHORT", eth_qty=0.2)
            result = await pt.execute_intent(intent)
            assert result.ok is True
            assert result.entry_filled is True
            assert pt.position_contracts > 0

    @pytest.mark.asyncio
    async def test_add_long_increases_position(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            intent1 = _FakeIntent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.1, layer_index=1)
            await pt.execute_intent(intent1)
            contracts_after_first = pt.position_contracts
            intent2 = _FakeIntent(intent_type="ADD_LONG", side="LONG", eth_qty=0.05, layer_index=2)
            await pt.execute_intent(intent2)
            assert pt.position_contracts > contracts_after_first

    @pytest.mark.asyncio
    async def test_tp_order_id_generated(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            intent = _FakeIntent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.1)
            result = await pt.execute_intent(intent)
            assert result.tp_order_id is not None
            assert result.tp_order_id.startswith("paper-tp-")
            assert pt.tp_order_id is not None


# ═══════════════════════════════════════════════════════════════════════════
# 8. replace_take_profit
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderReplaceTakeProfit:
    @pytest.mark.asyncio
    async def test_returns_fake_tp_id(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            intent = _FakeIntent(intent_type="UPDATE_TP", side="LONG", eth_qty=0.1)
            result = await pt.replace_take_profit(intent)
            assert result.ok is True
            assert result.action == "UPDATE_TP"
            assert result.tp_order_id is not None
            assert result.tp_order_id.startswith("paper-tp-")
            assert pt.tp_order_id == result.tp_order_id

    @pytest.mark.asyncio
    async def test_no_okx_call(self) -> None:
        """replace_take_profit must not import or call OKX client or trader module."""
        import src.execution.paper_trader as pt_module

        source = pt_module.__file__
        if source:
            with open(source) as f:
                lines = f.readlines()
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("from ") or stripped.startswith("import "):
                    assert "OkxPrivateClient" not in stripped
                    assert "aiohttp" not in stripped
                    assert "PrivateWriteRateLimiter" not in stripped
                    # Only match "src.execution.trader" when it is NOT followed
                    # by "_types" (which is trader_types, the shared DTO module).
                    if "src.execution.trader" in stripped and "src.execution.trader_types" not in stripped:
                        raise AssertionError(
                            f"PaperTrader must not import from src.execution.trader: {stripped!r}"
                        )


# ═══════════════════════════════════════════════════════════════════════════
# 9. execute_near_tp_reduce
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderNearTpReduce:
    @pytest.mark.asyncio
    async def test_reduces_contracts(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            # Build a position first
            intent1 = _FakeIntent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.2)
            await pt.execute_intent(intent1)
            before = pt.position_contracts

            reduce_intent = _FakeIntent(intent_type="NEAR_TP_REDUCE", side="LONG", eth_qty=0.05)
            result = await pt.execute_intent(reduce_intent)
            assert result.ok is True
            assert result.reduce_filled is True
            assert pt.position_contracts < before

    @pytest.mark.asyncio
    async def test_reduce_cannot_go_below_zero(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            intent1 = _FakeIntent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.01)
            await pt.execute_intent(intent1)
            reduce_intent = _FakeIntent(intent_type="NEAR_TP_REDUCE", side="LONG", eth_qty=1.0)
            await pt.execute_intent(reduce_intent)
            assert pt.position_contracts >= 0


# ═══════════════════════════════════════════════════════════════════════════
# 10. execute_market_exit_runner
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderMarketExitRunner:
    @pytest.mark.asyncio
    async def test_exit_flattens_position(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            intent1 = _FakeIntent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.1)
            await pt.execute_intent(intent1)
            assert pt.position_contracts > 0

            exit_intent = _FakeIntent(intent_type="MARKET_EXIT_RUNNER", side="LONG", eth_qty=0.1)
            result = await pt.execute_intent(exit_intent)
            assert result.ok is True
            assert result.near_tp_exit_all is True
            assert pt.position_contracts == 0
            assert pt.tp_order_id is None

    @pytest.mark.asyncio
    async def test_mark_flat_clears_all(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            pt.tp_order_id = "fake-tp"
            pt.near_tp_protective_sl_order_id = "fake-sl"
            pt.mark_flat()
            assert pt.position_contracts == 0
            assert pt.tp_order_id is None
            assert pt.near_tp_protective_sl_order_id is None


# ═══════════════════════════════════════════════════════════════════════════
# 11. eth_qty_to_contracts
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderEthQtyToContracts:
    def test_btc_metadata_calculation(self) -> None:
        """With BTC metadata (multiplier=0.01), eth_qty=0.1 → 10 contracts."""
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            contracts = pt.eth_qty_to_contracts(Decimal("0.1"))
            assert contracts == Decimal("10")  # 0.1 / 0.01

    def test_rounds_down_to_precision(self) -> None:
        """Contract calculation must round down to precision.
        eth_qty=0.00155 / 0.01 = 0.155 → floor(15.5) * 0.01 = 0.15."""
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            # 0.00155 / 0.01 = 0.155, floor(15.5)*0.01 = 0.15
            contracts = pt.eth_qty_to_contracts(Decimal("0.00155"))
            assert contracts == Decimal("0.15")

    def test_below_min_contracts_raises(self) -> None:
        """Less than min_contracts (0.01) must raise RuntimeError."""
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            with pytest.raises(RuntimeError, match="below minimum"):
                pt.eth_qty_to_contracts(Decimal("0.00005"))  # 0.00005/0.01 = 0.005 < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# 12. decimal / price formatting
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderFormatting:
    def test_decimal_to_str(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            assert pt.decimal_to_str(Decimal("1.5")) == "1.5"
            assert pt.decimal_to_str(10) == "10"
            assert pt.decimal_to_str("0.01") == "0.01"

    def test_price_to_str(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            assert pt.price_to_str(50000.123) == "50000.12"

    def test_price_to_str_invalid(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            with pytest.raises(RuntimeError, match="Invalid price"):
                pt.price_to_str(float("inf"))


# ═══════════════════════════════════════════════════════════════════════════
# 13. sidecar no-ops
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderSidecar:
    @pytest.mark.asyncio
    async def test_place_sidecar_market_order(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            result = await pt.place_sidecar_market_order(side="LONG", eth_qty=0.1)
            assert "order_id" in result
            assert result["order_id"].startswith("paper-sidecar-")
            assert "contracts" in result

    @pytest.mark.asyncio
    async def test_place_sidecar_fixed_tp(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            tp_id = await pt.place_sidecar_fixed_take_profit(
                side="LONG", contracts="1", tp_price=50000.0
            )
            assert tp_id.startswith("paper-sidecar-tp-")

    @pytest.mark.asyncio
    async def test_cancel_sidecar_tp(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            ok = await pt.cancel_sidecar_take_profit("any-id")
            assert ok is True

    @pytest.mark.asyncio
    async def test_fetch_sidecar_order_status(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            status = await pt.fetch_sidecar_order_status("any-id")
            assert status["status"] == "OPEN"


# ═══════════════════════════════════════════════════════════════════════════
# 14. forbidden imports guard
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderNoForbiddenImports:
    def test_no_okx_client_import(self) -> None:
        import src.execution.paper_trader as pt_module

        source_file = pt_module.__file__
        assert source_file is not None
        with open(source_file) as f:
            lines = f.readlines()
        # Check actual import lines, not docstring mentions
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("from ") or stripped.startswith("import "):
                assert "OkxPrivateClient" not in stripped, (
                    f"PaperTrader must not import OkxPrivateClient: {stripped!r}"
                )
                assert "OKX_CONFIG" not in stripped, (
                    f"PaperTrader must not import OKX_CONFIG: {stripped!r}"
                )
                assert "aiohttp" not in stripped, (
                    f"PaperTrader must not import aiohttp: {stripped!r}"
                )
                assert "PrivateWriteRateLimiter" not in stripped, (
                    f"PaperTrader must not import PrivateWriteRateLimiter: {stripped!r}"
                )

    def test_no_live_trading_false_error(self) -> None:
        """PaperTrader must not raise an error because LIVE_TRADING=false."""
        import src.execution.paper_trader as pt_module

        source_file = pt_module.__file__
        assert source_file is not None
        with open(source_file) as f:
            content = f.read()
        # Must not have the Trader live_trading gate
        assert "LIVE_TRADING is not true" not in content, (
            "PaperTrader must not refuse to initialize due to LIVE_TRADING=false"
        )

    def test_does_not_import_trader_module(self) -> None:
        """PaperTrader must NOT import from src.execution.trader (G08b fix).
        Importing from trader_types is fine — only an import from the live
        trader module (which pulls in OKX deps) is forbidden."""
        import src.execution.paper_trader as pt_module

        source_file = pt_module.__file__
        assert source_file is not None
        with open(source_file) as f:
            content = f.read()
        # Must not import from src.execution.trader (the live module).
        # "from src.execution.trader import" catches the old import.
        # "from src.execution.trader\n" catches a bare import line without _types.
        assert "from src.execution.trader import" not in content, (
            "PaperTrader must not import from src.execution.trader"
        )
        # Also verify the import from trader_types is present
        assert "from src.execution.trader_types import" in content, (
            "PaperTrader must import from src.execution.trader_types"
        )

    def test_imports_from_trader_types(self) -> None:
        """PaperTrader must import DTOs from trader_types (G08b fix)."""
        import src.execution.paper_trader as pt_module

        source_file = pt_module.__file__
        assert source_file is not None
        with open(source_file) as f:
            content = f.read()
        assert "from src.execution.trader_types import" in content, (
            "PaperTrader must import from src.execution.trader_types"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 15. protective stop no-ops
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderProtectiveStops:
    @pytest.mark.asyncio
    async def test_place_near_tp_protective_stop(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            ok, oid, err = await pt.place_near_tp_protective_stop_with_retries(
                side="LONG", contracts=Decimal("1"), stop_price=49000.0,
                retry_count=3, retry_interval_seconds=0.1,
            )
            assert ok is True
            assert oid is not None
            assert oid.startswith("paper-protective-sl-")

    @pytest.mark.asyncio
    async def test_cancel_protective_stops(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            pt = PaperTrader()
            assert await pt.cancel_near_tp_protective_stop("any") is True
            assert await pt.cancel_middle_runner_protective_stop("any") is True
            assert await pt.cancel_middle_bucket_fast_protective_stop("any") is True
            assert await pt.cancel_trend_runner_protective_stop("any") is True
            assert await pt.cancel_three_stage_post_tp1_protective_stop("any") is True
