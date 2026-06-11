#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G08b tests for trader_types backward compatibility.

These tests verify:
1. Trader still re-exports LiveTradeResult / PositionSnapshot / TraderInstrumentMetadata.
2. The re-exported names are identical objects (not copies).
3. trader_types module does NOT import OKX / aiohttp / PrivateWriteRateLimiter.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════
# 1. Backward-compat: Trader re-exports trader_types DTOs
# ═══════════════════════════════════════════════════════════════════════════


class TestTraderReExport:
    def test_live_trade_result_is_same_object(self) -> None:
        from src.execution.trader import LiveTradeResult
        from src.execution.trader_types import LiveTradeResult as TLiveTradeResult  # noqa: N813

        assert LiveTradeResult is TLiveTradeResult, (
            "Trader.LiveTradeResult must be the same object as trader_types.LiveTradeResult"
        )

    def test_position_snapshot_is_same_object(self) -> None:
        from src.execution.trader import PositionSnapshot
        from src.execution.trader_types import PositionSnapshot as TPositionSnapshot  # noqa: N813

        assert PositionSnapshot is TPositionSnapshot, (
            "Trader.PositionSnapshot must be the same object as trader_types.PositionSnapshot"
        )

    def test_trader_instrument_metadata_is_same_object(self) -> None:
        from src.execution.trader import TraderInstrumentMetadata
        from src.execution.trader_types import TraderInstrumentMetadata as TTraderInstrumentMetadata  # noqa: N813

        assert TraderInstrumentMetadata is TTraderInstrumentMetadata, (
            "Trader.TraderInstrumentMetadata must be the same object as trader_types.TraderInstrumentMetadata"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. trader_types has no forbidden imports
# ═══════════════════════════════════════════════════════════════════════════


class TestTraderTypesNoForbiddenImports:
    def test_no_okx_or_http_imports(self) -> None:
        import src.execution.trader_types as tt_module

        source_file = tt_module.__file__
        assert source_file is not None
        with open(source_file) as f:
            lines = f.readlines()

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("from ") or stripped.startswith("import "):
                assert "OkxPrivateClient" not in stripped, (
                    f"trader_types must not import OkxPrivateClient: {stripped!r}"
                )
                assert "OKX_CONFIG" not in stripped, (
                    f"trader_types must not import OKX_CONFIG: {stripped!r}"
                )
                assert "aiohttp" not in stripped, (
                    f"trader_types must not import aiohttp: {stripped!r}"
                )
                assert "PrivateWriteRateLimiter" not in stripped, (
                    f"trader_types must not import PrivateWriteRateLimiter: {stripped!r}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 3. TraderInstrumentMetadata validation is preserved
# ═══════════════════════════════════════════════════════════════════════════


class TestTraderInstrumentMetadataValidation:
    def test_valid_metadata(self) -> None:
        from decimal import Decimal
        from src.execution.trader_types import TraderInstrumentMetadata

        m = TraderInstrumentMetadata(
            inst_id="ETH-USDT-SWAP",
            contract_multiplier=Decimal("0.1"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
        )
        assert m.inst_id == "ETH-USDT-SWAP"
        assert m.contract_multiplier == Decimal("0.1")

    def test_empty_inst_id_raises(self) -> None:
        import pytest
        from decimal import Decimal
        from src.execution.trader_types import TraderInstrumentMetadata

        with pytest.raises(ValueError, match="inst_id must be a non-empty string"):
            TraderInstrumentMetadata(
                inst_id="",
                contract_multiplier=Decimal("0.1"),
                contract_precision=Decimal("0.01"),
                min_contracts=Decimal("0.01"),
            )

    def test_zero_multiplier_raises(self) -> None:
        import pytest
        from decimal import Decimal
        from src.execution.trader_types import TraderInstrumentMetadata

        with pytest.raises(ValueError, match="contract_multiplier must be > 0"):
            TraderInstrumentMetadata(
                inst_id="ETH-USDT-SWAP",
                contract_multiplier=Decimal("0"),
                contract_precision=Decimal("0.01"),
                min_contracts=Decimal("0.01"),
            )

    def test_zero_precision_raises(self) -> None:
        import pytest
        from decimal import Decimal
        from src.execution.trader_types import TraderInstrumentMetadata

        with pytest.raises(ValueError, match="contract_precision must be > 0"):
            TraderInstrumentMetadata(
                inst_id="ETH-USDT-SWAP",
                contract_multiplier=Decimal("0.1"),
                contract_precision=Decimal("0"),
                min_contracts=Decimal("0.01"),
            )

    def test_zero_min_contracts_raises(self) -> None:
        import pytest
        from decimal import Decimal
        from src.execution.trader_types import TraderInstrumentMetadata

        with pytest.raises(ValueError, match="min_contracts must be > 0"):
            TraderInstrumentMetadata(
                inst_id="ETH-USDT-SWAP",
                contract_multiplier=Decimal("0.1"),
                contract_precision=Decimal("0.01"),
                min_contracts=Decimal("0"),
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. TraderMarketSettings validation
# ═══════════════════════════════════════════════════════════════════════════


class TestTraderMarketSettingsValidation:
    def test_valid_settings(self) -> None:
        from decimal import Decimal
        from src.execution.trader_types import TraderMarketSettings

        s = TraderMarketSettings(
            inst_id="BTC-USDT-SWAP",
            td_mode="isolated",
            pos_side_mode="net",
            leverage=Decimal("15"),
        )
        assert s.inst_id == "BTC-USDT-SWAP"
        assert s.td_mode == "isolated"
        assert s.pos_side_mode == "net"
        assert s.leverage == Decimal("15")

    def test_leverage_accepts_decimal(self) -> None:
        from decimal import Decimal
        from src.execution.trader_types import TraderMarketSettings

        s = TraderMarketSettings(
            inst_id="BTC-USDT-SWAP",
            td_mode="isolated",
            pos_side_mode="net",
            leverage=Decimal("5"),
        )
        assert s.leverage == Decimal("5")

    def test_leverage_accepts_str(self) -> None:
        from decimal import Decimal
        from src.execution.trader_types import TraderMarketSettings

        s = TraderMarketSettings(
            inst_id="BTC-USDT-SWAP",
            td_mode="isolated",
            pos_side_mode="net",
            leverage="10",
        )
        assert s.leverage == Decimal("10")

    def test_leverage_accepts_int(self) -> None:
        from decimal import Decimal
        from src.execution.trader_types import TraderMarketSettings

        s = TraderMarketSettings(
            inst_id="BTC-USDT-SWAP",
            td_mode="isolated",
            pos_side_mode="net",
            leverage=20,
        )
        assert s.leverage == Decimal("20")

    def test_leverage_rejects_bool(self) -> None:
        import pytest
        from src.execution.trader_types import TraderMarketSettings

        with pytest.raises(ValueError, match="leverage must not be a boolean"):
            TraderMarketSettings(
                inst_id="BTC-USDT-SWAP",
                td_mode="isolated",
                pos_side_mode="net",
                leverage=True,  # type: ignore[arg-type]
            )

    def test_leverage_must_be_positive(self) -> None:
        import pytest
        from decimal import Decimal
        from src.execution.trader_types import TraderMarketSettings

        with pytest.raises(ValueError, match="leverage must be > 0"):
            TraderMarketSettings(
                inst_id="BTC-USDT-SWAP",
                td_mode="isolated",
                pos_side_mode="net",
                leverage=Decimal("0"),
            )

    def test_empty_inst_id_raises(self) -> None:
        import pytest
        from decimal import Decimal
        from src.execution.trader_types import TraderMarketSettings

        with pytest.raises(ValueError, match="inst_id must be a non-empty string"):
            TraderMarketSettings(
                inst_id="",
                td_mode="isolated",
                pos_side_mode="net",
                leverage=Decimal("15"),
            )

    def test_empty_td_mode_raises(self) -> None:
        import pytest
        from decimal import Decimal
        from src.execution.trader_types import TraderMarketSettings

        with pytest.raises(ValueError, match="td_mode must be a non-empty string"):
            TraderMarketSettings(
                inst_id="BTC-USDT-SWAP",
                td_mode="",
                pos_side_mode="net",
                leverage=Decimal("15"),
            )

    def test_empty_pos_side_mode_raises(self) -> None:
        import pytest
        from decimal import Decimal
        from src.execution.trader_types import TraderMarketSettings

        with pytest.raises(ValueError, match="pos_side_mode must be a non-empty string"):
            TraderMarketSettings(
                inst_id="BTC-USDT-SWAP",
                td_mode="isolated",
                pos_side_mode="",
                leverage=Decimal("15"),
            )

    def test_inst_id_is_stripped(self) -> None:
        from decimal import Decimal
        from src.execution.trader_types import TraderMarketSettings

        s = TraderMarketSettings(
            inst_id="  BTC-USDT-SWAP  ",
            td_mode="  isolated  ",
            pos_side_mode="  net  ",
            leverage=Decimal("15"),
        )
        assert s.inst_id == "BTC-USDT-SWAP"
        assert s.td_mode == "isolated"
        assert s.pos_side_mode == "net"

