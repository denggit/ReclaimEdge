from __future__ import annotations

import math
import os
from decimal import Decimal
from typing import Any

from config.env_loader import OKX_CONFIG
from src.execution import order_specs
from src.execution.okx_private_client import OkxPrivateClient, OkxPrivateClientConfig, PrivateWriteRateLimiter
from src.execution.trader_types import (
    LiveTradeResult,
    PositionSnapshot,
    TraderInstrumentMetadata,
    TraderMarketSettings,
)
from src.strategies.boll_cvd_reclaim_strategy import PositionSide, TradeIntent
from src.utils.log import get_logger

logger = get_logger(__name__)


def parse_allowed_live_symbols(raw: str | None) -> tuple[str, ...]:
    """Parse RECLAIM_ALLOWED_LIVE_SYMBOLS into a deduplicated tuple.

    Rules:
    - ``None`` or blank → ``("ETH-USDT-SWAP",)``
    - Comma‑separated, whitespace stripped, empty tokens dropped
    - Preserves order, removes duplicates
    - ``"*"`` anywhere → ValueError (wildcard disallowed)
    - Internal whitespace in a symbol → ValueError
    - Empty result after parsing → ``("ETH-USDT-SWAP",)``
    """
    if raw is None or raw.strip() == "":
        return ("ETH-USDT-SWAP",)

    if "*" in raw:
        raise ValueError("Wildcard '*' is not allowed in RECLAIM_ALLOWED_LIVE_SYMBOLS")

    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]

    if not parts:
        return ("ETH-USDT-SWAP",)

    for p in parts:
        if not isinstance(p, str) or not p:
            raise ValueError(f"Invalid symbol in RECLAIM_ALLOWED_LIVE_SYMBOLS: {p!r}")
        if any(ch.isspace() for ch in p):
            raise ValueError(f"Symbol must not contain whitespace: {p!r}")

    seen: set[str] = set()
    result: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result.append(p)

    return tuple(result)


#: Default metadata for ETH-USDT-SWAP – matches the hard-coded values that
#: Trader has always used before instrument metadata was configurable.
DEFAULT_ETH_INSTRUMENT_METADATA = TraderInstrumentMetadata(
    inst_id="ETH-USDT-SWAP",
    contract_multiplier=Decimal("0.1"),
    contract_precision=Decimal("0.01"),
    min_contracts=Decimal("0.01"),
)


def default_instrument_metadata_for(inst_id: str) -> TraderInstrumentMetadata:
    """Return the default instrument metadata for *inst_id*.

    Currently only ``ETH-USDT-SWAP`` is configured.  Raising for any other
    symbol is intentional – this gate prevents a symbol that passes the
    allowlist check from accidentally trading with wrong contract parameters.
    """
    normalized = (inst_id or "").strip()
    if normalized == "ETH-USDT-SWAP":
        return DEFAULT_ETH_INSTRUMENT_METADATA
    raise ValueError(
        f"No default instrument metadata for {normalized!r}. "
        "Only ETH-USDT-SWAP is configured for live trading metadata."
    )



class Trader:
    """Simple OKX live trader for ReclaimEdge.

    This module is intentionally small. It only supports the current strategy needs:
    - verify account balance cap
    - set leverage
    - market open long/short
    - place reduce-only take-profit limit order at BOLL target
    - replace take-profit when needed
    - recover existing ETH-USDT-SWAP position on restart
    """

    def __init__(
        self,
        *,
        instrument_metadata: TraderInstrumentMetadata | None = None,
        market_settings: TraderMarketSettings | None = None,
    ) -> None:
        self.symbol = os.getenv("OKX_INST_ID", "ETH-USDT-SWAP").strip() or "ETH-USDT-SWAP"
        self.allowed_live_symbols = parse_allowed_live_symbols(os.getenv("RECLAIM_ALLOWED_LIVE_SYMBOLS"))

        if self.symbol not in self.allowed_live_symbols:
            raise RuntimeError(
                "Live trader symbol is not allowed by RECLAIM_ALLOWED_LIVE_SYMBOLS: "
                f"symbol={self.symbol!r} allowed={self.allowed_live_symbols!r}"
            )

        # --- instrument metadata ---
        if instrument_metadata is None:
            metadata = default_instrument_metadata_for(self.symbol)
        else:
            if instrument_metadata.inst_id != self.symbol:
                raise ValueError(
                    f"instrument_metadata.inst_id {instrument_metadata.inst_id!r} "
                    f"does not match Trader symbol {self.symbol!r}"
                )
            metadata = instrument_metadata

        self.instrument_metadata = metadata
        self.contract_multiplier = metadata.contract_multiplier
        self.contract_precision = metadata.contract_precision
        self.min_contracts = metadata.min_contracts

        self.base_url = os.getenv("OKX_BASE_URL", "https://www.okx.com")

        # --- market settings (td_mode / pos_side_mode / leverage) ---
        if market_settings is not None:
            if market_settings.inst_id != self.symbol:
                raise ValueError(
                    f"market_settings.inst_id {market_settings.inst_id!r} "
                    f"does not match Trader symbol {self.symbol!r}"
                )
            self.td_mode = market_settings.td_mode
            self.pos_side_mode = market_settings.pos_side_mode
            self.leverage = str(market_settings.leverage)
        else:
            self.td_mode = os.getenv("OKX_TD_MODE", "isolated")
            self.leverage = os.getenv("LEVERAGE", "50")
            self.pos_side_mode = os.getenv("OKX_POS_SIDE_MODE", "net")
        self.live_trading = os.getenv("LIVE_TRADING", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
        self.max_live_equity_usdt = float(os.getenv("MAX_LIVE_EQUITY_USDT", "30"))

        self.api_key = OKX_CONFIG.get("api_key", "")
        self.secret_key = OKX_CONFIG.get("secret_key", "")
        self.passphrase = OKX_CONFIG.get("passphrase", "")

        self.tp_order_id: str | None = None
        self.near_tp_protective_sl_order_id: str | None = None
        self.middle_runner_protective_sl_order_id: str | None = None
        self.three_stage_post_tp1_protective_sl_order_id: str | None = None
        self.trend_runner_sl_order_id: str | None = None
        self.middle_bucket_fast_sl_order_id: str | None = None
        self.position_contracts = Decimal("0")
        self.account_equity_usdt: float = 0.0
        self._protected_reduce_only_order_ids: set[str] = set()
        self._managed_reduce_only_order_ids: set[str] = set()
        self._allow_cancel_unmanaged_reduce_only = True
        self._timeout_seconds = float(os.getenv("OKX_PRIVATE_REST_TIMEOUT_SECONDS", "10"))
        self._private_write_limiter = PrivateWriteRateLimiter()
        self._client = OkxPrivateClient(
            OkxPrivateClientConfig(
                base_url=self.base_url,
                api_key=self.api_key,
                secret_key=self.secret_key,
                passphrase=self.passphrase,
                timeout_seconds=self._timeout_seconds,
            )
        )
        from src.exchanges.okx.client import OkxBrokerClient
        from src.exchanges.okx.semantic_executor import OkxBrokerSemanticExecutor
        self._broker_client = OkxBrokerClient(self)
        self._broker_semantic_executor = OkxBrokerSemanticExecutor(self._broker_client)
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        self._tp_sl_manager = TpSlExecutionManager(self)

        if not self.api_key or not self.secret_key or not self.passphrase:
            raise ValueError("OKX API config is incomplete. Check OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHASE.")
        if not self.live_trading:
            raise RuntimeError("LIVE_TRADING is not true. Refusing to initialize live trader.")

    def __getattr__(self, name: str):
        if name == '_tp_sl_manager':
            from src.execution.tp_sl_execution_manager import TpSlExecutionManager
            mgr = TpSlExecutionManager(self)
            object.__setattr__(self, '_tp_sl_manager', mgr)
            return mgr
        if name == '_private_write_limiter':
            from src.execution.okx_private_client import PrivateWriteRateLimiter
            limiter = PrivateWriteRateLimiter()
            object.__setattr__(self, '_private_write_limiter', limiter)
            return limiter
        if name == '_broker_client':
            from src.exchanges.okx.client import OkxBrokerClient
            broker_client = OkxBrokerClient(self)
            object.__setattr__(self, '_broker_client', broker_client)
            return broker_client
        if name == '_broker_semantic_executor':
            from src.exchanges.okx.semantic_executor import OkxBrokerSemanticExecutor
            executor = OkxBrokerSemanticExecutor(self._broker_client)
            object.__setattr__(self, '_broker_semantic_executor', executor)
            return executor
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def broker_semantic_executor(self):
        return self._broker_semantic_executor

    async def start(self) -> None:
        await self._client.start()

    async def close(self) -> None:
        await self._client.close()

    async def initialize(self) -> None:
        equity = await self.fetch_usdt_equity()
        self.account_equity_usdt = equity
        if equity > self.max_live_equity_usdt:
            raise RuntimeError(
                f"USDT equity {equity:.4f} > MAX_LIVE_EQUITY_USDT {self.max_live_equity_usdt:.4f}. Refusing live trading."
            )
        await self.set_leverage()
        position = await self.fetch_position_snapshot()
        self.position_contracts = position.contracts
        logger.warning(
            "LIVE trader initialized | symbol=%s td_mode=%s leverage=%s equity=%.4f existing_side=%s existing_contracts=%s existing_avg=%.4f contract_multiplier=%s contract_precision=%s min_contracts=%s allowed_live_symbols=%s",
            self.symbol,
            self.td_mode,
            self.leverage,
            equity,
            position.side,
            self.position_contracts,
            position.avg_entry_price,
            self.contract_multiplier,
            self.contract_precision,
            self.min_contracts,
            self.allowed_live_symbols,
        )

    async def execute_intent(self, intent: TradeIntent) -> LiveTradeResult:
        if intent.intent_type == "NEAR_TP_REDUCE":
            return await self.execute_near_tp_reduce(intent)
        if intent.intent_type == "MARKET_EXIT_RUNNER":
            return await self.execute_market_exit_runner(intent)
        if intent.intent_type == "UPDATE_TP":
            return await self.replace_take_profit(intent)

        contracts = self.eth_qty_to_contracts(Decimal(str(intent.size.eth_qty)))
        from src.exchanges.models import BrokerOrderSide, BrokerPositionSide
        from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest

        semantic_result = await self.broker_semantic_executor.execute(
            BrokerSemanticRequest(
                exchange=self.broker_semantic_executor.exchange,
                symbol=self.symbol,
                action=(
                    BrokerSemanticAction.ADD_POSITION
                    if intent.intent_type in {"ADD_LONG", "ADD_SHORT"}
                    else BrokerSemanticAction.OPEN_POSITION
                ),
                role=(
                    BrokerSemanticOrderRole.ADD
                    if intent.intent_type in {"ADD_LONG", "ADD_SHORT"}
                    else BrokerSemanticOrderRole.ENTRY
                ),
                side=BrokerOrderSide.BUY if intent.side == "LONG" else BrokerOrderSide.SELL,
                position_side=BrokerPositionSide.LONG if intent.side == "LONG" else BrokerPositionSide.SHORT,
                quantity=contracts,
            )
        )
        order_id = semantic_result.order_id or ""
        if not order_id:
            raise RuntimeError(f"Missing order_id in broker semantic result: {semantic_result}")

        # From here on, assume the entry may already be live. Never let a later TP
        # failure look like a pre-entry failure to the caller.
        try:
            position = await self.fetch_position_snapshot()
            self.position_contracts = position.contracts if position.contracts > 0 else self.position_contracts + contracts
        except Exception:
            logger.exception("Failed to refresh position after entry; using requested contracts as fallback")
            self.position_contracts += contracts

        try:
            tp = await self.replace_take_profit(intent)
            if not tp.ok:
                return LiveTradeResult(
                    ok=False,
                    action=intent.intent_type,
                    order_id=order_id,
                    tp_order_id=tp.tp_order_id,
                    contracts=self.decimal_to_str(contracts),
                    tp_price=tp.tp_price,
                    message=f"entry_filled_but_tp_failed: {tp.message}",
                    entry_filled=True,
                    tp_ok=False,
                    tp_order_ids=tp.tp_order_ids,
                    protective_sl_order_id=tp.protective_sl_order_id,
                    protective_sl_price=tp.protective_sl_price,
                    protective_sl_ok=tp.protective_sl_ok,
                    middle_bucket_split_executed=tp.middle_bucket_split_executed,
                    middle_bucket_split_disabled_reason=tp.middle_bucket_split_disabled_reason,
                    middle_bucket_split_actual_order_mode=tp.middle_bucket_split_actual_order_mode,
                )
            return LiveTradeResult(
                ok=True,
                action=intent.intent_type,
                order_id=order_id,
                tp_order_id=tp.tp_order_id,
                contracts=self.decimal_to_str(contracts),
                tp_price=tp.tp_price,
                message="market order placed and take-profit protected",
                entry_filled=True,
                tp_ok=True,
                tp_order_ids=tp.tp_order_ids,
                protective_sl_order_id=tp.protective_sl_order_id,
                protective_sl_price=tp.protective_sl_price,
                protective_sl_ok=tp.protective_sl_ok,
                middle_bucket_split_executed=tp.middle_bucket_split_executed,
                middle_bucket_split_disabled_reason=tp.middle_bucket_split_disabled_reason,
                middle_bucket_split_actual_order_mode=tp.middle_bucket_split_actual_order_mode,
            )
        except Exception as exc:
            logger.exception("Entry appears filled, but TP placement raised an exception")
            return LiveTradeResult(
                ok=False,
                action=intent.intent_type,
                order_id=order_id,
                tp_order_id=None,
                contracts=self.decimal_to_str(contracts),
                tp_price=self.price_to_str(intent.tp_price),
                message=f"entry_filled_but_tp_exception: {exc}",
                entry_filled=True,
                tp_ok=False,
            )

    async def execute_near_tp_reduce(self, intent: TradeIntent) -> LiveTradeResult:
        return await self._tp_sl_manager.execute_near_tp_reduce(intent)

    async def execute_market_exit_runner(self, intent: TradeIntent) -> LiveTradeResult:
        return await self._tp_sl_manager.execute_market_exit_runner(intent)

    async def replace_take_profit(self, intent: TradeIntent) -> LiveTradeResult:
        return await self._tp_sl_manager.replace_take_profit(intent)

    async def _cancel_existing_take_profit_orders_for_intent(self, intent: TradeIntent) -> bool:
        return await self._tp_sl_manager._cancel_existing_take_profit_orders_for_intent(intent)

    async def _cancel_stale_runner_protective_stops_for_degrade(self, intent: TradeIntent) -> None:
        return await self._tp_sl_manager._cancel_stale_runner_protective_stops_for_degrade(intent)

    def _protected_order_ids_from_intent(self, intent: TradeIntent) -> set[str]:
        return self._tp_sl_manager._protected_order_ids_from_intent(intent)

    @staticmethod
    def _split_order_ids(value: str | None) -> set[str]:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        return TpSlExecutionManager._split_order_ids(value)

    def _managed_core_contracts_from_intent(self, intent: TradeIntent) -> Decimal | None:
        return self._tp_sl_manager._managed_core_contracts_from_intent(intent)

    def _build_take_profit_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        return self._tp_sl_manager._build_take_profit_order_specs(intent)

    def _build_three_stage_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        return self._tp_sl_manager._build_three_stage_order_specs(intent)

    def _trend_runner_sl_contracts(self, intent: TradeIntent, net_contracts_for_sl: Decimal) -> Decimal:
        return self._tp_sl_manager._trend_runner_sl_contracts(intent, net_contracts_for_sl)

    async def _place_reduce_only_take_profit_orders(self, intent: TradeIntent,
                                                    specs: list[tuple[str, Decimal, float]]) -> list[str]:
        return await self._tp_sl_manager._place_reduce_only_take_profit_orders(intent, specs)

    def _reduce_only_tp_order_body(self, side: PositionSide, contracts: Decimal, price: float) -> dict[str, Any]:
        return order_specs.build_reduce_only_tp_order_body(
            inst_id=self.symbol,
            td_mode=self.td_mode,
            side=side,
            contracts_text=self.decimal_to_str(contracts),
            price_text=self.price_to_str(price),
            pos_side_mode=self.pos_side_mode,
        )

    def _reduce_only_market_order_body(self, side: PositionSide, contracts: Decimal) -> dict[str, Any]:
        return order_specs.build_reduce_only_market_order_body(
            inst_id=self.symbol,
            td_mode=self.td_mode,
            side=side,
            contracts_text=self.decimal_to_str(contracts),
            pos_side_mode=self.pos_side_mode,
        )

    async def place_near_tp_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal | str | int | float,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        return await self._tp_sl_manager.place_near_tp_protective_stop_with_retries(
            side, contracts, stop_price, retry_count, retry_interval_seconds)

    async def place_middle_runner_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        return await self._tp_sl_manager.place_middle_runner_protective_stop_with_retries(
            side, contracts, stop_price, retry_count, retry_interval_seconds)

    async def place_middle_bucket_fast_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        return await self._tp_sl_manager.place_middle_bucket_fast_protective_stop_with_retries(
            side, contracts, stop_price, retry_count, retry_interval_seconds)

    async def place_trend_runner_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        return await self._tp_sl_manager.place_trend_runner_protective_stop_with_retries(
            side, contracts, stop_price, retry_count, retry_interval_seconds)

    async def place_three_stage_post_tp1_protective_stop_with_retries(
            self,
            side: PositionSide,
            contracts: Decimal,
            stop_price: float,
            retry_count: int,
            retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        return await self._tp_sl_manager.place_three_stage_post_tp1_protective_stop_with_retries(
            side, contracts, stop_price, retry_count, retry_interval_seconds)

    async def _cancel_unverified_near_tp_algo(self, algo_id: str, *, phase: str) -> None:
        return await self._tp_sl_manager._cancel_unverified_near_tp_algo(algo_id, phase=phase)

    async def verify_near_tp_protective_stop(self, algo_id: str, side: PositionSide, contracts: Decimal,
                                             stop_price: float) -> bool:
        return await self._tp_sl_manager.verify_near_tp_protective_stop(algo_id, side, contracts, stop_price)

    def _near_tp_protective_stop_matches(self, item: dict[str, Any], algo_id: str, side: PositionSide,
                                         contracts: Decimal, stop_price: float) -> bool:
        return self._tp_sl_manager._near_tp_protective_stop_matches(item, algo_id, side, contracts, stop_price)

    def _near_tp_protective_sl_algo_body(self, side: PositionSide, contracts: Decimal, stop_price: float) -> dict[
        str, Any]:
        return order_specs.build_conditional_protective_sl_algo_body(
            inst_id=self.symbol,
            td_mode=self.td_mode,
            side=side,
            contracts_text=self.decimal_to_str(contracts),
            stop_price_text=self.price_to_str(stop_price),
            pos_side_mode=self.pos_side_mode,
        )

    def _near_tp_fallback_conditional_close_body(self, side: PositionSide, contracts: Decimal, stop_price: float) -> \
            dict[str, Any]:
        return order_specs.build_conditional_protective_sl_algo_body(
            inst_id=self.symbol,
            td_mode=self.td_mode,
            side=side,
            contracts_text=self.decimal_to_str(contracts),
            stop_price_text=self.price_to_str(stop_price),
            pos_side_mode=self.pos_side_mode,
        )

    async def market_exit_remaining_position_with_retries(
        self,
        side: PositionSide,
        retry_count: int,
        *,
        context: str = "generic",
        retry_interval_seconds: float | None = None,
    ) -> tuple[bool, str]:
        return await self._tp_sl_manager.market_exit_remaining_position_with_retries(
            side, retry_count, context=context, retry_interval_seconds=retry_interval_seconds,
        )

    async def _cleanup_after_market_exit(self) -> None:
        return await self._tp_sl_manager._cleanup_after_market_exit()

    # Backward-compat alias
    async def _cleanup_after_near_tp_market_exit(self) -> None:
        return await self._cleanup_after_market_exit()

    def _tp_price_summary(self, specs: list[tuple[str, Decimal, float]]) -> str:
        return self._tp_sl_manager._tp_price_summary(specs)

    async def cancel_existing_reduce_only_orders(self, *, phase: str = "normal_cancel") -> bool:
        try:
            return await self._tp_sl_manager.cancel_existing_reduce_only_orders(phase=phase)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword" in message or "positional" in message or "argument" in message:
                return await self._tp_sl_manager.cancel_existing_reduce_only_orders()
            raise

    async def place_sidecar_market_order(self, *, side: PositionSide, eth_qty: float) -> dict[str, Any]:
        contracts = self.eth_qty_to_contracts(Decimal(str(eth_qty)))
        from src.execution.broker_semantic_helpers import get_broker_semantic_executor, require_semantic_order_id
        from src.exchanges.models import BrokerOrderSide, BrokerPositionSide
        from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest

        semantic_executor = get_broker_semantic_executor(self)
        if semantic_executor is not None:
            result = await semantic_executor.execute(
                BrokerSemanticRequest(
                    exchange=semantic_executor.exchange,
                    symbol=self.symbol,
                    action=BrokerSemanticAction.SIDECAR_ENTRY,
                    role=BrokerSemanticOrderRole.SIDECAR_ENTRY,
                    side=BrokerOrderSide.BUY if side == "LONG" else BrokerOrderSide.SELL,
                    position_side=BrokerPositionSide.LONG if side == "LONG" else BrokerPositionSide.SHORT,
                    quantity=contracts,
                )
            )
            order_id = require_semantic_order_id(result, action="SIDECAR_ENTRY")
        else:
            body = order_specs.build_market_entry_order_body(
                inst_id=self.symbol,
                td_mode=self.td_mode,
                side=side,
                contracts_text=self.decimal_to_str(contracts),
                pos_side_mode=self.pos_side_mode,
            )
            res = await self.request("POST", "/api/v5/trade/order", body)
            order_id = self.extract_order_id(res)
        return {
            "order_id": order_id,
            "contracts": self.decimal_to_str(contracts),
            "qty": float(contracts * self.contract_multiplier),
        }

    async def place_sidecar_fixed_take_profit(
            self,
            *,
            side: PositionSide,
            contracts: str | Decimal,
            tp_price: float,
            client_order_id: str | None = None,
    ) -> str:
        return await self._tp_sl_manager.place_sidecar_fixed_take_profit(
            side=side,
            contracts=contracts,
            tp_price=tp_price,
            client_order_id=client_order_id,
        )

    async def cancel_sidecar_take_profit(self, order_id: str | None) -> bool:
        return await self._tp_sl_manager.cancel_sidecar_take_profit(order_id)

    async def fetch_sidecar_order_status(self, order_id: str) -> dict[str, Any]:
        try:
            res = await self.request("GET", f"/api/v5/trade/order?instId={self.symbol}&ordId={order_id}")
        except Exception:
            return {"order_id": order_id, "status": "UNKNOWN", "filled_qty": None, "avg_fill_price": None}
        data = res.get("data", [])
        if not data:
            return {"order_id": order_id, "status": "NOT_FOUND", "filled_qty": None, "avg_fill_price": None}
        item = data[0]
        state = str(item.get("state") or "").lower()
        if state in {"live", "partially_filled"}:
            status = "OPEN"
        elif state == "filled":
            status = "FILLED"
        elif state in {"canceled", "cancelled"}:
            status = "CANCELED"
        else:
            status = "UNKNOWN"
        return {
            "order_id": order_id,
            "status": status,
            "filled_qty": _optional_float(item.get("accFillSz")),
            "avg_fill_price": _optional_float(item.get("avgPx")),
        }

    async def fetch_pending_orders_raw(self) -> list[dict[str, Any]]:
        res = await self.request("GET", f"/api/v5/trade/orders-pending?instId={self.symbol}")
        return list(res.get("data", []))

    async def fetch_pending_orders(self) -> list[dict[str, Any]]:
        result = await self.broker_semantic_executor.execute(
            _broker_semantic_fetch_request(self.symbol, "ordinary")
        )
        # Temporary bridge during broker semantic migration: startup recovery
        # and identity safety still inspect OKX raw fields via this facade.
        return [dict(order.raw) for order in result.orders]

    async def fetch_pending_algo_orders_raw(self) -> list[dict[str, Any]]:
        res = await self.request("GET", f"/api/v5/trade/orders-algo-pending?instId={self.symbol}&ordType=conditional")
        return list(res.get("data", []))

    async def fetch_pending_algo_orders(self) -> list[dict[str, Any]]:
        result = await self.broker_semantic_executor.execute(
            _broker_semantic_fetch_request(self.symbol, "algo")
        )
        # Temporary bridge during broker semantic migration: SL verification
        # still matches OKX raw algo fields while fetch is routed via broker semantics.
        return [dict(order.raw) for order in result.orders]

    async def cancel_near_tp_protective_stop(self, order_id: str | None) -> bool:
        return await self._tp_sl_manager.cancel_near_tp_protective_stop(order_id)

    async def cancel_middle_runner_protective_stop(self, order_id: str | None) -> bool:
        return await self._tp_sl_manager.cancel_middle_runner_protective_stop(order_id)

    async def cancel_middle_bucket_fast_protective_stop(self, order_id: str | None) -> bool:
        return await self._tp_sl_manager.cancel_middle_bucket_fast_protective_stop(order_id)

    async def cancel_trend_runner_protective_stop(self, order_id: str | None) -> bool:
        return await self._tp_sl_manager.cancel_trend_runner_protective_stop(order_id)

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id: str | None) -> bool:
        return await self._tp_sl_manager.cancel_three_stage_post_tp1_protective_stop(order_id)

    async def fetch_usdt_equity(self) -> float:
        res = await self.request("GET", "/api/v5/account/balance?ccy=USDT")
        data = res.get("data", [])
        if not data:
            return 0.0
        details = data[0].get("details", [])
        for item in details:
            if item.get("ccy") == "USDT":
                return float(item.get("eq") or item.get("availEq") or item.get("availBal") or 0.0)
        return float(data[0].get("totalEq") or 0.0)

    async def fetch_position_contracts(self) -> Decimal:
        return (await self.fetch_position_snapshot()).contracts

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        res = await self.request("GET", f"/api/v5/account/positions?instId={self.symbol}")
        best: PositionSnapshot | None = None
        for item in res.get("data", []):
            if item.get("instId") != self.symbol:
                continue
            raw_pos = Decimal(str(item.get("pos", "0")))
            if raw_pos == 0:
                continue
            contracts = abs(raw_pos)
            avg_entry = float(item.get("avgPx") or item.get("avgPxUsd") or 0.0)
            if self.pos_side_mode == "long_short":
                pos_side = str(item.get("posSide", "")).lower()
                side: PositionSide | None = "LONG" if pos_side == "long" else "SHORT" if pos_side == "short" else None
            else:
                side = "LONG" if raw_pos > 0 else "SHORT"
            best = PositionSnapshot(
                side=side,
                contracts=contracts,
                avg_entry_price=avg_entry,
                eth_qty=float(contracts * self.contract_multiplier),
                raw_pos=raw_pos,
            )
            break
        if best is None:
            return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))
        return best

    def mark_flat(self) -> None:
        self.position_contracts = Decimal("0")
        self.tp_order_id = None
        self.near_tp_protective_sl_order_id = None
        self.middle_runner_protective_sl_order_id = None
        self.three_stage_post_tp1_protective_sl_order_id = None
        self.trend_runner_sl_order_id = None
        self.middle_bucket_fast_sl_order_id = None

    async def set_leverage(self) -> None:
        bodies = order_specs.build_set_leverage_bodies(
            inst_id=self.symbol,
            td_mode=self.td_mode,
            leverage=self.leverage,
            pos_side_mode=self.pos_side_mode,
        )
        for body in bodies:
            await self.request("POST", "/api/v5/account/set-leverage", body)

    async def request(self, method: str, endpoint: str, payload: Any | None = None) -> dict[str, Any]:
        # Rate-limit all private write (POST) operations
        if method.upper() == "POST":
            await self._private_write_limiter.acquire()
        return await self._client.request(method, endpoint, payload)

    def headers(self, method: str, endpoint: str, body: str) -> dict[str, str]:
        return self._client.headers(method, endpoint, body)

    def eth_qty_to_contracts(self, eth_qty: Decimal) -> Decimal:
        raw_contracts = eth_qty / self.contract_multiplier
        contracts = self.round_contracts_down(raw_contracts)
        if contracts < self.min_contracts:
            raise RuntimeError(f"Order size {contracts} contracts is below minimum {self.min_contracts}")
        return contracts

    def round_contracts_down(self, contracts: Decimal) -> Decimal:
        return order_specs.round_contracts_down(contracts=contracts, contract_precision=self.contract_precision)

    def pos_side(self, side: str) -> str | None:
        return order_specs.pos_side_for_mode(side=side, pos_side_mode=self.pos_side_mode)

    @staticmethod
    def extract_order_id(res: dict[str, Any]) -> str:
        data = res.get("data", [])
        if not data or not data[0].get("ordId"):
            raise RuntimeError(f"Missing ordId in response: {res}")
        return str(data[0]["ordId"])

    @staticmethod
    def extract_algo_id(res: dict[str, Any]) -> str:
        data = res.get("data", [])
        if not data:
            raise RuntimeError(f"Missing algoId in response: {res}")
        algo_id = data[0].get("algoId") or data[0].get("ordId")
        if not algo_id:
            raise RuntimeError(f"Missing algoId in response: {res}")
        return str(algo_id)

    @staticmethod
    def _to_decimal(value: Decimal | str | int | float) -> Decimal:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    @staticmethod
    def decimal_to_str(value: Decimal | str | int | float) -> str:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        if not math.isfinite(price):
            raise RuntimeError(f"Invalid price: {price}")
        return f"{price:.2f}"


def _optional_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except Exception:
        return None


def _broker_semantic_fetch_request(symbol: str, source: str):
    from src.exchanges.models import ExchangeName
    from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest

    action = (
        BrokerSemanticAction.FETCH_ALGO_ORDERS
        if source == "algo"
        else BrokerSemanticAction.FETCH_OPEN_ORDERS
    )
    return BrokerSemanticRequest(
        exchange=ExchangeName.OKX,
        symbol=symbol,
        action=action,
        role=BrokerSemanticOrderRole.RECOVERY,
    )
