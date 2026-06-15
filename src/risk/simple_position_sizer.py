from __future__ import annotations

import os
from dataclasses import dataclass

from src.position_management.sidecar.model import calculate_core_margin_pct


@dataclass(frozen=True)
class SimplePositionSizerConfig:
    dry_run_equity_usdt: float = 1000.0
    layer_margin_pct: float = 0.03
    leverage: float = 20.0
    layer_multiplier_step: float = 0.15
    trade_risk_pct: float = 0.01
    fee_slippage_buffer_pct: float = 0.001
    max_order_notional_usdt: float = 0.0
    sidecar_enabled: bool = False
    sidecar_margin_pct: float = 0.01
    sidecar_tp_pct: float = 0.004
    sidecar_close_when_core_flat: bool = True
    sidecar_order_status_check_seconds: float = 5.0
    sidecar_max_legs: int = 10
    sidecar_skip_first_layer: bool = True

    @classmethod
    def from_env(cls) -> "SimplePositionSizerConfig":
        config = cls(
            dry_run_equity_usdt=float(os.getenv("DRY_RUN_EQUITY_USDT", "1000")),
            layer_margin_pct=float(os.getenv("LAYER_MARGIN_PCT", "0.03")),
            leverage=float(os.getenv("LEVERAGE", "20")),
            trade_risk_pct=float(os.getenv("TRADE_RISK_PCT", os.getenv("ENTRY_RISK_PCT", "0.01"))),
            fee_slippage_buffer_pct=float(
                os.getenv("ENTRY_FEE_SLIPPAGE_BUFFER_PCT", os.getenv("FEE_SLIPPAGE_BUFFER_PCT", "0.001"))
            ),
            max_order_notional_usdt=float(os.getenv("MAX_ORDER_NOTIONAL_USDT", "0")),
            sidecar_enabled=_env_bool("SIDECAR_ENABLED", False),
            sidecar_margin_pct=float(os.getenv("SIDECAR_MARGIN_PCT", "0.01")),
            sidecar_tp_pct=float(os.getenv("SIDECAR_TP_PCT", "0.004")),
            sidecar_close_when_core_flat=_env_bool("SIDECAR_CLOSE_WHEN_CORE_FLAT", True),
            sidecar_order_status_check_seconds=float(os.getenv("SIDECAR_ORDER_STATUS_CHECK_SECONDS", "5")),
            sidecar_max_legs=int(os.getenv("SIDECAR_MAX_LEGS", "10")),
            sidecar_skip_first_layer=_env_bool("SIDECAR_SKIP_FIRST_LAYER", True),
        )
        config.validate_sidecar()
        return config

    @classmethod
    def from_account_equity(cls, account_equity_usdt: float) -> "SimplePositionSizerConfig":
        config = cls(
            dry_run_equity_usdt=account_equity_usdt,
            layer_margin_pct=float(os.getenv("LAYER_MARGIN_PCT", "0.03")),
            leverage=float(os.getenv("LEVERAGE", "20")),
            trade_risk_pct=float(os.getenv("TRADE_RISK_PCT", os.getenv("ENTRY_RISK_PCT", "0.01"))),
            fee_slippage_buffer_pct=float(
                os.getenv("ENTRY_FEE_SLIPPAGE_BUFFER_PCT", os.getenv("FEE_SLIPPAGE_BUFFER_PCT", "0.001"))
            ),
            max_order_notional_usdt=float(os.getenv("MAX_ORDER_NOTIONAL_USDT", "0")),
            sidecar_enabled=_env_bool("SIDECAR_ENABLED", False),
            sidecar_margin_pct=float(os.getenv("SIDECAR_MARGIN_PCT", "0.01")),
            sidecar_tp_pct=float(os.getenv("SIDECAR_TP_PCT", "0.004")),
            sidecar_close_when_core_flat=_env_bool("SIDECAR_CLOSE_WHEN_CORE_FLAT", True),
            sidecar_order_status_check_seconds=float(os.getenv("SIDECAR_ORDER_STATUS_CHECK_SECONDS", "5")),
            sidecar_max_legs=int(os.getenv("SIDECAR_MAX_LEGS", "10")),
            sidecar_skip_first_layer=_env_bool("SIDECAR_SKIP_FIRST_LAYER", True),
        )
        config.validate_sidecar()
        return config

    @property
    def core_margin_pct(self) -> float:
        return calculate_core_margin_pct(self.layer_margin_pct, self.sidecar_enabled, self.sidecar_margin_pct)

    def validate_sidecar(self) -> None:
        if self.trade_risk_pct <= 0:
            raise RuntimeError("TRADE_RISK_PCT / ENTRY_RISK_PCT must be > 0")
        if self.fee_slippage_buffer_pct < 0:
            raise RuntimeError("ENTRY_FEE_SLIPPAGE_BUFFER_PCT must be >= 0")
        if self.leverage <= 0:
            raise RuntimeError("LEVERAGE must be > 0")
        if not self.sidecar_enabled:
            return
        if self.sidecar_margin_pct <= 0:
            raise RuntimeError("SIDECAR_ENABLED=true requires SIDECAR_MARGIN_PCT > 0")
        if self.sidecar_margin_pct >= self.layer_margin_pct:
            raise RuntimeError("SIDECAR_ENABLED=true requires SIDECAR_MARGIN_PCT < LAYER_MARGIN_PCT")
        if self.sidecar_tp_pct <= 0:
            raise RuntimeError("SIDECAR_ENABLED=true requires SIDECAR_TP_PCT > 0")
        if self.sidecar_max_legs < 1:
            raise RuntimeError("SIDECAR_ENABLED=true requires SIDECAR_MAX_LEGS >= 1")


@dataclass(frozen=True)
class PositionSize:
    margin_usdt: float
    notional_usdt: float
    eth_qty: float
    layer_index: int
    layer_multiplier: float
    sizing_mode: str = "margin"
    risk_usdt: float = 0.0
    stop_price: float | None = None
    stop_distance_pct: float = 0.0
    effective_risk_pct: float = 0.0


class SimplePositionSizer:
    def __init__(self, config: SimplePositionSizerConfig):
        self.config = config

    def update_account_equity(self, account_equity_usdt: float) -> None:
        self.config = SimplePositionSizerConfig(
            dry_run_equity_usdt=account_equity_usdt,
            layer_margin_pct=self.config.layer_margin_pct,
            leverage=self.config.leverage,
            layer_multiplier_step=self.config.layer_multiplier_step,
            trade_risk_pct=self.config.trade_risk_pct,
            fee_slippage_buffer_pct=self.config.fee_slippage_buffer_pct,
            max_order_notional_usdt=self.config.max_order_notional_usdt,
            sidecar_enabled=self.config.sidecar_enabled,
            sidecar_margin_pct=self.config.sidecar_margin_pct,
            sidecar_tp_pct=self.config.sidecar_tp_pct,
            sidecar_close_when_core_flat=self.config.sidecar_close_when_core_flat,
            sidecar_order_status_check_seconds=self.config.sidecar_order_status_check_seconds,
            sidecar_max_legs=self.config.sidecar_max_legs,
            sidecar_skip_first_layer=self.config.sidecar_skip_first_layer,
        )

    @property
    def account_equity_usdt(self) -> float:
        return self.config.dry_run_equity_usdt

    def calculate(self, price: float, layer_index: int = 1, stop_price: float | None = None) -> PositionSize:
        safe_layer_index = max(int(layer_index), 1)
        multiplier = 1.0 + (safe_layer_index - 1) * self.config.layer_multiplier_step
        if price <= 0:
            return PositionSize(
                margin_usdt=0.0,
                notional_usdt=0.0,
                eth_qty=0.0,
                layer_index=safe_layer_index,
                layer_multiplier=multiplier,
            )

        if stop_price is not None and stop_price > 0 and stop_price != price:
            stop_distance_pct = abs(float(price) - float(stop_price)) / float(price)
            effective_risk_pct = stop_distance_pct + self.config.fee_slippage_buffer_pct
            if effective_risk_pct <= 0:
                raise RuntimeError("risk_based_position_sizing_requires_positive_effective_risk_pct")
            risk_usdt = self.config.dry_run_equity_usdt * self.config.trade_risk_pct
            notional = risk_usdt / effective_risk_pct
            if self.config.max_order_notional_usdt > 0:
                notional = min(notional, self.config.max_order_notional_usdt)
            margin = notional / self.config.leverage
            eth_qty = notional / price
            return PositionSize(
                margin_usdt=margin,
                notional_usdt=notional,
                eth_qty=eth_qty,
                layer_index=safe_layer_index,
                layer_multiplier=1.0,
                sizing_mode="risk",
                risk_usdt=risk_usdt,
                stop_price=float(stop_price),
                stop_distance_pct=stop_distance_pct,
                effective_risk_pct=effective_risk_pct,
            )

        # Legacy fallback: retained for non-entry maintenance flows that only need
        # a size object for metadata. New entries must pass stop_price.
        base_margin = self.config.dry_run_equity_usdt * self.config.core_margin_pct
        margin = base_margin * multiplier
        notional = margin * self.config.leverage
        eth_qty = notional / price
        return PositionSize(
            margin_usdt=margin,
            notional_usdt=notional,
            eth_qty=eth_qty,
            layer_index=safe_layer_index,
            layer_multiplier=multiplier,
            sizing_mode="margin",
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
