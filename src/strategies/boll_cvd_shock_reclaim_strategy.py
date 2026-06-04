from __future__ import annotations

import os
import time

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    PositionSide,
    TradeIntent,
    TradeIntentType,
)
from src.utils.log import get_logger

logger = get_logger(__name__)


class BollCvdShockReclaimStrategy(BollCvdReclaimStrategy):
    """BOLL reclaim strategy gated by relative shock and latched BOLL width.

    BOLL width can flicker around the threshold when live candles are used. A
    strict per-tick switch check can miss good entries near the threshold. This
    strategy latches switch eligibility for the current candle and a short grace
    period after the last switch=True tick.
    """

    def __init__(self, config: BollCvdReclaimStrategyConfig, sizer: SimplePositionSizer):
        super().__init__(config, sizer)
        self.switch_grace_seconds = float(os.getenv("BOLL_SWITCH_GRACE_SECONDS", "300"))
        self.first_add_block_bypass_multiplier = float(os.getenv("FIRST_ADD_BLOCK_BYPASS_MULTIPLIER", "5"))
        self._last_switch_on_ts_ms: int = 0
        self._last_switch_on_candle_ts_ms: int = 0
        self._last_lower_outside_no_burst_log_monotonic: float = 0.0
        self._last_upper_outside_no_burst_log_monotonic: float = 0.0
        self.outside_no_burst_log_interval_seconds = float(os.getenv("OUTSIDE_NO_BURST_LOG_INTERVAL_SECONDS", "10"))

    def on_tick(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> list[TradeIntent]:
        intents: list[TradeIntent] = []

        if boll.alert_switch_on:
            self._last_switch_on_ts_ms = ts_ms
            self._last_switch_on_candle_ts_ms = boll.candle_ts_ms
        switch_eligible = self._switch_eligible(ts_ms, boll)

        # Existing armed state still needs expiry/middle reset even if BOLL width
        # switch turns off later. But new armed state requires switch eligibility.
        self._expire_armed_state(ts_ms)
        if switch_eligible:
            self._update_shock_armed_state(price, ts_ms, boll, cvd)
        else:
            self._reset_armed_if_middle_reclaimed(price, boll)

        tp_intent = self._maybe_update_tp(price, ts_ms, boll, cvd)
        if tp_intent is not None:
            intents.append(tp_intent)

        runner_exit_intent = self._maybe_trend_runner_market_exit(price, ts_ms, boll, cvd)
        if runner_exit_intent is not None:
            intents.append(runner_exit_intent)

        near_tp_intent = self._maybe_near_tp_reduce(price, ts_ms, boll, cvd)
        if near_tp_intent is not None:
            intents.append(near_tp_intent)

        if not switch_eligible:
            return intents

        if not self._cooldown_ok(ts_ms):
            return intents

        if self._long_setup(price, cvd):
            intent = self._maybe_open_or_add_long(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        if self._short_setup(price, cvd):
            intent = self._maybe_open_or_add_short(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        return intents

    def _open_position(
        self,
        side: PositionSide,
        intent_type: TradeIntentType,
        price: float,
        ts_ms: int,
        boll: BollSnapshot,
        cvd: CvdSnapshot,
        reason: str,
    ) -> TradeIntent:
        previous_layers = self.state.layers
        intent = super()._open_position(side, intent_type, price, ts_ms, boll, cvd, reason)
        if previous_layers <= 0:
            # Keep the first-entry clock stable. The first-add block must not be
            # reset by an early 5x-distance add; until this original 30m window
            # ends, every extra add still needs the same 5x distance exception.
            self.state.first_entry_ts_ms = ts_ms
        return intent

    def _add_timing_passed(self, side: PositionSide, price: float, ts_ms: int, target_layer: int) -> tuple[bool, str]:
        last = self.state.last_entry_price
        if last is None or last <= 0:
            return False, "missing_last_entry"

        first_elapsed_seconds = self._first_entry_elapsed_seconds(ts_ms)
        if self.state.layers >= 1 and first_elapsed_seconds < self.config.first_add_block_seconds:
            adverse_gap_pct = self._adverse_gap_pct(side, price)
            required_gap_pct = self._first_add_block_required_gap_pct_for_target_layer(target_layer)
            if adverse_gap_pct < required_gap_pct:
                return False, "first_add_block"
            logger.warning(
                "FIRST_ADD_BLOCK_BYPASSED | side=%s price=%.4f layers=%s target_layer=%s last_entry=%.4f first_elapsed_seconds=%.1f required_seconds=%s adverse_gap_pct=%.4f%% required_gap_pct=%.4f%% multiplier=%.2f",
                side,
                price,
                self.state.layers,
                target_layer,
                last,
                first_elapsed_seconds,
                self.config.first_add_block_seconds,
                adverse_gap_pct * 100,
                required_gap_pct * 100,
                self.first_add_block_bypass_multiplier,
            )
            return True, "first_add_block_bypassed"

        if self.state.layers == 1:
            return True, "ok"

        if self.state.layers >= 2:
            elapsed_seconds = self._add_elapsed_seconds(ts_ms)
            adverse_gap_pct = self._adverse_gap_pct(side, price)
            bypass_gap_pct = self._add_min_interval_bypass_gap_pct_for_target_layer(target_layer)
            if elapsed_seconds < self.config.add_min_interval_seconds and adverse_gap_pct < bypass_gap_pct:
                return False, "add_interval"

        return True, "ok"

    def _first_entry_elapsed_seconds(self, ts_ms: int) -> float:
        first_entry_ts_ms = int(getattr(self.state, "first_entry_ts_ms", 0) or 0)
        if first_entry_ts_ms <= 0:
            first_entry_ts_ms = int(self.state.last_order_ts_ms or 0)
        return max((ts_ms - first_entry_ts_ms) / 1000, 0.0)

    def _first_add_block_required_gap_pct_for_target_layer(self, target_layer: int) -> float:
        return self._add_layer_gap_pct_for_target_layer(target_layer) * self.first_add_block_bypass_multiplier

    def _log_add_timing_skipped(self, side: PositionSide, reason: str, price: float, ts_ms: int, target_layer: int) -> None:
        if reason == "first_add_block":
            last = self.state.last_entry_price if self.state.last_entry_price is not None else 0.0
            first_elapsed_seconds = self._first_entry_elapsed_seconds(ts_ms)
            adverse_gap_pct = self._adverse_gap_pct(side, price)
            required_gap_pct = self._first_add_block_required_gap_pct_for_target_layer(target_layer)
            logger.info(
                "ADD_SKIPPED | reason=first_add_block side=%s price=%.4f layers=%s target_layer=%s last_entry=%.4f first_elapsed_seconds=%.1f required_seconds=%s adverse_gap_pct=%.4f%% required_gap_pct=%.4f%% multiplier=%.2f",
                side,
                price,
                self.state.layers,
                target_layer,
                last,
                first_elapsed_seconds,
                self.config.first_add_block_seconds,
                adverse_gap_pct * 100,
                required_gap_pct * 100,
                self.first_add_block_bypass_multiplier,
            )
            return
        super()._log_add_timing_skipped(side, reason, price, ts_ms, target_layer)

    def _switch_eligible(self, ts_ms: int, boll: BollSnapshot) -> bool:
        if boll.alert_switch_on:
            return True
        if self._last_switch_on_candle_ts_ms == boll.candle_ts_ms and self._last_switch_on_ts_ms > 0:
            return True
        grace_ms = int(self.switch_grace_seconds * 1000)
        return self._last_switch_on_ts_ms > 0 and ts_ms - self._last_switch_on_ts_ms <= grace_ms

    def _update_shock_armed_state(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> None:
        if price < boll.lower:
            if not cvd.down_burst:
                self._log_lower_outside_no_burst(price, ts_ms, boll, cvd)
                if self.state.lower_armed:
                    self.state.lower_extreme_price = min(self.state.lower_extreme_price or price, price)
                    self._update_lower_deep_enough(boll)
                return
            if not self.state.lower_armed:
                self.state.lower_armed = True
                self.state.lower_armed_ts_ms = ts_ms
                self.state.lower_extreme_price = price
                logger.info(
                    "LOWER_ARMED | reason=down_shock price=%.4f lower=%.4f middle=%.4f switch_current=%s switch_latched=%s move_ratio=%.2f volume_ratio=%.2f burst_range=%.5f baseline_range=%.5f max_entry_distance=%.4f%% max_armed=%ss",
                    price,
                    boll.lower,
                    boll.middle,
                    boll.alert_switch_on,
                    not boll.alert_switch_on,
                    cvd.burst_move_ratio,
                    cvd.burst_volume_ratio,
                    cvd.burst_range_pct,
                    cvd.baseline_range_pct,
                    self.config.max_entry_distance_from_extreme_pct * 100,
                    self.config.max_armed_seconds,
                )
                self._update_lower_deep_enough(boll)
            else:
                old_extreme = self.state.lower_extreme_price or price
                self.state.lower_extreme_price = min(old_extreme, price)
                if self.state.lower_extreme_price < old_extreme:
                    logger.info(
                        "LOWER_ARMED_EXTREME_UPDATED | reason=down_shock extreme=%.4f price=%.4f move_ratio=%.2f volume_ratio=%.2f",
                        self.state.lower_extreme_price,
                        price,
                        cvd.burst_move_ratio,
                        cvd.burst_volume_ratio,
                    )
                self._update_lower_deep_enough(boll)
            self.state.lower_last_burst_ts_ms = ts_ms
            if self.state.upper_armed:
                logger.info("UPPER_ARMED_RESET | reason=opposite_lower_shock price=%.4f", price)
            self._reset_upper_armed()
            return

        if price > boll.upper:
            if not cvd.up_burst:
                self._log_upper_outside_no_burst(price, ts_ms, boll, cvd)
                if self.state.upper_armed:
                    self.state.upper_extreme_price = max(self.state.upper_extreme_price or price, price)
                    self._update_upper_deep_enough(boll)
                return
            if not self.state.upper_armed:
                self.state.upper_armed = True
                self.state.upper_armed_ts_ms = ts_ms
                self.state.upper_extreme_price = price
                logger.info(
                    "UPPER_ARMED | reason=up_shock price=%.4f upper=%.4f middle=%.4f switch_current=%s switch_latched=%s move_ratio=%.2f volume_ratio=%.2f burst_range=%.5f baseline_range=%.5f max_entry_distance=%.4f%% max_armed=%ss",
                    price,
                    boll.upper,
                    boll.middle,
                    boll.alert_switch_on,
                    not boll.alert_switch_on,
                    cvd.burst_move_ratio,
                    cvd.burst_volume_ratio,
                    cvd.burst_range_pct,
                    cvd.baseline_range_pct,
                    self.config.max_entry_distance_from_extreme_pct * 100,
                    self.config.max_armed_seconds,
                )
                self._update_upper_deep_enough(boll)
            else:
                old_extreme = self.state.upper_extreme_price or price
                self.state.upper_extreme_price = max(old_extreme, price)
                if self.state.upper_extreme_price > old_extreme:
                    logger.info(
                        "UPPER_ARMED_EXTREME_UPDATED | reason=up_shock extreme=%.4f price=%.4f move_ratio=%.2f volume_ratio=%.2f",
                        self.state.upper_extreme_price,
                        price,
                        cvd.burst_move_ratio,
                        cvd.burst_volume_ratio,
                    )
                self._update_upper_deep_enough(boll)
            self.state.upper_last_burst_ts_ms = ts_ms
            if self.state.lower_armed:
                logger.info("LOWER_ARMED_RESET | reason=opposite_upper_shock price=%.4f", price)
            self._reset_lower_armed()
            return

        self._reset_armed_if_middle_reclaimed(price, boll)

    def _log_lower_outside_no_burst(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> None:
        now_monotonic = time.monotonic()
        if self._last_lower_outside_no_burst_log_monotonic and now_monotonic - self._last_lower_outside_no_burst_log_monotonic < self.outside_no_burst_log_interval_seconds:
            return
        self._last_lower_outside_no_burst_log_monotonic = now_monotonic
        logger.info(
            "LOWER_OUTSIDE_NO_BURST | price=%.4f lower=%.4f middle=%.4f switch_current=%s switch_latched=%s lower_armed=%s lower_extreme=%s burst_net_move_pct=%.6f move_ratio=%.2f volume_ratio=%.2f burst_range_pct=%.6f baseline_range_pct=%.6f burst_volume=%.8f baseline_volume=%.8f up_burst=%s down_burst=%s",
            price,
            boll.lower,
            boll.middle,
            boll.alert_switch_on,
            not boll.alert_switch_on,
            self.state.lower_armed,
            self.state.lower_extreme_price,
            cvd.burst_net_move_pct,
            cvd.burst_move_ratio,
            cvd.burst_volume_ratio,
            cvd.burst_range_pct,
            cvd.baseline_range_pct,
            cvd.burst_volume,
            cvd.baseline_volume,
            cvd.up_burst,
            cvd.down_burst,
        )

    def _log_upper_outside_no_burst(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> None:
        now_monotonic = time.monotonic()
        if self._last_upper_outside_no_burst_log_monotonic and now_monotonic - self._last_upper_outside_no_burst_log_monotonic < self.outside_no_burst_log_interval_seconds:
            return
        self._last_upper_outside_no_burst_log_monotonic = now_monotonic
        logger.info(
            "UPPER_OUTSIDE_NO_BURST | price=%.4f upper=%.4f middle=%.4f switch_current=%s switch_latched=%s upper_armed=%s upper_extreme=%s burst_net_move_pct=%.6f move_ratio=%.2f volume_ratio=%.2f burst_range_pct=%.6f baseline_range_pct=%.6f burst_volume=%.8f baseline_volume=%.8f up_burst=%s down_burst=%s",
            price,
            boll.upper,
            boll.middle,
            boll.alert_switch_on,
            not boll.alert_switch_on,
            self.state.upper_armed,
            self.state.upper_extreme_price,
            cvd.burst_net_move_pct,
            cvd.burst_move_ratio,
            cvd.burst_volume_ratio,
            cvd.burst_range_pct,
            cvd.baseline_range_pct,
            cvd.burst_volume,
            cvd.baseline_volume,
            cvd.up_burst,
            cvd.down_burst,
        )

    def _reset_armed_if_middle_reclaimed(self, price: float, boll: BollSnapshot) -> None:
        if self.state.lower_armed and price >= boll.middle:
            logger.info("LOWER_ARMED_RESET | reason=middle_reclaimed price=%.4f middle=%.4f", price, boll.middle)
            self._reset_lower_armed()
        if self.state.upper_armed and price <= boll.middle:
            logger.info("UPPER_ARMED_RESET | reason=middle_reclaimed price=%.4f middle=%.4f", price, boll.middle)
            self._reset_upper_armed()

    def _long_setup(self, price: float, cvd: CvdSnapshot) -> bool:
        if not self.state.lower_armed or self.state.lower_extreme_price is None:
            return False
        if self.state.lower_last_burst_ts_ms <= self.state.last_order_ts_ms:
            return False
        return super()._long_setup(price, cvd)

    def _short_setup(self, price: float, cvd: CvdSnapshot) -> bool:
        if not self.state.upper_armed or self.state.upper_extreme_price is None:
            return False
        if self.state.upper_last_burst_ts_ms <= self.state.last_order_ts_ms:
            return False
        return super()._short_setup(price, cvd)

    def _reset_lower_armed(self) -> None:
        super()._reset_lower_armed()
        self.state.lower_last_burst_ts_ms = 0

    def _reset_upper_armed(self) -> None:
        super()._reset_upper_armed()
        self.state.upper_last_burst_ts_ms = 0
