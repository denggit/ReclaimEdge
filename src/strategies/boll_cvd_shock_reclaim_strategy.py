from __future__ import annotations

import os
import time
from collections import deque

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer
from src.strategies import add_freeze_chain
from src.strategies import add_layer_gates
from src.strategies import extreme_retest_add as _extreme_retest
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

    ADD_FREEZE_SKIP_LOG_INTERVAL_MS = 30_000

    def __init__(self, config: BollCvdReclaimStrategyConfig, sizer: SimplePositionSizer):
        super().__init__(config, sizer)
        self.switch_grace_seconds = float(os.getenv("BOLL_SWITCH_GRACE_SECONDS", "300"))
        self.first_add_block_bypass_multiplier = float(os.getenv("FIRST_ADD_BLOCK_BYPASS_MULTIPLIER", "5"))
        # ── ADD config fields (removed from BollCvdReclaimStrategyConfig, stored as instance attrs) ──
        self.add_gap_mode: str = str(getattr(config, "add_gap_mode", "linear") or "linear")
        self.add_gap_base_pct: float = float(getattr(config, "add_gap_base_pct", 0.003) or 0.003)
        self.add_gap_step_pct: float = float(getattr(config, "add_gap_step_pct", 0.001) or 0.001)
        self.first_add_block_seconds: int = int(getattr(config, "first_add_block_seconds", 1800) or 1800)
        self.add_min_interval_seconds: int = int(getattr(config, "add_min_interval_seconds", 600) or 600)
        self.add_freeze_chain_enabled: bool = bool(getattr(config, "add_freeze_chain_enabled", True))
        self.add_min_interval_bypass_multiplier: float = float(
            getattr(config, "add_min_interval_bypass_multiplier", 2.0) or 2.0)
        self._last_switch_on_ts_ms: int = 0
        self._last_switch_on_candle_ts_ms: int = 0
        self._last_add_freeze_skip_log_ts_ms: int = 0
        self._last_add_freeze_skip_log_key: tuple[str, int, int, int, float] | None = None
        self._last_lower_outside_no_burst_log_monotonic: float = 0.0
        self._last_upper_outside_no_burst_log_monotonic: float = 0.0
        self.outside_no_burst_log_interval_seconds = float(os.getenv("OUTSIDE_NO_BURST_LOG_INTERVAL_SECONDS", "10"))
        # ── Extreme Retest Add ────────────────────────────────────────────
        self._extreme_retest_config = _extreme_retest.ExtremeRetestConfig(
            enabled=getattr(config, "extreme_retest_add_enabled", False),
            pivot_left_bars=getattr(config, "extreme_retest_pivot_left_bars", 2),
            pivot_right_bars=getattr(config, "extreme_retest_pivot_right_bars", 2),
            anchor_max_age_candles=getattr(config, "extreme_retest_anchor_max_age_candles", 12),
            sweep_max_age_seconds=getattr(config, "extreme_retest_sweep_max_age_seconds", 900.0),
            near_extreme_pct=getattr(config, "extreme_retest_near_extreme_pct", 0.0015),
            reclaim_pct=getattr(config, "extreme_retest_reclaim_pct", 0.0005),
            min_reverse_ratio=getattr(config, "extreme_retest_min_reverse_ratio", 0.55),
            one_add_per_anchor=getattr(config, "extreme_retest_one_add_per_anchor", True),
        )
        self._extreme_retest_anchor = _extreme_retest.ExtremeRetestAnchor()
        self._prev_boll: BollSnapshot | None = None
        self._candle_buffer: deque[dict] = deque(maxlen=50)
        self._last_detected_pivot_ts_ms: int = 0
        self._last_extreme_retest_eval_log_ts_ms: int = 0
        self._last_extreme_retest_sweep_seen_logged: bool = False

    def on_tick(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> list[TradeIntent]:
        intents: list[TradeIntent] = []

        # ── Extreme Retest: track candle buffer on candle close ──────────
        if self._extreme_retest_config.enabled:
            self._track_candle_buffer(boll)

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

        if not switch_eligible:
            return intents

        if not self._cooldown_ok(ts_ms):
            return intents

        # ── Normal OUTER_BAND ADD (runs first) ───────────────────────────
        if self._long_setup(price, cvd, boll):
            intent = self._maybe_open_or_add_long(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        if self._short_setup(price, cvd, boll):
            intent = self._maybe_open_or_add_short(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        # ADD is intentionally disabled in the risk-first runtime.
        # Extreme Retest remains available for historical tests/helpers but is no
        # longer invoked by the live signal path.

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
    ) -> TradeIntent | None:
        previous_layers = self.state.layers
        was_active_freeze = self._add_freeze_active(ts_ms)
        intent = super()._open_position(side, intent_type, price, ts_ms, boll, cvd, reason)
        if intent is None:
            return None
        if previous_layers <= 0:
            self.state.first_entry_ts_ms = ts_ms
            decision = add_freeze_chain.start_add_freeze_after_first_entry(
                ts_ms=ts_ms,
                add_freeze_chain_enabled=self.add_freeze_chain_enabled,
                first_add_block_seconds=self.first_add_block_seconds,
            )
            if decision.enabled:
                self.state.add_freeze_until_ts_ms = decision.freeze_until_ts_ms
                self.state.add_freeze_penalty_count = decision.penalty_count
                logger.warning(
                    "ADD_FREEZE_STARTED | reason=first_entry side=%s layers=%s freeze_until_ts_ms=%s freeze_seconds=%s first_entry_ts_ms=%s",
                    side,
                    self.state.layers,
                    self.state.add_freeze_until_ts_ms,
                    self.first_add_block_seconds,
                    self.state.first_entry_ts_ms,
                )
        else:
            self._extend_add_freeze_after_successful_add(ts_ms, was_active_freeze=was_active_freeze)
        # ── Revalidate extreme retest anchor after any ADD ─────────────
        if previous_layers > 0 and self._extreme_retest_config.enabled:
            self._maybe_revalidate_extreme_retest_anchor_after_add()
        return intent

    # ── ADD layer gate helpers (delegate to add_layer_gates) ──────────────

    def _add_layer_gap_pct_for_target_layer(self, target_layer: int) -> float:
        return add_layer_gates.add_layer_gap_pct_for_target_layer(
            target_layer=target_layer,
            add_gap_mode=self.add_gap_mode,
            add_gap_base_pct=self.add_gap_base_pct,
            add_gap_step_pct=self.add_gap_step_pct,
        )

    def _add_min_interval_bypass_gap_pct_for_target_layer(self, target_layer: int) -> float:
        return add_layer_gates.add_min_interval_bypass_gap_pct_for_target_layer(
            target_layer=target_layer,
            add_gap_mode=self.add_gap_mode,
            add_gap_base_pct=self.add_gap_base_pct,
            add_gap_step_pct=self.add_gap_step_pct,
        )

    def _add_elapsed_seconds(self, ts_ms: int) -> float:
        return add_layer_gates.add_elapsed_seconds(ts_ms=ts_ms, last_order_ts_ms=self.state.last_order_ts_ms)

    def _adverse_gap_pct(self, side: PositionSide, price: float) -> float:
        return add_layer_gates.adverse_gap_pct(side=side, price=price, last_entry_price=self.state.last_entry_price)

    def _add_timing_passed(self, side: PositionSide, price: float, ts_ms: int, target_layer: int) -> tuple[bool, str]:
        last = self.state.last_entry_price
        if last is None or last <= 0:
            return False, "missing_last_entry"

        self._reset_add_freeze_if_expired(ts_ms)

        target_layer_gap_pct = self._add_layer_gap_pct_for_target_layer(target_layer)
        decision = add_freeze_chain.check_shock_add_timing(
            side=side,
            price=price,
            ts_ms=ts_ms,
            target_layer=target_layer,
            layers=self.state.layers,
            last_entry_price=self.state.last_entry_price,
            last_order_ts_ms=self.state.last_order_ts_ms,
            first_entry_ts_ms=int(getattr(self.state, "first_entry_ts_ms", 0) or 0),
            add_freeze_chain_enabled=self.add_freeze_chain_enabled,
            add_freeze_until_ts_ms=int(getattr(self.state, "add_freeze_until_ts_ms", 0) or 0),
            add_freeze_penalty_count=int(getattr(self.state, "add_freeze_penalty_count", 0) or 0),
            first_add_block_seconds=self.first_add_block_seconds,
            add_min_interval_seconds=self.add_min_interval_seconds,
            add_min_interval_bypass_multiplier=self.add_min_interval_bypass_multiplier,
            first_add_block_bypass_multiplier=self.first_add_block_bypass_multiplier,
            target_layer_gap_pct=target_layer_gap_pct,
        )

        if decision.ok and decision.reason == "first_add_block_bypassed":
            if not self.add_freeze_chain_enabled:
                logger.warning(
                    "FIRST_ADD_BLOCK_BYPASSED | side=%s price=%.4f layers=%s target_layer=%s last_entry=%.4f first_elapsed_seconds=%.1f required_seconds=%s adverse_gap_pct=%.4f%% required_gap_pct=%.4f%% multiplier=%.2f",
                    side,
                    price,
                    self.state.layers,
                    target_layer,
                    last,
                    decision.first_elapsed_seconds,
                    self.first_add_block_seconds,
                    decision.adverse_gap_pct * 100,
                    decision.required_gap_pct * 100,
                    decision.multiplier,
                )
            else:
                logger.warning(
                    "FIRST_ADD_BLOCK_BYPASSED | side=%s price=%.4f layers=%s target_layer=%s last_entry=%.4f freeze_remaining_seconds=%.1f required_seconds=%s adverse_gap_pct=%.4f%% required_gap_pct=%.4f%% multiplier=%.2f",
                    side,
                    price,
                    self.state.layers,
                    target_layer,
                    last,
                    decision.freeze_remaining_seconds,
                    self.first_add_block_seconds,
                    decision.adverse_gap_pct * 100,
                    decision.required_gap_pct * 100,
                    decision.multiplier,
                )
        elif decision.ok and decision.reason == "add_freeze_bypassed":
            logger.warning(
                "ADD_FREEZE_BYPASSED | side=%s price=%.4f layers=%s target_layer=%s last_entry=%.4f freeze_remaining_seconds=%.1f adverse_gap_pct=%.4f%% required_gap_pct=%.4f%% multiplier=%.2f penalty_count=%s",
                side,
                price,
                self.state.layers,
                target_layer,
                last,
                decision.freeze_remaining_seconds,
                decision.adverse_gap_pct * 100,
                decision.required_gap_pct * 100,
                decision.multiplier,
                self.state.add_freeze_penalty_count,
            )

        return decision.ok, decision.reason

    def _add_freeze_active(self, ts_ms: int) -> bool:
        return add_freeze_chain.add_freeze_active(
            add_freeze_chain_enabled=self.add_freeze_chain_enabled,
            add_freeze_until_ts_ms=int(getattr(self.state, "add_freeze_until_ts_ms", 0) or 0),
            ts_ms=ts_ms,
        )

    def _add_freeze_remaining_seconds(self, ts_ms: int) -> float:
        return add_freeze_chain.add_freeze_remaining_seconds(
            add_freeze_until_ts_ms=int(getattr(self.state, "add_freeze_until_ts_ms", 0) or 0),
            ts_ms=ts_ms,
        )

    def _reset_add_freeze_if_expired(self, ts_ms: int) -> None:
        if add_freeze_chain.should_reset_add_freeze_if_expired(
                add_freeze_until_ts_ms=int(getattr(self.state, "add_freeze_until_ts_ms", 0) or 0),
                ts_ms=ts_ms,
        ):
            self.state.add_freeze_until_ts_ms = 0
            self.state.add_freeze_penalty_count = 0

    def _active_add_freeze_bypass_multiplier(self) -> float:
        return add_freeze_chain.active_add_freeze_bypass_multiplier(
            layers=self.state.layers,
            penalty_count=int(getattr(self.state, "add_freeze_penalty_count", 0) or 0),
            first_add_block_bypass_multiplier=self.first_add_block_bypass_multiplier,
            add_min_interval_bypass_multiplier=self.add_min_interval_bypass_multiplier,
        )

    def _extend_add_freeze_after_successful_add(self, ts_ms: int, *, was_active_freeze: bool) -> None:
        decision = add_freeze_chain.extend_add_freeze_after_successful_add(
            ts_ms=ts_ms,
            add_freeze_chain_enabled=self.add_freeze_chain_enabled,
            add_min_interval_seconds=self.add_min_interval_seconds,
            add_freeze_until_ts_ms=int(getattr(self.state, "add_freeze_until_ts_ms", 0) or 0),
            add_freeze_penalty_count=int(getattr(self.state, "add_freeze_penalty_count", 0) or 0),
            was_active_freeze=was_active_freeze,
        )
        if not decision.changed:
            return
        self.state.add_freeze_until_ts_ms = decision.freeze_until_ts_ms
        self.state.add_freeze_penalty_count = decision.penalty_count
        logger.warning(
            "ADD_FREEZE_EXTENDED | layers=%s freeze_until_ts_ms=%s freeze_remaining_seconds=%.1f penalty_count=%s extension_seconds=%s was_active_freeze=%s",
            self.state.layers,
            self.state.add_freeze_until_ts_ms,
            self._add_freeze_remaining_seconds(ts_ms),
            self.state.add_freeze_penalty_count,
            self.add_min_interval_seconds,
            was_active_freeze,
        )

    def _first_entry_elapsed_seconds(self, ts_ms: int) -> float:
        return add_freeze_chain.first_entry_elapsed_seconds(
            ts_ms=ts_ms,
            first_entry_ts_ms=int(getattr(self.state, "first_entry_ts_ms", 0) or 0),
            last_order_ts_ms=int(self.state.last_order_ts_ms or 0),
        )

    def _first_add_block_required_gap_pct_for_target_layer(self, target_layer: int) -> float:
        target_layer_gap_pct = self._add_layer_gap_pct_for_target_layer(target_layer)
        return add_freeze_chain.first_add_block_required_gap_pct(
            target_layer_gap_pct=target_layer_gap_pct,
            first_add_block_bypass_multiplier=self.first_add_block_bypass_multiplier,
        )

    def _log_add_timing_skipped(self, side: PositionSide, reason: str, price: float, ts_ms: int,
                                target_layer: int) -> None:
        if reason == "add_freeze":
            multiplier = self._active_add_freeze_bypass_multiplier()
            current_key = add_freeze_chain.add_freeze_skip_log_key(
                side=side,
                layers=self.state.layers,
                target_layer=target_layer,
                penalty_count=int(getattr(self.state, "add_freeze_penalty_count", 0) or 0),
                multiplier=multiplier,
            )
            last_ts = int(getattr(self, "_last_add_freeze_skip_log_ts_ms", 0) or 0)
            last_key = getattr(self, "_last_add_freeze_skip_log_key", None)
            if not add_freeze_chain.should_emit_add_freeze_skip_log(
                    last_key=last_key,
                    current_key=current_key,
                    last_ts_ms=last_ts,
                    ts_ms=ts_ms,
                    interval_ms=self.ADD_FREEZE_SKIP_LOG_INTERVAL_MS,
            ):
                return
            self._last_add_freeze_skip_log_ts_ms = ts_ms
            self._last_add_freeze_skip_log_key = current_key
            last = self.state.last_entry_price if self.state.last_entry_price is not None else 0.0
            adverse_gap_pct = self._adverse_gap_pct(side, price)
            required_gap_pct = self._add_layer_gap_pct_for_target_layer(target_layer) * multiplier
            logger.info(
                "ADD_SKIPPED | reason=add_freeze side=%s price=%.4f layers=%s target_layer=%s last_entry=%.4f freeze_remaining_seconds=%.1f required_seconds=%s adverse_gap_pct=%.4f%% required_gap_pct=%.4f%% multiplier=%.2f penalty_count=%s",
                side,
                price,
                self.state.layers,
                target_layer,
                last,
                self._add_freeze_remaining_seconds(ts_ms),
                self.add_min_interval_seconds,
                adverse_gap_pct * 100,
                required_gap_pct * 100,
                multiplier,
                self.state.add_freeze_penalty_count,
            )
            return
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
                self.first_add_block_seconds,
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
                    self._check_lower_cvd_structure(cvd, boll, ts_ms)
                return
            if not self.state.lower_armed:
                self.state.lower_armed = True
                self.state.lower_armed_ts_ms = ts_ms
                self.state.lower_first_armed_ts_ms = ts_ms
                self.state.lower_extreme_price = price
                self.state.lower_reference_fast_cvd = cvd.fast_cvd
                logger.info(
                    "LOWER_ARMED | reason=down_shock price=%.4f lower=%.4f middle=%.4f switch_current=%s switch_latched=%s move_ratio=%.2f volume_ratio=%.2f burst_range=%.5f baseline_range=%.5f max_entry_distance=%.4f%% max_armed=%ss fast_cvd=%.8f",
                    price,
                    boll.lower,
                    boll.middle,
                    boll.alert_switch_on,
                    not boll.alert_switch_on,
                    cvd.burst_move_ratio,
                    cvd.burst_volume_ratio,
                    cvd.burst_range_pct,
                    cvd.baseline_range_pct,
                    0.0,
                    self.config.max_armed_seconds,
                    cvd.fast_cvd,
                )
                self._update_lower_deep_enough(boll)
                self._check_lower_cvd_structure(cvd, boll, ts_ms)
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
                self._check_lower_cvd_structure(cvd, boll, ts_ms)
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
                    self._check_upper_cvd_structure(cvd, boll, ts_ms)
                return
            if not self.state.upper_armed:
                self.state.upper_armed = True
                self.state.upper_armed_ts_ms = ts_ms
                self.state.upper_first_armed_ts_ms = ts_ms
                self.state.upper_extreme_price = price
                self.state.upper_reference_fast_cvd = cvd.fast_cvd
                logger.info(
                    "UPPER_ARMED | reason=up_shock price=%.4f upper=%.4f middle=%.4f switch_current=%s switch_latched=%s move_ratio=%.2f volume_ratio=%.2f burst_range=%.5f baseline_range=%.5f max_entry_distance=%.4f%% max_armed=%ss fast_cvd=%.8f",
                    price,
                    boll.upper,
                    boll.middle,
                    boll.alert_switch_on,
                    not boll.alert_switch_on,
                    cvd.burst_move_ratio,
                    cvd.burst_volume_ratio,
                    cvd.burst_range_pct,
                    cvd.baseline_range_pct,
                    0.0,
                    self.config.max_armed_seconds,
                    cvd.fast_cvd,
                )
                self._update_upper_deep_enough(boll)
                self._check_upper_cvd_structure(cvd, boll, ts_ms)
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
                self._check_upper_cvd_structure(cvd, boll, ts_ms)
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

    def _long_setup(self, price: float, cvd: CvdSnapshot, boll: BollSnapshot) -> bool:
        if not self.state.lower_armed or self.state.lower_extreme_price is None:
            return False
        if self.state.lower_last_burst_ts_ms <= self.state.last_order_ts_ms:
            return False
        return super()._long_setup(price, cvd, boll)

    def _short_setup(self, price: float, cvd: CvdSnapshot, boll: BollSnapshot) -> bool:
        if not self.state.upper_armed or self.state.upper_extreme_price is None:
            return False
        if self.state.upper_last_burst_ts_ms <= self.state.last_order_ts_ms:
            return False
        return super()._short_setup(price, cvd, boll)

    def _reset_lower_armed(self) -> None:
        super()._reset_lower_armed()
        self.state.lower_last_burst_ts_ms = 0

    def _reset_upper_armed(self) -> None:
        super()._reset_upper_armed()
        self.state.upper_last_burst_ts_ms = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Extreme Retest Add methods
    # ──────────────────────────────────────────────────────────────────────────

    def _track_candle_buffer(self, boll: BollSnapshot) -> None:
        """Detect candle close and push closed candle data into buffer."""
        if self._prev_boll is not None and boll.candle_ts_ms != self._prev_boll.candle_ts_ms:
            # Previous boll represents the just-closed candle at its final state
            prev = self._prev_boll
            high = prev.high if prev.high is not None else prev.close
            low = prev.low if prev.low is not None else prev.close
            self._candle_buffer.append({
                "ts_ms": prev.candle_ts_ms,
                "high": high,
                "low": low,
                "close": prev.close,
                "boll_upper": prev.upper,
                "boll_lower": prev.lower,
            })
            # Run pivot detection on candle close
            if self.state.side is not None:
                self._detect_pivot_on_new_candle(prev.candle_ts_ms, prev.upper, prev.lower)
        self._prev_boll = boll

    def _detect_pivot_on_new_candle(self, candle_ts_ms: int, boll_upper: float, boll_lower: float) -> None:
        """Detect extreme pivots from candle buffer on new candle close."""
        cfg = self._extreme_retest_config
        side = self.state.side

        # Skip if not in a position
        if side is None or self.state.layers <= 0:
            return

        anchor = self._extreme_retest_anchor
        self._sync_anchor_from_state()

        # Drop expired anchor
        _extreme_retest.drop_expired_anchor(anchor, candle_ts_ms, cfg.anchor_max_age_candles)

        # Need enough candles: left_bars + 1 (pivot) + right_bars
        min_candles = cfg.pivot_left_bars + 1 + cfg.pivot_right_bars
        buf = list(self._candle_buffer)
        if len(buf) < min_candles:
            return

        # Scan recent candles for pivots — check the rightmost candidate that has enough right bars
        # The pivot_idx is len(buf) - 1 - right_bars (most recent candle that has right_bars candles after it)
        pivot_idx = len(buf) - 1 - cfg.pivot_right_bars
        candidate = buf[pivot_idx]
        candidate_ts_ms = int(candidate["ts_ms"])

        # Dedup: skip if we already detected a pivot on this candidate candle
        if self._last_detected_pivot_ts_ms == candidate_ts_ms:
            return

        # Detect pivot
        if side == "SHORT":
            if not _extreme_retest.detect_pivot_high(buf, pivot_idx, cfg.pivot_left_bars, cfg.pivot_right_bars):
                return
            candidate_price = float(candidate["high"])
        else:
            if not _extreme_retest.detect_pivot_low(buf, pivot_idx, cfg.pivot_left_bars, cfg.pivot_right_bars):
                return
            candidate_price = float(candidate["low"])

        # Use candidate candle's own boll_upper/lower for strict outer-band check
        candidate_boll_upper = float(candidate.get("boll_upper") or 0)
        candidate_boll_lower = float(candidate.get("boll_lower") or 0)

        self._last_detected_pivot_ts_ms = candidate_ts_ms

        # Compute effective required gap
        effective_required_gap_pct = self._extreme_retest_effective_required_gap_pct(ts_ms=None)

        # Try to create or replace anchor
        _extreme_retest.try_create_or_replace_anchor(
            side=side,
            candidate_price=candidate_price,
            candle_ts_ms=candidate_ts_ms,
            boll_upper=candidate_boll_upper,
            boll_lower=candidate_boll_lower,
            last_entry_price=self.state.last_entry_price,
            effective_required_gap_pct=effective_required_gap_pct,
            anchor=anchor,
            config=cfg,
        )

        self._sync_anchor_to_state()

    def _evaluate_extreme_retest_add(
        self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot
    ) -> TradeIntent | None:
        """Evaluate extreme retest add trigger and run through original add gates."""
        self._sync_anchor_from_state()

        anchor = self._extreme_retest_anchor
        cfg = self._extreme_retest_config
        side = self.state.side
        if side is None:
            return None

        # Drop expired anchor based on candle age
        _extreme_retest.drop_expired_anchor(anchor, boll.candle_ts_ms, cfg.anchor_max_age_candles)

        # Evaluate on tick
        eval_result = _extreme_retest.evaluate_on_tick(
            side=side,
            price=price,
            ts_ms=ts_ms,
            boll_upper=boll.upper,
            boll_lower=boll.lower,
            anchor=anchor,
            config=cfg,
            buy_ratio=cvd.buy_ratio,
            sell_ratio=cvd.sell_ratio,
        )

        # Log evaluation — throttled: at most once per 60s when not triggered
        if anchor.is_active():
            now_ms = ts_ms
            interval_since_last = now_ms - self._last_extreme_retest_eval_log_ts_ms
            sweep_state_changed = (
                anchor.sweep_seen and not self._last_extreme_retest_sweep_seen_logged
            )
            should_log = (
                eval_result.triggered
                or sweep_state_changed
                or interval_since_last >= 60_000
            )
            if should_log:
                self._last_extreme_retest_eval_log_ts_ms = now_ms
                if sweep_state_changed:
                    self._last_extreme_retest_sweep_seen_logged = True
                elif not anchor.sweep_seen:
                    self._last_extreme_retest_sweep_seen_logged = False
                logger.info(
                    "EXTREME_RETEST_ADD_EVALUATED | side=%s layers=%s target_layer=%s "
                    "price=%s boll_upper=%s boll_lower=%s inside_band=%s "
                    "anchor_price=%s anchor_kind=%s pattern=%s "
                    "sweep_seen=%s sweep_extreme_price=%s reclaimed=%s "
                    "near_extreme=%s buy_ratio=%.4f sell_ratio=%.4f "
                    "reverse_ratio_ok=%s triggered=%s decision=%s reason=%s",
                    side,
                    self.state.layers,
                    self.state.layers + 1,
                    price,
                    boll.upper,
                    boll.lower,
                    eval_result.inside_band,
                    eval_result.anchor_price,
                    eval_result.anchor_kind,
                    eval_result.pattern,
                    eval_result.sweep_seen,
                    eval_result.sweep_extreme_price,
                    eval_result.reclaimed,
                    eval_result.near_extreme,
                    eval_result.buy_ratio,
                    eval_result.sell_ratio,
                    eval_result.reverse_ratio_ok,
                    eval_result.triggered,
                    eval_result.decision,
                    eval_result.reason,
                )

        if not eval_result.triggered:
            self._sync_anchor_to_state()
            return None

        logger.info(
            "ADD_SKIPPED | reason=add_disabled side=%s trigger_source=EXTREME_RETEST pattern=%s anchor_price=%s layers=%s",
            side,
            eval_result.pattern,
            eval_result.anchor_price,
            self.state.layers,
        )
        self._sync_anchor_to_state()
        return None

    def _log_extreme_retest_add_timing_skipped(
        self,
        side: PositionSide,
        reason: str,
        price: float,
        ts_ms: int,
        target_layer: int,
        *,
        pattern: str | None,
        anchor_price: float | None,
    ) -> None:
        """Log timing skip for extreme retest with trigger_source included."""
        last = self.state.last_entry_price if self.state.last_entry_price is not None else 0.0
        adverse_gap_pct = self._adverse_gap_pct(side, price)
        if reason == "add_freeze":
            multiplier = self._active_add_freeze_bypass_multiplier()
            logger.info(
                "ADD_SKIPPED | reason=add_freeze side=%s price=%s "
                "trigger_source=EXTREME_RETEST pattern=%s anchor_price=%s "
                "layers=%s target_layer=%s last_entry=%s "
                "freeze_remaining_seconds=%.1f adverse_gap_pct=%.4f%% "
                "required_gap_pct=%.4f%% multiplier=%.2f",
                side, price, pattern, anchor_price,
                self.state.layers, target_layer, last,
                self._add_freeze_remaining_seconds(ts_ms),
                adverse_gap_pct * 100,
                self._add_layer_gap_pct_for_target_layer(target_layer) * multiplier * 100,
                multiplier,
            )
            return
        if reason == "first_add_block":
            logger.info(
                "ADD_SKIPPED | reason=first_add_block side=%s price=%s "
                "trigger_source=EXTREME_RETEST pattern=%s anchor_price=%s "
                "layers=%s target_layer=%s last_entry=%s "
                "first_elapsed_seconds=%.1f required_seconds=%s "
                "adverse_gap_pct=%.4f%%",
                side, price, pattern, anchor_price,
                self.state.layers, target_layer, last,
                self._first_entry_elapsed_seconds(ts_ms),
                self.first_add_block_seconds,
                adverse_gap_pct * 100,
            )
            return
        if reason == "add_interval":
            logger.info(
                "ADD_SKIPPED | reason=add_interval side=%s price=%s "
                "trigger_source=EXTREME_RETEST pattern=%s anchor_price=%s "
                "layers=%s target_layer=%s last_entry=%s "
                "elapsed_seconds=%.1f required_seconds=%s "
                "adverse_gap_pct=%.4f%%",
                side, price, pattern, anchor_price,
                self.state.layers, target_layer, last,
                self._add_elapsed_seconds(ts_ms),
                self.add_min_interval_seconds,
                adverse_gap_pct * 100,
            )
            return
        logger.info(
            "ADD_SKIPPED | reason=%s side=%s price=%s "
            "trigger_source=EXTREME_RETEST pattern=%s anchor_price=%s "
            "layers=%s target_layer=%s last_entry=%s elapsed_seconds=%.1f",
            reason, side, price, pattern, anchor_price,
            self.state.layers, target_layer, last,
            self._add_elapsed_seconds(ts_ms),
        )

    def _sync_anchor_from_state(self) -> None:
        """Sync ExtremeRetestAnchor from strategy state fields."""
        anchor = self._extreme_retest_anchor
        anchor.side = self.state.extreme_retest_anchor_side
        anchor.kind = self.state.extreme_retest_anchor_kind
        anchor.price = self.state.extreme_retest_anchor_price
        anchor.candle_ts_ms = self.state.extreme_retest_anchor_candle_ts_ms
        anchor.boll_upper = self.state.extreme_retest_anchor_boll_upper
        anchor.boll_lower = self.state.extreme_retest_anchor_boll_lower
        anchor.sweep_seen = self.state.extreme_retest_sweep_seen
        anchor.sweep_extreme_price = self.state.extreme_retest_sweep_extreme_price
        anchor.sweep_first_seen_ts_ms = self.state.extreme_retest_sweep_first_seen_ts_ms
        anchor.sweep_last_seen_ts_ms = self.state.extreme_retest_sweep_last_seen_ts_ms
        anchor.consumed_watermark_price = self.state.extreme_retest_consumed_watermark_price
        anchor.consumed_anchor_ts_ms = self.state.extreme_retest_consumed_anchor_ts_ms

    def _sync_anchor_to_state(self) -> None:
        """Sync ExtremeRetestAnchor back to strategy state fields."""
        anchor = self._extreme_retest_anchor
        self.state.extreme_retest_anchor_side = anchor.side
        self.state.extreme_retest_anchor_kind = anchor.kind
        self.state.extreme_retest_anchor_price = anchor.price
        self.state.extreme_retest_anchor_candle_ts_ms = anchor.candle_ts_ms
        self.state.extreme_retest_anchor_boll_upper = anchor.boll_upper
        self.state.extreme_retest_anchor_boll_lower = anchor.boll_lower
        self.state.extreme_retest_sweep_seen = anchor.sweep_seen
        self.state.extreme_retest_sweep_extreme_price = anchor.sweep_extreme_price
        self.state.extreme_retest_sweep_first_seen_ts_ms = anchor.sweep_first_seen_ts_ms
        self.state.extreme_retest_sweep_last_seen_ts_ms = anchor.sweep_last_seen_ts_ms
        self.state.extreme_retest_consumed_watermark_price = anchor.consumed_watermark_price
        self.state.extreme_retest_consumed_anchor_ts_ms = anchor.consumed_anchor_ts_ms

    def _maybe_revalidate_extreme_retest_anchor_after_add(self) -> None:
        """Revalidate extreme retest anchor after any ADD changes last_entry_price."""
        self._sync_anchor_from_state()
        anchor = self._extreme_retest_anchor
        effective_required_gap_pct = self._extreme_retest_effective_required_gap_pct()
        _extreme_retest.revalidate_anchor_after_add(
            anchor=anchor,
            last_entry_price=self.state.last_entry_price,
            effective_required_gap_pct=effective_required_gap_pct,
        )
        self._sync_anchor_to_state()

    def restore_extreme_retest_state_from_saved(self, trusted: bool) -> None:
        """Restore extreme retest state from saved state fields during startup recovery.

        Called externally (e.g., from startup recovery) after strategy state is loaded.

        When trusted=False, drops the saved anchor but preserves consumed watermark.
        """
        self._sync_anchor_from_state()
        anchor = self._extreme_retest_anchor
        if not trusted:
            # Drop anchor but keep consumed watermark
            consumed_price = anchor.consumed_watermark_price
            consumed_ts = anchor.consumed_anchor_ts_ms
            anchor.clear()
            anchor.consumed_watermark_price = consumed_price
            anchor.consumed_anchor_ts_ms = consumed_ts
            logger.info(
                "EXTREME_RETEST_STATE_DROPPED | reason=untrusted_saved_state_or_no_valid_anchor"
            )
            self._sync_anchor_to_state()
            return

        if anchor.is_active():
            logger.info("EXTREME_RETEST_STATE_RESTORED")
        self._sync_anchor_to_state()

    def rebuild_extreme_retest_anchor_from_candles(self) -> bool:
        """Try to rebuild anchor from closed candle buffer after startup.

        Returns True if an anchor was successfully rebuilt.
        """
        if self.state.side is None or self.state.layers <= 0:
            return False

        self._sync_anchor_from_state()
        anchor = self._extreme_retest_anchor
        cfg = self._extreme_retest_config
        side = self.state.side

        effective_required_gap_pct = self._extreme_retest_effective_required_gap_pct()

        buf = list(self._candle_buffer)
        if len(buf) < cfg.pivot_left_bars + 1 + cfg.pivot_right_bars:
            logger.info(
                "EXTREME_RETEST_STATE_DROPPED | reason=untrusted_saved_state_or_no_valid_anchor"
            )
            self._sync_anchor_to_state()
            return False

        # Get current boll band from most recent candle in buffer
        latest = buf[-1]
        boll_upper = latest.get("boll_upper", 0)
        boll_lower = latest.get("boll_lower", 0)

        rebuilt = _extreme_retest.rebuild_anchor_from_closed_candles(
            side=side,
            candles=buf,
            boll_upper=boll_upper,
            boll_lower=boll_lower,
            last_entry_price=self.state.last_entry_price,
            effective_required_gap_pct=effective_required_gap_pct,
            consumed_watermark_price=anchor.consumed_watermark_price,
            config=cfg,
        )

        if rebuilt is not None:
            self._extreme_retest_anchor = rebuilt
            self._sync_anchor_to_state()
            return True

        self._sync_anchor_to_state()
        return False
