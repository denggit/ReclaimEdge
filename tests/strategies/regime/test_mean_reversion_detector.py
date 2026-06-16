from __future__ import annotations

from src.strategies.regime.anchored_cvd import (
    AnchoredCvdConfig,
    build_anchored_cvd_state,
)
from src.strategies.regime.mean_reversion_detector import (
    MeanReversionDetector,
    MeanReversionDetectorConfig,
)
from src.strategies.regime.types import (
    BandSnapshot,
    BreakoutSnapshot,
    TrendState,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _breakout(direction: str = "UP", ts_ms: int = 10000, price: float = 2110.0,
               upper: float = 2100.0, middle: float = 2000.0, lower: float = 1900.0) -> BreakoutSnapshot:
    return BreakoutSnapshot(
        direction=direction,  # type: ignore[arg-type]
        ts_ms=ts_ms,
        price=price,
        band=BandSnapshot(upper=upper, middle=middle, lower=lower, candle_ts_ms=ts_ms),
        anchor_cvd=100.0,
        anchor_volume=1000.0,
    )


def _cvd_diverging_up() -> "AnchoredCvdState":
    """CVD going down while price goes up (mean-reversion signal for UP breakout)."""
    return build_anchored_cvd_state(
        anchor_ts_ms=10000, current_ts_ms=20000,
        anchor_cvd=100.0, current_cvd=85.0,
        episode_buy_volume=30.0, episode_sell_volume=70.0,
        episode_cvd_max=100.0, episode_cvd_min=85.0,
    )


def _cvd_diverging_down() -> "AnchoredCvdState":
    """CVD going up while price goes down (mean-reversion signal for DOWN breakout)."""
    return build_anchored_cvd_state(
        anchor_ts_ms=10000, current_ts_ms=20000,
        anchor_cvd=200.0, current_cvd=250.0,
        episode_buy_volume=70.0, episode_sell_volume=30.0,
        episode_cvd_max=250.0, episode_cvd_min=200.0,
    )


def _make_detector(mr_cfg: dict | None = None, cvd_cfg: dict | None = None) -> MeanReversionDetector:
    return MeanReversionDetector(
        MeanReversionDetectorConfig(**(mr_cfg or {})),
        AnchoredCvdConfig(**(cvd_cfg or {})),
    )


# ── Tests ─────────────────────────────────────────────────────────────


class TestTrendUpCandidateBlocksShort:
    """Test 1: TREND_UP_CANDIDATE not failed → blocks MEAN_REVERSION_SHORT."""

    def test_candidate_active_blocks_short(self):
        detector = _make_detector()
        bo = _breakout("UP")
        cvd = _cvd_diverging_up()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_UP_CANDIDATE,
            trend_failed=False,
            trend_failure_reason=None,
            trend_blocks_mean_reversion=True,
            price_reclaimed_inside=True,
            price_new_extreme=True,
        )
        assert gate.allowed is False
        assert "blocks_mean_reversion" in gate.reason


class TestTrendUpConfirmedBlocksShort:
    """Test 2: TREND_UP_CONFIRMED → blocks MEAN_REVERSION_SHORT."""

    def test_confirmed_blocks_short(self):
        detector = _make_detector()
        bo = _breakout("UP")
        cvd = _cvd_diverging_up()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_UP_CONFIRMED,
            trend_failed=False,
            trend_failure_reason=None,
            trend_blocks_mean_reversion=True,
            price_reclaimed_inside=True,
            price_new_extreme=True,
        )
        assert gate.allowed is False
        assert "blocks_mean_reversion" in gate.reason


class TestTrendUpFailedAllowsShort:
    """Test 3: TREND_UP_CANDIDATE failed + inside reclaim + CVD divergence → allows SHORT."""

    def test_failed_with_reclaim_and_divergence_allows_short(self):
        detector = _make_detector()
        bo = _breakout("UP")
        cvd = _cvd_diverging_up()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_UP_CANDIDATE,
            trend_failed=True,
            trend_failure_reason="cvd_diverges_from_price",
            trend_blocks_mean_reversion=False,
            price_reclaimed_inside=True,
            price_new_extreme=True,
        )
        assert gate.allowed is True
        assert gate.side == "SHORT"

    def test_failed_but_no_reclaim_blocks_short(self):
        detector = _make_detector()
        bo = _breakout("UP")
        cvd = _cvd_diverging_up()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_UP_CANDIDATE,
            trend_failed=True,
            trend_failure_reason="cvd_diverges_from_price",
            trend_blocks_mean_reversion=False,
            price_reclaimed_inside=False,  # no reclaim
            price_new_extreme=True,
        )
        assert gate.allowed is False
        assert "not_reclaimed" in gate.reason


class TestTrendDownCandidateBlocksLong:
    """Test 4: TREND_DOWN_CANDIDATE → blocks MEAN_REVERSION_LONG. Down failed → allows LONG."""

    def test_candidate_active_blocks_long(self):
        detector = _make_detector()
        bo = _breakout("DOWN")
        cvd = _cvd_diverging_down()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_DOWN_CANDIDATE,
            trend_failed=False,
            trend_failure_reason=None,
            trend_blocks_mean_reversion=True,
            price_reclaimed_inside=True,
            price_new_extreme=True,
        )
        assert gate.allowed is False
        assert "blocks_mean_reversion" in gate.reason

    def test_failed_allows_long(self):
        detector = _make_detector()
        bo = _breakout("DOWN")
        cvd = _cvd_diverging_down()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_DOWN_CANDIDATE,
            trend_failed=True,
            trend_failure_reason="cvd_diverges_from_price",
            trend_blocks_mean_reversion=False,
            price_reclaimed_inside=True,
            price_new_extreme=True,
        )
        assert gate.allowed is True
        assert gate.side == "LONG"


class TestNoTrendCandidateAllowsReversion:
    """Test 5: No trend candidate → ordinary mean-reversion gate passes."""

    def test_no_trend_allows_short_on_up_breakout(self):
        detector = _make_detector()
        bo = _breakout("UP")
        cvd = _cvd_diverging_up()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.NO_TREND,
            trend_failed=False,
            trend_failure_reason=None,
            trend_blocks_mean_reversion=False,
            price_reclaimed_inside=False,
            price_new_extreme=False,
        )
        assert gate.allowed is True
        assert gate.side == "SHORT"

    def test_no_trend_allows_long_on_down_breakout(self):
        detector = _make_detector()
        bo = _breakout("DOWN")
        cvd = _cvd_diverging_down()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.NO_TREND,
            trend_failed=False,
            trend_failure_reason=None,
            trend_blocks_mean_reversion=False,
            price_reclaimed_inside=False,
            price_new_extreme=False,
        )
        assert gate.allowed is True
        assert gate.side == "LONG"


# ── New tests ────────────────────────────────────────────────────────────


class TestTrendBlocksMeanReversion:
    """trend_blocks_mean_reversion=True blocks MR regardless of trend_state."""

    def test_blocks_short_on_up_breakout(self):
        detector = _make_detector()
        bo = _breakout("UP")
        cvd = _cvd_diverging_up()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_UP_CANDIDATE,
            trend_failed=False,
            trend_failure_reason=None,
            trend_blocks_mean_reversion=True,
            price_reclaimed_inside=True,
            price_new_extreme=True,
        )
        assert gate.allowed is False
        assert "blocks_mean_reversion" in gate.reason
        assert gate.side == "SHORT"

    def test_blocks_long_on_down_breakout(self):
        detector = _make_detector()
        bo = _breakout("DOWN")
        cvd = _cvd_diverging_down()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_DOWN_CANDIDATE,
            trend_failed=False,
            trend_failure_reason=None,
            trend_blocks_mean_reversion=True,
            price_reclaimed_inside=True,
            price_new_extreme=True,
        )
        assert gate.allowed is False
        assert "blocks_mean_reversion" in gate.reason
        assert gate.side == "LONG"


class TestHistoricalCvdDivergence:
    """cvd_divergence_seen allows reclaim tick without new extreme."""

    def test_historical_divergence_allows_short_without_new_extreme(self):
        detector = _make_detector()
        bo = _breakout("UP")
        cvd = _cvd_diverging_up()

        # Reclaim tick: price is back inside, NOT a new extreme,
        # but CVD divergence was seen during the outside phase
        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_UP_CANDIDATE,
            trend_failed=True,
            trend_failure_reason="fast_reclaim_with_cvd_divergence",
            trend_blocks_mean_reversion=False,
            price_reclaimed_inside=True,
            price_new_extreme=False,  # reclaim tick is NOT new extreme
            cvd_divergence_seen=True,  # but divergence was seen earlier
        )
        assert gate.allowed is True
        assert gate.side == "SHORT"

    def test_historical_divergence_allows_long_without_new_extreme(self):
        detector = _make_detector()
        bo = _breakout("DOWN")
        cvd = _cvd_diverging_down()

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_DOWN_CANDIDATE,
            trend_failed=True,
            trend_failure_reason="fast_reclaim_with_cvd_divergence",
            trend_blocks_mean_reversion=False,
            price_reclaimed_inside=True,
            price_new_extreme=False,
            cvd_divergence_seen=True,
        )
        assert gate.allowed is True
        assert gate.side == "LONG"

    def test_no_divergence_at_all_blocks(self):
        detector = _make_detector()
        bo = _breakout("UP")
        # CVD NOT diverging — confirms trend
        cvd = build_anchored_cvd_state(
            anchor_ts_ms=10000, current_ts_ms=20000,
            anchor_cvd=100.0, current_cvd=160.0,
            episode_buy_volume=80.0, episode_sell_volume=20.0,
            episode_cvd_max=160.0, episode_cvd_min=100.0,
        )

        gate = detector.evaluate(
            breakout=bo,
            anchored_cvd=cvd,
            trend_state=TrendState.TREND_UP_CANDIDATE,
            trend_failed=True,
            trend_failure_reason="inside_reclaim_too_long",
            trend_blocks_mean_reversion=False,
            price_reclaimed_inside=True,
            price_new_extreme=False,
            cvd_divergence_seen=False,  # no divergence ever seen
        )
        assert gate.allowed is False
        assert "cvd_not_diverging" in gate.reason
