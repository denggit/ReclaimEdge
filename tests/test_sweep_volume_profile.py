"""Unit tests for SweepVolumeProfile."""
from __future__ import annotations

import pytest
from src.strategies.sweep_volume_profile import SweepVolumeProfile


def test_empty_profile_returns_none() -> None:
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    assert sp.poc_price() is None


def test_single_bucket_is_poc() -> None:
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(1900.12, 10.0)
    poc = sp.poc_price()
    assert poc is not None
    # Should round to nearest bucket
    assert poc > 0


def test_poc_returns_highest_volume_bucket() -> None:
    sp = SweepVolumeProfile(bucket_pct=0.001)  # 0.1% buckets
    sp.add(100.0, 5.0)
    sp.add(100.5, 20.0)  # most volume here
    sp.add(101.0, 3.0)
    poc = sp.poc_price()
    assert poc is not None
    # The bucket with 20.0 volume should be the POC
    bucket_size = 100.5 * 0.001  # ~0.1005
    expected_bucket = round(100.5 / bucket_size) * bucket_size
    assert abs(poc - expected_bucket) < 0.0001


def test_tied_buckets_return_midpoint() -> None:
    sp = SweepVolumeProfile(bucket_pct=0.001)
    sp.add(100.0, 10.0)
    sp.add(101.0, 10.0)
    poc = sp.poc_price()
    assert poc is not None
    # Midpoint of the two bucket centers
    bucket_size_100 = 100.0 * 0.001
    bucket_size_101 = 101.0 * 0.001
    bp1 = round(100.0 / bucket_size_100) * bucket_size_100
    bp2 = round(101.0 / bucket_size_101) * bucket_size_101
    assert abs(poc - (bp1 + bp2) / 2.0) < 0.0001


def test_reset_clears_buckets() -> None:
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(1900.0, 10.0)
    assert sp.poc_price() is not None
    sp.reset()
    assert sp.poc_price() is None
    assert len(sp.buckets) == 0


def test_negative_volume_clamped_to_zero() -> None:
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(1900.0, -5.0)
    assert sp.poc_price() is None  # no positive volume recorded


def test_zero_price_ignored() -> None:
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(0.0, 10.0)
    assert sp.poc_price() is None


def test_bucket_rounding() -> None:
    sp = SweepVolumeProfile(bucket_pct=0.001)
    # bucket_size at ~100.0 = 0.1
    # Adding at 100.05 and 100.07 should go to the same bucket
    sp.add(100.05, 5.0)
    sp.add(100.07, 3.0)
    sp.add(100.15, 10.0)  # This is in a different bucket, more volume
    poc = sp.poc_price()
    assert poc is not None
    # The 100.15 bucket should be POC (10.0 volume)
    bucket_size = 100.15 * 0.001
    expected = round(100.15 / bucket_size) * bucket_size
    assert abs(poc - expected) < 0.0001


def test_invalid_bucket_pct_raises() -> None:
    with pytest.raises(ValueError):
        SweepVolumeProfile(bucket_pct=0.0)
    with pytest.raises(ValueError):
        SweepVolumeProfile(bucket_pct=-0.001)
