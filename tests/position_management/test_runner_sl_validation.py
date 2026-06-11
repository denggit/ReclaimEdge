from decimal import Decimal

from src.position_management.runner.sl_validation import validate_runner_protective_sl_price


def test_runner_sl_validation_skips_when_current_price_missing() -> None:
    result = validate_runner_protective_sl_price(
        side="LONG",
        current_price=Decimal("0"),
        new_sl_price=Decimal("1679.22"),
    )

    assert result.valid
    assert result.reason == "missing_current_price_skip_validation"


def test_runner_sl_validation_rejects_invalid_prices_when_current_price_known() -> None:
    long_result = validate_runner_protective_sl_price(
        side="LONG",
        current_price=Decimal("1678.18"),
        new_sl_price=Decimal("1679.22"),
    )
    short_result = validate_runner_protective_sl_price(
        side="SHORT",
        current_price=Decimal("1678.18"),
        new_sl_price=Decimal("1677.00"),
    )

    assert not long_result.valid
    assert long_result.reason == "long_sl_not_below_last_price"
    assert not short_result.valid
    assert short_result.reason == "short_sl_not_above_last_price"
