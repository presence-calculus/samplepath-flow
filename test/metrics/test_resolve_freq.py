# tests/test_resolve_freq.py
import pytest
import pandas as pd

from spath.metrics import _resolve_freq


@pytest.mark.parametrize(
    "bucket,expected",
    [
        ("day", "D"),
        ("DAY", "D"),
        (" dAy  ", "D"),
        ("month", "MS"),
        ("MONTH", "MS"),
    ],
)
def test_basic_day_month(bucket, expected):
    assert _resolve_freq(bucket) == expected


@pytest.mark.parametrize(
    "anchor,expected",
    [
        ("SUN", "W-SUN"),
        ("Mon", "W-MON"),
        ("tue", "W-TUE"),
        ("FRI", "W-FRI"),
    ],
)
def test_week_with_anchor(anchor, expected):
    assert _resolve_freq("week", week_anchor=anchor) == expected


@pytest.mark.parametrize(
    "anchor,expected",
    [
        ("JAN", "QS-JAN"),
        ("feb", "QS-FEB"),
        ("MAR", "QS-MAR"),
    ],
)
def test_quarter_with_anchor(anchor, expected):
    assert _resolve_freq("quarter", quarter_anchor=anchor) == expected


@pytest.mark.parametrize(
    "anchor,expected",
    [
        ("JAN", "YS-JAN"),
        ("apr", "YS-APR"),
        ("DEC", "YS-DEC"),
    ],
)
def test_year_with_anchor(anchor, expected):
    assert _resolve_freq("year", year_anchor=anchor) == expected


# in test_passthrough_valid_pandas_aliases
@pytest.mark.parametrize("alias", [
    "D",
    "h",       # was "H"
    "W-MON",
    "W-SUN",
    "MS",
    "ME",      # was "M"
    "QS-JAN",
    "QE",      # was "Q"
    "YS-JAN",
    "YE",      # was "Y"
])
def test_passthrough_valid_pandas_aliases(alias):
    out = _resolve_freq(alias)
    assert out == alias
    pd.tseries.frequencies.to_offset(out)



@pytest.mark.parametrize(
    "bucket_kwargs",
    [
        ({"bucket": "week", "week_anchor": "XYZ"}),        # questionable week anchor
        ({"bucket": "quarter", "quarter_anchor": "???"}),  # questionable quarter anchor
        ({"bucket": "year", "year_anchor": "13"}),         # invalid month code
    ],
)
def test_invalid_anchor_values_return_string_for_downstream_validation(bucket_kwargs):
    # Helper just formats; downstream consumers validate with pandas
    out = _resolve_freq(**bucket_kwargs)
    assert isinstance(out, str)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "fortnight",
        "every-other-wed",
        "wkly",
        "biweekly",
    ],
)
def test_unknown_frequency_raises_value_error(bad):
    with pytest.raises(ValueError) as ei:
        _resolve_freq(bad)
    msg = str(ei.value)
    assert "Unknown frequency" in msg
    assert "pandas alias" in msg
    assert "{day, week, month, quarter, year}" in msg
