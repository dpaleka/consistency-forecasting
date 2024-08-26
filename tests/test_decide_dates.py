import pytest
import datetime as dt
import sys
from common.path_utils import get_scripts_path

sys.path.append(str(get_scripts_path()))

from pipeline.decide_dates import (
    noneify_if_not_in_range,
    decide_resolution_date,
)


@pytest.fixture
def date_range():
    return dt.datetime(2024, 3, 1), dt.datetime(2024, 7, 1)


def test_noneify_if_not_in_range(date_range):
    min_date, max_date = date_range

    assert noneify_if_not_in_range(
        dt.datetime(2024, 4, 1), min_date, max_date
    ) == dt.datetime(2024, 4, 1)
    assert noneify_if_not_in_range(dt.datetime(2024, 2, 1), min_date, max_date) is None
    assert noneify_if_not_in_range(dt.datetime(2024, 8, 1), min_date, max_date) is None
    assert noneify_if_not_in_range(None, min_date, max_date) is None


def test_decide_resolution_date_both_in_close_after(date_range):
    """
    If both are in the range, and resolve_date < close_date, we return close_date because it means that the market resolved early (and would induce Navalny bias)
    Of course, it's possible that the close_date is still earlier than the true logical resolution date, but we can't know that without reasoning about the question in depth.
    """
    min_date, max_date = date_range
    close_date = dt.datetime(2024, 5, 1)
    resolve_date = dt.datetime(2024, 4, 1)

    assert (
        decide_resolution_date(close_date, resolve_date, min_date, max_date)
        == close_date
    )


def test_decide_resolution_date_both_in_close_before(date_range):
    """
    If both are in the range, and close_date < resolve_date, we return resolve_date because it's possible that it was closing early.
    Of course, resolve_date could still be earlier than the true logical resolution date (and thus induce Navalny bias), but we can't know that without reasoning about the question in depth.
    """
    min_date, max_date = date_range
    close_date = dt.datetime(2024, 4, 1)
    resolve_date = dt.datetime(2024, 5, 1)

    assert (
        decide_resolution_date(close_date, resolve_date, min_date, max_date)
        == resolve_date
    )


def test_decide_resolution_date_close_after(date_range):
    """
    If the close date is after the max date, we should always return None, no matter what
    """
    min_date, max_date = date_range
    close_date = dt.datetime(2024, 8, 1)
    for resolve_date in [
        dt.datetime(2024, 2, 1),
        dt.datetime(2024, 4, 1),
        dt.datetime(2024, 6, 1),
        dt.datetime(2024, 8, 15),
    ]:
        assert (
            decide_resolution_date(close_date, resolve_date, min_date, max_date) is None
        )


def test_decide_resolution_date_resolve_after(date_range):
    """
    If the resolve date is after the max date, but close_date is in the range, what do we do? Was it an early close or a late resolve?
    We should return None to be safe.
    """
    min_date, max_date = date_range
    resolve_date = dt.datetime(2024, 8, 15)
    for close_date in [
        dt.datetime(2024, 2, 1),
        dt.datetime(2024, 4, 1),
        dt.datetime(2024, 6, 1),
        dt.datetime(2024, 8, 1),
        dt.datetime(2024, 9, 1),
    ]:
        assert (
            decide_resolution_date(close_date, resolve_date, min_date, max_date) is None
        )


def test_decide_resolution_date_only_resolve_in_close_before(date_range):
    """
    If the close date is before the min date, but the resolve date is in the range:
    It could be that the market was late to resolve and the true resolution date is actually < min_date
    We decide to pass only resolve_date > min_date + 7 days; as a week feels like a reasonable buffer for most markets to resolve.
    """
    min_date, max_date = date_range
    close_date = dt.datetime(2024, 2, 1)
    barely_in_range_resolve_date = dt.datetime(
        2024, 3, 6
    )  # We don't care about whether the boundary is inclusive or not
    assert (
        decide_resolution_date(
            close_date, barely_in_range_resolve_date, min_date, max_date
        )
        is None
    )
    clearly_in_range_resolve_date = dt.datetime(2024, 3, 8)
    assert (
        decide_resolution_date(
            close_date, clearly_in_range_resolve_date, min_date, max_date
        )
        == clearly_in_range_resolve_date
    )


def test_decide_resolution_date_resolve_before_min_date(date_range):
    """
    If the market resolved before min_date, we return None whatever happens.
    The question needs to be unresolved for forecasters with information up to min_date.
    """
    min_date, max_date = date_range
    resolve_date = dt.datetime(2024, 2, 1)
    for close_date in [
        dt.datetime(2024, 1, 15),
        dt.datetime(2024, 2, 15),
        dt.datetime(2024, 3, 15),
        dt.datetime(2024, 8, 15),
    ]:
        assert (
            decide_resolution_date(close_date, resolve_date, min_date, max_date) is None
        )


def test_decide_resolution_date_neither_valid(date_range):
    min_date, max_date = date_range
    close_date = dt.datetime(2024, 2, 1)
    resolve_date = dt.datetime(2024, 8, 1)

    assert decide_resolution_date(close_date, resolve_date, min_date, max_date) is None


def test_decide_resolution_date_no_max_date(date_range):
    """
    Test only with min_date bound. For the reasons discussed above, we return the max of close_date and resolve_date.
    """
    min_date = date_range[0]
    max_date = None
    close_date = dt.datetime(2024, 4, 1)
    resolve_date = dt.datetime(2024, 5, 1)

    assert decide_resolution_date(close_date, resolve_date) == resolve_date

    close_date = dt.datetime(2024, 5, 1)
    resolve_date = dt.datetime(2024, 4, 1)

    assert decide_resolution_date(close_date, resolve_date) == close_date


def test_no_bounds():
    """
    Test with no min or max date bounds. For the reasons discussed above, we return the max of close_date and resolve_date.
    """
    close_date = dt.datetime(2024, 4, 1)
    resolve_date = dt.datetime(2024, 5, 1)
    assert decide_resolution_date(close_date, resolve_date) == resolve_date

    close_date = dt.datetime(2024, 5, 1)
    resolve_date = dt.datetime(2024, 4, 1)
    assert decide_resolution_date(close_date, resolve_date) == close_date
