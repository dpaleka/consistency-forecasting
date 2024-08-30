import datetime as dt


def noneify_if_not_in_range(
    date: dt.datetime | None, min_date: dt.datetime | None, max_date: dt.datetime | None
) -> dt.datetime | None:
    if date is None:
        return None
    if min_date is not None and date < min_date:
        return None
    if max_date is not None and date > max_date:
        return None
    return date


def decide_resolution_date(
    close_date: dt.datetime,
    resolve_date: dt.datetime,
    min_date: dt.datetime = None,
    max_date: dt.datetime = None,
) -> dt.datetime | None:
    """
    Decides the logical a priori resolution date between close_date and resolve_date.

    Args:
    close_date (dt.datetime): The close time of the question.
    resolve_date (dt.datetime): The resolve time of the question.
    min_date (dt.datetime, optional): The minimum allowed date for the logical a priori resolution_date
    max_date (dt.datetime, optional): The maximum allowed date for the logical a priori resolution_date

    Returns:
    dt.datetime | None: The chosen resolution date or None if no valid date is found.
    """
    print("Parameters:")
    print(f"close_date: {close_date}")
    print(f"resolve_date: {resolve_date}")
    print(f"min_date: {min_date}")
    print(f"max_date: {max_date}")

    later_date = (
        max(close_date, resolve_date)
        if close_date is not None and resolve_date is not None
        else (close_date or resolve_date)
    )

    if min_date and later_date and later_date < min_date:
        print("Later of close_date and resolve_date is before min_date, returning None")
        return None

    if max_date and later_date and later_date > max_date:
        print("Later of close_date and resolve_date is after max_date, returning None")
        return

    if min_date and resolve_date and resolve_date < min_date:
        print("Resolve date is before min_date, returning None")
        return None

    close_date = noneify_if_not_in_range(close_date, min_date, max_date)
    resolve_date = noneify_if_not_in_range(resolve_date, min_date, max_date)

    if min_date and close_date is None and resolve_date:
        if resolve_date < min_date + dt.timedelta(days=7):
            print("Resolve date is within 7 days of min_date, returning None")
            return None

    if resolve_date is not None and close_date is not None:
        if resolve_date < close_date:
            print("Resolve date is before close date, returning close date")
            return close_date
        else:
            print("Close date is before resolve date, returning resolve date")
            return resolve_date
    elif close_date is not None:
        print("Only close date is valid, returning close date")
        return close_date
    elif resolve_date is not None:
        print("Only resolve date is valid, returning resolve date")
        return resolve_date
    else:
        print("No valid close or resolve date found, returning None")
        return None


def too_close_dates(
    question_created: dt.datetime | None,
    resolution_date: dt.datetime,
) -> bool:
    """
    We discard questions where the resolution date is within 3 days of the creation date,
    for likely being not serious enough to include in any testing.
    """
    assert resolution_date is not None
    if question_created is None:
        return False
    assert question_created <= resolution_date
    if resolution_date - question_created <= dt.timedelta(days=3):
        return True
