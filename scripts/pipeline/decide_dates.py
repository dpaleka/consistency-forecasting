import datetime as dt


def noneify_if_not_in_range(
    date: dt.datetime | None, min_date: dt.datetime | None, max_date: dt.datetime | None
) -> dt.datetime | None:
    if date is None:
        return None
    if min_date is not None:
        if date < min_date:
            return None
    if max_date is not None:
        if date > max_date:
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
    last_updated_date (dt.datetime, optional): The last updated time of the question.
    min_date (dt.datetime, optional): The minimum allowed date for the logical a priori resolution_date
    max_date (dt.datetime, optional): The maximum allowed date for the logical a priori resolution_date

    Returns:
    dt.datetime: The chosen resolution date.
    """
    print("Parameters:")
    print(f"close_date: {close_date}")
    print(f"resolve_date: {resolve_date}")
    print(f"min_date: {min_date}")
    print(f"max_date: {max_date}")
    close_date = noneify_if_not_in_range(close_date, min_date, max_date)
    resolve_date = noneify_if_not_in_range(resolve_date, min_date, max_date)

    if resolve_date is not None and close_date is not None:
        if resolve_date < close_date:
            # We pick close_date because it's possible it resolved early
            return close_date
        else:
            # We pick resolve_date because it's possible close_date was early
            return resolve_date
    elif resolve_date is None and close_date is not None:
        return close_date
    elif resolve_date is not None and close_date is None:
        print("No valid close date found, using resolve date")
        return resolve_date
    elif resolve_date is None and close_date is None:
        print("No valid close or resolve date found, returning None")
        return None


def decide_question_created_date(
    created_date: dt.datetime,
    published_date: dt.datetime | None,
    min_date: dt.datetime = None,
    max_date: dt.datetime = None,
) -> dt.datetime:
    return NotImplementedError
