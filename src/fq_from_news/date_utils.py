from datetime import datetime, timedelta
import calendar


def get_month_date_range(year: int, month: int) -> tuple:
    """
    Given a year and a month, returns the first and last date of that month.

    Args:
        year (int): The year as an integer.
        month (int): The month as an integer.

    Returns:
        Tuple[date, date]: Tuple containing the start date and end date of the month as date objects.
    """
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    return start_date.date(), end_date.date()


def parse_date(date_str: str):
    """
    Given a date in the format YYYY-MM-DD, returns the corresponding date object.
    Has validation for the correct type.

    Args:
        date_str (str): The date as a string.

    Returns:
        date: The date as a date object.

    Raises:
        TypeError: If the date string is not in the YYYY-MM-DD format.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise TypeError(
            f"Invalid date format: {date_str}. Date must be in YYYY-MM-DD format."
        )


def last_datetime_of_month(dt: datetime) -> datetime:
    """
    Given a datetime object, returns the last datetime of the last day in the same month.
    The time is set to 23:59:59 of that day.

    Args:
        dt (datetime): A datetime object representing any date in the month.

    Returns:
        datetime: A datetime object representing the last datetime of the last day in the month.
    """
    # Get the last day of the month
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    # Create a datetime object for the last day of the month at 23:59:59
    return datetime(dt.year, dt.month, last_day, 23, 59, 59)


def format_news_range_date(date: datetime):
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    month_name = months[date.month - 1]
    return f"{month_name}-{date.day}-{date.year}"
