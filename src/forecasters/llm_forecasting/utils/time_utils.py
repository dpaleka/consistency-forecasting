# Related third-party imports
from datetime import datetime, timedelta


def convert_date_string_to_tuple(date_string):
    """
    Convert a date string of the form 'year-month-day' to a tuple (year, month, day).

    Args:
        date_string (str): A string representing the date in 'year-month-day' format.

    Returns:
        tuple: A tuple containing the year, month, and day as integers.
    """
    # Split the date string by '-'
    parts = date_string.split("-")
    # Check that the date string is in the correct format
    assert len(parts) == 3, "Date string must be in 'year-month-day' format."
    # Convert the parts to integers and return as a tuple
    return tuple(map(int, parts))


def is_more_recent(first_date_str, second_date_str, or_equal_to=False):
    """
    Determine if |second_date_str| is more recent than |first_date_str|.

    Args:
        first_date_str (str): A string representing the first date to compare against. Expected format: 'YYYY-MM-DD'.
        second_date_str (str): A string representing the second date. Expected format: 'YYYY-MM-DD'.

    Returns:
        bool: True if the second date is more recent than the first date, False otherwise.
    """
    first_date_obj = datetime.strptime(first_date_str, "%Y-%m-%d")
    second_date_obj = datetime.strptime(second_date_str, "%Y-%m-%d")
    if or_equal_to:
        return second_date_obj >= first_date_obj
    return second_date_obj > first_date_obj


def get_todays_date():
    """
    Get today's date in Year-Month-Day format.

    Returns:
        str: Year-Month-Day
    """
    today = datetime.today()
    # Format the date as year-month-day
    formatted_date = today.strftime("%Y-%m-%d")

    return formatted_date


def subtract_days_from_date(date_str, k):
    """
    Subtract k days from the given date.

    Args:
        date_str (str): The date to subtract from in the format 'YYYY-MM-DD'.
        k (int): The number of days to subtract.

    Returns:
        str: The resulting date after subtracting k days in the format 'YYYY-MM-DD'.
    """
    # Convert the input date string to a datetime object
    input_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Subtract k days from the input date
    result_date = input_date - timedelta(days=k)

    # Format the result date as year-month-day
    formatted_date = result_date.strftime("%Y-%m-%d")

    return formatted_date