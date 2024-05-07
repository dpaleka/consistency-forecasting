# Standard library imports
from collections import Counter


def flatten_list(nested_list):
    flat_list = [item for sublist in nested_list for item in sublist]
    return flat_list


def most_frequent_item(lst):
    """
    Return the most frequent item in the given list.

    If there are multiple items with the same highest frequency, one of them is
    returned.

    Args:
        lst (list): The list from which to find the most frequent item.

    Returns:
        The most frequent item in the list.
    """
    if not lst:
        return None  # Return None if the list is empty
    # Count the frequency of each item in the list
    count = Counter(lst)
    # Find the item with the highest frequency
    most_common = count.most_common(1)
    return most_common[0][0]  # Return the item (not its count)