import re
import os


def submission_log_only_stats(filename):
    if not os.path.exists(filename):
        print("Log file does not exist:", filename)
        return []

    with open(filename, "r") as file:
        lines = file.readlines()

    date_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"

    extracted_lines = []
    for line in lines:
        if re.match(date_pattern, line):
            extracted_lines.append(line.strip())

    return extracted_lines


def extract_element(line, element_name):
    parts = line.split(" - ", 1)
    if len(parts) != 2:
        return ""
    element_mapping = {}

    element_mapping["datetime"] = parts[0]

    elements = parts[1].split(", ")

    for element in elements:
        if ": " in element:
            key, value = element.split(": ", 1)
            element_mapping[key] = value
        else:
            element_mapping[element] = ""

    return element_mapping.get(element_name, "")


def convert_to_decimal(percentage_str):
    if percentage_str:
        return float(percentage_str.strip("%")) / 100
    else:
        return None
