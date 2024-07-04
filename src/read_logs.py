import re


def submission_log_only_stats(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    date_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"

    extracted_lines = []
    for line in lines:
        if re.match(date_pattern, line):
            extracted_lines.append(line.strip())

    return extracted_lines


def extract_element(line, element_name):
    pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - id: ([\d,]+), url: (https?://[^\s,]+)?, title: ([^,]+)?, samples: ([\d,]+)?, adv_prediction: ([\d.,%]+)?, basic_prediction: ([\d.,%]+)?, meta_prediction: ([\d.,%]+)?, submission: ([\d.,%]+)?,"
    match = re.match(pattern, line)
    if match:
        groups = match.groups()
        element_mapping = {
            "datetime": groups[0],
            "id": groups[1] or "",
            "url": groups[2] or "",
            "title": groups[3] or "",
            "samples": groups[4] or "",
            "adv_prediction": groups[5] or "",
            "basic_prediction": groups[6] or "",
            "meta_prediction": groups[7] or "",
            "submission": groups[8] or "",
        }
        if element_name in element_mapping:
            return element_mapping[element_name]
        else:
            return None
    else:
        return None


def convert_to_decimal(percentage_str):
    if percentage_str:
        return float(percentage_str.strip("%")) / 100
    else:
        return None
