import os
import re
import json
from datetime import datetime, timezone


def filter_jsonl_files(directory):
    for filename in os.listdir(directory):
        parts = filename.split("_")
        date_parts = [part for part in parts if re.match(r"^\d{8}$", part)]
        if "__" in filename:
            continue

        if len(date_parts) == 2:
            start_date, end_date = date_parts
        elif len(date_parts) == 1:
            date_part = date_parts[0]
            start_date = None
            end_date = None

            if parts.index(date_part) == len(parts) - 1:
                end_date = date_part
            else:
                start_date = date_part
        else:
            continue

        if start_date is not None:
            start_datetime = datetime.strptime(start_date, "%Y%m%d").replace(
                tzinfo=timezone.utc
            )
        else:
            start_datetime = datetime.min.replace(tzinfo=timezone.utc)

        if end_date is not None:
            end_datetime = (
                datetime.strptime(end_date, "%Y%m%d") + datetime.timedelta(days=1)
            ).replace(tzinfo=timezone.utc)
        else:
            end_datetime = datetime.max.replace(tzinfo=timezone.utc)

        filter_jsonl_file(
            os.path.join(directory, filename), start_datetime, end_datetime
        )


def filter_jsonl_file(file_path, start_datetime, end_datetime):
    with open(file_path, "r") as input_file, open(
        f"{file_path}__date_filtered.jsonl", "w"
    ) as output_file:
        for line in input_file:
            entry = json.loads(line)
            resolution_date = datetime.fromisoformat(
                entry["resolution_date"].replace("Z", "+00:00")
            ).replace(tzinfo=timezone.utc)
            if start_datetime <= resolution_date < end_datetime:
                output_file.write(line)


if __name__ == "__main__":
    filter_jsonl_files(os.getcwd())
