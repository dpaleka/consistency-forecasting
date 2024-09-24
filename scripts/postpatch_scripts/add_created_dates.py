"""Add created_date to tuples"""

import os
import json
from datetime import datetime


def load_jsonl(file_path) -> list[dict]:
    """Load the JSONL file into a list of dicts."""
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def get_earliest_created_date(metadata: dict) -> datetime | None:
    """Extract the earliest created_date from the metadata's base_sentences."""
    base_sentences = metadata.get("base_sentences", {})
    created_dates = []
    for key, question_data in base_sentences.items():
        created_date = question_data.get("created_date")
        if created_date:
            try:
                created_dates.append(datetime.fromisoformat(created_date.rstrip("Z")))
            except ValueError:
                pass
    return min(created_dates) if created_dates else None


def update_created_dates(data: list[dict]):
    """Update each question's created_date if it's null, based on the metadata's earliest created_date."""
    for item in data:
        metadata = item.get("metadata", {})
        earliest_created_date = get_earliest_created_date(metadata)

        # Update the created_date for all fields except 'metadata'
        for key, question_data in item.items():
            if key != "metadata" and isinstance(question_data, dict):
                if question_data.get("created_date") is None and earliest_created_date:
                    question_data["created_date"] = (
                        earliest_created_date.isoformat() + "Z"
                    )


def save_jsonl(data: list[dict], output_file_path: str):
    """Save the updated data back to a JSONL file."""
    with open(output_file_path, "w") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


def process_jsonl_files(input_directory: str, output_directory: str):
    """Process all .jsonl files in the input_directory and save them to output_directory."""
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each .jsonl file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".jsonl"):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)

            # Load, update, and save the file
            data = load_jsonl(input_file_path)
            update_created_dates(data)
            save_jsonl(data, output_file_path)

            print(f"Processed {filename} and saved to {output_file_path}")


input_directory = "data/tuples_scraped"
output_directory = "data/tuples_scraped_"

process_jsonl_files(input_directory, output_directory)
