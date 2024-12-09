import json
import sys
import os
import uuid


def generate_random_uid():
    uid = str(uuid.uuid4())[
        :16
    ]  # Generate a random UUID and take the first 16 characters
    return uid


# Define a function to extract relevant fields
def extract_fields(data):
    result = {}
    result["id"] = generate_random_uid()
    result["title"] = data["question"]
    result["body"] = data["background"]
    result["resolution_date"] = data["date_resolve_at"]
    result["question_type"] = data["question_type"].lower()
    result["data_source"] = data["data_source"]
    result["url"] = data["url"]
    result["metadata"] = {}
    for key, value in data.items():
        if key not in [
            "id",
            "question",
            "background",
            "date_resolve_at",
            "question_type",
            "data_source",
            "url",
            "resolution",
            "community_predictions",
        ]:
            result["metadata"][key] = value
    result["resolution"] = (
        bool(data["resolution"]) if data["resolution"] is not None else None
    )
    return result


# Check if the input file is provided
if len(sys.argv) < 2:
    print("Usage: python script.py input_file.json")
    sys.exit(1)

# Get the input file name
input_file = sys.argv[1]

# Load the input JSON file
with open(input_file, "r") as f:
    data = json.load(f)

# Generate the output file name
output_file = os.path.splitext(input_file)[0] + "_formatted.jsonl"

# Convert the data and write to the output file
with open(output_file, "w") as f:
    for item in data:
        formatted_data = extract_fields(item)
        f.write(json.dumps(formatted_data) + "\n")

print(f"Output written to {output_file}")
