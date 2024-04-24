#!/usr/bin/env python3
import subprocess
import sys

# Define the script to run and the list of JSONL files to validate
BASE_DIR = "src/"
script = f"{BASE_DIR}/validate_fq_jsonl.py"
REAL_DATA_DIR = f"{BASE_DIR}/data/fq/real"
# TODO add all
jsonl_files = [
    f"{REAL_DATA_DIR}/questions_cleaned_formatted.jsonl",
]


# Function to run the validation script on each file
def validate_file(file_path):
    result = subprocess.run(
        ["python3", script, "--filename", file_path], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Validation failed for {file_path}.")
        print(result.stderr)
        return False
    return True


# Main logic
if __name__ == "__main__":
    all_valid = True
    for file_path in jsonl_files:
        if not validate_file(file_path):
            all_valid = False
            break  # Stop on first failure

    if not all_valid:
        sys.exit(1)  # Exit with a non-zero status code to indicate failure

    print("Validation successful for all files.")
