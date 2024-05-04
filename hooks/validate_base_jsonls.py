#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

# Define the script to run and the list of JSONL files to validate
# Intended to be run from the root of the repository, as when running the pre-commit hook

# Need to do this hack to import stuff
sys.path.append(".")
from src.common.path_utils import get_data_path, get_src_path

script = f"{get_src_path()}/validate_fq_jsonl.py"
REAL_DATA_DIR = f"{get_data_path()}/fq/real"
SYNTHETIC_DATA_DIR = f"{get_data_path()}/fq/synthetic"

jsonl_files = []
real_data_dir = Path(REAL_DATA_DIR)
synthetic_data_dir = Path(SYNTHETIC_DATA_DIR)

for file in real_data_dir.rglob("*.jsonl"):
    jsonl_files.append(file)

for file in synthetic_data_dir.rglob("*.jsonl"):
    jsonl_files.append(file)

print(f"{len(jsonl_files)} jsonl files found")


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
