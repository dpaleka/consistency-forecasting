#!/usr/bin/env python3
import subprocess
import sys
import os
import git

# Define the script to run and the list of JSONL files to validate
# Intended to be run from the root of the repository, as when running the pre-commit hook

# Need to do this hack to import stuff
sys.path.append(".")
from src.common.path_utils import get_data_path, get_src_path

validate_all = os.getenv("VALIDATE_ALL", False)

script = f"{get_src_path()}/validate_fq_jsonl.py"
jsonl_files = []
real_data_dir = get_data_path() / "fq" / "real"
synthetic_data_dir = get_data_path() / "fq" / "synthetic"
tuple_data_dir = get_data_path() / "tuples"


for file in real_data_dir.rglob("*.jsonl"):
    jsonl_files.append(
        {
            "file": str(file),
            "tuple": False,
        }
    )

for file in synthetic_data_dir.rglob("*.jsonl"):
    jsonl_files.append(
        {
            "file": str(file),
            "tuple": False,
        }
    )

for file in tuple_data_dir.rglob("*.jsonl"):
    jsonl_files.append(
        {
            "file": str(file),
            "tuple": True,
        }
    )

print(f"{len(jsonl_files)} jsonl files found")

# now get the git diff ones

repo = git.Repo(search_parent_directories=True)
diff = repo.git.diff("HEAD", name_only=True)

diff_files: list[str] = diff.splitlines()
diff_files = [f for f in diff_files if f.endswith(".jsonl")]

if validate_all:
    jsonl_files_to_check = jsonl_files
else:
    jsonl_files_to_check = []
    for file_path in jsonl_files:
        for diff_file in diff_files:
            if file_path["file"].endswith(diff_file):
                jsonl_files_to_check.append(file_path)
                break

print(f"{len(jsonl_files_to_check)} jsonl files to check")


# Function to run the validation script on each file
def validate_file(file_path):
    command = ["python3", script, "--filename", str(file_path["file"])]
    if file_path["tuple"]:
        command.append("--tuple")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"Validation failed for {str(file_path['file'])}, tuple={file_path['tuple']}."
        )
        print(result)
        return False
    return True


all_valid = True
for file_path in jsonl_files_to_check:
    print(f"{file_path}")
    if not validate_file(file_path):
        all_valid = False
        break  # Stop on first failure

if not all_valid:
    sys.exit(1)  # Exit with a non-zero status code to indicate failure

print("Validation successful for all files.")
