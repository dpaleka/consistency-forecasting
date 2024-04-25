#!/usr/bin/env python3
"""
pytest will run all files of the form test_*.py or *_test.py in the current directory and its subdirectories.
This hook is meant to be used with pre-commit, so that we can catch any files that are not named correctly.
"""
import sys
import os

BASE_DIR = "src/"


# Function to run the validation script
def validate_naming_convention(file_path):
    if (file_path.startswith("test_") and file_path.endswith(".py")) or (
        file_path.endswith("_test.py")
    ):
        print(
            f"File {file_path} does not follow the naming convention. Avoid naming files *_test.py or test_*.py."
        )
        return False
    return True


# Main logic
if __name__ == "__main__":
    all_valid = True
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if not validate_naming_convention(os.path.join(root, file)):
                all_valid = False
                break  # Stop on first failure

    if not all_valid:
        sys.exit(1)  # Exit with a non-zero status code to indicate failure

    print("Naming convention successful for all files.")
