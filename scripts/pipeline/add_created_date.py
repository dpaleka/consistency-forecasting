import argparse
import json
from pathlib import Path


def add_created_date_to_jsonl(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        with open(file_path, "w") as file:
            for line in lines:
                data = json.loads(line)
                if "created_date" not in data:
                    # add created_date field, just after "data_source"
                    # Find the index of "data_source" in the dict
                    data_source_index = list(data.keys()).index("data_source")
                    print(data.keys())
                    # Create a new dict with the items before "data_source"
                    new_data = {
                        k: data[k] for k in list(data.keys())[: data_source_index + 1]
                    }
                    # Add the "created_date" field
                    new_data["created_date"] = None
                    # Add the remaining items from the original dict
                    new_data.update(
                        {k: data[k] for k in list(data.keys())[data_source_index + 1 :]}
                    )
                    # Replace the original data with the new dict
                    data = new_data
                    print(data.keys())
                file.write(json.dumps(data) + "\n")

        print(f"Processed file: {file_path}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def process_directory(directory):
    for item in directory.iterdir():
        if item.is_file() and item.suffix == ".jsonl":
            add_created_date_to_jsonl(item)
        elif item.is_dir():
            process_directory(item)  # Recurse into subdirectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add 'created_date': null to all JSON objects in JSONL files."
    )
    parser.add_argument(
        "path", help="Path to the JSONL file or directory containing JSONL files"
    )
    args = parser.parse_args()
    path = Path(args.path)

    if path.is_dir():
        process_directory(path)
    elif path.is_file() and path.suffix == ".jsonl":
        add_created_date_to_jsonl(path)
    else:
        print("Please provide a valid JSONL file or directory.")
