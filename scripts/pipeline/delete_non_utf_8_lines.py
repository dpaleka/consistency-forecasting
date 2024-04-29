import argparse
from pathlib import Path


def check_utf8_and_prompt_deletion(file_path):
    try:
        with open(file_path, "rb") as file:
            lines = file.readlines()

        # Track the line number
        line_number = 0
        for line in lines:
            line_number += 1
            try:
                line.decode("utf-8")
            except UnicodeDecodeError:
                # Prompt user for action on the non-UTF-8 line
                print(f"Non-UTF-8 line detected at line {line_number}: {line}")
                user_input = input("Do you want to delete this line? (y/n): ")
                if user_input.lower() == "y":
                    # Remove the line if user agrees
                    lines.remove(line)
                    print("Line deleted.")
                else:
                    print("Line kept.")

        # Write back the potentially modified lines
        with open(file_path, "wb") as file:
            file.writelines(lines)
        print("File processing complete.")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check and optionally delete non-UTF-8 lines from a file."
    )
    parser.add_argument("file_path", help="Path to the file to be checked")
    args = parser.parse_args()
    file_path = Path(args.file_path)
    if file_path.is_dir():
        # If the provided path is a directory, process all .jsonl files within it
        for jsonl_file in file_path.glob("*.jsonl"):
            print(f"Checking {jsonl_file}")
            check_utf8_and_prompt_deletion(jsonl_file)
    else:
        # If the provided path is a file, process it directly
        check_utf8_and_prompt_deletion(file_path)
