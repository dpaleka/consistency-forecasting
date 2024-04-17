import json
import argparse


def prettify_json_with_bold_keys(json_obj, indent=2) -> str:
    # ANSI escape code for bold text
    bold_start = "\033[1m"
    bold_end = "\033[0m"

    def prettify(obj, depth=0):
        if isinstance(obj, dict):
            result = "{\n"
            indent_str = " " * (indent * (depth + 1))
            for i, (key, value) in enumerate(obj.items()):
                formatted_key = f'{bold_start}"{key}"{bold_end}'
                formatted_value = (
                    prettify(value, depth + 1)
                    if isinstance(value, dict)
                    else json.dumps(value)
                )
                result += f"{indent_str}{formatted_key}: {formatted_value}"
                if i < len(obj) - 1:
                    result += ","
                result += "\n"
            result += " " * (indent * depth) + "}"
            return result
        elif isinstance(obj, list):
            result = "[\n"
            indent_str = " " * (indent * (depth + 1))
            for i, item in enumerate(obj):
                formatted_item = prettify(item, depth + 1)
                result += f"{indent_str}{formatted_item}"
                if i < len(obj) - 1:
                    result += ","
                result += "\n"
            result += " " * (indent * depth) + "]"
            return result
        else:
            return json.dumps(obj)

    return prettify(json_obj)


def main(jsonl_file_path):
    try:
        # Open and read the JSONL file
        with open(jsonl_file_path, "r") as file:
            lines = file.readlines()

        print(
            f"Loaded {len(lines)} items. Press Enter to scroll through them, type 'exit' or Ctrl+C to quit."
        )

        # ANSI escape code for bold text
        bold_start = "\033[1m"
        bold_end = "\033[0m"

        # Iterate through each line in the file
        line_index = 0
        while True:
            line = lines[line_index]
            try:
                # Parse JSON from each line
                json_obj = json.loads(line)

                # Pretty-printing JSON object with bold keys
                print(f"\nItem {line_index}:")
                print(prettify_json_with_bold_keys(json_obj))

                # Wait for user input to continue
                user_input = input("[n]/p/q:")
                if user_input == "q" or user_input == "exit":
                    break
                elif user_input == "p":
                    line_index -= 1
                else:
                    line_index += 1
                    if line_index >= len(lines):
                        line_index = 0

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")

    except FileNotFoundError:
        print(f"File not found: {jsonl_file_path}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Inspect a JSONL file with the ability to scroll through elements on keypress."
    )

    # Add an argument for the JSONL file path
    parser.add_argument("file", help="Path to the JSONL file to be inspected")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the JSONL file path
    main(args.file)
