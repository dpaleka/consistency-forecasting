import json


def load_jsonl(file_path):
    """Load the JSONL file into a list of dicts."""
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def save_jsonl(data, output_file_path):
    """Save the updated data back to a JSONL file."""
    with open(output_file_path, "w") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


def make_hashable(value):
    """Recursively convert dictionaries into tuples to make them hashable."""
    if isinstance(value, dict):
        return tuple((key, make_hashable(val)) for key, val in sorted(value.items()))
    elif isinstance(value, list):
        return tuple(make_hashable(item) for item in value)
    return value


def remove_duplicates(jsonl_data):
    """Remove duplicates based on the 'P' field, ignoring the 'id'."""
    seen = set()
    unique_data = []

    for entry in jsonl_data:
        # Get the 'P' field and remove 'id' from it before comparison
        p_data = entry.get("P", {}).copy()
        p_data.pop("id", None)

        # Make the entire 'P' field hashable
        p_tuple = make_hashable(p_data)

        if p_tuple not in seen:
            seen.add(p_tuple)
            unique_data.append(entry)

    return unique_data


def process_jsonl(input_file, output_file):
    """Load, remove duplicates, and save the JSONL file."""
    data = load_jsonl(input_file)
    unique_data = remove_duplicates(data)
    save_jsonl(unique_data, output_file)
    print(f"Duplicates removed. Output saved to {output_file}")


def output_file_name(input_file):
    return input_file.replace(".jsonl", "_unique.jsonl")


if __name__ == "__main__":
    # File paths
    input_files = [
        "data/tuples_scraped/NegChecker.jsonl",
        "data/tuples_scraped/ParaphraseChecker.jsonl",
        "data/tuples_newsapi/NegChecker.jsonl",
        "data/tuples_newsapi/ParaphraseChecker.jsonl",
    ]
    output_files = [output_file_name(input_file) for input_file in input_files]

    # Process the file
    for input_file, output_file in zip(input_files, output_files):
        process_jsonl(input_file, output_file)
