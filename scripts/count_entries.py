import json
import argparse

def count_entries(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return len(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count the number of entries in a JSON file.')
    parser.add_argument('-f', '--file', type=str, help='Path to the JSON file', required=True)
    
    args = parser.parse_args()
    file_path = args.file
    entries_count = count_entries(file_path)
    print(f"Total entries in the JSON file: {entries_count}")