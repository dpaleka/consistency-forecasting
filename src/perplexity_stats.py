import json

def analyze_jsonl(file_path):
    # Initialize counters
    can_resolve_no_answer = 0
    answer_agrees_resolution = 0
    answer_disagrees_resolution = 0
    cannot_resolve = 0
    resolution_none = 0

    # Read and process the JSONL file
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            
            if entry['can_resolve_question']:
                if entry['answer'] is None:
                    can_resolve_no_answer += 1
                elif entry['resolution'] is None:
                    resolution_none += 1
                elif entry['answer'] == entry['resolution']:
                    answer_agrees_resolution += 1
                else:
                    answer_disagrees_resolution += 1
            else:
                cannot_resolve += 1

    # Print results
    print(f"Entries with 'can_resolve_question' true and answer null: {can_resolve_no_answer}")
    print(f"Questions where answer is not null and agrees with resolution: {answer_agrees_resolution}")
    print(f"Questions where answer is not null and disagrees with resolution: {answer_disagrees_resolution}")
    print(f"Questions with 'can_resolve_question' as false: {cannot_resolve}")
    print(f"Questions with resolution as null: {resolution_none}")

# Usage
file_path = 'src/output_results_5.jsonl'
analyze_jsonl(file_path)