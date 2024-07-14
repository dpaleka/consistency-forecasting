#!/bin/bash

# Call manifold.py script
python ../manifold.py -start 20240301 -end 20240601 -num 500
python ../count_entries.py -f manifold_20240301_20240601.json
python3 ../add_body.py manifold_20240301_20240601.json
python3 ../reshape_questions.py --filename manifold_20240301_20240601.json
python3 ../../../src/format_and_verify_questions.py -f manifold_20240301_20240601.jsonl -d real -o manifold_cleaned_formatted_20240301_20240601.jsonl --max_questions 500 --overwrite -F True -M gpt-4o