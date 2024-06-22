#!/bin/bash

# Call manifold.py script
python ../manifold.py -start 20240701 -end 20241001 -num 200
python ../count_entries.py -f manifold_20240701_20241001.json
python3 ../add_body.py manifold_20240701_20241001.json
python3 ../reshape_questions.py --filename manifold_20240701_20241001.json
python3 ../../../src/format_and_verify_questions.py -f manifold_20240701_20241001.jsonl -d real -o manifold_cleaned_formatted_20240701_20241001.jsonl --max_questions 500 --overwrite 