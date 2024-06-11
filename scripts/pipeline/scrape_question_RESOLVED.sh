#!/bin/bash

# Call metaculus.py script
python metaculus.py -start 20240301 -end 20240601 -num 200
python count_entries.py -f metaculus_20240301_20240601.json
python3 add_body.py metaculus_20240301_20240601.json
python3 reshape_metaculus.py --filename metaculus_20240301_20240601.json
python3 ../../src/format_and_verify_questions.py -f metaculus_20240301_20240601.json -d real -o questions_cleaned_formatted_20240301_20240601.jsonl -m 400 