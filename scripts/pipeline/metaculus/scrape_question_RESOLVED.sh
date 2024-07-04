#!/bin/bash

# Call metaculus.py script
python ../metaculus.py -start 20240301 -end 20240601 -num 200
python ../count_entries.py -f metaculus_20240301_20240601.json
python3 ../add_body.py metaculus_20240301_20240601.json
python3 ../reshape_questions.py --filename metaculus_20240301_20240601.json
python3 ../../../src/format_and_verify_questions.py -f metaculus_20240301_20240601.jsonl -d real -o metaculus_cleaned_formatted_20240301_20240601.jsonl --max_questions 25 --overwrite -M gpt-3.5-turbo