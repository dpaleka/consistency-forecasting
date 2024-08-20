#!/bin/bash

# Call metaculus.py script
python ../metaculus.py -start 20240701 -end 20241001 -num 500
python ../count_entries.py -f metaculus_20240701_20241001.json
python3 ../add_body.py metaculus_20240701_20241001.json
python3 ../reshape_questions.py --filename metaculus_20240701_20241001.json
python3 ../../../src/format_and_verify_questions.py -f metaculus_20240701_20241001.jsonl -d real -o metaculus_cleaned_formatted_20240701_20241001.jsonl --max_questions 500 --overwrite -M gpt-4o