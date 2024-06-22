#!/bin/bash

# Call metaculus.py script
python3 ../metaculus.py
# cp metaculus.json QUESTIONS_CLEANED.json
# python3 reformat_entries.py
python ../count_entries.py -f metaculus.json
python3 ../add_body.py metaculus.json
python3 ../reshape_questions.py --filename metaculus.json
python3 ../../../src/format_and_verify_questions.py -f metaculus.jsonl -d real -o metaculus_cleaned_formatted.jsonl --max_questions 500 --overwrite