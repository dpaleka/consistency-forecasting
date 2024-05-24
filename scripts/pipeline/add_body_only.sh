#!/bin/bash
python3 add_body.py QUESTIONS_CLEANED_MODIFIED.json
python3 reshape_metaculus.py --filename QUESTIONS_CLEANED_MODIFIED.json
python3 ../../src/format_and_verify_questions.py -f QUESTIONS_CLEANED_MODIFIED.jsonl -d real -o questions_cleaned_formatted.jsonl -m 400 