#!/bin/bash

# Call metaculus.py script
python3 metaculus.py
cp metaculus.json QUESTIONS_CLEANED.json
python3 reformat_entries.py
python3 add_body.py --filename QUESTIONS_CLEANED_MODIFIED.json
python3 reshape_metaculus.py --filename QUESTIONS_CLEANED_MODIFIED.json

