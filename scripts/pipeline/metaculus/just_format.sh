#!/bin/bash

python3 ../../../src/format_and_verify_questions.py -f metaculus_20240301_20240601.jsonl -d real -o metaculus_cleaned_formatted_20240301_20240601.jsonl --max_questions 500 --overwrite -M gpt-4o
python3 ../../../src/format_and_verify_questions.py -f metaculus_20240701_20241001.jsonl -d real -o metaculus_cleaned_formatted_20240701_20241001.jsonl --max_questions 500 --overwrite -M gpt-4o
python3 ../../../src/format_and_verify_questions.py -f metaculus.jsonl -d real -o metaculus_cleaned_formatted.jsonl --max_questions 500 --overwrite -M gpt-4o