NUM_LINES=242
INPUT_FILE="src/data/fq/real/20240501_20240815.jsonl"
MODEL="gpt-4o-mini-2024-07-18"

# ConsistentForecaster with NegChecker, depth=4
OUTPUT_DIRNAME="ConsistentForecaster_N4_scraped"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file $INPUT_FILE --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[NegChecker]' -o depth=4 -o model=$MODEL --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# ConsistentForecaster with ParaphraseChecker, depth=4
OUTPUT_DIRNAME="ConsistentForecaster_P4_scraped"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file $INPUT_FILE --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ParaphraseChecker]' -o depth=4 -o model=$MODEL --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# # ConsistentForecaster with NegChecker and ParaphraseChecker, depth=4
OUTPUT_DIRNAME="ConsistentForecaster_NP4_scraped"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file $INPUT_FILE --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[NegChecker, ParaphraseChecker]' -o depth=4 -o model=$MODEL --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# # ConsistentForecaster with ExpectedEvidenceChecker x 4, depth=1
OUTPUT_DIRNAME="ConsistentForecaster_4xEE1_scraped"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file $INPUT_FILE --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker]' -o depth=1 -o model=$MODEL --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true