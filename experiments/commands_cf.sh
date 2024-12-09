NUM_LINES=500
INPUT_FILE="src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl"
MODEL="gpt-4o-mini-2024-07-18"

# ConsistentForecaster with NegChecker, depth=4
# OUTPUT_DIRNAME="ConsistentForecaster_N4_20240701_20240831"
# USE_OPENROUTER=False python src/ground_truth_run.py --input_file $INPUT_FILE --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[NegChecker]' -o depth=4 -o model=$MODEL --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
# already ran as a test

# ConsistentForecaster with ParaphraseChecker, depth=4
OUTPUT_DIRNAME="ConsistentForecaster_P4_20240701_20240831"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file $INPUT_FILE --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ParaphraseChecker]' -o depth=4 -o model=$MODEL --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# # ConsistentForecaster with NegChecker and ParaphraseChecker, depth=4
OUTPUT_DIRNAME="ConsistentForecaster_NP4_20240701_20240831"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file $INPUT_FILE --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[NegChecker, ParaphraseChecker]' -o depth=4 -o model=$MODEL --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# # ConsistentForecaster with ExpectedEvidenceChecker x 4, depth=1
OUTPUT_DIRNAME="ConsistentForecaster_4xEE1_20240701_20240831"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file $INPUT_FILE --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker]' -o depth=1 -o model=$MODEL --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true