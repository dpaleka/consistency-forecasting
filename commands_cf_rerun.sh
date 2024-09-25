MODEL="gpt-4o-mini-2024-07-18"

# NUM_LINES=180
# INPUT_FILE="src/data/fq/rerun/newsapi.jsonl"

# # # ConsistentForecaster with ExpectedEvidenceChecker x 4, depth=1, NewsAPI
# OUTPUT_DIRNAME="ConsistentForecaster_4xEE1_20240701_20240831"
# USE_OPENROUTER=False python src/ground_truth_run.py --input_file $INPUT_FILE --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker]' -o depth=1 -o model=$MODEL --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

NUM_LINES=42
INPUT_FILE="src/data/fq/rerun/scraped.jsonl"

# # ConsistentForecaster with NegChecker and ParaphraseChecker, depth=4, scraped
OUTPUT_DIRNAME="ConsistentForecaster_NP4_scraped"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file $INPUT_FILE --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[NegChecker, ParaphraseChecker]' -o depth=4 -o model=$MODEL --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true