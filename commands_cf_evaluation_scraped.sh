NUM_LINES=200
INPUT_DIR="src/data/tuples_scraped"
MODEL="gpt-4o-mini-2024-07-18"

# ConsistentForecaster with ExpectedEvidenceChecker x 4, depth=1
# 4 min, $0.12 per concurrent
# OUTPUT_DIRNAME="ConsistentForecaster_4xEE1_tuples_scraped"
# USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker]' -o depth=1 -o model=$MODEL -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# ConsistentForecaster with NegChecker, depth=4
# 15 min, $0.12, 930 calls per concurrent
OUTPUT_DIRNAME="ConsistentForecaster_N4_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[NegChecker]' -o depth=4 -o model=$MODEL -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# ConsistentForecaster with ParaphraseChecker, depth=4
OUTPUT_DIRNAME="ConsistentForecaster_P4_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ParaphraseChecker]' -o depth=4 -o model=$MODEL -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# ConsistentForecaster with NegChecker and ParaphraseChecker, depth=4
# 18 min, $0.60, 4830 calls per concurrent
OUTPUT_DIRNAME="ConsistentForecaster_NP4_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[NegChecker, ParaphraseChecker]' -o depth=4 -o model=$MODEL -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
