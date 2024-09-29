source venv/bin/activate

INPUT_DIR="src/data/tuples_synthetic"
MODEL="gpt-4o-mini-2024-07-18"
NUM_LINES=200

# OUTPUT_DIRNAME="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_synthetic"
# USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker]' -o depth=1 -o model=$MODEL -k all --output_dir $OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# OUTPUT_DIRNAME="src/data/forecasts/ConsistentForecaster_N4_tuples_synthetic"
# USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[NegChecker]' -o depth=4 -o model=$MODEL -k all --output_dir $OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# OUTPUT_DIRNAME="src/data/forecasts/ConsistentForecaster_P4_tuples_synthetic"
# USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ParaphraseChecker]' -o depth=4 -o model=$MODEL -k all --output_dir $OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

OUTPUT_DIRNAME="src/data/forecasts/ConsistentForecaster_NP4_tuples_synthetic"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[NegChecker, ParaphraseChecker]' -o depth=4 -o model=$MODEL -k all --output_dir $OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true