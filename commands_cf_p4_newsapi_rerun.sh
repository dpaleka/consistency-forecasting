INPUT_DIR="src/data/tuples_rerun"
MODEL="gpt-4o-mini-2024-07-18"
OUTPUT_DIRNAME="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_rerun"

CHECKER="CondCondChecker"
NUM_LINES=150
USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ParaphraseChecker]' -o depth=4 -o model=$MODEL -k $CHECKER --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

CHECKER="ExpectedEvidenceChecker"
NUM_LINES=200
USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[ParaphraseChecker]' -o depth=4 -o model=$MODEL -k $CHECKER --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true