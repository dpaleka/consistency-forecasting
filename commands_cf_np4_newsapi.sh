INPUT_DIR="src/data/tuples_newsapi"
MODEL="gpt-4o-mini-2024-07-18"
OUTPUT_DIRNAME="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi"
NUM_LINES=200

USE_OPENROUTER=False python src/evaluation.py --tuple_dir $INPUT_DIR --num_lines $NUM_LINES --run --async -f ConsistentForecaster -o checks='[NegChecker, ParaphraseChecker]' -o depth=4 -o model=$MODEL -k all --output_dir $OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true