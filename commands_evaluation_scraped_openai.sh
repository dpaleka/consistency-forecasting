#!/bin/bash

# BaselineForecaster with p=0.4
OUTPUT_DIRNAME="BaselineForecaster_p0.4_tuples_scraped"
#python src/evaluation.py --tuple_dir src/data/tuples_scraped -p src/forecasters/various.py::BaselineForecaster --forecaster_options p=0.4 --num_lines 400 --run --async -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

#Summary written to /Users/daniel/code/consistency-forecasting/src/data/forecasts/A_UniformRandomForecaster_most_recent/ground_truth_summary.json
# UniformRandomForecaster with n_buckets=100

# BasicForecaster with gpt-4o-2024-08-06 model
OUTPUT_DIRNAME="BasicForecaster_gpt4o_2024-08-06_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=gpt-4o-2024-08-06 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with gpt-4o-2024-05-13 model
OUTPUT_DIRNAME="BasicForecaster_gpt4o_2024-05-13_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=gpt-4o-2024-05-13 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with gpt-4o-mini-2024-07-18 model
OUTPUT_DIRNAME="BasicForecaster_gpt4o_mini_2024-07-18_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=gpt-4o-mini-2024-07-18 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with gpt-4o-2024-08-06 model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_tuples_scraped" 
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-2024-08-06 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with gpt-4o-mini-2024-07-18 model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-mini-2024-07-18 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with o1-mini model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_o1-mini_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-mini -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with o1-preview model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_o1-preview_tuples_scraped"
python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 50 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-preview -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true


