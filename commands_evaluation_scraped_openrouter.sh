#!/bin/bash


OUTPUT_DIRNAME="UniformRandomForecaster_n_buckets100_tuples_scraped"
#python src/evaluation.py --tuple_dir src/data/tuples_scraped -p src/forecasters/various.py::UniformRandomForecaster --forecaster_options n_buckets=100 --num_lines 200 --run --async -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with anthropic/claude-3.5-sonnet model (with OpenRouter)
OUTPUT_DIRNAME="BasicForecaster_claude-3.5-sonnet_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=anthropic/claude-3.5-sonnet -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with anthropic/claude-3.5-sonnet model (with OpenRouter)
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=anthropic/claude-3.5-sonnet -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-8B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-8B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-70B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-70B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-405B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-405B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-405B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# ResolverBasedForecaster with perplexity/llama-3.1-sonar-large-128k-online model (with OpenRouter)
OUTPUT_DIRNAME="ResolverBasedForecaster_large_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped -p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-large-128k-online -o model=perplexity/llama-3.1-sonar-large-128k-online -o n_attempts=1 --num_lines 200 --run --async -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-8B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-8B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-70B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-70B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with meta-llama/Meta-Llama-3.1-405B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-405B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-405B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
