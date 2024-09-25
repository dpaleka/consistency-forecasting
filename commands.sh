#!/bin/bash

# BaselineForecaster with p=0.4
OUTPUT_DIRNAME="BaselineForecaster_p0.4_20240701_20240831"
#python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl -p src/forecasters/various.py::BaselineForecaster --forecaster_options p=0.4 --num_lines 1000 --run --async --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

#Summary written to /Users/daniel/code/consistency-forecasting/src/data/forecasts/A_UniformRandomForecaster_most_recent/ground_truth_summary.json
# UniformRandomForecaster with n_buckets=100
OUTPUT_DIRNAME="UniformRandomForecaster_n_buckets100_20240701_20240831"
#python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl -p src/forecasters/various.py::UniformRandomForecaster --forecaster_options n_buckets=100 --num_lines 1000 --run --async --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with gpt-4o-2024-08-06 model
OUTPUT_DIRNAME="BasicForecaster_gpt4o_2024-08-06_20240701_20240831"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=gpt-4o-2024-08-06 --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with gpt-4o-2024-05-13 model
OUTPUT_DIRNAME="BasicForecaster_gpt4o_2024-05-13_20240701_20240831"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=gpt-4o-2024-05-13 --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with gpt-4o-mini-2024-07-18 model
OUTPUT_DIRNAME="BasicForecaster_gpt4o_mini_2024-07-18_20240701_20240831"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=gpt-4o-mini-2024-07-18 --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with anthropic/claude-3.5-sonnet model (with OpenRouter)
OUTPUT_DIRNAME="BasicForecaster_claude-3.5-sonnet_20240701_20240831"
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=anthropic/claude-3.5-sonnet --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with o1-mini model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_o1-mini_20240701_20240831"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-mini --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with o1-preview model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_o1-preview_20240701_20240831"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-preview --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with gpt-4o-2024-08-06 model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_20240701_20240831" 
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-2024-08-06 --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with gpt-4o-mini-2024-07-18 model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_20240701_20240831"
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-mini-2024-07-18 --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with anthropic/claude-3.5-sonnet model (with OpenRouter)
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_20240701_20240831"
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=anthropic/claude-3.5-sonnet --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-8B_20240701_20240831"
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-70B_20240701_20240831"
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-70B-Instruct --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-405B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-405B_20240701_20240831"
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-405B-Instruct --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# ResolverBasedForecaster with perplexity/llama-3.1-sonar-huge-128k-online model (with OpenRouter)
OUTPUT_DIRNAME="ResolverBasedForecaster_huge_20240701_20240831" 
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl -p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-huge-128k-online -o model=perplexity/llama-3.1-sonar-huge-128k-online -o n_attempts=1 --num_lines 1000 --run --async --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# ResolverBasedForecaster with perplexity/llama-3.1-sonar-large-128k-online model (with OpenRouter)
OUTPUT_DIRNAME="ResolverBasedForecaster_large_20240701_20240831"
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl -p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-large-128k-online -o model=perplexity/llama-3.1-sonar-large-128k-online -o n_attempts=1 --num_lines 1000 --run --async --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-8B_20240701_20240831"
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-70B_20240701_20240831"
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-70B-Instruct --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with meta-llama/Meta-Llama-3.1-405B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-405B_20240701_20240831"
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-405B-Instruct --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
