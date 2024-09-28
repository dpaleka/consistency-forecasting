## General notes

For evaluation.py, we log the output in a file called logs/evaluation_{some experiment details}.log
We also name the dirs in a meaningful way, like `src/data/forecasts/BasicForecaster_0501_0815_model_gpt-4o-2024-05-13/`

## Run ground_truth_run.py on [20240501_20240815.jsonl](src/data/fq/real/20240501_20240815.jsonl) and the corresponding [tuples_scraped](src/data/tuples_scraped/)

### BaselineForecaster with p=0.4
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::BaselineForecaster --forecaster_options p=0.4 --num_lines 242 --run --async
```
-> [`src/data/forecasts/BaselineForecaster_09-23-13-41/ground_truth_summary.json`](src/data/forecasts/BaselineForecaster_09-23-13-41/ground_truth_summary.json)

- [x] evaluation
```
python src/evaluation.py --tuple_dir src/data/tuples_scraped/ -p src/forecasters/various.py::BaselineForecaster -o p=0.4 -k all --num_lines 500 --run --async
```
-> [`src/data/forecasts/BaselineForecaster_p0.4_tuples_scraped/stats_summary.json`](src/data/forecasts/BaselineForecaster_p0.4_tuples_scraped/stats_summary.json)


### ResolverBasedForecaster with perplexity/llama-3.1-sonar-huge-128k-online model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-huge-128k-online -o model=perplexity/llama-3.1-sonar-huge-128k-online -o n_attempts=1 --num_lines 242 --run --async
```
-> [`src/data/forecasts/ResolverBasedForecaster_09-23-18-15/ground_truth_summary.json`](src/data/forecasts/ResolverBasedForecaster_09-23-18-15/ground_truth_summary.json)

- [ ] evaluation
```
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped/ -p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-large-128k-online -o model=perplexity/llama-3.1-sonar-large-128k-online -o n_attempts=1 -k all --num_lines 500 --run --async
```

### ResolverBasedForecaster with perplexity/llama-3.1-sonar-large-128k-online model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-large-128k-online -o model=perplexity/llama-3.1-sonar-large-128k-online -o n_attempts=1 --num_lines 242 --run --async
```
-> [`src/data/forecasts/ResolverBasedForecaster_09-23-21-55/ground_truth_summary.json`](src/data/forecasts/ResolverBasedForecaster_09-23-21-55/ground_truth_summary.json)

### BasicForecaster with gpt-4o-2024-08-06 model
- [x] ground_truth_run
```
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=gpt-4o-2024-08-06
```
-> [`src/data/forecasts/BasicForecaster_09-23-13-46/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_09-23-13-46/ground_truth_summary.json)


- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_gpt4o_2024-08-06_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=gpt-4o-2024-08-06 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/BasicForecaster_gpt4o_2024-08-06_tuples_scraped/stats_summary.json`](src/data/forecasts/BasicForecaster_gpt4o_2024-08-06_tuples_scraped/stats_summary.json)

### BasicForecaster with gpt-4o-2024-05-13 model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=gpt-4o-2024-05-13
```
-> [`src/data/forecasts/BasicForecaster_09-24-23-30/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_09-24-23-30/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_gpt4o_2024-05-13_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=gpt-4o-2024-05-13 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/BasicForecaster_gpt4o_2024-05-13_tuples_scraped/stats_summary.json`](src/data/forecasts/BasicForecaster_gpt4o_2024-05-13_tuples_scraped/stats_summary.json)


### BasicForecaster with gpt-4o-mini-2024-07-18 model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=gpt-4o-mini-2024-07-18
```
-> [`src/data/forecasts/BasicForecaster_09-24-19-10/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_09-24-19-10/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_gpt4o_mini_2024-07-18_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=gpt-4o-mini-2024-07-18 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/BasicForecaster_gpt4o_mini_2024-07-18_tuples_scraped/stats_summary.json`](src/data/forecasts/BasicForecaster_gpt4o_mini_2024-07-18_tuples_scraped/stats_summary.json)


### CoT_ForecasterTextBeforeParsing with gpt-4o-2024-08-06 model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-2024-08-06
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-19-30/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-19-30/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_tuples_scraped" 
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-2024-08-06 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_tuples_scraped/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_tuples_scraped/stats_summary.json)

### CoT_ForecasterTextBeforeParsing with gpt-4o-mini-2024-07-18 model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-mini-2024-07-18
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-19-44/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-19-44/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-mini-2024-07-18 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_tuples_scraped/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_tuples_scraped/stats_summary.json)

### CoT_ForecasterTextBeforeParsing with o1-mini model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-mini
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-23-22-25/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-23-22-25/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_o1-mini_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-mini -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-mini_tuples_scraped/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-mini_tuples_scraped/stats_summary.json)


### CoT_ForecasterTextBeforeParsing with o1-preview model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-preview
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-19-12/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-19-12/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_o1-preview_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 50 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-preview -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

```
**Error: ButChecker and ExpectedEvidence checker have some stuff violating usage policy.**
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-preview_tuples_scraped/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-preview_tuples_scraped/stats_summary.json)


### BasicForecaster with anthropic/claude-3.5-sonnet model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=anthropic/claude-3.5-sonnet
```
-> [`src/data/forecasts/BasicForecaster_09-24-19-09/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_09-24-19-09/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_claude-3.5-sonnet_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=anthropic/claude-3.5-sonnet -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/BasicForecaster_claude-3.5-sonnet_tuples_scraped/stats_summary.json`](src/data/forecasts/BasicForecaster_claude-3.5-sonnet_tuples_scraped/stats_summary.json)

### CoT_ForecasterTextBeforeParsing with anthropic/claude-3.5-sonnet model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=anthropic/claude-3.5-sonnet
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-22-42/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-22-42/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=anthropic/claude-3.5-sonnet -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
**Errors abound in checkers, maybe need to rerun with some stability, or recompute**
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_scraped/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_scraped/stats_summary.json)

### CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-8B-Instruct
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-23-36/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-23-36/ground_truth_summary.json)


- [x] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-8B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-8B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
**CondCondChecker failed, see the stats_summary.json for details.**
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-8B_tuples_scraped/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-8B_tuples_scraped/stats_summary.json)

### CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-70B-Instruct
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-23-09/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-23-09/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-70B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-70B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-70B_tuples_scraped/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-70B_tuples_scraped/stats_summary.json)

### CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-405B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-405B-Instruct
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-23-25/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-23-25/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-405B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-405B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
**Note: ButChecker and CondCondChecker failed, see the stats_summary.json for details.**
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-405B_tuples_scraped/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-405B_tuples_scraped/stats_summary.json)


### BasicForecaster with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-8B-Instruct
```
-> [`src/data/forecasts/BasicForecaster_09-24-19-12/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_09-24-19-12/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-8B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-8B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
**Some checks failed, see the stats_summary.json for details. I recommend not reporting this model.**
-> [`src/data/forecasts/BasicForecaster_llama-3.1-8B_tuples_scraped/stats_summary.json`](src/data/forecasts/BasicForecaster_llama-3.1-8B_tuples_scraped/stats_summary.json)

### BasicForecaster with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-70B-Instruct
```
-> [`src/data/forecasts/BasicForecaster_09-24-19-29/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_09-24-19-29/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-70B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-70B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/BasicForecaster_llama-3.1-70B_tuples_scraped/stats_summary.json`](src/data/forecasts/BasicForecaster_llama-3.1-70B_tuples_scraped/stats_summary.json)

### BasicForecaster with meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-405B-Instruct
```
-> [`src/data/forecasts/BasicForecaster_09-24-22-40/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_09-24-22-40/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-405B_tuples_scraped"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 200 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-405B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/BasicForecaster_llama-3.1-405B_tuples_scraped/stats_summary.json`](src/data/forecasts/BasicForecaster_llama-3.1-405B_tuples_scraped/stats_summary.json)




# Run ground_truth_run.py on [20240701_20240831.jsonl](src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl) and the corresponding [tuples_news_api](src/data/tuples_news_api/)
(There is a slight distribution shift between the ground truth data and what was used for tuple generation, see README.md for details, but that's OK, we are interested in generalization here.)

```
# BaselineForecaster with p=0.4
OUTPUT_DIRNAME="BaselineForecaster_p0.4_tuples_newsapi"
#python src/evaluation.py --tuple_dir src/data/tuples_newsapi -p src/forecasters/various.py::BaselineForecaster --forecaster_options p=0.4 --num_lines 400 --run --async -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

#Summary written to /Users/daniel/code/consistency-forecasting/src/data/forecasts/A_UniformRandomForecaster_most_recent/ground_truth_summary.json
# UniformRandomForecaster with n_buckets=100
OUTPUT_DIRNAME="UniformRandomForecaster_n_buckets100_tuples_newsapi"
python src/evaluation.py --tuple_dir src/data/tuples_newsapi -p src/forecasters/various.py::UniformRandomForecaster --forecaster_options n_buckets=100 --num_lines 300 --run --async -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with gpt-4o-2024-08-06 model
OUTPUT_DIRNAME="BasicForecaster_gpt4o_2024-08-06_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=gpt-4o-2024-08-06 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with gpt-4o-2024-05-13 model
OUTPUT_DIRNAME="BasicForecaster_gpt4o_2024-05-13_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=gpt-4o-2024-05-13 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with gpt-4o-mini-2024-07-18 model
OUTPUT_DIRNAME="BasicForecaster_gpt4o_mini_2024-07-18_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=gpt-4o-mini-2024-07-18 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# BasicForecaster with anthropic/claude-3.5-sonnet model (with OpenRouter)
OUTPUT_DIRNAME="BasicForecaster_claude-3.5-sonnet_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=anthropic/claude-3.5-sonnet -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with o1-mini model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_o1-mini_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-mini -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with o1-preview model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_o1-preview_tuples_newsapi"
python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 50 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-preview -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with gpt-4o-2024-08-06 model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_tuples_newsapi" 
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-2024-08-06 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with gpt-4o-mini-2024-07-18 model
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-mini-2024-07-18 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with anthropic/claude-3.5-sonnet model (with OpenRouter)
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=anthropic/claude-3.5-sonnet -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true

# CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-8B_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-8B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```

### BaselineForecaster with p=0.4
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl -p src/forecasters/various.py::BaselineForecaster --forecaster_options p=0.4 --num_lines 1000 --run --async --output_dir src/data/forecasts/BaselineForecaster_p0.4_20240701_20240831
```
-> [`src/data/forecasts/BaselineForecaster_p0.4_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/BaselineForecaster_p0.4_20240701_20240831/ground_truth_summary.json)


### UniformRandomForecaster with n_buckets=100
- [x] ground_truth_run
```
OUTPUT_DIRNAME="UniformRandomForecaster_n_buckets100_20240701_20240831"
python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl -p src/forecasters/various.py::UniformRandomForecaster --forecaster_options n_buckets=100 --num_lines 1000 --run --async --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/UniformRandomForecaster_n_buckets100_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/UniformRandomForecaster_n_buckets100_20240701_20240831/ground_truth_summary.json)

**Separated in two json object, ConcCond in one, and the others in other.**
- [x] evaluation
```
python src/evaluation.py --tuple_dir src/data/tuples_newsapi -p src/forecasters/various.py::UniformRandomForecaster --forecaster_options n_buckets=100 --num_lines 2 --run --async --output_dir src/data/forecasts/UniformRandomForecaster_n_buckets100_tuples_newsapi
```
-> [`src/data/forecasts/UniformRandomForecaster_n_buckets100_tuples_newsapi/stats_summary.json`](src/data/forecasts/UniformRandomForecaster_n_buckets100_tuples_newsapi/stats_summary.json)

### BasicForecaster with gpt-4o-2024-08-06 model
- [x] ground_truth_run
```
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=gpt-4o-2024-08-06 --output_dir src/data/forecasts/BasicForecaster_gpt4o_2024-08-06_20240701_20240831
```
-> [`src/data/forecasts/BasicForecaster_gpt4o_2024-08-06_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_gpt4o_2024-08-06_20240701_20240831/ground_truth_summary.json)

**Not valid json, two objects, one with CondCond, and one with everything else (including CondCond)**
- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_gpt4o_2024-08-06_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=gpt-4o-2024-08-06 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/BasicForecaster_gpt4o_2024-08-06_tuples_newsapi/stats_summary.json`](src/data/forecasts/BasicForecaster_gpt4o_2024-08-06_tuples_newsapi/stats_summary.json)

### BasicForecaster with gpt-4o-2024-05-13 model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=gpt-4o-2024-05-13 --output_dir src/data/forecasts/BasicForecaster_gpt4o_2024-05-13_20240701_20240831
```
-> [`src/data/forecasts/BasicForecaster_gpt4o_2024-05-13_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_gpt4o_2024-05-13_20240701_20240831/ground_truth_summary.json)

**Two json objects with two sets of results**
- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_gpt4o_2024-05-13_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=gpt-4o-2024-05-13 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/BasicForecaster_gpt4o_2024-05-13_tuples_newsapi/stats_summary.json`](src/data/forecasts/BasicForecaster_gpt4o_2024-05-13_tuples_newsapi/stats_summary.json) 

### BasicForecaster with gpt-4o-mini-2024-07-18 model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=gpt-4o-mini-2024-07-18 --output_dir src/data/forecasts/BasicForecaster_gpt4o_mini_2024-07-18_20240701_20240831
```
-> [`src/data/forecasts/BasicForecaster_gpt4o_mini_2024-07-18_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_gpt4o_mini_2024-07-18_20240701_20240831/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_gpt4o_mini_2024-07-18_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=gpt-4o-mini-2024-07-18 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/BasicForecaster_gpt4o_mini_2024-07-18_tuples_newsapi/stats_summary.json`](src/data/forecasts/BasicForecaster_gpt4o_mini_2024-07-18_tuples_newsapi/stats_summary.json) 

### BasicForecaster with anthropic/claude-3.5-sonnet model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=anthropic/claude-3.5-sonnet --output_dir src/data/forecasts/BasicForecaster_claude-3.5-sonnet_20240701_20240831
```
-> [`src/data/forecasts/BasicForecaster_claude-3.5-sonnet_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_claude-3.5-sonnet_20240701_20240831/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_claude-3.5-sonnet_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=anthropic/claude-3.5-sonnet -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/BasicForecaster_claude-3.5-sonnet_tuples_newsapi/stats_summary.json`](src/data/forecasts/BasicForecaster_claude-3.5-sonnet_tuples_newsapi/stats_summary.json) 

### BasicForecaster with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir src/data/forecasts/BasicForecaster_llama-3.1-8B_20240701_20240831
```
-> [`src/data/forecasts/BasicForecaster_llama-3.1-8B_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_llama-3.1-8B_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-8B_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-8B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```

### BasicForecaster with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-70B-Instruct --output_dir src/data/forecasts/BasicForecaster_llama-3.1-70B_20240701_20240831
```
-> [`src/data/forecasts/BasicForecaster_llama-3.1-70B_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_llama-3.1-70B_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-70B_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-70B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```

### BasicForecaster with meta-llama/Meta-Llama-3.1-405B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-405B-Instruct --output_dir src/data/forecasts/BasicForecaster_llama-3.1-405B_20240701_20240831
```
-> [`src/data/forecasts/BasicForecaster_llama-3.1-405B_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_llama-3.1-405B_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="BasicForecaster_llama-3.1-405B_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-405B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```

### CoT_ForecasterTextBeforeParsing with o1-mini model
- [x] ground_truth_run
```
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-mini --output_dir src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-mini_20240701_20240831
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-mini_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-mini_20240701_20240831/ground_truth_summary.json)

**Two Json objects, one with CondCond, and another with all the checkers**
- [x] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_o1-mini_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-mini -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-mini_tuples_newsapi/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-mini_tuples_newsapi/stats_summary.json)   

### CoT_ForecasterTextBeforeParsing with o1-preview model
- [x] ground_truth_run
```
USE_OPENROUTER=False python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-preview --output_dir src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-preview_20240701_20240831
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-preview_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-preview_20240701_20240831/ground_truth_summary.json)


- [ ] evaluation

(we run 50 per check because it is really expensive)
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_o1-preview_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 50 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-preview -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
**NOTE: some checks in this file have failed due to lack of credits, need to rerun these checks, look in the file**
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-preview_tuples_newsapi/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-preview_tuples_newsapi/stats_summary.json)   


### CoT_ForecasterTextBeforeParsing with gpt-4o-2024-08-06 model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-2024-08-06 --output_dir src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_20240701_20240831
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-2024-08-06 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
**NOTE: some checks in this file have failed due to lack of credits, need to rerun these checks, look in the file**
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_tuples_newsapi/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_tuples_newsapi/stats_summary.json)

### CoT_ForecasterTextBeforeParsing with gpt-4o-mini-2024-07-18 model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-mini-2024-07-18 --output_dir src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_20240701_20240831
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-mini-2024-07-18 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
**NOTE: some checks in this file have failed due to lack of credits, need to rerun these checks, look in the file**
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_tuples_newsapi/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_tuples_newsapi/stats_summary.json)


### CoT_ForecasterTextBeforeParsing with anthropic/claude-3.5-sonnet model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=anthropic/claude-3.5-sonnet --output_dir src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_20240701_20240831
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=anthropic/claude-3.5-sonnet -k all --output_dir src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_newsapi 2>&1 | tee logs/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_newsapi_$(date +%Y%m%d_%H%M).log || true
```
**NOTE: some checks in this file have failed due to unknown reasons, need to rerun those checks, look in the file**
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_newsapi/stats_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_newsapi/stats_summary.json)

### CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-8B_20240701_20240831
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-8B_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-8B_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-8B_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-8B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```

### CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-70B-Instruct --output_dir src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-70B_20240701_20240831
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-70B_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-70B_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-70B_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-70B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```

### CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-405B-Instruct model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-405B-Instruct --output_dir src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-405B_20240701_20240831
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-405B_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-405B_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="CoT_ForecasterTextBeforeParsing_llama-3.1-405B_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-405B-Instruct -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```

### ResolverBasedForecaster with perplexity/llama-3.1-sonar-huge-128k-online model (with OpenRouter)
(maybe skip this one?)

- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl -p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-huge-128k-online -o model=perplexity/llama-3.1-sonar-huge-128k-online -o n_attempts=1 --num_lines 1000 --run --async --output_dir src/data/forecasts/ResolverBasedForecaster_huge_20240701_20240831
```

### ResolverBasedForecaster with perplexity/llama-3.1-sonar-large-128k-online model (with OpenRouter)
- [x] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl -p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-large-128k-online -o model=perplexity/llama-3.1-sonar-large-128k-online -o n_attempts=1 --num_lines 1000 --run --async --output_dir src/data/forecasts/ResolverBasedForecaster_large_20240701_20240831
```
-> [`src/data/forecasts/ResolverBasedForecaster_large_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/ResolverBasedForecaster_large_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="ResolverBasedForecaster_large_tuples_newsapi"
USE_OPENROUTER=True python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-large-128k-online -o model=perplexity/llama-3.1-sonar-large-128k-online -o n_attempts=1 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```

## ConsistentForecaster jobs


## PromptedToCons_Forecaster jobs

### PromptedToCons_Forecaster with gpt-4o-mini-2024-07-18 model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 100 --run --async -f PromptedToCons_Forecaster -o model=gpt-4o-mini-2024-07-18 --output_dir src/data/forecasts/PromptedToCons_Forecaster_gpt4o_mini_2024-07-18_20240501_20240815
```
-> [`src/data/forecasts/PromptedToCons_Forecaster_gpt4o_mini_20240501_20240815/ground_truth_summary.json`](src/data/forecasts/PromptedToCons_Forecaster_gpt4o_mini_20240501_20240815/ground_truth_summary.json)

- [x] evaluation
```
OUTPUT_DIRNAME="PromptedToCons_Forecaster_gpt4o_mini_2024-07-18_tuples_scraped"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_scraped --num_lines 100 --run --async -f PromptedToCons_Forecaster -o model=gpt-4o-mini-2024-07-18 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```
-> [`src/data/forecasts/PromptedToCons_Forecaster_gpt4o_mini_2024-07-18_tuples_scraped/stats_summary.json`](src/data/forecasts/PromptedToCons_Forecaster_gpt4o_mini_2024-07-18_tuples_scraped/stats_summary.json)



### PromptedToCons_Forecaster with gpt-4o-mini-2024-07-18 model
- [] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --num_lines 1000 --run --async -f PromptedToCons_Forecaster -o model=gpt-4o-mini-2024-07-18 --output_dir src/data/forecasts/PromptedToCons_Forecaster_gpt4o_mini_2024-07-18_20240701_20240831
```
-> [`src/data/forecasts/PromptedToCons_Forecaster_gpt4o_mini_2024-07-18_20240701_20240831/ground_truth_summary.json`](src/data/forecasts/PromptedToCons_Forecaster_gpt4o_mini_2024-07-18_20240701_20240831/ground_truth_summary.json)

- [ ] evaluation
```
OUTPUT_DIRNAME="PromptedToCons_Forecaster_gpt4o_mini_2024-07-18_tuples_newsapi"
USE_OPENROUTER=False python src/evaluation.py --tuple_dir src/data/tuples_newsapi --num_lines 300 --run --async -f PromptedToCons_Forecaster -o model=gpt-4o-mini-2024-07-18 -k all --output_dir src/data/forecasts/$OUTPUT_DIRNAME 2>&1 | tee logs/{$OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M).log || true
```


