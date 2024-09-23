## Run ground_truth_run.py on [20240501_20240815.jsonl](src/data/fq/real/20240501_20240815.jsonl) and the corresponding [tuples_scraped](src/data/tuples/tuples_scraped/)

### BaselineForecaster with p=0.4
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::BaselineForecaster --forecaster_options p=0.4 --num_lines 242 --run --async
```
-> [`src/data/forecasts/BaselineForecaster_09-23-13-41/ground_truth_summary.json`](src/data/forecasts/BaselineForecaster_09-23-13-41/ground_truth_summary.json)

- [x] evaluation
```
python src/evaluation.py --tuples_dir src/data/tuples/tuples_scraped/ -p src/forecasters/various.py::BaselineForecaster --forecaster_options p=0.4 -k all --num_lines 500 --run --async
```
-> [`src/data/evaluation/BaselineForecaster_09-23-14-12/stats_summary.json`](src/data/forecasts/BaselineForecaster_09-23-14-12/stats_summary.json)


### BaselineForecaster with p=0.6
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::BaselineForecaster- --forecaster_options p=0.6 --num_lines 242 --run --async
```

- [ ] evaluation
```
python src/evaluation.py --tuples_dir src/data/tuples/tuples_scraped/ -p src/forecasters/various.py::BaselineForecaster --forecaster_options p=0.6 -k all --num_lines 500 --run --async
```

### ResolverBasedForecaster with perplexity/llama-3.1-sonar-huge-128k-online model
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::ResolverBasedForecaster --forecaster_options resolver_model=perplexity/llama-3.1-sonar-huge-128k-online model=perplexity/llama-3.1-sonar-huge-128k-online n_attempts=1 --num_lines 242 --run --async
```

- [ ] evaluation
```
python src/evaluation.py --tuples_dir src/data/tuples/tuples_scraped/ -p src/forecasters/various.py::ResolverBasedForecaster --forecaster_options resolver_model=perplexity/llama-3.1-sonar-large-128k-online model=perplexity/llama-3.1-sonar-large-128k-online n_attempts=1 -k all --num_lines 500 --run --async
```

### ResolverBasedForecaster with perplexity/llama-3.1-sonar-large-128k-online model
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::ResolverBasedForecaster --forecaster_options resolver_model=perplexity/llama-3.1-sonar-large-128k-online model=perplexity/llama-3.1-sonar-large-128k-online n_attempts=1 --num_lines 242 --run --async
```

### BasicForecaster with gpt-4o-2024-08-06 model
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=gpt-4o-2024-08-06
```
-> [`src/data/forecasts/BasicForecaster_09-23-13-46/ground_truth_summary.json`](src/data/forecasts/BasicForecaster_09-23-13-46/ground_truth_summary.json)


### BasicForecaster with gpt-4o-2024-05-13 model
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -f --num_lines 242 --run --async -f BasicForecaster -o model=gpt-4o-2024-05-13
```

### BasicForecaster with gpt-4o-mini-2024-07-18 model
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=gpt-4o-mini-2024-07-18
```

### BasicForecaster with anthropic/claude-3.5-sonnet model (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=anthropic/claude-3.5-sonnet
```

### BasicForecaster with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-8B-Instruct
```

### BasicForecaster with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-70B-Instruct
```

### BasicForecaster with meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo model (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
```

### CoT_ForecasterTextBeforeParsing with o1-mini model
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-mini
```

### CoT_ForecasterTextBeforeParsing with o1-preview model
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-preview
```

### CoT_ForecasterTextBeforeParsing with gpt-4o-2024-08-06 model
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-2024-08-06
```

### CoT_ForecasterTextBeforeParsing with gpt-4o-mini-2024-07-18 model
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-mini-2024-07-18
```

### CoT_ForecasterTextBeforeParsing with anthropic/claude-3.5-sonnet model (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=anthropic/claude-3.5-sonnet
```

### CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-8B-Instruct
```

### CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-70B-Instruct
```

### CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo model (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
```

### AdvancedForecaster with [cheap_haiku.yaml](src/forecasters/forecaster_configs/advanced/cheap_haiku.yaml) config (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/cheap_haiku.yaml
```

### AdvancedForecaster with [cheap_gpt4o-mini.yaml](src/forecasters/forecaster_configs/advanced/cheap_gpt4o-mini.yaml) config
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/cheap_gpt4o-mini.yaml
```
-> [`src/data/forecasts/AdvancedForecaster_09-23-13-52/ground_truth_summary.json`](src/data/forecasts/AdvancedForecaster_09-23-13-52/ground_truth_summary.json)

### AdvancedForecaster with [default_gpt-4o-2024-08-06.yaml](src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-08-06.yaml) config
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-08-06.yaml
```
-> [`src/data/forecasts/AdvancedForecaster_09-23-14-32/ground_truth_summary.json`](src/data/forecasts/AdvancedForecaster_09-23-14-32/ground_truth_summary.json)

### AdvancedForecaster with [default_gpt-4o-2024-05-13.yaml](src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-05-13.yaml) config
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -f --num_lines 242 --run --async AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-05-13.yaml
```

### AdvancedForecaster with [default_sonnet.yaml](src/forecasters/forecaster_configs/advanced/default_sonnet.yaml) config (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -f --num_lines 242 --run --async AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/default_sonnet.yaml
```

### PromptedToCons_Forecaster with gpt-4o-mini-2024-07-18
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -f --num_lines 242 --run --async PromptedToCons_Forecaster -o model=gpt-4o-mini-2024-07-18
```



## TODO: ConsistentForecaster jobs

