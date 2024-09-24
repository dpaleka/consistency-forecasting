## General notes

For evaluation.py, we log the output in a file called logs/evaluation_{some experiment details}.log
We also name the dirs in a meaningful way, like `src/data/forecasts/BasicForecaster_0501_0815_model_gpt-4o-2024-05-13/
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
-> [`src/data/evaluation/BaselineForecaster_09-23-14-12/stats_summary.json`](src/data/forecasts/BaselineForecaster_09-23-14-12/stats_summary.json)


### BaselineForecaster with p=0.6
- [ ] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::BaselineForecaster- --forecaster_options p=0.6 --num_lines 242 --run --async
```

- [ ] evaluation
```
python src/evaluation.py --tuple_dir src/data/tuples_scraped/ -p src/forecasters/various.py::BaselineForecaster -o p=0.6 -k all --num_lines 500 --run --async
```

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
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=gpt-4o-2024-08-06
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
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-mini
```
-> [`src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-23-22-25/ground_truth_summary.json`](src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-23-22-25/ground_truth_summary.json)


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
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl  --num_lines 242 --run --async -f AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-05-13.yaml
```

### AdvancedForecaster with [default_sonnet.yaml](src/forecasters/forecaster_configs/advanced/default_sonnet.yaml) config (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/default_sonnet.yaml
```


## PromptedToCons_Forecaster  (WIP)

### PromptedToCons_Forecaster with gpt-4o-mini-2024-07-18
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f PromptedToCons_Forecaster -o model=gpt-4o-mini-2024-07-18
```
-> [`src/data/forecasts/PromptedToCons_Forecaster_09-23-21-49/ground_truth_summary.json`](src/data/forecasts/PromptedToCons_Forecaster_09-23-21-49/ground_truth_summary.json)

worse than random.

- [x] evaluation
Just NegChecker and CondChecker, two samples:
```
python src/evaluation.py --tuple_dir src/data/tuples_scraped/ -f PromptedToCons_Forecaster -o model=gpt-4o-mini-2024-07-18 -k NegChecker -k CondChecker --num_lines 2 --run --async
``` 
-> [`src/data/forecasts/PromptedToCons_Forecaster_09-24-15-16/stats_summary.json`](src/data/forecasts/PromptedToCons_Forecaster_09-24-15-16/stats_summary.json)
It doesn't seem like it's making it consistent, nor that it is reporting valid reasoning. See [`src/data/forecasts/PromptedToCons_Forecaster_09-24-15-16/NegChecker.jsonl`](src/data/forecasts/PromptedToCons_Forecaster_09-24-15-16/NegChecker.jsonl)


### PromptedToCons_Forecaster with gpt-4o-2024-05-13
- [x] ground_truth_run
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f PromptedToCons_Forecaster -o model=gpt-4o-2024-05-13
```
-> [`src/data/forecasts/PromptedToCons_Forecaster_09-24-13-11/ground_truth_summary.json`](src/data/forecasts/PromptedToCons_Forecaster_09-24-13-11/ground_truth_summary.json)
worse than random.

- [ ] evaluation
```
python src/evaluation.py --tuple_dir src/data/tuples_scraped/ -f PromptedToCons_Forecaster -o model=gpt-4o-2024-05-13 -k all --num_lines 500 --run --async
``` 




## ConsistentForecaster jobs

