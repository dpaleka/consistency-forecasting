# Run ground_truth_run.py on the 242-size dataset for each forecaster described in the README

- [x] BaselineForecaster with p=0.4
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::BaselineForecaster --forecaster_options p=0.4 --num_lines 242 --run --async
```
-> [`src/data/forecasts/BaselineForecaster_09-23-13-41/`](src/data/forecasts/BaselineForecaster_09-23-13-41/)

- [ ] BaselineForecaster with p=0.6
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::BaselineForecaster- --forecaster_options p=0.6 --num_lines 242 --run --async
```


- [ ] ResolverBasedForecaster with perplexity/llama-3.1-sonar-huge-128k-online model
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::ResolverBasedForecaster --forecaster_options resolver_model=perplexity/llama-3.1-sonar-huge-128k-online model=perplexity/llama-3.1-sonar-huge-128k-online n_attempts=1 --num_lines 242 --run --async
```

- [ ] ResolverBasedForecaster with perplexity/llama-3.1-sonar-large-128k-online model
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/various.py::ResolverBasedForecaster --forecaster_options resolver_model=perplexity/llama-3.1-sonar-large-128k-online model=perplexity/llama-3.1-sonar-large-128k-online n_attempts=1 --num_lines 242 --run --async
```

- [ ] BasicForecaster with gpt-4o-2024-08-06 model
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=gpt-4o-2024-08-06
```
-> [`src/data/forecasts/BasicForecaster_09-23-13-46/`](src/data/forecasts/BasicForecaster_09-23-13-46/)


- [ ] BasicForecaster with gpt-4o-2024-05-13 model
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -f --num_lines 242 --run --async -f BasicForecaster -o model=gpt-4o-2024-05-13
```

- [ ] BasicForecaster with gpt-4o-mini-2024-07-18 model
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=gpt-4o-mini-2024-07-18
```

- [ ] BasicForecaster with anthropic/claude-3.5-sonnet model (with OpenRouter)
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=anthropic/claude-3.5-sonnet
```

- [ ] BasicForecaster with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-8B-Instruct
```

- [ ] BasicForecaster with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-70B-Instruct
```

- [ ] BasicForecaster with meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo model (with OpenRouter)
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
```

- [ ] CoT_ForecasterTextBeforeParsing with o1-mini model
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-mini
```

- [ ] CoT_ForecasterTextBeforeParsing with o1-preview model
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=o1-preview
```

- [ ] CoT_ForecasterTextBeforeParsing with gpt-4o-2024-08-06 model
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-2024-08-06
```

- [ ] CoT_ForecasterTextBeforeParsing with gpt-4o-mini-2024-07-18 model
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-mini-2024-07-18
```

- [ ] CoT_ForecasterTextBeforeParsing with anthropic/claude-3.5-sonnet model (with OpenRouter)
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=anthropic/claude-3.5-sonnet
```

- [ ] CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-8B-Instruct model (with OpenRouter)
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-8B-Instruct
```

- [ ] CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-70B-Instruct model (with OpenRouter)
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-70B-Instruct
```

- [ ] CoT_ForecasterTextBeforeParsing with meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo model (with OpenRouter)
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
```

- [ ]  AdvancedForecaster with cheap_haiku.yaml configuration (with OpenRouter)
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/cheap_haiku.yaml
```

- [ ] AdvancedForecaster with default_gpt4o_mini.yaml configuration
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/default_gpt4o_mini.yaml
```

- [ ] AdvancedForecaster with default_gpt-4o-2024-08-06.yaml configuration
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-08-06.yaml
```

- [ ] AdvancedForecaster with default_gpt-4o-2024-05-13.yaml configuration
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -f --num_lines 242 --run --async AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-05-13.yaml
```

- [ ] AdvancedForecaster with default_sonnet.yaml configuration (with OpenRouter)
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -f --num_lines 242 --run --async AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/default_sonnet.yaml
```

- [ ] PromptedToCons_Forecaster with gpt-4o-mini-2024-07-18 model
```
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -f --num_lines 242 --run --async PromptedToCons_Forecaster -o model=gpt-4o-mini-2024-07-18
```

