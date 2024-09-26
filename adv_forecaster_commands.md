
### AdvancedForecaster with [cheap_haiku.yaml](src/forecasters/forecaster_configs/advanced/cheap_haiku.yaml) config (with OpenRouter)
- [ ] ground_truth_run
```
USE_OPENROUTER=True python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --num_lines 242 --run --async -f AdvancedForecaster -c src/forecasters/forecaster_configs/advanced/cheap_haiku.yaml
```

## AdvancedForecaster
The following runs are wrong, the search was silently failing.
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