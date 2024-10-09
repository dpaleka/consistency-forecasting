Instructions for how to test your forecasting system (a function that takes in text questions and gives probabilities between 0 and 1) on our Consistency and Ground Truth benchmarks. 

In the vast majority of cases, what you want to do is: 

1. subclass our `Forecaster` class to use your system to produce forecasts on the `ForecastingQuestion` datatype
2. generate forecasts on our datasets in `src/data/tuples_...`
3. measure and report violations. This is described under ***Typical usage***.

In some situations you might also like to measure consistency violations on your own custom data set. This is described under ***Bring your own data***.


```bash
## preliminaries
git clone https://github.com/dpaleka/consistency-forecasting.git

pip install -r consistency-forecasting/requirements.txt
```

## Typical usage


1. Create a file `your_forecaster.py` like so:

```python
from consistency-forecasting.src.forecasters import Forecaster
from consistency-forecasting.src.common.datatypes import ForecastingQuestion, Forecast

class YourForecaster(Forecaster):
	
	# subclass the following methods

	def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
		# WRAPPER FOR YOUR LOGIC HERE

	async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
		# WRAPPER FOR YOUR LOGIC HERE

	# optionally also subclass __init__() with any configuration options your forecaster takes, e.g. LLM model name

```

2. Evaluate its consistency on any one of the three data sets we provide (`tuples_scraped`, `tuples_newsapi`, `tuples_2028`), by running:

```bash
TUPLE_DIR=consistency-forecasting/src/data/tuples_scraped
# TUPLE_DIR=consistency-forecasting/src/data/tuples_newsapi
# TUPLE_DIR=consistency-forecasting/src/data/tuples_2028

FORECASTER_PATH=/path/to/your_forecaster.py
# FORECASTER_PATH=/path/to/your_forecaster.py::YourForecaster

# max number of data points you want to evaluate on
NUM_LINES=300 # max 200 for NegChecker and ParaphraseChecker; 500 for all else. will just generate the max if you exceed it or set to -1

OUTPUT_DIR=/folder/where/you/want/results/written/to

python consistency-forecasting/src/evaluation.py --tuple_dir TUPLE_DIR -p FORECASTER_PATH --num_lines NUM_LINES --run --async -k all --output_dir OUTPUT_DIR
```

NOTES: 
- if `your_forecaster.py` contains definitions of more than one subclass of `Forecaster`, you must point to the one you want to test by replacing `-p FORECASTER_PATH` with `-p FORECASTER_PATH::YourForecaster`.
- if your forecaster takes any options in `__init__`, you can supply them with `-o` (`--forecaster-options`) e.g. `-o model=gpt-4o -o temperature=0`.
- `-k all` can be replaced with e.g. `-k NegChecker -k ExpectedEvidenceChecker` if you only want to evaluate on some checks.

3. Optionally, evaluate it on ground truth on the scraped and NewsAPI data sets (the synthetic questions, by design, do not have ground truth resolutions)

```bash
FQ_FILE=consistency-forecasting/src/data/fq/real/20240501_20240815.jsonl # scraped
# FQ_FILE=consistency-forecasting/src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl # newsapi


FORECASTER_PATH=/path/to/your_forecaster.py
# FORECASTER_PATH=/path/to/your_forecaster.py::YourForecaster

OUTPUT_DIR=/folder/where/you/want/results/written/to

# max number of data points you want to evaluate on
NUM_LINES=200 # max 242 for scraped, 1000 for newsAPI; will just generate the max if you exceed it or set to -1

python consistency-forecasting/src/ground_truth_run.py --input_file FQ_FILE -p FORECASTER_PATH --num_lines 1000 --run --async --output_dir OUTPUT_DIR
```

## Bring your own data

Alternatively you might want to evaluate your forecaster for consistency on your own data. If you have a dataset consisting of forecasting questions, then:

- First transform it into a jsonl file where each line is a Forecasting Question (see our [scraped](src/data/fq/real/20240501_20240815.jsonl) dataset as an example) — this is critical because the instantiation process expects, and has only been optimally designed around the `ForecastingQuestion` data type.
- Run the following to form logical tuples out of these questions:

```bash
FQ_FILE=/path/to/ForecastingQuestions.jsonl

# model that generates the tuples
MODEL_MAIN=gpt-4o-2024-08-06
# model that first checks if some questions are suitable to tuple up
MODEL_RELEVANCE=gpt-4o-mini-2024-07-18

# number of potential tuples to check if are worth making tuples of
N_RELEVANCE=5000
# number of tuples to create
N_WRITE=500

TUPLE_DIR=/dir/you/want/to/write/tuples/to

python consistency-forecasting/src/instantiation.py --data-path FQ_FILE --model_main=MODEL_MAIN --model_relevance=MODEL_RELEVANCE --n_relevance=N_RELEVANCE --n_write=N_WRITE --tuple_dir=TUPLE_DIR -k all --seed=42  
```

Again, `-k all` may be replaced with a specific list of consistency checks.

(TODO: add description of `instantiate_related` option)

Then proceed to the Step 2 (evaluation) of “Typical usage”.
