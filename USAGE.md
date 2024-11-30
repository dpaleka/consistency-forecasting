# TL;DR

## Our consistency check datasets
See [Important data files](README.md#consistency-tuple-files) for details on:

- `tuples_2028`: consistency checks generated from the 2028 dataset
- `tuples_scraped`: from questions scraped from the web
- `tuples_newsapi`: questions generated from the NewsAPI

## Measure consistency violations on a set of consistency checks
Take a dataset of questions and forecast them in any way you want. Then dump the forecasts in the following format:
```
// Each line in the JSONL file is a JSON object with this structure:
{
  "line": {
    // Keys vary by checker type
    [component_key: string]: {
      "question": {
        "title": string,
      },
      "forecast": {
        "prob": float,  // probability value between 0 and 1
      }
    }
  }
}
```
For example:
```
{
    "line": {"P": {"question": {"title": "Test P"}, "forecast": {"prob": 0.7}}, "not_P": {"question": {"title": "Test not_P"}, "forecast": {"prob": 0.4}}}
}
```

| Checker Type | Required Components |
|--------------|-------------------|
| NegChecker | `P`, `not_P` |
| ParaphraseChecker | `P`, `Q` |
| AndChecker | `P`, `Q`, `P_and_Q` |
| OrChecker | `P`, `Q`, `P_or_Q` |
| CondChecker | `P`, `Q_given_P`, `P_and_Q` |
| CondCondChecker | `P`, `Q_given_P`, `R_given_P_and_Q`, `P_and_Q_and_R` |
| AndOrChecker | `P`, `Q`, `P_and_Q`, `P_or_Q` |
| ButChecker | `P`, `Q_and_not_P`, `P_or_Q` |
| ConsequenceChecker | `P`, `Q`|
| ExpectedEvidenceChecker | `P`, `Q`, `P_given_Q` , `P_given_not_Q` |


The forecasts should be in a directory, with filenames corresponding to the checker type. Any subset of the following files is fine:

```
/path/to/forecasts/
    /NegChecker.jsonl
    /ParaphraseChecker.jsonl
	/AndChecker.jsonl
	/OrChecker.jsonl
	/CondChecker.jsonl
	/CondCondChecker.jsonl
	/AndOrChecker.jsonl
	/ButChecker.jsonl
	/ConsequenceChecker.jsonl
	/ExpectedEvidenceChecker.jsonl
```

Then run:
```
python consistency-forecasting/src/evaluation.py --load /path/to/forecasts
```
and look at the `stats_{CheckerType}.json` and 

**This appends the consistency metrics in-place to the forecast files, without modifying the forecasts. The above command is idempotent.**

## Measure forecasting accuracy on a set of questions with known resolution

Evaluate on a set of questions with known resolution:

The input file should be a JSONL file called `ground_truth_results.jsonl` where each line contains a JSON object with the following format:
```json
{
    "question": {
        "title": "Question title",
        "resolution": true  // or false
    },
    "forecast": {
        "prob": 0.7  // probability between 0 and 1
    }
}
```

Example:
```json
{"question": {"title": "Test Question 1", "resolution": true}, "forecast": {"prob": 0.7}}
{"question": {"title": "Test Question 2", "resolution": false}, "forecast": {"prob": 0.5}}
{"question": {"title": "Test Question 3", "resolution": true}, "forecast": {"prob": 0.8}}
```

Then run:
```
python consistency-forecasting/src/ground_truth_run.py --load /path/to/forecasts
```

**This appends the ground truth metrics in-place to the forecast files, without modifying the forecasts. The above command is idempotent.**

## Create consistency checks from a set of questions
If you have a dataset consisting of forecasting questions, and you want to generate consistency checks from them, then:

- Transform it into a jsonl file where each line is a Forecasting Question (see our [scraped](src/data/fq/real/20240501_20240815.jsonl) dataset as an example) — this is critical because the instantiation process expects, and has only been optimally designed around the `ForecastingQuestion` data type.
- Run the following to form logical tuples out of these questions:

```bash
FQ_FILE=/path/to/ForecastingQuestions.jsonl

# model that generates the tuples
MODEL_MAIN=gpt-4o-2024-08-06

# model to check relevance of different questions to form tuples
MODEL_RELEVANCE=gpt-4o-mini-2024-07-18

# number of potential tuples to check if are worth making tuples of
N_RELEVANCE=5000

# number of tuples to create
N_WRITE=500

TUPLE_DIR=/dir/you/want/to/write/consistency/checks/to

python consistency-forecasting/src/instantiation.py \
--data-path FQ_FILE \
--model_main=MODEL_MAIN \
--model_relevance=MODEL_RELEVANCE \
--n_relevance=N_RELEVANCE \
--n_write=N_WRITE \
--tuple_dir=TUPLE_DIR \
--relevant_checks=all \
--seed=42  
```

The parameter `--relevant_checks=all` may be replaced with a specific list of consistency checks; run `python src/instantiation.py --help` for more information.

(TODO: add description of `instantiate_related` option)



# Full writeup
Here are the instructions for how to test your forecaster (a function that takes in text questions and gives probabilities between 0 and 1) on our consistency and ground truth benchmarks. 

The typical usage is:

1. take our datasets in `src/data/tuples_...`
2. produce forecasts by subclassing our `Forecaster` class to use your system to produce forecasts on the `ForecastingQuestion` datatype
  - or, dump your forecasts in the required format as described above
3. measure and report violations.



```bash
## preliminaries
git clone https://github.com/dpaleka/consistency-forecasting.git

pip install -r consistency-forecasting/requirements.txt
```

## Implementing and evaluating a Forecaster


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


## Re-evaluation

If whatever reason you have elicited forecasts for each tuple but not the actual inconsistency metrics (similarly, if you have elicited forecasts for each forecasting question but not the actual ground truth metrics), or you want to just recalculate those metrics—

—both `evaluation`  and `ground_truth_run` provide a `--load_dir` option to simply load existing forecasts and recalculate metrics cheaply, writing them back into their respective files.

```bash
python evaluation.py --load_dir="consistency-forecasting/src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped/" -k all
```

```bash
python consistency-forecasting/src/ground_truth_run.py --load_dir="consistency-forecasting/src/data/forecasts/recalc_test/groundtruth/"
```
