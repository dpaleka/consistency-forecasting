# Consistency & Forecasting

## Installation requirements
Create a virtual environment, and ensure it has Python 3.11 installed.

```python
import sys
assert sys.version_info[:2] == (3, 11), "Python 3.11 is required."
```

Then do:
```
pip install -r requirements.txt
pre-commit install
```

Then, create your `.env` based on [`.env.example`](.env.example). By default, use `NO_CACHE=True`.

### VS Code / Cursor settings
Copy the settings in [`.vscode/settings.example.json`](.vscode/settings.example.json) to your workspace `settings.json`,
or just do `cp .vscode/settings.example.json .vscode/settings.json` if you have no other settings nor an existing workspace.

## docs
- [Meeting and Agenda doc](https://docs.google.com/document/d/1_amt7CQK_aadKciMJuNmedEyf07ubIAL_b5ru_mS0nw/edit)
- [Datatypes and Pipeline doc](https://docs.google.com/document/d/19CDHfwKHfouttiXPc7UNp8iBeYE4KD3H1Hw8_kqnnL4/edit)
- [Overleaf](https://www.overleaf.com/project/661ef8533d19f47ba8b0b3b6)
- [Key considerations doc](https://docs.google.com/document/d/1VR39XE--JPel8dMpwnFxPhoqiMxQDzC9sDdqsjH4IoI/edit)
- [Poster for ICML workshops](https://docs.google.com/presentation/d/1lWDL7pZcyjFtwLM6Gd9tw2uOPRyJHHoM1FLpQVDMcq8/edit#slide=id.p)
- [April 16 writeup](https://docs.google.com/document/d/1849L5P9JNZEjBp4s4TsivJOG2iS98Ru6conx9jE0wPE/edit)
- [Instructor vs BAML](https://docs.google.com/document/d/1x4uwVMZ9Dgf0Y6OxtKID9W-txCN7ya18z2QXWGV3rsA/edit)

## Coding guidelines

### Utils
**Please read [LLM call utils](/src/common/README.md).**
Feel free to add more utils in `utils.py`, `llm_utils.py`, or other files, as you need them.


### Running code
The preferred way to test anything is either from `playground.py`, or creating a new file / Jupyter notebook in the `src` directory.
Do not create scripts in subfolders of the `src` directory.
Do not run files with actual logic (e.g. anything in `static_checks/` ) directly; this runs into Python import / path issues.

### Testing Before Submitting PRs
Before submitting a pull request that deals with the core code in `src/`, please ensure that you run the test suite to check that your changes do not break any existing functionality. 
You can run the tests with the following command from the root directory of the project:
```
NO_CACHE=True python -m pytest -s
```
This will run all tests located in the `tests/` directory. Please fix any failing tests before submitting your PR.
As `pytest` also runs all files named `test_*.py` or `*_test.py`, please do not name anything in `src/` like this if you don't think it should run on every PR.

#### Skipping tests
The following tests are skipped by default. You can run them by enabling the corresponding flags:

- [`tests/test_verify_question.py`](tests/test_verify_question.py) checks that ForecastingQuestion verification works as expected. The flag is `TEST_FQ_VERIFICATION`.
- [`tests/test_verify_tuple.py`](tests/test_verify_tuple.py) checks that all consistency tuple verification works as expected. The flag is `TEST_TUPLE_VERIFICATION`.
- [`tests/test_adv_forecaster.py`](tests/test_adv_forecaster.py) checks that AdvancedForecaster works as expected. The flag is `TEST_ADV_FORECASTER`.
- [`tests/test_consistent_forecaster.py`](tests/test_consistent_forecaster.py) checks that ConsistentForecaster works as expected. The flag is `TEST_CONSISTENT_FORECASTER`.

All other tests are enabled by default.

### Paths
Use [`src/common/path_utils.py`](/src/common/path_utils.py) to specify paths in code, Jupyter notebooks, etc.
Do not hardcode paths, except relative to `pathlib.Path` objects returned by the utils in `path_utils.py`.

### Relative imports
Try to not have relative imports (meaning: put entry points to the code in `src/`.). 
If you must, here is a prototype of how to modify your path to import something:
```
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
```

### Validation of data
(Note: *validation* is about data format, *verification* is about the semantics of the data.)

Our base data directory is `src/data/`. Inside this, we have the following schema:
```
src/data
├── fq
│  ├── real             # ForecastingQuestions made from real scraped data. Formatting validated upon commit.
│  └── synthetic        # ForecastingQuestions made from synthetic data. Formatting validated upon commit.
├── feedback            # Feedback data on real and synhetic questions. TODO Validate upon commit.
├── tuple*              # Tuples of (question, answer) pairs. Formatting validated upon commit. TODO we need to expand this section and clean up where tuples go.
├── other               # All other data, e.g. raw scrapes, or intermediate steps for synthetic questions. Not validated. TODO move some stuff out of here to somewhere where it makes sense.
├── check_tuple_logs    # Where forecasting of the already instantiated consistency checks + violation is logged. In .gitignore, do not commit. 
├── forecasts           # Where forecast results on tuples are saved. Not validated. Commit only full-fledged experimental results.
├── verification        # Logging question verification. In .gitignore, do not commit.
└── test                # Where tests write data. In .gitignore, do not commit.
```

This schema is not final. In particular:
- We might add other directories, e.g. for forecasts, later. 
- If we figure out a need for some data to be committed, we can remove the corresponding .gitignore entry.

TODO we need to fix this schema, too many things are in `data/other`.

Please install `pre-commit`, so the validation hooks in `hooks/` can check that all data in the validated directories is in the correct format.
The script that validates the data is [`hooks/validate_jsonls.py`](hooks/validate_jsonls.py). The pre-commit hooks runs this on everything that is changed in the commit. 
To validate all data outside of the pre-commit hook process, execute the following command:
```
VALIDATE_ALL=True python hooks/validate_jsonls.py
```



## Labeling tool for questions
The streamlit app [`data_labeling/feedback_form.py`](data_labeling/feedback_form.py) is used to label questions.
Do not try to install its dependencies in the main Python environment.
Instead, make a new virtual environment and install the requirements with:
```
pip install -r data_labeling/streamlit_requirements.txt
```
Alternatively, for now you can use [pipx](https://github.com/pypa/pipx), run `pipx install streamlit`, and continue to use the Python environment you have been using so far.


The feedback form app can be run with:
```
cd data_labeling
streamlit run feedback_form.py -- -f ../src/data/fq/synthetic/{filename}.jsonl
```
It writes into `src/data/feedback/`.

## Bot for the Metaculus competition
We run the forecasters on the Metaculus [AI Forecasting Benchmark Series (July 2024)](https://www.metaculus.com/notebooks/25525/announcing-the-ai-forecasting-benchmark-series--july-8-120k-in-prizes/) (contact @amaster97 for details).
The bot to call the forecasters is developed in `src/metaculus_competition_fast.py`.  If that doesn't "work" due to issues with concurrency, a similar fallback script `src/metaculus_competition_slow.py`
 can be substituted.  It can be run as a single job on [Modal](https://modal.com/) by doing:
```
python competition_bot/modal_daily_job.py
```

Furthermore, modal also allows the deployment of timed runs for the program (in this case daily).  To deploy a recurring job use:

```
modal deploy competition_bot/modal_daily_job.py
```


To set up, you need Modal credentials.
Do not try to install `modal` in the main Python environment.
Instead, make a new virtual environment and install the requirements with:
```
pip install -r competition_bot/modal_requirements.txt
modal token new
```
and follow the instructions.

In addition to the LLM API costs, each daily run costs Modal credits for the CPU time occupied. Modal gives $30 in credits to new users, and it should be enough for this competition.

### Logs
We store logs of both the question - submissions as well as any potential errors.
In the modal app we have these stored at:
LOG_FILE_PATH = "/mnt/logs/metaculus_submissions.log"
ERROR_LOG_FILE_PATH = "/mnt/logs/metaculus_submission_errors.log"

## Entry points to the code

- [`src/format_and_verify_questions.py`](src/format_and_verify_questions.py) reads from a file with (potentially incomplete) ForecastingQuestions, optionally fills `body` and `resolution_date`, and verifies basic sanity checks on the `body` using a LLM call. It raises a ValidationError if the file contains incorrect data types, e.g. an invalid JSONL, or incorrect datetime for `resolution_date`, or non-string types where strings are needed. Thus, this should always be run on files containing a valid subset of ForecastingQuestion entries; it won't fix any formatting errors except missing `body` and `resolution_date` fields. If you want it to fill in the body (resolution criteria), use the `--fill_in_body` flag. *It is mandatory to read and understand all flags before running this script*. Writes to `src/data/fq/{appropiate_dir}...`

- [`src/validate_fq_jsonl.py`](src/validate_fq_jsonl.py) Validates that a JSONL file contains only valid ForecastingQuestions, in the sense of having the correct data types. Does not write anything.

- [`scripts/pipeline/scrape_question.py`](scripts/pipeline/scrape_question.py) runs pipeline to scrape a given data source for questions resolving in a given range, process and optionally verify them, and store them in `src/data/fq/real/`.  It is highly recommended to check the options given in the script before running it. Any part of this pipeline can be skipped, which is particularly useful if the data has already been scraped. Example command:
```
python scrape_question.py -d manifold -s 20240501 -e 20240815 -n 500 -o cleaned_formatted -m gpt-4o-2024-08-06 --verification_level none --skip scrape
```

### Synthetic FQ generation
- [`src/generate_topic_questions.py`](src/generate_topic_questions.py) Generates "raw" synthetic questions from topics. Note: this script has to have `OPENAI_JSON_STRICT=False` in `.env` to work if using OpenAI, because the datatype is too complex for OpenAI's strict JSON mode.
  
- [`src/generate_related_questions.py`](src/generate_related_questions.py) Generates "raw" synthetic questions from source questions. See [`tests/test_evaluation_pipeline.py`](tests/test_evaluation_pipeline.py) for an example command.

- [`src/generate_fqs_from_news.py`](src/generate_fqs_from_news.py) generates FQs with ground-truth resolution using NewsAPI scraped data
  - Usage has been described in [`src/fq_from_news/README.md`](src/fq_from_news/README.md)

- [`src/generate_fqs_using_reference_class.py`](src/generate_fqs_using_reference_class.py) Creates new forecasting questions from some source FQs following the same there and structure as the original questions.

- [`src/perplexity_resolver_script.py`](src/perplexity_resolver_script.py) processes a JSONL file of forecasting questions using Perplexity AI models. It resolves each question and writes the results to a new JSONL file with `_resolved.jsonl` as suffix. The script supports various command-line arguments for customization, run with `--help` to see all options. Example usage:
  ```
  USE_OPENROUTER=True python src/perplexity_resolver_script.py --input_file path/to/input.jsonl --max_questions 10 --include_unresolvable --n_attempts 2
  ```

### Tuple instantiation
- [`src/instantiation.py`](src/instantiation.py) Runs instantiation. Takes a JSONL file (a list of ForecastingQuestions), and writes multiple JSONL files (each a list of QuestionTuples) into `src/data/tuples`.

### Forecasting and evaluation
- [`src/evaluation.py`](src/evaluation.py) runs forecasters on checks and scores them. 
Takes the JSONL files in `src/data/tuples/{self.__class__.__name__}.jsonl` (for each Checker class we have), feeds them their respective Checker.elicit methods.
Please run `python src/evaluation.py --help` and read what it says before using this script.
  - Run example (TODO unclear if this is working): 
```
python src/evaluation.py -f AdvancedForecaster -c src/forecasters/forecaster_configs/cheap_haiku.yaml -num_lines 3 --relevant_checks all [--run] [--load_dir src/data/forecasts/...] [--async] 
```
  - Run example on some directory: see the commands in [tests/test_evaluation_pipeline.py](tests/test_evaluation_pipeline.py). Those are working if the tests are passing.

- [`src/reevaluation.py`](src/reevaluation.py) recomputes violation metrics from files of forecasts made with `src/evaluation.py`, 
and aggregates metrics across multiple forecast files. The `forecasts/` directories it draws from are given in the file, edit them as needed.

- [`src/ground_truth_run.py`](src/ground_truth_run.py) runs the ground truth forecasting evaluation pipeline end-to-end.
See the commands in [tests/test_ground_truth_run.py](tests/test_ground_truth_run.py), or just run something like:
```
python src/ground_truth_run.py --input_file src/data/fq/real/metaculus_cleaned_formatted_20240501_20240815.jsonl --forecaster_class BasicForecaster --forecaster_options model=gpt-4o-mini --num_lines 10 --run --async
```

Any Python class that inherits from `Forecaster` can be used as a forecaster in both `src/evaluation.py` and `src/ground_truth_run.py`.
For example:
```
python src/ground_truth_run.py --input_file src/data/fq/real/metaculus_cleaned_formatted_20240501_20240815.jsonl --custom_path src/forecasters/basic_forecaster.py::BasicForecasterWithExamples --forecaster_options model=gpt-4o-mini --num_lines 10 --run --async
```

### Misc
- [`src/forecaster_demo.py`](src/forecaster_demo.py) is a method to run the strong LLM forecasters on a file of ForecastingQuestions. Does not write anything. Writes to `src/data/forecasts/stats_*.jsonl`.

- [`src/playground.py`](src/playground.py) various testing and playing around.


This list not include the entry points already mentioned in previous sections (feedback form, tests).

## Important data files
- [`src/data/fq/real/20240501_20240815.jsonl`](src/data/fq/real/20240501_20240815.jsonl) is a file of 257 scraped and FQ-verified questions from Manifold and Metaculus that were both scheduled to resolve *and* actually resolved between May 1, 2024 and August 15, 2024, inclusive.
- [`src/data/fq/real/20240501_20240815_unverified.jsonl`](src/data/fq/real/20240501_20240815_unverified.jsonl) is close to a superset of the above, 627 questions, but not FQ-verified, and certain originally multiple-choice Metaculus question might have different wording. May contain weird questions that are not answerable from general world knowledge, such as meta-questions about prediction markets, or joke questions.

- [`src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831_gpt-4o_spanned_resolved.jsonl`](src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831_gpt-4o_spanned_resolved.jsonl) is a file of 2621 synthetic ForecastingQuestions generated from NewsAPI data and reference spanning, using gpt-4o, between July 1, 2024 and August 31, 2024, inclusive. The resolutions are all produced by the Perplexity resolver using the command:
```
USE_OPENROUTER=True python src/perplexity_resolver_script.py -i src/data/fq/synthetic/news_api_generated_fqs/.../strict_res_checking_fqs_cleaned-ref-class-spanned-basic.jsonl --models perplexity/llama-3.1-sonar-huge-128k-online --start_from 0 -n [file_size] --batch_size 30 --n_attempts 1 --include_unresolvable
```
and then merged using [`src/merge_fq_files.py`](src/merge_fq_files.py).

- [`src/data/tuples_scraped/`](src/data/tuples_scraped/) contains the tuples generated from the scraped Metaculus and Manifold FQs described above.

- [`src/data/tuples_newsapi/`](src/data/tuples_newsapi/) contains the tuples generated from the NewsAPI FQs described above

## Experiments

### Forecasters used for the experiments
(Draft, for now)

- `-p src/forecasters/various.py::BaselineForecaster -o p=0.4`
- `-p src/forecasters/various.py::BaselineForecaster -o p=0.6`
- `-p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-huge-128k-online -o model=perplexity/llama-3.1-sonar-huge-128k-online -o n_attempts=1` (with OpenRouter)
- `-p src/forecasters/various.py::ResolverBasedForecaster -o resolver_model=perplexity/llama-3.1-sonar-huge-128k-online -o model=perplexity/llama-3.1-sonar-huge-128k-online -o n_attempts=1` (with OpenRouter)
- `-f BasicForecaster -o model=gpt-4o-2024-08-06`
- `-f BasicForecaster -o model=gpt-4o-2024-05-13`
- `-f BasicForecaster -o model=gpt-4o-mini-2024-07-18`
- `-f BasicForecaster -o model=anthropic/claude-3.5-sonnet` (with OpenRouter)
- `-f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-8B-Instruct` (with OpenRouter)
- `-f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-70B-Instruct` (with OpenRouter)
- `-f BasicForecaster -o model=meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo` (with OpenRouter)
- `-f CoT_ForecasterTextBeforeParsing -o model=o1-mini`
- `-f CoT_ForecasterTextBeforeParsing -o model=o1-preview`
- `-f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-2024-08-06`
- `-f CoT_ForecasterTextBeforeParsing -o model=gpt-4o-mini-2024-07-18`
- `-f CoT_ForecasterTextBeforeParsing -o model=anthropic/claude-3.5-sonnet` (with OpenRouter)
- `-f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-8B-Instruct` (with OpenRouter)
- `-f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-70B-Instruct` (with OpenRouter)
- `-f CoT_ForecasterTextBeforeParsing -o model=meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo` (with OpenRouter)
- `-f AdvancedForecaster --config_path` [`src/forecasters/forecaster_configs/advanced/cheap_haiku.yaml`](src/forecasters/forecaster_configs/advanced/cheap_haiku.yaml) (with OpenRouter)
- `-f AdvancedForecaster --config_path` [`src/forecasters/forecaster_configs/advanced/default_gpt4o_mini.yaml`](src/forecasters/forecaster_configs/advanced/default_gpt4o_mini.yaml)
- `-f AdvancedForecaster --config_path` [`src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-08-06.yaml`](src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-08-06.yaml)
- `-f AdvancedForecaster --config_path` [`src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-05-13.yaml`](src/forecasters/forecaster_configs/advanced/default_gpt-4o-2024-05-13.yaml)
- `-f AdvancedForecaster --config_path` [`src/forecasters/forecaster_configs/advanced/default_sonnet.yaml`](src/forecasters/forecaster_configs/advanced/default_sonnet.yaml) (with OpenRouter)
- `-f PromptedToCons_Forecaster -o model=gpt-4o-mini-2024-07-18`

Forecasters that run a JSON mode call: `BasicForecaster`, `CoT_Forecaster`.
The other forecasters ask a native call and then parse the answer into an output format with an LLM (or otherwise, in case of `AdvancedForecaster`).
The parsing model is always `gpt-4o-mini-2024-07-18`.

### Evaluation

```
python evaluation.py -f BasicForecaster -o model=gpt-4o-2024-08-06 --run -n 100 -k all --async
# gpt-4o-2024-08-06 is the latest and is cheaper I think
# ... ADD other models e.g. llamas
# ... ADD COT_Forecaster etc.

python evaluation.py -f AdvancedForecaster -c forecasters/forecaster_configs/advanced/cheap_haiku.yaml --run -n 100 -k all --async
# ... perhaps with more configurations

python evaluation.py -f ConsistentForecaster -o model=gpt-4o-mini -o checks='[NegChecker]' -o depth=4 --run -n 100 --relevant_checks NegChecker ParaphraseChecker --async #*
python evaluation.py -f ConsistentForecaster -o model=gpt-4o-mini -o checks='[ParaphraseChecker]' -o depth=4 --run -n 100 --relevant_checks NegChecker ParaphraseChecker --async #*
python evaluation.py -f ConsistentForecaster -o model=gpt-4o-mini -o checks='[NegChecker, ParaphraseChecker]' -o depth=4 --run -n 100 --relevant_checks all --async #*

python evaluation.py -f ConsistentForecaster -o model=gpt-4o-mini -o checks='[ExpectedEvidenceChecker]' -o depth=1 --run -n 100 --relevant_checks all --async
python evaluation.py -f ConsistentForecaster -o model=gpt-4o-mini -o checks='[ExpectedEvidenceChecker, ExpectedEvidenceChecker]' -o depth=1 --run -n 100 --relevant_checks all --async
python evaluation.py -f ConsistentForecaster -o model=gpt-4o-mini -o checks='[ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker]' -o depth=1 --run -n 100 --relevant_checks all --async
python evaluation.py -f ConsistentForecaster -o model=gpt-4o-mini -o checks='[ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker]' -o depth=1 --run -n 100 --relevant_checks all --async
```

Those marked `#*` should then be evaluated with `rcf_evaluation.py`. 

Presumably all the above are also exactly what we want to run ground truth evaluation on?
