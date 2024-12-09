# Usage of the benchmark
See [USAGE.md](USAGE.md) for instructions on how to:
1. evaluate your AI forecasters on our consistency benchmarks; or
2. generate your own consistency benchmarks from a dataset of forecasting questions.

# Download the datasets
The datasets used in the paper are also available on Hugging Face: [dpaleka/ccflmf](https://huggingface.co/datasets/dpaleka/ccflmf).


# Development guide
Note: this section is for if you want to extend or use our codebase in a way other than evaluating your LLM forecaster on the consistency benchmarks.
In case you just want to evaluate your forecaster, see [USAGE.md](USAGE.md).

## Installation requirements
We use `uv` to manage our Python environment. In case you use something else, just remove the `uv` prefix from any commands below.

[Create a virtual environment](https://docs.astral.sh/uv/pip/environments/), and ensure it has Python 3.11 installed.

```python
import sys
assert sys.version_info[:2] == (3, 11), "Python 3.11 is required."
```

Then do:
```
uv pip install -r requirements.txt
pre-commit install
```


Then, create your `.env` based on [`.env.example`](.env.example). By default, use `NO_CACHE=True`. If you want to reuse LLM calls, set `NO_CACHE=False` and e.g. `LOCAL_CACHE=.cache`.


### VS Code / Cursor settings
Copy the settings in [`.vscode/settings.example.json`](.vscode/settings.example.json) to your workspace `settings.json`,
or just do `cp .vscode/settings.example.json .vscode/settings.json` if you have no other settings nor an existing workspace.
Optionally, append the contents of `.cursorrules` to your LLM coding instructions.


## Coding guidelines

### Utils
**Please read [LLM call utils](/src/common/README.md) and [.cursorrules](.cursorrules).**
Feel free to add more utils in `utils.py`, `llm_utils.py`, or other files, as you need them.

### Running code
The preferred way to test functionality is via a test in `tests/`,  or creating a new file / Jupyter notebook in the `src` directory.
Do not create scripts in subfolders of the `src` directory.
Do not run files with actual logic (e.g. anything in `static_checks/` ) directly; this runs into Python import / path issues.

### Testing Before Submitting PRs
Before submitting a pull request that deals with the core code in `src/`, please ensure that you run the test suite to check that your changes do not break any existing functionality. 
You can rerun the tests with the following command from the root directory of the project:
```
NO_CACHE=True python -m pytest -s
```
This will run all tests located in the `tests/` directory. 
As `pytest` also runs all files named `test_*.py` or `*_test.py`, please do not name anything in `src/` like this if you don't think it should run on every PR.
If you want to reuse the LLM calls you made in a previous test run, use the `LOCAL_CACHE=cache_dir` flag.
Please fix any failing tests before submitting your PR.

#### Skipping tests
The following tests are skipped by default. You can run them by enabling the corresponding flags:

- [`tests/test_verify_question.py`](tests/test_verify_question.py) checks that ForecastingQuestion verification works as expected. The flag is `TEST_FQ_VERIFICATION`.
- [`tests/test_verify_tuple.py`](tests/test_verify_tuple.py) checks that all consistency tuple verification works as expected. The flag is `TEST_TUPLE_VERIFICATION`.
- [`tests/test_adv_forecaster.py`](tests/test_adv_forecaster.py) checks that AdvancedForecaster (from [Halawi et al. 2024](https://arxiv.org/abs/2402.18563)) works as expected. The flag is `TEST_ADV_FORECASTER`.
- [`tests/test_consistent_forecaster.py`](tests/test_consistent_forecaster.py) checks that ConsistentForecaster (called ArbitrageForecaster in the paper) works as expected. The flag is `TEST_CONSISTENT_FORECASTER`.
- [`tests/test_perplexity_resolver.py`](tests/test_perplexity_resolver.py) checks that the automated Perplexity resolver of questions works as expected. The flag is `TEST_PERPLEXITY_RESOLVER`.

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
(Note: in our terminology, *validation* is about data format, *verification* is about the semantics of the data.)

Our base data directory is `src/data/`. Inside this, we have the following schema:
```
src/data
├── fq
│  ├── real             # ForecastingQuestions (FQs) made from scraped Manifold and Metaculus questions. Formatting validated upon commit.
│  └── synthetic        # ForecastingQuestions (FQs) made from synthetic data. Formatting validated upon commit.
├── feedback            # Feedback data on real and synhetic questions. TODO Validate upon commit.
├── tuples              # Consistency checks, consisting of named dicts of ForecastingQuestions. Formatting validated upon commit. 
│   |- scraped           # From FQs scraped from Manifold and Metaculus.
│   |- newsapi           # From news-generated synthetic FQs.
│   |- 2028              # From synthetic FQs resolving in 2028.
├── other               # All other non-final data, e.g. raw scrapes, or intermediate steps for synthetic questions. Not validated. 
├── check_tuple_logs    # Where forecasting of the already instantiated consistency checks + violation is logged. In .gitignore, do not commit. 
├── forecasts           # Where forecast results on tuples are saved. Not validated. Commit only full-fledged experimental results.
├── verification        # Logging question verification. In .gitignore, do not commit.
└── test                # Where tests write data. In .gitignore, do not commit.
```

The script that validates the data is [`hooks/validate_jsonls.py`](hooks/validate_jsonls.py). The pre-commit hooks runs this on everything that is changed in the commit. 
To validate all data without running pre-commit, execute the following command:
```
VALIDATE_ALL=True python hooks/validate_jsonls.py
```


## Labeling tool for questions
The streamlit app [`data_labeling/feedback_form.py`](data_labeling/feedback_form.py) is used to label questions.
Do not try to install its dependencies in the main Python environment.
The simplest way to run the labeling tool is to create a new virtual environment and install the requirements with:
```
uv pip install -r data_labeling/streamlit_requirements.txt
```
Alternatively, you can use [pipx](https://github.com/pypa/pipx), run `pipx install streamlit`, and continue to use the Python environment you have been using so far.


The feedback form app can be run with:
```
cd data_labeling
streamlit run feedback_form.py -- -f ../src/data/fq/synthetic/{filename}.jsonl
```
It writes into `src/data/feedback/`.


## Entry points to the code

- [`src/format_and_verify_questions.py`](src/format_and_verify_questions.py) reads from a file with (potentially incomplete) ForecastingQuestions, optionally fills `body` and `resolution_date`, and verifies basic sanity checks on the `body` using a LLM call. It raises a ValidationError if the file contains incorrect data types, e.g. an invalid JSONL, or incorrect datetime for `resolution_date`, or non-string types where strings are needed. Hence, this script should always be run on files containing a valid subset of ForecastingQuestion entries, because it won't fix any formatting errors except missing `body` and `resolution_date` fields. If you want it to fill in the body (resolution criteria), use the `--fill_in_body` flag. *Please read and understand all the flags before running the script*. Writes to `src/data/fq/{appropriate_dir}...`
  - Note: verification is really aggressive (https://github.com/dpaleka/consistency-forecasting/issues/182, https://github.com/dpaleka/consistency-forecasting/issues/199), no matter the model used. It discards many questions with very slight or nonexistent flaws, with the corresponding benefit of a low false negative rate. There is no way to move the decision boundary in the current implementation.
  
- [`src/validate_fq_jsonl.py`](src/validate_fq_jsonl.py) Validates that a JSONL file contains only valid ForecastingQuestions, in the sense of having the correct data types. Does not write anything.

- [`scripts/pipeline/scrape_question.py`](scripts/pipeline/scrape_question.py) runs pipeline to scrape a given data source for questions resolving in a given range, process and optionally verify them, and store them in `src/data/fq/real/`.  It is highly recommended to check the options given in the script before running it. Any part of this pipeline can be skipped, which is particularly useful if the data has already been scraped. Example command:
```
python scrape_question.py -d manifold -s 20240501 -e 20240815 -n 500 -o cleaned_formatted -m gpt-4o-2024-08-06 --verification_level none --skip scrape
```

### Synthetic FQ generation
- [`src/generate_topic_questions.py`](src/generate_topic_questions.py) Generates "raw" synthetic questions from topics. Note: this script has to have `OPENAI_JSON_STRICT=False` in `.env` to work if using OpenAI, because the datatype is too complex for OpenAI's strict JSON mode.
  
- [`src/generate_related_questions.py`](src/generate_related_questions.py) Generates "raw" synthetic questions from source questions. See [`tests/test_evaluation_pipeline.py`](tests/test_evaluation_pipeline.py) for an example command.

- [`src/generate_fqs_from_news.py`](src/generate_fqs_from_news.py) generates FQs with ground-truth resolution using NewsAPI scraped data. See  [`src/fq_from_news/README.md`](src/fq_from_news/README.md) for how to use it.

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
Please `python src/evaluation.py --help` and read what it says before using this script.
  - Run example: see the commands in [tests/test_evaluation_pipeline.py](tests/test_evaluation_pipeline.py). If the tests are passing, similar commands should work.

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


This list not include the entry points already mentioned in previous sections (feedback form, tests).

## Important data files

### ForecastingQuestion datasets
- [`src/data/fq/real/20240501_20240815.jsonl`](src/data/fq/real/20240501_20240815.jsonl) is a file of 242 scraped and FQ-verified questions from Manifold and Metaculus that were both scheduled to resolve *and* actually resolved between May 1, 2024 and August 15, 2024, inclusive.
- [`src/data/fq/real/20240501_20240815_unverified.jsonl`](src/data/fq/real/20240501_20240815_unverified.jsonl) has 627 questions, but not FQ-verified. Certain originally multiple-choice Metaculus questions might have different wording in here and in the above file. May contain weird questions that are not answerable from general world knowledge, such as meta-questions about prediction markets, or joke questions.

- [`src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831_gpt-4o_spanned_resolved.jsonl`](src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831_gpt-4o_spanned_resolved.jsonl) is a file of 2621 synthetic ForecastingQuestions generated from NewsAPI data and reference spanning, using gpt-4o and claude-3.5-sonnet, between July 1, 2024 and August 31, 2024, inclusive. The resolutions are all produced by the Perplexity resolver using the command:
```
USE_OPENROUTER=True python src/perplexity_resolver_script.py -i src/data/fq/synthetic/news_api_generated_fqs/.../strict_res_checking_fqs_cleaned-ref-class-spanned-basic.jsonl --models perplexity/llama-3.1-sonar-huge-128k-online --start_from 0 -n [file_size] --batch_size 30 --n_attempts 1 --include_unresolvable
```
and then merged using [`src/merge_fq_files.py`](src/merge_fq_files.py).
- [`src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl`](src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl) is a subset of the above, containing 1000 questions, including all 150 NewsAPI-generated questions and 850 reference-class-spanned questions.
  The command to generate this file is:
```shell
python src/filter_fqs.py --input_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831_gpt-4o_spanned_resolved.jsonl --output_file src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831.jsonl --filter_score original --only_preference --random_sample 1000 
```

- [`src/data/fq/synthetic/questions_resolving_2028.jsonl`](src/data/fq/synthetic/questions_resolving_2028.jsonl) is a file of 900 FQ-verified synthetic ForecastingQuestions, each resolving by or in 2028, produced by [`src/generate_topic_questions.py`](src/generate_topic_questions.py). 

### Consistency benchmark datasets

- [`src/data/tuples/scraped/`](src/data/tuples/scraped/) contains the tuples generated from the scraped Metaculus and Manifold FQs in [`src/data/fq/real/20240501_20240815.jsonl`](src/data/fq/real/20240501_20240815.jsonl). There are 500 tuples per check, except for NegChecker and ParaphraseChecker, where we restrict to the number of questions in the source if less than 500.

- [`src/data/tuples/newsapi/`](src/data/tuples/newsapi/) contains the tuples generated from the NewsAPI FQs in [`src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831_gpt-4o_spanned_resolved.jsonl`](src/data/fq/synthetic/news_api_generated_fqs/20240701_20240831_gpt-4o_spanned_resolved.jsonl) described above. There are 500 tuples per check.

- [`src/data/tuples/2028`](src/data/tuples/2028) contains 300 tuples per check, generated from the 2028 FQs in [`src/data/fq/synthetic/questions_resolving_2028.jsonl`](src/data/fq/synthetic/questions_resolving_2028.jsonl), using the following command:
```
python src/instantiation.py -d src/data/fq/synthetic/questions_resolving_2028.jsonl --n_relevance=3000 --n_write=300 --seed=42 --tuple_dir=src/data/tuples/2028 -k all --model_main=gpt-4o-2024-08-06 --model_relevance=gpt-4o-mini-2024-07-18
```

