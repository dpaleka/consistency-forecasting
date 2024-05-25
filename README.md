# Consistency & Forecasting

## Installation requirements
Create a virtual environment, and ensure it has Python 3.11 installed.

```python
import sys
assert sys.version_info >= (3, 11), "Python 3.11 or later is required."
```

Then do:
```
pip install -r requirements.txt
pre-commit install
```

Then, create your `.env` based on [`.env.example`](.env.example). By default, use `NO_CACHE=True`.

## docs
- [Meeting and Agenda doc](https://docs.google.com/document/d/1_amt7CQK_aadKciMJuNmedEyf07ubIAL_b5ru_mS0nw/edit)
- [Datatypes and Pipeline doc](https://docs.google.com/document/d/19CDHfwKHfouttiXPc7UNp8iBeYE4KD3H1Hw8_kqnnL4/edit)
- [Overleaf](https://www.overleaf.com/project/661ef8533d19f47ba8b0b3b6)

## Apr 16 writeup
[writeup doc](https://docs.google.com/document/d/1849L5P9JNZEjBp4s4TsivJOG2iS98Ru6conx9jE0wPE/edit)

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
Our base data directory is `src/data/`. Inside this, we have the following scheme:
```
src/data
├── fq
│  ├── real             # ForecastingQuestions made from real scraped data. Formatting validated upon commit.
│  └── synthetic        # ForecastingQuestions made from synthetic data. Formatting validated upon commit.
├── feedback            # Feedback data on real and synhetic questions. TODO Validate upon commit.
├── tuples              # Tuples of (question, answer) pairs. Formatting validated upon commit.
├── other               # All other data, e.g. raw scrapes, or intermediate steps for synthetic questions. Not validated.
├── check_tuple_logs    # Where forecasting of the already instantiated consistency checks + violation is logged. In .gitignore, do not commit. 
├── forecasts           # Where forecast results on tuples are saved. Not validated. Commit only full-fledged experimental results.
├── verification        # Logging question verification. In .gitignore, do not commit.
└── test                # Where tests write data. In .gitignore, do not commit.
```

This scheme is not final. In particular:
- We might add other directories, e.g. for forecasts, later. 
- If we figure out a need for some data to be committed, we can remove the corresponding .gitignore entry.

Please install `pre-commit`, so the validation hooks in `hooks/` can check that all data in the validated directories is in the correct format.


## Labeling tool for questions
The streamlit app [`data_labeling/feedback_form.py`](data_labeling/feedback_form.py) is used to label questions.
Do not try to install its dependencies in the main Python environment.
Instead, make a new virtual environment and install the requirements with:
```
pip install -r data_labeling/requirements.txt
```
Alternatively, for now you can use [pipx](https://github.com/pypa/pipx), run `pipx install streamlit`, and continue to use the Python environment you have been using so far.


The feedback form app can be run with:
```
cd data_labeling
streamlit run feedback_form.py -- -f ../src/data/fq/synthetic/{filename}.jsonl
```
It writes into `src/data/feedback/`.


## Entry points to the code

- [`scripts/pipeline/scrape_questions.sh`](scripts/pipeline/scrape_questions.sh) runs pipeline to scrape metaculus for questions and stores them in `questions_cleaned_formatted.jsonl`.

- [`src/generate_topic_questions.py`](src/generate_topic_questions.py) Generates "raw" synthetic questions.

- [`src/format_and_verify_questions.py`](src/format_and_verify_questions.py) reads from a file with (potentially incomplete) questions, optionally fills the rest of the fields for a ForecastingQuestion, and verifies basic sanity checks using a LLM call. If you want it to fill in the resolution criteria, use the `--fill_in_body` flag. *Read and understand all flags before running this script*. Writes to `src/data/fq/{appropiate_dir}...`

- [`src/validate_fq_jsonl.py`](src/validate_fq_jsonl.py) Validates that a JSONL file contains only valid ForecastingQuestions. Does not write anything.


- [`src/instantiation.py`](src/instantiation.py) Runs instantiation. Takes a JSONL file (a list of ForecastingQuestions), and writes multiple JSONL files (each a list of QuestionTuples) into `src/data/tuples`.

- [`evaluation.py`](src/evaluation.py)
elicits forecasts and scores them. 
Takes the JSONL files in `src/data/tuples/{self.__class__.__name__}.jsonl` (for each Checker class we have), feeds them their respective Checker.elicit_and_violation methods.
  - Run: `python src/evaluation.py | tee src/data/evaluation.txt`


- [`src/forecaster_demo.py`](src/forecaster_demo.py) is a method to run the strong LLM forecasters on a file of ForecastingQuestions. Does not write anything. Not merged yet.

- [`src/playground.py`](src/playground.py) various testing and playing around.

This does not include the ones already mentioned in previous sections (feedback form, tests).
