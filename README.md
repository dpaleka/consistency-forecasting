# Consistency & Forecasting

## Installation requirements
First ensure your Python environment has Python 3.11 installed.

```python
import sys
assert sys.version_info >= (3, 11), "Python 3.11 or later is required."
```

Then do:
```
pip install -r requirements.txt
pre-commit install
```

## Docs
[doc](https://docs.google.com/document/d/1_amt7CQK_aadKciMJuNmedEyf07ubIAL_b5ru_mS0nw/edit)

## Apr 16 writeup
[writeup doc](https://docs.google.com/document/d/1849L5P9JNZEjBp4s4TsivJOG2iS98Ru6conx9jE0wPE/edit)

## Utils
We have some [LLM call utils](/src/common/README.md). (Please read that file!)
Feel free to add more utils in `utils.py`, `llm_utils.py`, or other files, as you need them.

## Running code
The preferred way to test anything is either from `playground.py`, or creating a new file / Jupyter notebook in the `src` directory.
Do not run files with actual logic (e.g. anything in `static_checks/` ) directly; this runs into Python import / path issues.
**Please do not name anything in `src/` with `test_something.py` if it's not in `tests/`.**

## Data Formats
Currently being discussed in the [Datatypes design doc](https://docs.google.com/document/d/19CDHfwKHfouttiXPc7UNp8iBeYE4KD3H1Hw8_kqnnL4/edit). 

## Testing Before Submitting PRs
Before submitting a pull request that deals with the core code in `src/`, please ensure that you run the test suite to check that your changes do not break any existing functionality. 
You can run the tests with the following command from the root directory of the project:
```
python -m pytest
```
This will run all tests located in the `tests/` directory. Please fix any failing tests before submitting your PR.

## Validation of data
Our base data directory is `src/data/`. Inside this, we have the following scheme:
```
src/data
├── fq
│  ├── real             # ForecastingQuestions made from real scraped data. Formatting validated upon commit.
│  └── synthetic        # ForecastingQuestions made from synthetic data. Formatting validated upon commit.
├── feedback            # Feedback data on real and synhetic questions. TODO Validate upon commit.
├── tuples              # Tuples of (question, answer) pairs. TODO Validate upon commit.
├── other               # All other data, e.g. raw scrapes, or intermediate steps for synthetic questions. Unvalidated.
├── test                # Directory where tests write data. In .gitignore, do not commit.
```

(We might add other directories for other data, e.g. for forecasts, later.)

Please install `pre-commit`, so the validation hooks in `hooks/` can check that all data in the validated directories is in the correct format.
