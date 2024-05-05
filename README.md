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
[Meeting and Agenda doc](https://docs.google.com/document/d/1_amt7CQK_aadKciMJuNmedEyf07ubIAL_b5ru_mS0nw/edit)
[Datatypes and Pipeline doc](https://docs.google.com/document/d/19CDHfwKHfouttiXPc7UNp8iBeYE4KD3H1Hw8_kqnnL4/edit). 

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
NO_CACHE=True python -m pytest
```
This will run all tests located in the `tests/` directory. Please fix any failing tests before submitting your PR.
As `pytest` also runs all files named `test_*.py` or `*_test.py`, please do not name anything in `src/` like this if you don't think it should run on every PR.

### Paths
Use [`src/common/path_utils.py`](/src/common/path_utils.py) to specify paths in code, Jupyter notebooks, etc.
Do not hardcode paths except relative to `pathlib.Path` objects returned by the utils in `path_utils.py`.

### Validation of data
Our base data directory is `src/data/`. Inside this, we have the following scheme:
```
src/data
├── fq
│  ├── real             # ForecastingQuestions made from real scraped data. Formatting validated upon commit.
│  └── synthetic        # ForecastingQuestions made from synthetic data. Formatting validated upon commit.
├── feedback            # Feedback data on real and synhetic questions. TODO Validate upon commit.
├── tuples              # Tuples of (question, answer) pairs. TODO Validate upon commit.
├── other               # All other data, e.g. raw scrapes, or intermediate steps for synthetic questions. Unvalidated.
├── check_tuple_logs    # Where results of the already instantiated consistency checks + violation are saved. In .gitignore, do not commit. 
└── test                # Where tests write data. In .gitignore, do not commit.
```

This scheme is not final. In particular:
- We might add other directories, e.g. for forecasts, later. 
- If we figure out a need for some data to be committed, we can remove the corresponding .gitignore entry.

Please install `pre-commit`, so the validation hooks in `hooks/` can check that all data in the validated directories is in the correct format.

