# Consistency & Forecasting

## Installation requirements
Python 3.11 is required for all operations within this project. Ensure you have it installed before proceeding.
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
Running the files with actual logic in them may run into Python import / path issues.

## Data Formats
Currently being discussed in the [Datatypes design doc](https://docs.google.com/document/d/19CDHfwKHfouttiXPc7UNp8iBeYE4KD3H1Hw8_kqnnL4/edit). 

## Testing Before Submitting PRs
Before submitting a Pull Request, please ensure that you run the test suite to check that your changes do not break any existing functionality. You can run the tests with the following command from the root directory of the project:
```
python3.11 -m pytest
```
This will run all tests located in the `tests/` directory. Please fix any failing tests before submitting your PR.
