# Consistency & Forecasting

## Installation requirements
```
pip install -r requirements.txt
pip install -r src/common/requirements.txt
pre-commit install
```

## Docs
[doc](https://docs.google.com/document/d/1_amt7CQK_aadKciMJuNmedEyf07ubIAL_b5ru_mS0nw/edit)

## Utils
We have some [LLM call utils](/src/common/README.md). (Please read that file!)
Feel free to add more utils in `utils.py`, `llm_utils.py`, or other files, as you need them.

## Running code
The preferred way to test anything is either from `playground.py`, or creating a new file / Jupyter notebook in the `src` directory.
Running the files with actual logic in them may run into Python import / path issues.

## Data Formats
Currently being discussed in the [Datatypes design doc](https://docs.google.com/document/d/19CDHfwKHfouttiXPc7UNp8iBeYE4KD3H1Hw8_kqnnL4/edit). 

