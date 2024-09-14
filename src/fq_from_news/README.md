# FQ Generation from News

A module that provides functionality for generating Forecasting Questions using downloaded news.

## Usage

The entry point into the utilities described here is the [`src/generate_fqs_from_news.py`](../generate_fqs_from_news.py) script.

The following commands have been written assuming that you run it from within the [`src`](..) directory.

### Most Frequent Uses

#### Generating FQs for a given month

If you wish the resolution to not be checked _strictly_ (whether the resolution follows unambiguously from the source news article) for the month of July (7):

```python
python generate_fqs_from_news.py \
--news-month 7 \
--rough-fq-gen-model-name gpt-4o-2024-05-13 \
--final-fq-gen-model-name anthropic/claude-3.5-sonnet \
--final-fq-verification-model-name gpt-4o-2024-05-13
```

If you wish to be slightly leninent in the resolution checking (generated more number of final forecasting questions), add the `-lax` argument.

If you had already run the entire command once before without the `-lax` argument, the rough FQs would already have been generated. Following this, you may generate the FQs using

1. Generate the unverified final questions:

    ```python
    python generate_fqs_from_news.py \
    --news-month 7 \
    --rough-fq-gen-model-name gpt-4o-2024-05-13 \
    --final-fq-gen-model-name anthropic/claude-3.5-sonnet \
    --final-fq-verification-model-name gpt-4o-2024-05-13 \
    --only-gen-final
    ```

2. Generate the verfied final questions:

    ```python
    python generate_fqs_from_news.py \
    --news-month 7 \
    --rough-fq-gen-model-name gpt-4o-2024-05-13 \
    --final-fq-gen-model-name anthropic/claude-3.5-sonnet \
    --final-fq-verification-model-name gpt-4o-2024-05-13 \
    --only-verify-fq
    ```

#### Generating FQs for a given date range

Replace the `--news-month` option with `--start-date YYYY-MM-DD --end-date YYYY-MM-DD`

#### Arguments for generating only steps of the entire pipeline

1. `--only-download-news`: Only downloads and processes the news
2. `--only-gen-rough`: Only generates the rough, intermediate forecasting question data.
3. `--only-gen-final`: Only generates the final, resolution checked (in the context of the source article) forecasting questions.
4. `--only-verify-fq`: Only verifies the final forecasting questions (generated in the previous step) using the common FQ verification module.

The actual pipeline runs the above steps in order.

### Help

Outputs all the arguments available.

```python
python generate_fqs_from_news.py --help
```