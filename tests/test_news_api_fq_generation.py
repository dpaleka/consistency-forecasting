"""
Test individual outputs and overall pipline for NewsAPI -> FQs
"""

import sys
from common.path_utils import get_src_path

sys.path.append(str(get_src_path()))

import pytest
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from fq_from_news.date_utils import parse_date, last_datetime_of_month


load_dotenv()


# Date utils tests
def test_parse_date():
    assert parse_date("2024-01-01") == datetime(2024, 1, 1).date()
    assert parse_date("2024-02-09") == datetime(2024, 2, 9).date()


def test_last_datetime_of_month() -> datetime:
    assert last_datetime_of_month(datetime(2024, 1, 9).date()) == datetime(
        2024, 1, 31, 23, 59, 59
    )
    assert last_datetime_of_month(datetime(2024, 2, 11).date()) == datetime(
        2024, 2, 29, 23, 59, 59
    )
    assert last_datetime_of_month(datetime(2023, 2, 11).date()) == datetime(
        2023, 2, 28, 23, 59, 59
    )


# Pipeline specific
def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


def run_pipeline_command(command):
    returncode, stdout, stderr = run_command(command)

    print(f"Return Code: {returncode}")
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")

    assert returncode == 0, f"Command failed: {command}\nError: {stderr}"


commands = [
    "python -m spacy download en_core_web_sm",
    """
    python src/generate_fqs_from_news.py \
    --start-date 2024-09-01 --end-date 2024-09-04 \
    --rough-fq-gen-model-name gpt-4o-2024-05-13 \
    --final-fq-gen-model-name anthropic/claude-3.5-sonnet \
    --final-fq-verification-model-name gpt-4o-2024-05-13 \
    -lax
    """,
    """
    python src/generate_fqs_from_news.py \
    --start-date 2024-09-01 --end-date 2024-09-04 \
    --rough-fq-gen-model-name gpt-4o-2024-05-13 \
    --final-fq-gen-model-name anthropic/claude-3.5-sonnet \
    --final-fq-verification-model-name gpt-4o-2024-05-13 \
    --only-gen-final
    """,
    """
    python src/generate_fqs_from_news.py \
    --start-date 2024-09-01 --end-date 2024-09-04 \
    --rough-fq-gen-model-name gpt-4o-2024-05-13 \
    --final-fq-gen-model-name anthropic/claude-3.5-sonnet \
    --final-fq-verification-model-name gpt-4o-2024-05-13 \
    --only-verify-fq
    """,
]


# Expected files
def expected_files(test_exist: bool = False):
    files = [
        "src/data/fq/synthetic/news_api_generated_fqs/verified_final_fq_using_gpt-4o-2024-05-13_lax_res_checking_from_September-1-2024_to_September-4-2024_num_pages_1_num_articles_all.jsonl",
        "src/data/fq/synthetic/news_api_generated_fqs/verified_final_fq_using_gpt-4o-2024-05-13_strict_res_checking_from_September-1-2024_to_September-4-2024_num_pages_1_num_articles_all.jsonl",
        "src/data/news_feed_fq_generation/news_api/final_unverified_forecasting_questions/final_fq_using_anthropic__claude-3.5-sonnet_lax_res_checking_from_September-1-2024_to_September-4-2024_num_pages_1_num_articles_all.jsonl",
        "src/data/news_feed_fq_generation/news_api/final_unverified_forecasting_questions/final_fq_using_anthropic__claude-3.5-sonnet_strict_res_checking_from_September-1-2024_to_September-4-2024_num_pages_1_num_articles_all.jsonl",
        "src/data/news_feed_fq_generation/news_api/news_feed_data_dump/consolidated_news_api_from_2024-09-01_to_2024-09-04_num_pages_1.jsonl",
        "src/data/news_feed_fq_generation/news_api/news_feed_data_dump/processed_news_api_from_2024-09-01_to_2024-09-04_num_pages_1.jsonl",
        "src/data/news_feed_fq_generation/news_api/rough_forecasting_question_data/rough_fq_using_gpt-4o-2024-05-13_from_September-1-2024_to_September-4-2024_num_pages_1_num_articles_all.jsonl",
        "src/data/news_feed_fq_generation/news_api/rough_forecasting_question_data/validated_news_articles_using_gpt-4o-2024-05-13_from_September-1-2024_to_September-4-2024_num_pages_1_num_articles_all.jsonl",
    ]

    if test_exist:
        for file_path in files:
            assert Path(
                file_path
            ).exists(), f"Expected output file does not exist: {file_path}"

    return files


def test_pipeline_end_to_end():
    output_files = expected_files(test_exist=False)
    print("\033[1mDeleting the following files:\033[0m")
    for file_path in output_files:
        print(f"  {file_path}")

    # delete all produced files
    for file_path in output_files:
        if Path(file_path).exists():
            Path(file_path).unlink()

    for command in commands:
        print(f"\033[1mRunning command: {command}\033[0m")
        run_pipeline_command(command)

    expected_files(test_exist=True)


if __name__ == "__main__":
    asyncio.run(pytest.main([__file__]))
