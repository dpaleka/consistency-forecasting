from datetime import datetime
import os
import asyncio
from common.datatypes import ForecastingQuestion
from common.utils import write_jsonl, append_question, load_questions, load_jsonl
from common.path_utils import get_src_path
from question_generators.question_formatter import verify_question
from .final_fq_generator import NewsApiFinalForecastingQuestionGenerator
from .rough_fq_generator import NewsApiRoughForecastingQuestionGenerator


# *************************************************************************************************************************
#                                                   Common Utils
# *************************************************************************************************************************


def set_save_directories(rough_fq_save_directory: str, final_fq_save_directory: str):
    NewsApiRoughForecastingQuestionGenerator.set_save_directory(rough_fq_save_directory)
    NewsApiFinalForecastingQuestionGenerator.set_save_directory(final_fq_save_directory)


# *************************************************************************************************************************
#                                                   Rough FQ Generation
# *************************************************************************************************************************
def _rough_forecasting_data_save_path(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
    should_exist: bool = False,
) -> str:
    """
    Returns the path to save the rough fq data.
    Raises an error if the save path already exists.

    :returns: str
    """
    # check if the save path for the rough forecasting questions already exists
    rough_fq_save_path = NewsApiRoughForecastingQuestionGenerator.article_to_rough_forecasting_question_download_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
    )
    if not should_exist and os.path.exists(rough_fq_save_path):
        raise RuntimeError(
            f"The rough forecasting questions data has possibly already been generated at {rough_fq_save_path}! Delete it first"
        )
    if should_exist and not os.path.exists(rough_fq_save_path):
        raise RuntimeError(
            f"The rough forecasting questions data was NOT found at {rough_fq_save_path}!"
        )

    return rough_fq_save_path


def generate_rough_forecasting_data_sync(
    articles_download_path: str,
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
    pose_date: datetime,
) -> None:
    """
    Wrapper for calling functionality to generate rough forecasting questions data in a sync manner.

    :returns: None
    """
    rough_fq_save_path = _rough_forecasting_data_save_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
    )

    articles = load_jsonl(articles_download_path)

    num_articles_processed = 0
    rough_forecasting_question_data = []

    for article in articles:
        if num_articles_processed >= num_articles:
            break

        rough_forecasting_questions = NewsApiRoughForecastingQuestionGenerator.article_to_rough_forecasting_question_sync(
            article, rough_fq_gen_model_name, end_date, pose_date
        )

        num_articles_processed += 1

        # Append the data as dicts
        for rough_forecasting_question in rough_forecasting_questions:
            rough_forecasting_question_data.append(rough_forecasting_question)

    write_jsonl(rough_fq_save_path, rough_forecasting_question_data)

    print(f"Rough forecasting question data has been saved to {rough_fq_save_path}")


async def generate_rough_forecasting_data(
    articles_download_path: str,
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
    pose_date: datetime,
) -> None:
    """
    Wrapper for calling functionality to generate rough forecasting questions data in an async manner.

    :returns: None
    """
    rough_fq_save_path = _rough_forecasting_data_save_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
    )

    news_articles_validation_path = (
        NewsApiRoughForecastingQuestionGenerator.validated_news_articles_save_path(
            start_date,
            end_date,
            num_pages,
            num_articles,
            rough_fq_gen_model_name,
        )
    )

    initial_articles = load_jsonl(articles_download_path)

    tasks = []
    for article in initial_articles:
        if len(tasks) >= num_articles:
            break

        tasks.append(
            NewsApiRoughForecastingQuestionGenerator.validate_articles_for_fq_generation(
                article, rough_fq_gen_model_name, end_date, pose_date
            )
        )

    article_validation_results = await asyncio.gather(*tasks)

    write_jsonl(news_articles_validation_path, article_validation_results)
    print(
        f"News artciles validation results have been saved to {news_articles_validation_path}"
    )

    tasks = []
    for article_val_result in article_validation_results:
        if article_val_result["validation_result"]:
            tasks.append(
                NewsApiRoughForecastingQuestionGenerator.article_to_rough_forecasting_question(
                    article_val_result, rough_fq_gen_model_name, end_date, pose_date
                )
            )

    results = await asyncio.gather(*tasks)

    # Save the rough forecasting question data
    rough_forecasting_question_data = []
    for rough_forecasting_questions in results:
        for rough_forecasting_question in rough_forecasting_questions:
            rough_forecasting_question_data.append(rough_forecasting_question)

    write_jsonl(rough_fq_save_path, rough_forecasting_question_data)

    print(f"Rough forecasting question data has been saved to {rough_fq_save_path}")


def _check_if_rough_fq_was_rejected(rough_fq: dict):
    return "fqRejectionReason" in rough_fq


# *************************************************************************************************************************
#                                                    Final FQ Generation
# *************************************************************************************************************************
def _final_forecasting_questions_save_path(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    final_fq_gen_model_name: str,
    be_lax_in_resolution_checking: bool,
    should_exist: bool = False,
) -> str:
    """
    Returns the path to save the final fq data.
    Raises an error if the save path already exists.

    :returns: str
    """
    # check if the save path for the final forecasting questions already exists
    final_fq_save_path = (
        NewsApiFinalForecastingQuestionGenerator.rough_fq_to_final_fq_download_path(
            start_date,
            end_date,
            num_pages,
            num_articles,
            final_fq_gen_model_name,
            be_lax_in_resolution_checking,
        )
    )
    if not should_exist and os.path.exists(final_fq_save_path):
        raise RuntimeError(
            f"The Final forecasting questions are possibly already at at {final_fq_save_path}! Delete it first."
        )
    if should_exist and not os.path.exists(final_fq_save_path):
        raise RuntimeError(
            f"The final forecasting questions were NOT found at {final_fq_save_path}!"
        )
    return final_fq_save_path


def _save_forecasting_question_in_jsonl(
    forecasting_question: ForecastingQuestion, fq_save_path: str
) -> None:
    if forecasting_question is not None:
        append_question(forecasting_question, fq_save_path)


def generate_final_forecasting_questions_sync(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
    final_fq_gen_model_name: str,
    pose_date: datetime,
) -> None:
    """
    Wrapper for calling functionality to generate final forecasting questions in a sync manner.

    :returns: None
    """
    final_fq_save_path = _final_forecasting_questions_save_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        final_fq_gen_model_name,
    )

    rough_fq_save_path = _rough_forecasting_data_save_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
        should_exist=True,
    )

    rough_fqs = load_jsonl(rough_fq_save_path)
    for rough_fq in rough_fqs:
        if not _check_if_rough_fq_was_rejected(rough_fq):
            final_forecasting_question = (
                NewsApiFinalForecastingQuestionGenerator.rough_fq_to_final_fq_sync(
                    rough_fq, final_fq_gen_model_name, end_date, pose_date
                )
            )

            _save_forecasting_question_in_jsonl(
                final_forecasting_question, final_fq_save_path
            )

    print(f"Final forecasting questions have been saved to {final_fq_save_path}")


async def generate_final_forecasting_questions(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
    final_fq_gen_model_name: str,
    pose_date: datetime,
    creation_date: datetime,
    be_lax_in_resolution_checking: bool,
) -> None:
    """
    Wrapper for calling functionality to generate final forecasting questions in an async manner.

    :returns: None
    """
    final_fq_save_path = _final_forecasting_questions_save_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        final_fq_gen_model_name,
        be_lax_in_resolution_checking,
    )

    rough_fq_save_path = _rough_forecasting_data_save_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
        should_exist=True,
    )

    tasks = []
    rough_fqs = load_jsonl(rough_fq_save_path)
    for rough_fq in rough_fqs:
        if not _check_if_rough_fq_was_rejected(rough_fq):
            tasks.append(
                NewsApiFinalForecastingQuestionGenerator.rough_fq_to_final_fq(
                    rough_fq,
                    final_fq_gen_model_name,
                    end_date,
                    pose_date,
                    creation_date,
                    be_lax_in_resolution_checking,
                )
            )

    results = await asyncio.gather(*tasks)
    for final_forecasting_question in results:
        _save_forecasting_question_in_jsonl(
            final_forecasting_question, final_fq_save_path
        )

    print(f"Final forecasting questions have been saved to {final_fq_save_path}")


# *************************************************************************************************************************
#                                                   Verified Final FQ Generation
# *************************************************************************************************************************
def _final_verified_questions_save_dir(news_source, given_directory_path):
    if news_source == "NewsAPI":
        if given_directory_path is None or len(given_directory_path.strip()) == 0:
            dir_path = os.path.join(
                get_src_path(), "data/fq/synthetic/news_api_generated_fqs"
            )
        else:
            dir_path = given_directory_path
    else:
        raise ValueError("Not a valid news source")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def _final_verified_forecasting_questions_save_path(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    final_fq_verification_model_name: str,
    news_source: str,
    was_lax_in_resolution_checking: bool,
    verified_fq_save_directory: str,
) -> str:
    """
    Returns the path to save the final fq data.
    Raises an error if the save path already exists.

    :returns: str
    """
    num_pages_str = "all" if num_pages == -1 else str(num_pages)
    num_articles_str = (
        "all"
        if num_articles == -1 or num_articles == float("inf")
        else str(num_articles)
    )

    lax_str = (
        "lax_res_checking" if was_lax_in_resolution_checking else "strict_res_checking"
    )

    final_fq_verification_model_name_cleaned = final_fq_verification_model_name.replace(
        "/", "__"
    ).replace("\\", "__")

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    directory_structure = os.path.join(
        _final_verified_questions_save_dir(news_source, verified_fq_save_directory),
        final_fq_verification_model_name_cleaned,
        f"{start_date_str}_to_{end_date_str}",
        f"num_pages_{num_pages_str}",
        f"num_articles_{num_articles_str}",
    )

    os.makedirs(directory_structure, exist_ok=True)
    news_save_file_name = f"{lax_str}_fqs.jsonl"

    # TODO - refactor for non News API things
    final_verfied_fq_save_path = os.path.join(
        os.path.join(directory_structure, news_save_file_name)
    )
    if os.path.exists(final_verfied_fq_save_path):
        raise RuntimeError(
            f"The Final forecasting questions are possibly already at at {final_verfied_fq_save_path}! Delete it first."
        )
    return final_verfied_fq_save_path


async def _verify_final_fq(
    question: ForecastingQuestion, model_name: str
) -> ForecastingQuestion:
    result = await verify_question(question=question, model=model_name)

    if result.valid:
        return question
    return None


async def verify_final_forecasting_questions(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    final_fq_gen_model_name: str,
    final_fq_verification_model_name: str,
    news_source: str,
    was_lax_in_resolution_checking: bool,
    verified_fq_save_directory: str,
) -> None:
    """
    Verifies the generated final forecasting questions and saved them to the ./data/fq/synthetic directory.

    :returns: None
    """
    final_fq_save_path = _final_forecasting_questions_save_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        final_fq_gen_model_name,
        was_lax_in_resolution_checking,
        should_exist=True,
    )

    verified_final_fq_save_path = _final_verified_forecasting_questions_save_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        final_fq_verification_model_name,
        news_source,
        was_lax_in_resolution_checking,
        verified_fq_save_directory,
    )

    final_unverified_fqs = load_questions(final_fq_save_path)

    tasks = []
    for final_unverified_fq in final_unverified_fqs:
        tasks.append(
            _verify_final_fq(final_unverified_fq, final_fq_verification_model_name)
        )

    results = await asyncio.gather(*tasks)
    for verified_final_forecasting_question in results:
        _save_forecasting_question_in_jsonl(
            verified_final_forecasting_question, verified_final_fq_save_path
        )

    print(
        f"Final verified forecasting questions have been saved to {verified_final_fq_save_path}"
    )
