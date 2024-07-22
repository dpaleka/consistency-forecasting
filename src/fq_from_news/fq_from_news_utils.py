from datetime import datetime
import json
import os
import asyncio
from .final_fq_generator import FinalForecastingQuestionGenerator
from .rough_fq_generator import RoughForecastingQuestionGenerator


def _rough_forecasting_data_save_path(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
) -> str:
    """
    Returns the path to save the rough fq data.
    Raises an error if the save path already exists.

    :returns: str
    """
    # check if the save path for the rough forecasting questions already exists
    rough_fq_save_path = RoughForecastingQuestionGenerator.article_to_rough_forecasting_question_download_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
    )
    if os.path.exists(rough_fq_save_path):
        raise RuntimeError(
            f"The rough forecasting questions data has possibly already been generated at {rough_fq_save_path}! Delete it first"
        )

    return rough_fq_save_path


def generate_rough_forecasting_data_sync(
    articles_download_path: str,
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
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

    num_articles_processed = 0
    with open(articles_download_path, "r") as jsonl_file:
        for line in jsonl_file:
            if num_articles_processed >= num_articles:
                break

            article = json.loads(line.strip())
            rough_forecasting_questions = RoughForecastingQuestionGenerator.article_to_rough_forecasting_question_sync(
                article, rough_fq_gen_model_name, start_date
            )

            num_articles_processed += 1

            # Save the rough forecasting question data
            with open(rough_fq_save_path, "a") as jsonl_file:
                for rough_forecasting_question in rough_forecasting_questions:
                    jsonl_file.write(json.dumps(rough_forecasting_question) + "\n")

    print(f"Rough forecasting question data has been saved to {rough_fq_save_path}")


async def generate_rough_forecasting_data(
    articles_download_path: str,
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
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

    tasks = []
    with open(articles_download_path, "r") as jsonl_file:
        for line in jsonl_file:
            if len(tasks) >= num_articles:
                break

            article = json.loads(line.strip())
            tasks.append(
                RoughForecastingQuestionGenerator.article_to_rough_forecasting_question(
                    article, rough_fq_gen_model_name, start_date
                )
            )

    # TODO can this result in an MLE?
    results = await asyncio.gather(*tasks)

    # Save the rough forecasting question data
    with open(rough_fq_save_path, "a") as jsonl_file:
        for rough_forecasting_questions in results:
            for rough_forecasting_question in rough_forecasting_questions:
                jsonl_file.write(json.dumps(rough_forecasting_question) + "\n")

    print(f"Rough forecasting question data has been saved to {rough_fq_save_path}")


def _final_forecasting_questions_save_path(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    final_fq_gen_model_name: str,
) -> str:
    """
    Returns the path to save the final fq data.
    Raises an error if the save path already exists.

    :returns: str
    """
    # check if the save path for the final forecasting questions already exists
    final_fq_save_path = (
        FinalForecastingQuestionGenerator.rough_fq_to_final_fq_download_path(
            start_date,
            end_date,
            num_pages,
            num_articles,
            final_fq_gen_model_name,
        )
    )
    if os.path.exists(final_fq_save_path):
        raise RuntimeError(
            f"The Final forecasting questions are possibly already at at {final_fq_save_path}! Delete it first."
        )
    return final_fq_save_path


def _rough_fq_path_for_final_fq_generation(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
) -> str:
    """
    Returns the path that the rough forecasting question data has been saved at
        and will be used for generating final forecasting questions.
    Raises an error if the save path does NOT exist.

    :returns: str
    """
    # Need the rough forecasting questions data to exist
    rough_fq_save_path = RoughForecastingQuestionGenerator.article_to_rough_forecasting_question_download_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
    )
    if not os.path.exists(rough_fq_save_path):
        raise RuntimeError(
            "The rough forecasting question data has not been generated yet! Generate it first."
        )
    return rough_fq_save_path


def _save_final_fq(final_forecasting_question, final_fq_save_path: str) -> None:
    if final_forecasting_question is not None:
        with open(final_fq_save_path, "a") as jsonl_file:
            jsonl_file.write(
                json.dumps(
                    {
                        "id": str(final_forecasting_question.id),
                        "title": final_forecasting_question.title,
                        "body": final_forecasting_question.body,
                        "resolution": final_forecasting_question.resolution,
                        "question_type": final_forecasting_question.question_type,
                        "data_source": final_forecasting_question.data_source,
                        "url": final_forecasting_question.url,
                        "resolution_date": final_forecasting_question.resolution_date.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "metadata": final_forecasting_question.metadata,
                    }
                )
                + "\n"
            )


def generate_final_forecasting_question_sync(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
    final_fq_gen_model_name: str,
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

    rough_fq_save_path = _rough_fq_path_for_final_fq_generation(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
    )

    with open(rough_fq_save_path, "r") as jsonl_file:
        for line in jsonl_file:
            rough_fq = json.loads(line.strip())
            if "fqRejectionReason" not in rough_fq:
                final_forecasting_question = (
                    FinalForecastingQuestionGenerator.rough_fq_to_final_fq_sync(
                        rough_fq, final_fq_gen_model_name, start_date
                    )
                )

                _save_final_fq(final_forecasting_question, final_fq_save_path)

    print(f"Final forecasting questions have been saved to {final_fq_save_path}")


async def generate_final_forecasting_question(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
    final_fq_gen_model_name: str,
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

    rough_fq_save_path = _rough_fq_path_for_final_fq_generation(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
    )

    tasks = []
    with open(rough_fq_save_path, "r") as jsonl_file:
        for line in jsonl_file:
            rough_fq = json.loads(line.strip())
            if "fqRejectionReason" not in rough_fq:
                tasks.append(
                    FinalForecastingQuestionGenerator.rough_fq_to_final_fq(
                        rough_fq, final_fq_gen_model_name, start_date
                    )
                )

    results = await asyncio.gather(*tasks)
    for final_forecasting_question in results:
        _save_final_fq(final_forecasting_question, final_fq_save_path)

    print(f"Final forecasting questions have been saved to {final_fq_save_path}")
