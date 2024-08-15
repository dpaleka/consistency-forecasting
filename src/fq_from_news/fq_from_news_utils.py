from datetime import datetime
import json
import os
import asyncio
from common.datatypes import ForecastingQuestion
from question_generators.question_formatter import verify_question
from .final_fq_generator import NewsApiFinalForecastingQuestionGenerator
from .rough_fq_generator import NewsApiRoughForecastingQuestionGenerator


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

    num_articles_processed = 0
    with open(articles_download_path, "r") as jsonl_file:
        for line in jsonl_file:
            if num_articles_processed >= num_articles:
                break

            article = json.loads(line.strip())
            rough_forecasting_questions = NewsApiRoughForecastingQuestionGenerator.article_to_rough_forecasting_question_sync(
                article, rough_fq_gen_model_name, end_date, pose_date
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

    tasks = []
    with open(articles_download_path, "r") as jsonl_file:
        for line in jsonl_file:
            if len(tasks) >= num_articles:
                break

            article = json.loads(line.strip())
            tasks.append(
                NewsApiRoughForecastingQuestionGenerator.article_to_rough_forecasting_question(
                    article, rough_fq_gen_model_name, end_date, pose_date
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


# *************************************************************************************************************************
#                                                    Final FQ Generation
# *************************************************************************************************************************
def _final_forecasting_questions_save_path(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    final_fq_gen_model_name: str,
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
        with open(fq_save_path, "a") as jsonl_file:
            jsonl_file.write(
                json.dumps(
                    {
                        "id": str(forecasting_question.id),
                        "title": forecasting_question.title,
                        "body": forecasting_question.body,
                        "resolution": forecasting_question.resolution,
                        "question_type": forecasting_question.question_type,
                        "data_source": forecasting_question.data_source,
                        "url": forecasting_question.url,
                        "resolution_date": forecasting_question.resolution_date.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "metadata": forecasting_question.metadata,
                    }
                )
                + "\n"
            )


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

    with open(rough_fq_save_path, "r") as jsonl_file:
        for line in jsonl_file:
            rough_fq = json.loads(line.strip())
            if "fqRejectionReason" not in rough_fq:
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
) -> None:
    """
    Wrapper for calling functionality to generate final forecasting questions in an async manner.

    :returns: None
    """
    final_fq_save_path = _final_forecasting_questions_save_path(
        start_date, end_date, num_pages, num_articles, final_fq_gen_model_name
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
    with open(rough_fq_save_path, "r") as jsonl_file:
        for line in jsonl_file:
            rough_fq = json.loads(line.strip())
            if "fqRejectionReason" not in rough_fq:
                tasks.append(
                    NewsApiFinalForecastingQuestionGenerator.rough_fq_to_final_fq(
                        rough_fq, final_fq_gen_model_name, end_date, pose_date
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
def _final_verified_questions_save_dir(news_source):
    if news_source == "NewsAPI":
        dir_path = "./data/fq/synthetic/news_api_generated_fqs"
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
) -> str:
    """
    Returns the path to save the final fq data.
    Raises an error if the save path already exists.

    :returns: str
    """
    if num_pages == -1:
        num_pages = "all"
    if num_articles == -1 or num_articles == float("inf"):
        num_articles = "all"

    news_save_file_name = f"verified_final_fq_using_{final_fq_verification_model_name}_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_{num_pages}_num_articles_{num_articles}.jsonl"

    # TODO - refactor for non News API things
    final_verfied_fq_save_path = os.path.join(
        _final_verified_questions_save_dir(news_source),
        news_save_file_name,
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
        should_exist=True,
    )

    verified_final_fq_save_path = _final_verified_forecasting_questions_save_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        final_fq_verification_model_name,
        news_source,
    )

    tasks = []
    with open(final_fq_save_path, "r") as jsonl_file:
        for line in jsonl_file:
            final_unverified_fq_dict = json.loads(line.strip())
            final_unverified_fq = ForecastingQuestion(**final_unverified_fq_dict)
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
