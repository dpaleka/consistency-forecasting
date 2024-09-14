import argparse
import os
import asyncio
from uuid import uuid4
from datetime import datetime

from common.llm_utils import answer
from common.datatypes import (
    ForecastingQuestion_stripped_list,
    ForecastingQuestion,
    ForecastingQuestion_stripped,
)
from common.utils import load_questions, write_questions
from fq_from_news.date_utils import parse_date


class BinaryFQReferenceClassSpanner:
    preface = """
    You are an AI assistant tasked with generating unbiased forecasting questions based on a given example. Your role is to create new questions that belong to the same reference class as the provided question, maintaining similar thematic and structural elements.

    ### Key Guidelines:

    1. **Resolution Criteria**: Questions must have a definitive outcome from the current date until the resolution date. Assume the current date is {pose_date}, and generate questions with adequate context so that they remain answerable based on known events as of this date.

    2. **Avoid Time-Specific Indicators**: Do not reference the date of question formation or mention the source forecasting question in your output.

    ### Definition of Reference Class:
    A reference class refers to a group of similar entities or events that share common characteristics, helping you frame new questions in a consistent thematic and structural manner.

    ### Forecasting Question Structure:

    - **Title**: A concise question with a clear YES or NO answer.
    - **Body**: Additional context or details supporting the question, avoiding ambiguity and unnecessary information.

    ### Title Guidelines:
    - **Definitive**: The title must elicit a YES or NO answer.
    - **Neutral**: Avoid sensitive or biased topics such as religion, politics, race, or gender.
    - **Clarity**: Titles must be straightforward and precise, avoiding ambiguity.
    - **Named Entities**: Include at least one named entity, but no more than four, for specificity.

    ### Body Guidelines:
    - **Disambiguation**: Include only information directly supporting the title, avoiding overly detailed or confusing content.
    - **Relevance**: Focus on information essential to resolving the question.

    ### Resolution Guidelines:
    - **Binary**: The question must resolve to either YES or NO.
    - **Stable Outcome**: The question must remain unresolved from the current date to the resolution date.

    ### General Rules:
    - **Avoid Specific Knowledge**: The question should not rely on specific post-current-date events.
    - **Named Events**: The question should avoid referring to events not known as of the current date.
    - **Numerical Values**: Use clear thresholds for numerical questions, avoiding complex calculations.

    ### Example Forecasting Questions:
    1. **Title**: "Will a significant political figure endorse a theory related to string theory by July 2024?"
    **Body**: This question resolves as YES if a notable figure publicly endorses string theory by July 31, 2024, based on reports by at least two reputable outlets (e.g., BBC, NY Times).

    2. **Title**: "Will a First Crystal Tier Market be created by August 2024?"
    **Body**: Resolves YES if a Crystal-tier market is created before August 2024, or if Manifold adjusts the tier system with similar costs.

    3. **Title**: "Will TIME's 100 Most Influential Companies list be released in May 2024?"
    **Body**: Resolves YES if the list is published by May 31, 2024. Otherwise, it resolves NO.
    """

    prompt = """
    ### Steps for Generating New Questions Using the Reference Class:

    1. **Analyze the Original Question**: Begin by examining the provided question.
        {source_forecasting_question}
    
    2. **Define the Reference Class**: Identify key elements of the original question’s context:
    - Subject matter (e.g., economic indicators, legal rulings).
    - Geographical scope (e.g., specific countries, regions).
    - Event type (e.g., elections, corporate mergers).
    - Domain (e.g., technology, agriculture).
    - Importance of entities (e.g., multinational corporations).
    - Time sensitivity and scale (e.g., rapid developments, large-scale events).

    3. **Generate New Questions**: Using the reference class, create questions with definitive YES or NO outcomes, while adjusting multiple classes (e.g., changing location or subject matter) to avoid bias. Maintain the same resolution date.

    Here is an example forecasting question:
    - **Title**: "Will Colorado hold a referendum on enshrining abortion rights by July 2024?"
    - **Body**: Resolves YES if a referendum on abortion rights is held and reported by at least two reputable sources by July 31, 2024.

    Example question titles by varying classes:
    - "Will California hold a referendum on regulating the gig economy by July 2024?"
    - "Will Nevada hold a referendum on stricter water laws by July 2024?"
    - "Will Texas hold a referendum on raising property tax exemptions by July 2024?"

    Your task is to generate at least {num_questions} new forecasting questions based on the same reference class.
    """

    @classmethod
    def _create_fq_from_stripped_fq(
        cls,
        new_stripped_fq: ForecastingQuestion_stripped,
        source_fq: ForecastingQuestion,
        pose_date: datetime,
    ):
        return ForecastingQuestion(
            id=uuid4(),
            title=new_stripped_fq.title,
            body=new_stripped_fq.body,
            resolution=None,
            question_type="binary",
            data_source="synthetic",
            url=None,
            resolution_date=source_fq.resolution_date,
            created_date=pose_date,
            metadata={
                "generated_from_ref_class_spanner": True,
                "original_question": source_fq.to_dict(),
            },
        )

    @classmethod
    async def generate_spanned_fqs(
        cls,
        source_fq: ForecastingQuestion,
        model_name: str,
        num_questions: int,
        pose_date: datetime,
    ) -> list[ForecastingQuestion]:
        """
        Class method to create the final ForecastingQuestion from rough forecasting question data asynchronously.

        Args:
            source_fq (ForecastingQuestion): The origincal FQ
            model_name (str): The model being used to create the rough forecasting question.
            num_questions (int): Minimum number of questions to generate for the given FQ
            pose_date (datetime): The question creattion date

        Returns:
            list[ForecastingQuestion]: List of generated FQs
        """

        if source_fq.created_date is not None:
            pose_date = source_fq.created_date

        (
            forecasting_preface,
            forecasting_prompt,
        ) = (
            cls.preface.format(pose_date=pose_date.strftime("%B %d, %Y")),
            cls.prompt.format(
                num_questions=num_questions,
                source_forecasting_question=source_fq.cast_stripped(),
            ),
        )

        generated_stripped_forecasting_question_list = (
            await answer(
                prompt=forecasting_prompt,
                preface=forecasting_preface,
                model=model_name,
                response_model=ForecastingQuestion_stripped_list,
            )
        ).questions

        return [
            cls._create_fq_from_stripped_fq(generated_stripped_fq, source_fq, pose_date)
            for generated_stripped_fq in generated_stripped_forecasting_question_list
        ]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--source-fqs-path",
        type=str,
        help="Path to the JSONL file of the forecasting questions being used to create new questions.",
        required=True,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        help="Model used to generate the new FQs",
        default="anthropic/claude-3.5-sonnet",
    )

    parser.add_argument(
        "--num-questions",
        type=int,
        help="Minimum number of question to be generated per given FQ",
        default=5,
    )

    parser.add_argument(
        "--gen-fqs-save-path",
        type=str,
        help="""
        Path where the generated Forecasting questions should be saved.

        Defaults to ${source-fqs-path without extension}-ref-class-spanned.jsonl
        """,
        default="",
    )

    parser.add_argument(
        "--pose-date",
        type=parse_date,
        help="""
        The question creation date. NOTE - not used if the source FQ has a not None `created_date`.
        """,
        default=datetime(2023, 10, 1),
    )

    args = parser.parse_args()

    return args


def _get_final_save_path(source_fqs_path, generated_fqs_save_path):
    if generated_fqs_save_path is None or len(generated_fqs_save_path.strip()) == 0:
        directory, filename = os.path.split(source_fqs_path)
        name, _ = os.path.splitext(filename)
        new_filename = f"{name}-ref-class-spanned.jsonl"
        new_path = os.path.join(directory, new_filename)
        return new_path

    return generated_fqs_save_path


async def main(args: argparse.Namespace) -> None:
    """
    Pipeline for generating forecasting questions using the reference class spanner.

    :args: Arguments supplied to the main function

    :returns: None
    """
    generated_fqs_save_path = _get_final_save_path(
        args.source_fqs_path, args.gen_fqs_save_path
    )
    if os.path.exists(generated_fqs_save_path):
        raise RuntimeError(f"Save path {generated_fqs_save_path} already exists!")

    tasks = []
    for source_fq in load_questions(args.source_fqs_path):
        tasks.append(
            BinaryFQReferenceClassSpanner.generate_spanned_fqs(
                source_fq, args.model_name, args.num_questions, args.pose_date
            )
        )

    results = await asyncio.gather(*tasks)

    # Save even the initial FQs
    final_fqs = load_questions(args.source_fqs_path)
    for cur_fqs in results:
        final_fqs.extend(cur_fqs)

    write_questions(list(final_fqs), generated_fqs_save_path)

    print(f"FQs have been saved to {generated_fqs_save_path}")


if __name__ == "__main__":
    args = get_args()

    asyncio.run(main(args))
