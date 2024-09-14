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
    You are an AI assistant tasked with generating unbiased, high-quality forecasting questions based on a provided example. Create new questions that align with the thematic and structural elements of the example question, i.e., span the reference class of the example question.

    ## Key Guidelines:

    1. **Resolution Criteria**: Questions must have a definitive outcome from the current date until the resolution date. Assume the current date is {pose_date}. Ensure questions are answerable based on events known as of this date.

    2. **Avoid Time-Specific Indicators**: Do not reference the date of question formation or the example question in your output.

    ## Reference Class:
    A reference class is a group of entities or events that share common traits, used as a benchmark to inform and guide the analysis of new situations or questions. This approach enhances consistency and accuracy in decision-making by providing a relevant context for comparison.

	## Forecasting Question Structure:
	- **Title**: A clear, concise question with a YES or NO answer.
	- **Body**: Provide context directly supporting the question, avoiding ambiguity and excess information.
	
		### Title Guidelines:
		- **Definitive**: Must elicit a YES or NO answer.
		- **Neutral**: Avoid sensitive topics like politics, religion, or race.
		- **Clarity**: Ensure straightforward and precise wording.
		
		### Body Guidelines:
		- **Disambiguation**: Provide information that directly supports the title.
		- **Relevance**: Focus on essential information.
		
		### Resolution Guidelines:
		- **Binary Outcome**: Must resolve to YES or NO.
		- **Stable Outcome**: Remains unresolved from the current date to the resolution date.

		### General Rules:
		- **Avoid Specific Knowledge**: Do not rely on events post-current-date.
		- **Named Events**: Avoid referring to events not known as of the current date.
		- **Numerical Values**: Use clear thresholds, avoiding complex calculations.
    """

    # example_high_quality_fq_1 = {
    #     "title": "Will a major cryptocurrency be named after a cricket term by July 2025?",
    #     "body": 'This question will resolve as Yes if, by 31 July, 2025, a cryptocurrency that is ranked within the top 100 by market \
    #         capitalization according to a recognized cryptocurrency market analysis platform (e.g., CoinMarketCap, CoinGecko) is named after a \
    #         cricket term. The term must be widely recognized within the cricket community and must directly relate to the sport (e.g., "Wicket", \
    #         "Bowler", "Century"). The naming of the cryptocurrency must be intentional, with clear references to its cricket-related origin in its \
    #         official documentation or announcements by its creators. In the event of multiple cryptocurrencies meeting these criteria, the question will \
    #         resolve as Yes if at least one of them is within the top 100 by market capitalization. This question resolves as NO if no such cryptocurrency \
    #         exists by the specified date.',
    # }

    # example_high_quality_fq_2 = {
    #     "title": "Will a Formula 1 Grand Prix be hosted in a country currently under international sanctions by December 2025?",
    #     "body": 'This question will resolve as Yes if, by December 31, 2025, a Formula 1 Grand Prix is officially announced and \
    #         scheduled to take place in a country that, at the time of the announcement, is under international sanctions by the United Nations, \
    #         the European Union, the United States, or any other major international body recognized for imposing sanctions.\n\nFor the purpose of \
    #         this question, "international sanctions" refer to financial, trade, or other sanctions imposed by international bodies or coalitions\
    #         of countries against a nation for political, economic, or human rights reasons. The sanctions must be widely reported and recognized by \
    #         reputable news sources (BBC, The Guardian, New York Times, Washington Post).\n\nIn the event of a Grand Prix being announced in a \
    #         country that later has sanctions lifted before the race occurs, the question will still resolve as Yes if the sanctions were in place \
    #         at the time of the announcement. Temporary or partial lifting of sanctions for the event does not affect the resolution.\n\nThis question does \
    #         not consider unofficial or speculative announcements. Confirmation must come from the Formula 1 organization or the sanctioned country\'s government.',
    # }

    # example_high_quality_fq_3 = {
    #     "title": "Will South Korea become the leader in global digital governance by December 2030?",
    #     "body": "This question will resolve as Yes if, by December 31, 2030, South Korea is recognized as the global leader in digital governance. \
    #         Recognition must come from at least two of the following authoritative sources: the United Nations, the World Bank, the Digital Nations \
    #         (formerly known as the D5), or a consensus among at least three major technology-focused publications (e.g., Wired, TechCrunch, The Verge). \
    #         Criteria for leadership in digital governance include but are not limited to: - Implementation of advanced digital services across government \
    #         sectors. - Adoption of cutting-edge technologies in public administration. - Demonstrable impact of digital governance on improving public \
    #         services and citizen engagement. - Leadership in international digital policy discussions and agreements. In the event of a tie or close \
    #         competition with another nation, the question will resolve as Yes only if South Korea is clearly distinguished as the leader by the majority \
    #         of the aforementioned sources. Edge cases, such as temporary leadership positions or recognition in a single aspect of digital governance, \
    #         do not meet the resolution criteria.",
    # }

    prompt = {
        "single": """
        ### Steps for Generating New Questions:

        1. **Analyze the Original Question**: Review the provided example forecasting question in detail.
        {source_forecasting_question}

        2. **Identify Key Components**: Decompose the original question into the following essential components:
        - **Subject Matter** (e.g., economic trends, technological advancements)
        - **Geographical Scope** (e.g., specific countries, regions)
        - **Event Type** (e.g., elections, policy decisions, innovations)
        - **Domain** (e.g., finance, technology, governance)
        - **Entity Importance** (e.g., prominent organizations, influential individuals)
        - **Time Sensitivity** (e.g., time-bound predictions, rapid developments)

        3. **Generate New Questions**: Create questions that yield a definitive YES or NO answer, ensuring that only **one class** (such as geographical scope, subject matter, or event type) is altered per variation while maintaining consistency with the original reference class. Each question should have the same resolution date as the original.

        ### Example Variations by Changing Only One Class:

        - **Original Title**: "Will South Korea become the leader in global digital governance by December 2030?"

        #### Variation by Geographical Scope:
        - **New Title**: "Will the European Union become the leader in global digital governance by December 2030?" (Only location has changed)

        #### Variation by Achievement:
        - **New Title**: "Will South Korea surpass all other countries in the number of AI research publications by December 2030?" (Only the achievement has changed)

        #### Variation by Location:
        - **New Title**: "Will Japan become the leader in global digital governance by December 2030?" (Only the location has changed)

        Your task is to generate at least {num_questions} unique forecasting questions, ensuring that each new question reflects variation along only **one dimension** from the original. This ensures consistency with the reference class.
        """,
        "multiple": """
        ### Steps for Generating New Questions:

        1. **Analyze the Original Question**: Carefully review the provided example question.
        {source_forecasting_question}

        2. **Define the Reference Class**: Break down the key components of the original question, focusing on:
        - **Subject Matter**: (e.g., economic trends, geopolitical events)
        - **Geographical Scope**: (e.g., specific countries, regions)
        - **Event Type**: (e.g., elections, technological breakthroughs)
        - **Domain**: (e.g., finance, technology, governance)
        - **Entity Importance**: (e.g., global organizations, industry leaders)
        - **Time Sensitivity**: (e.g., rapid or large-scale developments)

        3. **Generate New Questions**: Create questions that yield a definitive YES or NO answer, while ensuring diversity across multiple elements like **location**, **subject matter**, and **entities** to avoid any potential bias. When varying elements, ensure that **multiple classes** are changed simultaneously (e.g., adjusting both geographical scope and subject matter) to maintain neutrality. All questions should keep the same resolution date as the original example.

        ### Example variations over a title:
        - **Original Title**: "Will South Korea become the leader in global digital governance by December 2030?"
        - **New Title**: "Will the European Union implement a unified digital currency by December 2030?"
        - **New Title**: "Will China surpass the U.S. in total AI research publications by December 2030?"
        - **New Title**: "Will Japan become the top exporter of autonomous vehicle technology by December 2030?"

        Your task is to generate at least {num_questions} unique new forecasting questions that remain within the reference class but reflect variation across dimensions to ensure unbiased forecasting.
        """,
    }

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
        spanning_type: str,
    ) -> list[ForecastingQuestion]:
        """
        Class method to create the final ForecastingQuestion from rough forecasting question data asynchronously.

        Args:
            source_fq (ForecastingQuestion): The origincal FQ
            model_name (str): The model being used to create the rough forecasting question.
            num_questions (int): Minimum number of questions to generate for the given FQ
            pose_date (datetime): The question creattion date
            spanning_type (str): whether "multiple" or "single"

        Returns:
            list[ForecastingQuestion]: List of generated FQs
        """

        if source_fq.created_date is not None:
            pose_date = source_fq.created_date

        (
            forecasting_preface,
            forecasting_prompt,
        ) = (
            cls.preface.format(
                pose_date=pose_date.strftime("%B %d, %Y"),
            ),
            cls.prompt[spanning_type].format(
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
        default="gpt-4o-2024-05-13",
    )

    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        help="Minimum number of question to be generated per given FQ",
        default=3,
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

    parser.add_argument(
        "-m",
        "--span-multiple",
        action="store_true",
        help="""
        Set to span multiple entities in the spanned FQs
        """,
        default=False,
    )

    args = parser.parse_args()

    return args


def _get_final_save_path(source_fqs_path, generated_fqs_save_path, span_type):
    if generated_fqs_save_path is None or len(generated_fqs_save_path.strip()) == 0:
        directory, filename = os.path.split(source_fqs_path)
        name, _ = os.path.splitext(filename)
        new_filename = f"{name}-ref-class-spanned-{span_type}.jsonl"
        new_path = os.path.join(directory, new_filename)
        return new_path

    return generated_fqs_save_path


async def main(args: argparse.Namespace) -> None:
    """
    Pipeline for generating forecasting questions using the reference class spanner.

    :args: Arguments supplied to the main function

    :returns: None
    """
    span_type = "single"
    if args.span_multiple:
        span_type = "multiple"

    generated_fqs_save_path = _get_final_save_path(
        args.source_fqs_path, args.gen_fqs_save_path, span_type
    )
    if os.path.exists(generated_fqs_save_path):
        raise RuntimeError(f"Save path {generated_fqs_save_path} already exists!")

    tasks = []
    for source_fq in load_questions(args.source_fqs_path):
        tasks.append(
            BinaryFQReferenceClassSpanner.generate_spanned_fqs(
                source_fq,
                args.model_name,
                args.num_questions,
                args.pose_date,
                span_type,
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
