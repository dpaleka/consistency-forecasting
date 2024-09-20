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
    VerificationResult,
)
from common.utils import load_questions, write_questions
from fq_from_news.date_utils import parse_date
from question_generators.question_formatter import verify_question


class BinaryFQReferenceClassSpanner:
    ref_spaning_preface = """
**Objective:** Generate high-quality forecasting questions (FQs) by spanning the reference class of a given source question. Your goal is to enhance the diversity of the dataset while minimizing bias. 

### Understanding the Task:

In probability theory, a **reference class** refers to a group of similar events or outcomes that share common features. These features form the basis for estimating probabilities. For example, all state-level referendums on policy issues could be part of the same reference class when forecasting whether a state will hold a specific referendum.

**Your task:** Create new forecasting questions by varying key components (e.g., location, topic, action, or subject) of the source question, ensuring the new questions stay within the same reference class and do not already have a known resolution.

### Key Requirements:

1. **Consistency in Structure:**  
- The newly generated questions must maintain the same structure and thematic integrity as the original source question. 

2. **Reference Class Spanning:**  
- Vary only one to two key elements (e.g., location, subject, or entity) to create a new question. The changes should be meaningful yet retain the logical consistency of the original question.

3. **Resolution Status:**  
- Ensure that the newly generated questions, to the best of your knowledge, remain unresolved as of now.

4. **Maintain Resolution Date:**
- Use the same resolution date as the one given in the source question.

---

### Forecasting Question Structure:

#### Title Guidelines:
1. **YES/NO Clarity:**  
- Each question must be answerable with a definitive YES or NO.  
- *Good Example:* "Will Tesla launch its electric bike by March 2025?"  
- *Bad Example:* "Will Tesla’s electric bike launch be a success?"

2. **Avoid Sensitive Topics:**  
- Do not reference religion, politics, gender, or race.

3. **Direct and Precise:**  
- Titles should be straightforward and unambiguous, avoiding vague or unclear terms.  
- *Acceptable:* "Will Apple release a new iPhone model by September 2024?"  
- *Not Acceptable:* "Will Apple make a significant product announcement next year?"

4. **Context for Clarity:**  
- Provide sufficient context to ensure the question is clear and unambiguous.  
- *Acceptable:* "Will Microsoft announce a new AI framework at its annual Build event by May 2024?"  
- *Not Acceptable:* "Will a company announce new software at a future conference?"

#### Resolution Date:
- Retain the same resolution date as the source forecasting question.

#### Body Guidelines:
1. **Disambiguation:**  
- Focus on details that clarify the core question. Avoid unrelated or extraneous information.

2. **Conciseness:**  
- Keep the body concise and relevant. Do not include unnecessary details that could complicate or confuse the question.

#### Resolution Guidelines:
1. **Binary Outcome:**  
- Resolutions must be clearly marked as True for YES and False for NO.

2. **Stable Outcome:**  
- Ensure the resolution remains consistent and unchangeable until the resolution date.

---

### Examples of High-Quality Forecasting Questions:

    **Example 1:**  
    - **Title:** "Will a major cryptocurrency be named after a cricket term by July 2025?"  
    - **Body:**  
    "This question will resolve as Yes if, by 31 July 2025, a cryptocurrency ranked within the top 100 by market capitalization according to a recognized cryptocurrency market analysis platform (e.g., CoinMarketCap, CoinGecko) is named after a cricket term. The term must be widely recognized within the cricket community and must directly relate to the sport (e.g., 'Wicket', 'Bowler', 'Century'). The naming must be intentional, with clear references to its cricket-related origin in official documentation or announcements by its creators. If multiple cryptocurrencies meet these criteria, the question will resolve as Yes if at least one is within the top 100 by market capitalization. This question resolves as NO if no such cryptocurrency exists by the specified date."

    **Example 2:**  
    - **Title:** "Will a Formula 1 Grand Prix be hosted in a country currently under international sanctions by December 2025?"  
    - **Body:**  
    "This question will resolve as Yes if, by December 31, 2025, a Formula 1 Grand Prix is officially announced and scheduled to take place in a country that, at the time of the announcement, is under international sanctions by the United Nations, the European Union, the United States, or any other major international body recognized for imposing sanctions. 'International sanctions' refer to financial, trade, or other sanctions imposed by international bodies or coalitions of countries against a nation for political, economic, or human rights reasons. The sanctions must be widely reported and recognized by reputable news sources (e.g., BBC, The Guardian, New York Times). If a Grand Prix is announced in a country that later has sanctions lifted before the race occurs, the question will still resolve as Yes if the sanctions were in place at the time of the announcement. Temporary or partial lifting of sanctions for the event does not affect the resolution. This question does not consider unofficial or speculative announcements. Confirmation must come from the Formula 1 organization or the sanctioned country's government."

---

**Next Steps:**  
Using the guidelines and examples provided, span the reference class of the source forecasting question to generate new questions. Ensure that each generated question is logically consistent, contextually relevant, and unbiased. The steps to span the reference class have been outlined below.
    """

    ref_spanning_prompt = {
        "basic": """
The source forecasting question is:
    {source_forecasting_question}

**Instructions:**

1. **Reference Class Identification:**  
- Determine the core components of the source question, such as the event type, location, key subjects, or outcomes. These elements define the question's context and reference class.

2. **Component Substitution:**  
- Replace one to two significant elements of the source question (e.g., location, organization, or key subject) with a similar entity. Ensure that the substitution maintains the question's logical structure and meaningfulness. Do NOT change the resolution date of the source forecasting question.

3. **Balance and Neutrality:**  
- The new questions should present a range of possible outcomes, avoiding any bias towards likely YES resolutions. Aim for a diverse probability distribution of potential results.

4. **Realism and Plausibility:**  
- Ensure that the new questions remain realistic and relevant to the context. Avoid creating absurd or nonsensical scenarios. The event must be forecastable, with a clear and plausible outcome. 

5. **Quality Control:**  
- Verify that the generated questions maintain the integrity of the source question. Ensure consistency in resolution dates and avoid introducing illogical elements.

**Objective:**  
Create **{num_questions}** forecasting questions by spanning the reference class of the provided source question. New forecasting questions must maintain logical consistency and relevance.

**Examples:**
    **Source Question 1:**  
    - **Title:** Will NASA astronauts switch from Boeing to SpaceX for their return to Earth by August 2024?  
    - **Acceptable Variations:**  
    1. Will ESA astronauts switch from SpaceX to Boeing for their return to Earth by August 2024?  
    2. Will NASA announce a crewed mission to Mars using SpaceX technology by August 2025?  
    3. Will NASA astronauts return to Earth from the International Space Station using a non-US vehicle by August 2025?  
    4. Will Russian astronauts use a SpaceX vehicle to return from the International Space Station by August 2024?  
    5. Will astronauts return to Earth using a reusable space capsule developed by India by August 2024?

    - **Unacceptable Variations:**  
    1. Will Chinese astronauts return to Earth from their space station using a SpaceX vehicle by September 2025?  
        - *Issue:* The resolution date differs from the source question.  
    2. Will NASA astronauts switch from SpaceX to Boeing for their return to Earth by August 2024?  
        - *Issue:* The question is vacuously true and lacks meaningful context since the original question is about a shift from Boeing to SpaceX.

    **Source Question 2:**  
    - **Title:** Will Tesla complete a major software upgrade for over 1.5 million vehicles in China by July 2024?  
    - **Acceptable Variations:**  
    1. Will Tesla complete a major software upgrade for over 1 million vehicles in the United States by July 2024?  
    2. Will Toyota complete a major software upgrade for over 1 million vehicles in the US by July 2024?  
    3. Will Tesla complete a major battery software upgrade for over 2 million vehicles in China by July 2024?  
    4. Will Toyota complete a major software upgrade for over 1 million vehicles in Japan by July 2024?  
    5. Will Tesla complete a major safety-related software upgrade for over 500,000 vehicles in Australia by July 2024?

    - **Unacceptable Variations:**  
    1. Will Tesla complete a major safety-related software upgrade for over 500,000 vehicles in Australia by the next few months?  
        - *Issue:* The resolution date "by the next few months" is inconsistent with the original date of "July 2024."
        """,
    }

    resolution_date_checking_preface = """
You are an expert in forecasting question analysis. Your task is to determine if two given forecasting questions resolve on the same date. Provide a clear Yes or No answer. Consider all the details in each question, focusing on the resolution date specifically mentioned in their descriptions.
    """

    resolution_date_checking_prompt = """
Here are two forecasting questions:
First Forecasting Question:
    {source_fq}
Second Forecasting Question:
    {spanned_fq}

Do these two forecasting questions resolve on the same date? Provide a clear Yes or No answer.
"""

    @classmethod
    def _create_fq_from_stripped_fq(
        cls,
        new_stripped_fq: ForecastingQuestion_stripped,
        source_fq: ForecastingQuestion,
        created_date: datetime,
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
            created_date=created_date,
            metadata={
                "generated_from_ref_class_spanner": True,
                "original_question": source_fq.to_dict(),
            },
        )

    @classmethod
    async def _verifiy_resolution_date(
        cls,
        source_fq: ForecastingQuestion_stripped,
        spanned_fq: ForecastingQuestion_stripped,
        model_name: str,
    ) -> VerificationResult:
        (
            forecasting_preface,
            forecasting_prompt,
        ) = (
            cls.resolution_date_checking_preface,
            cls.resolution_date_checking_prompt.format(
                source_fq=source_fq.cast_stripped(),
                spanned_fq=spanned_fq.cast_stripped(),
            ),
        )

        result = await answer(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=VerificationResult,
        )

        return result

    @classmethod
    async def _verify_spanned_question(
        cls, source_question, spanned_question: ForecastingQuestion, model_name: str
    ) -> ForecastingQuestion:
        resolution_date_verification_result = await cls._verifiy_resolution_date(
            source_question, spanned_question, model_name
        )

        if not resolution_date_verification_result.valid:
            return None

        generic_verification_result = await verify_question(
            question=spanned_question, model=model_name
        )

        if not generic_verification_result.valid:
            return None

        return spanned_question

    @classmethod
    async def generate_spanned_fqs(
        cls,
        source_fq: ForecastingQuestion,
        model_name: str,
        num_questions: int,
        created_date: datetime,
        spanning_type: str,
    ) -> list[ForecastingQuestion]:
        """
        Class method to create the final ForecastingQuestion from rough forecasting question data asynchronously.

        Args:
            source_fq (ForecastingQuestion): The original FQ
            model_name (str): The model being used to create the rough forecasting question.
            num_questions (int): Minimum number of questions to generate for the given FQ
            created_date (datetime): The question creattion date
            spanning_type (str): whether "multiple" or "single"

        Returns:
            list[ForecastingQuestion]: List of generated FQs
        """

        if source_fq.created_date is not None:
            created_date = source_fq.created_date

        (
            forecasting_preface,
            forecasting_prompt,
        ) = (
            cls.ref_spaning_preface,
            cls.ref_spanning_prompt[spanning_type].format(
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

        formed_forecasting_questions = [
            cls._create_fq_from_stripped_fq(
                generated_stripped_fq, source_fq, created_date
            )
            for generated_stripped_fq in generated_stripped_forecasting_question_list
        ]

        tasks = []
        for spanned_fq in formed_forecasting_questions:
            tasks.append(
                BinaryFQReferenceClassSpanner._verify_spanned_question(
                    source_fq, spanned_fq, model_name
                )
            )

        verified_spanned_fqs = await asyncio.gather(*tasks)

        results = [fq for fq in verified_spanned_fqs if fq is not None]

        return results


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
        default="openai/gpt-4o-2024-08-06",
    )

    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        help="Minimum number of question to be generated per given FQ",
        default=10,
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
        "--created-date",
        type=parse_date,
        help="""
        The question creation date. NOTE - not used if the source FQ has a not None `created_date`.
        """,
        default=datetime(2023, 10, 1),
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
    # NOTE - adding this to account for different forms of spanning (was earlier set to either single for multiple)
    span_type = "basic"

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
                args.created_date,
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
