from dotenv import load_dotenv
import os
from common.datatypes import ForecastingQuestion, VerificationResult
import asyncio
import uuid
from common.llm_utils import answer
from common.path_utils import get_data_path
from common.utils import write_jsonl_async_from_str
from typing import Optional
from datetime import datetime

load_dotenv()
write_verification = os.getenv("WRITE_VERIFICATION", "False") == "True"
print(
    f"The setting for WRITE_VERIFICATION in question_formatter.py: {write_verification}"
)


resolution_criteria_date_prompt = """\
I want you to help me craft a resolution criteria and an appropiate resolution date for a forecasting question,
as on the sites like Metaculus / PredictIt.
The resolution criteria should:
- Unambiguously define what outcomes will be considered for the outcome of the question.
- Define any ambigious terms in the question.
- When possible, use authoritative sources.
- Consider edge cases and fall-back scenarios.

The resolution date is the date when the outcome of the question will be decided, come up with a resolution date that is
consistent with the question.

The output format should be the following:

resolution_criteria: [resolution criteria here].
resolution_date: [dd/mm/yyyy]


Example 1:
Question: Will there be any progress in human lifespan enhancement by 2100?

->

resolution_criteria: 
This question will resolve as Yes if either of the following occur before January 1, 2100:
- At least 10 people have confirmed lifespans of at least 130 years of age;
- At least 100 people have confirmed lifespans of at least 120 years of age;
- Ray Kurzweil lives to the age of 120.

The verification process on these individuals' lifespans should be at least as strict as the standards used during 2018.
People resuscitated after being cryonically frozen will not be included in this question's resolution.

resolution_date: 01/01/2100


Example 2:
Question: When will the first general AI system be devised, tested, and publicly announced?

resolution_criteria: An "AI system" is defined as a unified software system that must meet the following requirements:
- Adversarial Turing Test: It should pass a challenging 2-hour Turing test using text, images, and audio, with a single successful test considered sufficient if approved by Metaculus Admins.
- Robotic Assembly Skill: The AI must autonomously assemble a detailed scale model of a Ferrari 312 T4, demonstrating its ability in a practical task.
- Broad Expertise: It must achieve at least 75% accuracy in each task and 90% average accuracy across all tasks in a specific Q&A dataset.
- Problem Solving Efficiency: The AI should attain a top-1 strict accuracy of at least 90.0% on complex benchmark problems.
- System Integration: The AI needs to function as a cohesive whole rather than as separate specialized systems, capable of explaining its actions and decisions.

Compliance with these criteria can be confirmed through direct demonstration, developer statements, or a special committee's decision, with the official recognition based on public disclosure of these capabilities.

resolution_date: 05/11/2031

-----

Question: {title}
"""

validate_forecasting_question_prompt = """\
I want you to help me validate if a forecasting question (as on sites like Metaculus / PredictIt) is well defined. 
The question will ask about an event in the future.
The fields are:
- title: The title of the question.
- body: The resolution criteria of the question.
- resolution_date: The date when the outcome of the question will be decided.

I want you to validate according to the following criteria:
- The resolution date should be consistent with the question. 
- The resolution date should be in the future. The current date is {current_date}.
- The resolution criteria should not be excessively vague or ambiguous, and should be consistent with the question.

The format of your response should be:
reasoning: [reasoning here]
valid: [True/False]

Example 1:
title: Will Kanye West become the president of the United States by 2020?
body: This question will resolve as Yes if Kanye West is elected as the president of the United States before January 1, 2020.
resolution_date: 01/01/2020

->

reasoning: The resolution date is in the past.
valid: False


Example 2:
title: Will Kanye West become the president of the United States by 2030?
body: This question will resolve as Yes if it happens.
resolution_date: 01/01/2030

->

reasoning: The resolution criteria is too vague.
valid: False


Example 3:
title: Will Kanye West become the president of the United States by 2030?
body: This question will resolve as Yes if Kanye West is elected and inaugurated as the president of the United States before January 1, 2030.
resolution_date: 01/01/2030

->

reasoning: The resolution criteria are clear and consistent with the title.
The resolution date is consistent with the title and in the future.
valid: True

-----

title: {title}
body: {body}
resolution_date: {resolution_date}
"""

from common.datatypes import BodyAndDate


async def get_criteria_and_date(
    title: str, model: str = "gpt-4o-2024-05-13", **kwargs
) -> BodyAndDate:
    prompt = resolution_criteria_date_prompt.format(
        title=title, response_model=BodyAndDate
    )
    return await answer(prompt, response_model=BodyAndDate, model=model, **kwargs)


async def from_string(
    question: str,
    data_source: str,
    question_type: Optional[str] = None,
    url: Optional[str] = None,
    metadata: Optional[dict] = None,
    body: Optional[str] = None,
    date: str = None,
    model: str = "gpt-4o-2024-05-13",
    fill_in_body: bool = False,
    **kwargs,
) -> ForecastingQuestion:
    if not question_type:
        question_type = "binary"

    if date is not None:
        try:
            date = datetime.strptime(date, "%d/%m/%Y")
        except ValueError:
            date = None

    if not fill_in_body and body is None:
        raise ValueError("No question body provided and fill_in_body is False")

    if body is None or date is None:
        match (body, date):
            case (None, None):
                print(
                    "No body, getting criteria and date from the title with an LLM call"
                )
            case (None, _):
                print("No body, getting criteria from the title with an LLM call")
            case (_, None):
                print("No date, getting date from the title with an LLM call")
            case _:
                assert False

        for attempt in range(3):
            try:
                bodyAndDate = await get_criteria_and_date(
                    question, model=model, **kwargs
                )
                if body is None:
                    body = bodyAndDate.resolution_criteria
                if date is None:
                    date = bodyAndDate.resolution_date
                break
            except Exception as e:
                print(f"An error has occurred: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(1)

    print(f"\nquestion_formatter.from_string: {bodyAndDate=}")

    return ForecastingQuestion(
        id=uuid.uuid4(),
        title=question,
        body=body,
        resolution_date=date,
        question_type=question_type,
        data_source=data_source,
        url=url,
        metadata=metadata,
        resolution=None,
    )


async def verify_question(question: ForecastingQuestion, **kwargs):
    current_date = datetime.now()
    print(f"question.body: {question.body}")
    print(f"current_date: {current_date}")
    prompt = validate_forecasting_question_prompt.format(
        title=question.title,
        body=question.body,
        resolution_date=question.resolution_date,
        current_date=current_date,
    )
    print(f"kwargs: {kwargs}")
    verification = await answer(prompt, response_model=VerificationResult, **kwargs)

    if write_verification:
        filename = get_data_path() / "verification/question_verification.jsonl"
        dict_to_write = question.model_dump_json()
        verification_flat = (
            dict_to_write[:-1]
            + ","
            + '"verification": '
            + verification.model_dump_json()
            + "}"
        )
        dict_to_write = verification_flat
        await write_jsonl_async_from_str(filename, [dict_to_write], append=True)

    return verification
