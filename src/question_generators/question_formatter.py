from common.datatypes import ForecastingQuestion, ValidationResult
import asyncio
import uuid
from common.llm_utils import answer
from typing import Optional
from datetime import datetime


resolution_criteria_date_prompt = """
I want you to help me craft a resolution criteria and an appropiate resolution date for a forecasting question.
The question will ask about an event in the future, like the quesitons in places like metaculus. The resolution criteria
should:
- Unambiguously define what outcomes will be considered for the outcome of the question.
- Define concretely the terms used in the question.
- When possible, use authoritative sources.
- Consider edge cases and fall-back scenarios.

The resolution date is the date when the outcome of the question will be decided, come up with a resolution date that is
consistent with the question.

The format should be the following:

resolution_criteria: [resolution criteria here].
date: [dd/mm/yyyy]

Examples

Question: Will there be any progress in human lifespan enhancement by 2100?

Resolution criteria: This question will resolve as Yes if either of the following occur before January 1, 2100:

At least 10 people have confirmed lifespans of at least 130 years of age

At least 100 people have confirmed lifespans of at least 120 years of age

Ray Kurzweil lives to the age of 120

The verification process on these individuals' lifespans should be at least as strict as the standards used during 2018.
People resuscitated after being cryonically frozen will not be included in this question's resolution.
resolution_date: 01/01/2100


Question: When will the first general AI system be devised, tested, and publicly announced?


Resolution criteria: 
An "AI system" is defined as a unified software system that must meet the following requirements:

Adversarial Turing Test: It should pass a challenging 2-hour Turing test using text, images, and audio, with a single successful test considered sufficient if approved by Metaculus Admins.

Robotic Assembly Skill: The AI must autonomously assemble a detailed scale model of a Ferrari 312 T4, demonstrating its ability in a practical task.

Broad Expertise: It must achieve at least 75% accuracy in each task and 90% average accuracy across all tasks in a specific Q&A dataset.

Problem Solving Efficiency: The AI should attain a top-1 strict accuracy of at least 90.0% on complex benchmark problems.

System Integration: The AI needs to function as a cohesive whole rather than as separate specialized systems, capable of explaining its actions and decisions.

Compliance with these criteria can be confirmed through direct demonstration, developer statements, or a special committee's decision, with the official recognition based on public disclosure of these capabilities.
----
resolution_date: 05/11/2031


Question: {question}
"""

validate_forecasting_question_prompt = """

I want you to help me validate if a forecasting question is well defined. The question will ask about an event in the future, like the quesitons in places like metaculus.
The fields will be:
- title: The title of the question.
- body: The resolution criteria of the question.
- resolution_date: The date when the outcome of the question will be decided.

I want you to validate according to the following criteria:
- The resolution date should be consistent with the question. And in the future. The current date is {current_date}
- The resolution criteria should not be excessively vague or ambiguous. And should be consistent with the question.

The format of your response should be:
reasoning: [reasoning here]
valid: [True/False]

Examples:

title: Will Kanye West become the president of the United States by 2030?
body: This question will resolve as Yes if Kanye West is elected as the president of the United States before January 1, 2030.
resolution_date: 01/01/2020

Response:
reasoning: The resolution date is in the past.
valid: False

title: Will Kanye West become the president of the United States by 2030?
body: This question will resolve as Yes if it happens.
resolution_date: 01/01/2020

Response:
reasoning: The resolution criteria is too vague.
valid: False

title: Will Kanye West become the president of the United States by 2030?
body: This question will resolve as Yes if Kanye West is elected as the president of the United States before January 1, 2030.
esolution_date: 01/01/2030

Response:
reasoning: The resolution date is consistent with the question.
valid: True

"""

from common.datatypes import BodyAndDate


async def get_criteria_and_date(question: str):
    prompt = resolution_criteria_date_prompt.format(
        question=question, response_model=BodyAndDate
    )  # Assuming definition elsewhere
    return await answer(prompt, response_model=BodyAndDate)


async def from_string(
    question: str,
    data_source: str,
    question_type: Optional[str] = None,
    url: Optional[str] = None,
    metadata: Optional[dict] = None,
    body: Optional[str] = None,
    date: str = None,
) -> ForecastingQuestion:
    if not question_type:
        question_type = "binary"

    if date is not None:
        try:
            date = datetime.strptime(date, "%d/%m/%Y")
        except ValueError:
            date = None

    for attempt in range(3):
        try:
            bodyAndDate = await get_criteria_and_date(question)
            break
        except Exception as e:
            print(f"An error has occurred: {e}")
            if attempt == 2:
                raise
            await asyncio.sleep(1)

    print(f"\n{bodyAndDate=}")

    return ForecastingQuestion(
        id=uuid.uuid4(),
        title=question,
        body=body or bodyAndDate.resolution_criteria,
        resolution_date=date or bodyAndDate.resolution_date,
        question_type=question_type,
        data_source=data_source,
        url=url,
        metadata=metadata,
        resolution=None,
    )


async def validate_question(question: ForecastingQuestion):
    current_date = datetime.now()
    prompt = validate_forecasting_question_prompt.format(current_date=current_date)
    return await answer(prompt, response_model=ValidationResult)
