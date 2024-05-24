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
- When possible, use authoritative sources. For example if the question is about a scientific discovery, use peer-reviewed papers. When its a news event, use reputable news sources. Like BBC, CNN, etc.
- Consider edge cases and fall-back scenarios.
- As concise as possible, while still being clear and comprehensive.

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
Question: Will the United Kingdom deploy an AI-powered public surveillance system with real-time facial recognition capabilities by 2025?

resolution_criteria: This question will resolve as Yes if, by December 31, 2025, the United Kingdom has officially deployed a public surveillance system
that utilizes artificial intelligence for real-time facial recognition across any part of its public spaces.
The system must be capable of identifying individuals and matching them to a database with at least an 85% accuracy rate. 
The deployment of such a system must be confirmed by an official government announcement or documentation. In the absence of a public announcement,
 credible reports from at least three major news organizations (BBC, The Guardian, Reuters, Bloomberg, New York Times, Washington Post) will be considered sufficient evidence.
The system must be operational and not in a trial phase. If multiple systems are deployed, the resolution will consider the first system that meets these criteria.
In the event of a partial deployment (e.g., limited to specific cities or areas), the question will resolve as Yes if the system is intended to be expanded nationwide. 
Edge cases, such as temporary deployments for specific events or the use of similar technology in private spaces, will not count towards this question's resolution.

resolution_date: 31/12/2025

Example 3:
Question: Will NASA discover definitive evidence of past or present life on Mars by 2030?

resolution_criteria: This question will resolve as Yes if, by December 31, 2030, NASA (or any entity recognized by NASA) publicly announces and provides evidence that meets the following criteria for definitive evidence of past or present life on Mars:

Microbial Life: Discovery of microbial life forms that are conclusively identified as having originated on Mars, through genetic or biochemical analysis.

Fossilized Life: Unambiguous identification of fossils that are conclusively determined to be of Martian origin, indicating past life.

Biochemical Markers: Detection of biochemical markers such as specific isotopes or molecules that are uniquely associated with biological processes and are indisputably Martian in origin.

Atmospheric or Soil Analysis: Results from atmospheric or soil analysis that provide incontrovertible evidence of biological activity on Mars.

This question will resolve as YES only if evidence is published in a peer-reviewed scientific journal or officially announced at a major scientific conference. Otherwise it resolves NO.

resolution_date: 31/12/2030

Example 4:

Question: Will a machine learning model be the first to prove the Riemann Hypothesis by 2030?

resolution_criteria: A "machine learning model" is defined as any computational system that utilizes machine learning techniques to analyze, infer, or predict outcomes based on data. This question will resolve as Yes if, before January 1, 2030:

A machine learning model successfully proves the Riemann Hypothesis, and this proof is accepted by a recognized mathematical authority, such as the Clay Mathematics Institute or an equivalent organization.

The proof must be published in a peer-reviewed mathematical journal or presented at a major mathematical conference and subsequently verified by independent experts in the field.

In the event that multiple proofs are presented, the resolution will be based on the first proof that is verified and accepted by the mathematical community.

If the Riemann Hypothesis is disproved by a machine learning model, this will also result in a Yes resolution.

We will define a machine learning model as proving it if either the model is cited as a first authorship or equivalent in the peer-reviewed paper, or said paper explicitly mentions that it was the model that completed the majority of the proof work.

resolution_date: 01/01/2030

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
        print(
            "No body or no date, getting criteria and date from the title with an LLM call"
        )
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
