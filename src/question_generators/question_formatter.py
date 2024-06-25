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


resolution_date_specification = """\
The resolution date is the date when the outcome of the question will be decided. 
You should come up with a resolution date that is consistent with the question.  
In case of "before YYYY", default to 31/12/(YYYY-1).
In case of "by YYYY", default to 31/12/YYYY.
In case of "by DD MM YYYY" or "on DD MM YYYY", of course, default to DD MM YYYY."""

# We use this mainly for questions that are scraped and do have body/resolution criteria, but do not have a resolution date.
resolution_date_prompt = """\
I want you to help me craft an appropriate resolution date for a forecasting question,
as on the sites like Metaculus / PredictIt.

{resolution_date_specification}

Example 1:
Question: Will there be any progress in human lifespan enhancement before 2100?

->

resolution_date: 2099-12-31 00:00:00


Example 2:
Question: Will the United Kingdom deploy an AI-powered public surveillance system with real-time facial recognition capabilities by June 2026?

->

resolution_date: 2026-06-30 00:00:00

---

Question: {title}
"""


resolution_criteria_date_prompt = """\
I want you to help me craft a resolution criteria and an appropiate resolution date for a forecasting question,
as on the sites like Metaculus / PredictIt.
The resolution criteria should:
- Unambiguously define what outcomes will be considered for the outcome of the question.
- Define any ambigious terms in the question.
- When possible, use authoritative sources, or consensus of the scientific community. For a newsworthy event, use reputable news sources like (BBC, The Guardian, New York Times, Washington Post, etc.)
- Consider edge cases and fall-back scenarios.
- Be as concise as possible, while still being clear and comprehensive.

{resolution_date_specification}

The output format should be the following:

resolution_criteria: [resolution criteria here].
resolution_date: [dd/mm/yyyy]


Example 1:
Question: Will there be any progress in human lifespan enhancement before 2100?

->

resolution_criteria: 
This question will resolve as Yes if any of the following occur before January 1, 2100:
- At least 10 people have confirmed lifespans of at least 130 years of age;
- At least 100 people have confirmed lifespans of at least 120 years of age;

The verification process must meet or exceed the standards used in 2024.
People resuscitated after being cryonically frozen do not count towards this question's resolution.

resolution_date: 2099-12-31 00:00:00


Example 2:
Question: Will the United Kingdom deploy an AI-powered public surveillance system with real-time facial recognition capabilities by 2025?

->

resolution_criteria: This question will resolve as Yes if, by December 31, 2025, the United Kingdom has officially deployed a public surveillance system that utilizes artificial intelligence for real-time facial recognition across any part of its public spaces.
The system must be capable of identifying individuals and matching them to a database with at least an 85% accuracy rate. 
The deployment of such a system must be confirmed by an official government announcement or documentation. 
In the absence of a public announcement, credible reports from at least three major news organizations (BBC, The Guardian, Reuters, Bloomberg, New York Times, Washington Post) are sufficient.
The system must be operational, not in a trial phase. 
In the event of a partial deployment (e.g., limited to specific cities or areas), the question will resolve as Yes if the system is intended for nationwide expansion. 
Temporary deployments for specific events or the use of similar technology in private spaces do not count towards this question's resolution.

resolution_date: 2025-12-31 00:00:00


Example 3:
Question: Will we discover definitive evidence of past or present life on Mars by 2030?

->

resolution_criteria: This question will resolve as Yes if, by December 31, 2030, evidence that meets the following criteria for definitive evidence of past or present life on Mars is announced and recognized by the scientific community:

- Microbial Life: Discovery of Martian-origin microbial life through genetic or biochemical analysis.
- Fossilized Life: Unambiguous identification of Martian-origin fossils indicating past life.
- Biochemical Markers: Detection of biochemical markers uniquely associated with Martian biological processes.
- Atmospheric or Soil Analysis: Incontrovertible evidence of biological activity on Mars from atmospheric or soil analysis.

This question resolves as YES upon official recognition of the discovery by both NASA and ESA, or by one of these organizations together with a publication in a peer-reviewed scientific journal or major conference. 
Otherwise, this question resolves as NO.

resolution_date: 2030-12-31 00:00:00


Example 4:

Question: Will a machine learning model be the first to prove or disprove the Riemann Hypothesis by 2030?

->

resolution_criteria: A "machine learning model" is any (artificial) computational system using machine learning techniques to analyze, infer, or predict outcomes based on data.
This question resolves as Yes if, before January 1, 2031, a machine learning model successfully settles the Riemann Hypothesis, and the proof is both:
(i) Accepted by a recognized mathematical authority, such as the Clay Mathematics Institute.
(ii) Published in a peer-reviewed mathematical journal or presented at a major mathematical conference and verified by independent experts.

The proof must be the first to satisfy these criteria. In case of simultaneous discovery, the resolution is based on the official announcement date.

A machine learning model is considered to have proved it if cited as the most important author in the peer-reviewed paper, or if human authors explicitly state the model completed the majority of the work.

resolution_date: 2030-12-31 00:00:00

-----

Question: {title}
"""

verify_forecasting_question_prompt = """\
I want you to help me validate if a forecasting question (as on sites like Metaculus / Manifold) is well defined. 
The question will ask about an event in the future or past.
The fields are:
- title: The title of the question.
- body: The resolution criteria of the question.
- resolution_date: The date when the outcome of the question will be decided.

I want you to validate according to the following criteria:
- The resolution date should be consistent with the question. 
- The resolution criteria should not be excessively vague or ambiguous, and should be consistent with the question.

The format of your response should be:
reasoning: [reasoning here]
valid: [True/False]

Example 1:
title: Will Kanye West become the president of the United States by 2020?
body: This question will resolve as Yes if Kanye West is elected as the president of the United States before January 1, 2020.
resolution_date: 2020-01-01 00:00:00

->

reasoning: The resolution criteria are clear and consistent with the title.
valid: True


Example 2:
title: Will Kanye West become the president of the United States by 2030?
body: This question will resolve as Yes if it happens.
resolution_date: 2030-01-01 00:00:00

->

reasoning: The resolution criteria is too vague.
valid: False


Example 3:
title: Will Kanye West become the president of the United States by 2030?
body: This question will resolve as Yes if Kanye West is elected and inaugurated as the president of the United States before January 1, 2030.
resolution_date: 2030-01-01 00:00:00

->

reasoning: The resolution criteria are clear and consistent with the title.
The resolution date is consistent with the title and in the future.
valid: True


Example 4:
title: Will any member of Kanye West's family become the president of the United States by 2035?
body: This question will resolve as Yes if Kanye West or any of his family members (including his wife, children, siblings and parents) is elected and... Show More\n
resolution_date: 2035-01-01 00:00:00

->

reasoning: There is a "Show More" instead of the end of the resolution criteria.
valid: False

-----

title: {title}
body: {body}
resolution_date: {resolution_date}
"""

from common.datatypes import BodyAndDate, ResolutionDate


async def get_criteria_and_date(
    title: str, model: str = "gpt-4o-2024-05-13", **kwargs
) -> BodyAndDate:
    prompt = resolution_criteria_date_prompt.format(
        title=title, resolution_date_specification=resolution_date_specification
    )
    return await answer(prompt, response_model=BodyAndDate, model=model, **kwargs)


async def get_date(
    title: str, model: str = "gpt-4o-2024-05-13", **kwargs
) -> ResolutionDate:
    prompt = resolution_date_prompt.format(
        title=title, resolution_date_specification=resolution_date_specification
    )
    return await answer(prompt, response_model=ResolutionDate, model=model, **kwargs)


async def from_string(
    question: str,
    data_source: str,
    question_type: Optional[str] = None,
    url: Optional[str] = None,
    metadata: Optional[dict] = None,
    body: Optional[str] = None,
    date: Optional[str] = None,
    resolution: Optional[bool] = None,
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

    if body is None:
        print("No body, getting criteria and date from the title with an LLM call")
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

    elif date is None:
        print("No date, getting date from the title with an LLM call")
        resolution_date = await get_date(question, model=model, **kwargs)
        print(f"\nquestion_formatter.from_string: {resolution_date=}")
        date = resolution_date.resolution_date

    return ForecastingQuestion(
        id=uuid.uuid4(),
        title=question,
        body=body,
        resolution_date=date,
        question_type=question_type,
        data_source=data_source,
        url=url,
        metadata=metadata,
        resolution=resolution,
    )


async def verify_question_llm(
    question: ForecastingQuestion, current_date: datetime, **kwargs
) -> VerificationResult:
    prompt = verify_forecasting_question_prompt.format(
        title=question.title,
        body=question.body,
        resolution_date=question.resolution_date,
        current_date=current_date,
    )
    print(f"kwargs: {kwargs}")
    verification = await answer(prompt, response_model=VerificationResult, **kwargs)
    return verification


async def verify_question_filter_known_smells(
    question: ForecastingQuestion,
) -> VerificationResult:
    """
    This function checks if the question matches any of the known smells (issues) with the question body or title.
    If the question matches a known smell, the verification result is set to False.
    If the question does not match any known smell, the verification result is set to True.
    """
    if "Sister questions" in question.body:
        return VerificationResult(
            reasoning="The question body contains references to sister questions",
            valid=False,
        )

    if "Ragnarök Question Series" in question.title:
        return VerificationResult(
            reasoning="The Ragnarök Question Series questions from Metaculus do not include enough information about the resolution criteria on their own.",
            valid=False,
        )

    return VerificationResult(
        reasoning="The question does not match any known smells", valid=True
    )


async def verify_question_all_methods(
    question: ForecastingQuestion, **kwargs
) -> VerificationResult:
    current_date = datetime.now()
    print(f"question.body: {question.body}")
    print(f"current_date: {current_date}")

    verification = await verify_question_filter_known_smells(question)
    if not verification.valid:
        return verification

    verification = await verify_question_llm(question, current_date, **kwargs)
    return verification


async def verify_question(
    question: ForecastingQuestion, **kwargs
) -> VerificationResult:
    verification = await verify_question_all_methods(question, **kwargs)

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
