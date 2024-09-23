from common.datatypes import ForecastingQuestion
import asyncio
import uuid
from common.llm_utils import answer
from typing import Optional
from common.utils import normalize_date_format
from fq_generation.multi_to_binary import reformat_metaculus_question


resolution_date_specification = """\
The resolution date is the date when the outcome of the question will be decided. 
You should come up with a resolution date that is consistent with the question.  
In case of "before YYYY", default to 31/12/(YYYY-1).
In case of "by YYYY", default to 31/12/YYYY.
In case of "by DD MM YYYY" or "on DD MM YYYY", of course, default to DD/MM/YYYY."""

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
    created_date: Optional[str] = None,
    question_type: Optional[str] = None,
    url: Optional[str] = None,
    metadata: Optional[dict] = None,
    body: Optional[str] = None,
    resolution_date: Optional[str] = None,
    resolution: Optional[bool] = None,
    model: str = "gpt-4o-2024-05-13",
    fill_in_body: bool = False,
    **kwargs,
) -> ForecastingQuestion:
    if not question_type:
        question_type = "binary"

    if resolution_date is not None:
        resolution_date = normalize_date_format(resolution_date)

    if created_date is not None:
        created_date = normalize_date_format(created_date)

    if not fill_in_body and body is None:
        raise ValueError("No question body provided and fill_in_body is False")

    if data_source == "metaculus":
        reformat_result: dict[
            str, str | None | bool
        ] = await reformat_metaculus_question(question, body, model=model)

        if reformat_result["did_change"]:
            print(
                f"Reformatted Metaculus question: {question=} -> {reformat_result['title']=}"
            )
            if metadata is None:
                metadata = {}
            metadata["reformat_metaculus_question"] = {
                "original_question": question,
                "original_body": body,
            }
            question = reformat_result["title"]
            body = reformat_result["body"]

    if body is None:
        print("No body, getting criteria and date from the title with an LLM call")
        for attempt in range(3):
            try:
                bodyAndDate = await get_criteria_and_date(
                    question, model=model, **kwargs
                )
                if body is None:
                    body = bodyAndDate.resolution_criteria
                if resolution_date is None:
                    resolution_date = bodyAndDate.resolution_date
                break
            except Exception as e:
                print(f"An error has occurred: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(1)
        print(f"\nfq_body_generator.from_string: {bodyAndDate=}")

    elif resolution_date is None:
        print("No date, getting date from the title with an LLM call")
        resolution_date = await get_date(question, model=model, **kwargs)
        print(f"\nfq_body_generator.from_string: {resolution_date=}")
        resolution_date = resolution_date.resolution_date

    return ForecastingQuestion(
        id=uuid.uuid4(),
        title=question,
        body=body,
        resolution_date=resolution_date,
        question_type=question_type,
        data_source=data_source,
        created_date=created_date,
        url=url,
        metadata=metadata,
        resolution=resolution,
    )
