from common.datatypes import ForecastingQuestion
import asyncio
import uuid
from common.llm_utils import answer
from typing import Optional
import datetime

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

Resolution criteria: [resolution criteria here].
-----
Resolution date: [dd/mm/yyyy]

Examples:

Question: Will there be any progress in human lifespan enhancement by 2100?

Resolution criteria: This question will resolve as Yes if either of the following occur before January 1, 2100:

At least 10 people have confirmed lifespans of at least 130 years of age

At least 100 people have confirmed lifespans of at least 120 years of age

Ray Kurzweil lives to the age of 120

The verification process on these individuals' lifespans should be at least as strict as the standards used during 2018.
People resuscitated after being cryonically frozen will not be included in this question's resolution.
----
Resolution date: 01/01/2100


Question: When will the first general AI system be devised, tested, and publicly announced?


Resolution criteria: We will thus define "an AI system" as a single unified software system that can satisfy the following criteria, all completable by at least some humans.

Able to reliably pass a 2-hour, adversarial Turing test during which the participants can send text, images, and audio files (as is done in ordinary text messaging applications) during the course of their conversation.
 An 'adversarial' Turing test is one in which the human judges are instructed to ask interesting and difficult questions, designed to advantage human participants, and to successfully unmask the computer as an impostor.
  A single demonstration of an AI passing such a Turing test, or one that is sufficiently similar, will be sufficient for this condition, so long as the test is well-designed to the estimation of Metaculus Admins.

Has general robotic capabilities, of the type able to autonomously, when equipped with appropriate actuators and when given human-readable instructions, 
satisfactorily assemble a (or the equivalent of a) circa-2021 Ferrari 312 T4 1:8 scale automobile model. A single demonstration of this ability, or a sufficiently similar demonstration, will be considered sufficient.

High competency at a diverse fields of expertise, as measured by achieving at least 75% accuracy in every task and 90% mean accuracy across all tasks in the Q&A dataset developed by Dan Hendrycks et al..

Able to get top-1 strict accuracy of at least 90.0% on interview-level problems found in the APPS benchmark introduced by Dan Hendrycks, Steven Basart et al.
 Top-1 accuracy is distinguished, as in the paper, from top-k accuracy in which k outputs from the model are generated, and the best output is selected.

By "unified" we mean that the system is integrated enough that it can, for example, explain its reasoning on a Q&A task, or verbally report its progress and identify objects during model assembly.
 (This is not really meant to be an additional capability of "introspection" so much as a provision that the system not simply be cobbled together as a set of sub-systems specialized to tasks like the above, but rather a single system applicable to many problems.)

Resolution will come from any of three forms, whichever comes first: (1) direct demonstration of such a system achieving ALL of the above criteria,
(2) confident credible statement by its developers that an existing system is able to satisfy these criteria,
or (3) judgement by a majority vote in a special committee composed of the question author and two AI experts chosen in good faith by him, for the sole purpose of resolving this question.
Resolution date will be the first date at which the system (subsequently judged to satisfy the criteria) and its capabilities are publicly described in a talk, press release, paper, or other report available to the general public.
----
Resolution date: 05/11/2031


Question: {question}
"""



#TODO(ALejandro): Improve parse, request json.
async def get_criteria_and_date(question: str) -> (str, str):
    prompt = resolution_criteria_date_prompt.format(question=question)  # Assuming definition elsewhere
    r = await answer(prompt)  
    parts = r.split("---")
    cleaned_parts = [part.strip("-") for part in parts]
    if len(cleaned_parts) != 2:
        raise ValueError("Error when parsing response from get_criteria_and_date")
    resolution_criteria, date_str = cleaned_parts
    return resolution_criteria, date_str

async def from_string(question: str, data_source: str, question_type: Optional[str] = None, url: Optional[str] = None, metadata: Optional[dict] = None) -> ForecastingQuestion:
    if not question_type:
        question_type = "binary"

    for attempt in range(3):
        try:
            resolution_criteria, resolution_date_str = await get_criteria_and_date(question)
            break  
        except Exception as e:
            print(f"An error has occurred: {e}")
            if attempt == 2:  
                raise  
            await asyncio.sleep(1)  

    question_dict = {
        "id": str(uuid.uuid4()),
        "title": question,
        "body": resolution_criteria,
        "resolution_date": resolution_date_str,  # ISO 8601 format string expected
        "question_type": question_type,
        "data_source": data_source,
        "url": url,
        "metadata": metadata,
        "resolution": None, 
    }

    return ForecastingQuestion.from_dict(question_dict)
