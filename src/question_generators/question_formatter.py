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

# From Halawi et al.
verify_forecasting_title_prompt = """
I’m trying to assess the quality of an old forecasting dataset.
Here is a forecasting question from the dataset: {question}.
Please flag questions that don’t sound like binary forecasting questions by outputting "flag". If it sounds
like a reasonable question, output valid is True.
Examples of strings that should be flagged:
• Will I finish my homework tonight?
• Metaculus party 2023
• Will Hell freeze over?
• Heads or tails
• Will this video reach 100k views by the EOD?
Examples of strings that should not be flagged:
• Will Megan Markle and Prince Harry have a baby by the end of the year?
• Will the Brain Preservation Foundation’s Large Mammal preservation prize be won by Feb 9th, 2017?
• Will there be more novel new drugs approved by the FDA in 2016 than in 2015?
If a question is already resolved, that doesn’t mean it should be flagged. When in doubt, mark it as "ok".
Your response should take the following structure:
Reasoning: {{ insert your concise thoughts here }}
Valid: {{ True/False }}
"""

verify_forecasting_body_prompt = """
I'm assessing the quality of the body (resolution criteria) of a forecasting question. Here's the body of the question:

{body}

Please evaluate the body according to the following criteria:

1. Clarity: The resolution criteria should be clear and unambiguous.
2. Objectivity: The criteria should be objectively measurable or verifiable.
3. Completeness: All necessary information for resolving the question should be included.
4. Appropriateness: The criteria should not mention market probabilities, betting odds, or other inappropriate elements.

Examples of valid bodies:
- This question will resolve as Yes if Kanye West is elected and inaugurated as the president of the United States before January 1, 2030.
- France has last won the FIFA World Cup in 2018. This question will resolve as Yes if, by December 31, 2026, the French national football team (senior, not youth or women's) has won either the FIFA World Cup or the UEFA European Championship.

Examples of invalid bodies:
- This question will resolve as Yes if it happens.
- {"resolution_criteria": "nini","background_info": "50% chance. Resolves NO if dismissed, dropped, or still in process.\n\nResolves YES if there is a settlement or judgement for either side.\n\nhttps://www.politico.com/news/2024/06/17/george-norcross-indictment-new-jersey-corruption-00163700\n\n[image]"}
- This question will resolve as Yes if, by December 31, 2030, a new strain of influenza virus is identified that meets the following criteria: 
  - The strain must be genetically distinct from previously identified strains, with genetic differences significant enough to be considered a new strain by the World Health Organization (WHO) or a similar authoritative body.
  - The strain must have demonstrated the ability to cause human infection and have the potential for sustained human-to-human transmission.
  - The strain must be assessed by the WHO or a similar authoritative body as having pandemic potential, meaning it has a significant risk of causing widespread illness across multiple countries or continents.
  In the absence of a formal declaration by the WHO or a similar authoritative body, a consensus among at least three leading infectious disease research institutions will be considered sufficient for resolution.

Issues to flag:
- Vague or subjective terms without clear definitions
- Inconsistencies within the body itself
- Missing crucial information needed for resolution
- Multiple distinct pathways to resolution that aren't all important for the question
- Any mention of market probabilities, betting odds, or similar concepts
- Use of external links or images that are crucial for understanding the resolution criteria
- Overly complex or ambiguous resolution criteria

Your response should take the following structure:
Reasoning: {{ insert your concise evaluation here, addressing the criteria above }}
Valid: {{ True if the body meets all criteria, False otherwise }}

Remember, when in doubt, it's better to flag potential issues.
"""


verify_forecasting_question_prompt = """\
I want you to help me validate if a forecasting question (as on sites like Metaculus / Manifold) is well defined. 
The question will ask about an event in the future or past.
The fields are:
- title: The title of the question.
- body: The resolution criteria of the question.
- resolution_date: The date when the outcome of the question will be decided.

I want you to validate according to the following criteria:

- The resolution date should be consistent with the question. In particular:
   - In case of events that happen at a fixed time (e.g. "Will France win the FIFA World Cup in 2026"), it should *never* be *before* the date when the event is scheduled to happen. Ideally it is exactly at the date. There can be *at most a month of leeway* to account for resolution issues and delays. If the delay is too long, the question should be marked invalid.
   - In case of events that could resolve at an uncertain date (will Warren Buffet die of cancer?), the resolution date should be such that it is highly likely the event will happen before it. In this case, the `body` should say the question resolves N/A if the resolution date comes and there was no resolution.

- The resolution criteria should not be excessively vague or ambiguous, and should be consistent with the question.


The format of your response should be:
reasoning: [reasoning here]
valid: [True/False]

Example 1:
title: Will Kanye West become the president of the United States by 2020?
body: This question will resolve as Yes if Kanye West is elected as the president of the United States before January 1, 2020.
resolution_date: 2020-01-01 00:00:00

->

reasoning: The body (resolution criteria) is clear and consistent with the title.
valid: True


Example 2:
title: Will Kanye West become the president of the United States by 2030?
body: This question will resolve as Yes if it happens.
resolution_date: 2030-01-01 00:00:00

->

reasoning: The body is too vague.
valid: False


Example 3:
title: Will Kanye West become the president of the United States before January 2030?
body: This question will resolve as Yes if Kanye West is elected and inaugurated as the president of the United States before January 1, 2030.
resolution_date: 2030-01-01 00:00:00

->

reasoning: The resolution criteria are clear and consistent with the title.
The resolution date is consistent with the title and in the future.
valid: True


Example 4:
title: Will Kanye West become the president of the United States by 2030?
body: This question will resolve as Yes if Kanye West is elected and inaugurated as the president of the United States before January 1, 2031.
resolution_date: 2031-12-01 00:00:00

->

reasoning: The resolution date is too late compared to when the body says the question will resolve. It is in December 2031, while the body says the question should resolve by January 2031.
valid: False


Example 5:
title: Will any member of Kanye West's family become the president of the United States by 2035?
body: This question will resolve as Yes if Kanye West or any of his family members (including his wife, children, siblings and parents) is elected and... Show More\n
resolution_date: 2035-01-01 00:00:00

->

reasoning: There is a "Show More" instead of the end of the resolution criteria.
valid: False


Example 6:
title: Will a new strain of influenza virus with pandemic potential be identified by 2030?
body: This question will resolve as Yes if, by December 31, 2030, a new strain of influenza virus is identified that meets the following criteria: 
- The strain must be genetically distinct from previously identified strains, with genetic differences significant enough to be considered a new strain by the World Health Organization (WHO) or a similar authoritative body.
- The strain must have demonstrated the ability to cause human infection and have the potential for sustained human-to-human transmission.
- The strain must be assessed by the WHO or a similar authoritative body as having pandemic potential, meaning it has a significant risk of causing widespread illness across multiple countries or continents.
In the absence of a formal declaration by the WHO or a similar authoritative body, a consensus among at least three leading infectious disease research institutions will be considered sufficient for resolution.
resolution_date: 2030-12-31 23:59:59

->

reasoning: "Pandemic potential" is way too subjective; it is not a clearly objective threshold. 
In the absence of a formal declaration by the WHO or a similar authoritative body, a consensus among at least three leading infectious disease research institutions will be considered sufficient for resolution." is not good because it is not clear on what exactly they should agree on.
valid: False


Example 7:
title: Will DeepMind develop an AI with the capability to significantly disrupt at least one major industry by 2030?
body: This question will resolve as Yes if, by December 31, 2030, DeepMind has developed an artificial intelligence (AI) system that has been publicly recognized to significantly disrupt at least one major industry.  The AI system must be a primary factor in the disruption, as opposed to a contributing technology among others. The industry in question must be one of the following: healthcare, automotive, finance, entertainment, or energy. Disruption is considered significant if it leads to a change in at least 20%% of the market share within the industry or a comparable metric of impact, such as a 20%% increase in efficiency or productivity. The resolution will rely on reports and analyses from the specified industry sources published by December 31, 2030. In the absence of clear industry consensus, a panel of experts from the affected industry may be consulted to determine the resolution.

->

reasoning: "to significantly disrupt at least one major industry by 2030" is too fuzzy. The resolution criteria must be quantifiable. "20%% increase in efficiency or productivity" is clearly not measurable.
In addition, we do not want multiple distinct pathways to resolution in a question where it is not important there are multiple pathways.
Here, we have both the market share and "a comparable metric of impact" as resolution criteria, and notwithstanding the fact that impact is not defined clearly, we should not have both.
valid: False


Example 8:
title: Will France win any major football tournament by 2026?
body: France has last won the FIFA World Cup in 2018. This question will resolve as Yes if, by December 31, 2026, the French national football team (senior, not youth or women's) has won either the FIFA World Cup or the UEFA European Championship.
resolution_date: 2026-12-31 23:59:59

->

reasoning: The resolution date is fine. The body is clear; there are multiple distinct pathways to resolution, but all are important for the question. The body includes a bit of background information that is not necessary for the question, but it is not harmful.
valid: True
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
    # Verify the question title
    title_prompt = verify_forecasting_title_prompt.format(question=question.title)
    title_verification = await answer(title_prompt, response_model=VerificationResult, **kwargs)
    
    if not title_verification.valid:
        return VerificationResult(
            reasoning=f"Title validation failed: {title_verification.reasoning}",
            valid=False
        )
    
    # Verify the question body
    body_prompt = verify_forecasting_body_prompt.format(body=question.body)
    body_verification = await answer(body_prompt, response_model=VerificationResult, **kwargs)
    
    if not body_verification.valid:
        return VerificationResult(
            reasoning=f"Body validation failed: {body_verification.reasoning}\nTitle was valid.",
            valid=False
        )
    
    # Verify the full question (title, body, and resolution date)
    full_prompt = verify_forecasting_question_prompt.format(
        title=question.title,
        body=question.body,
        resolution_date=question.resolution_date,
    )
    full_verification = await answer(full_prompt, response_model=VerificationResult, **kwargs)
    
    if not full_verification.valid:
        return VerificationResult(
            reasoning=f"Full question validation failed: {full_verification.reasoning}\nTitle and body were valid individually.",
            valid=False
        )
    
    # If all verifications passed
    combined_reasoning = (
        f"Title: {title_verification.reasoning}\n"
        f"Body: {body_verification.reasoning}\n"
        f"Full: {full_verification.reasoning}"
    )
    
    return VerificationResult(reasoning=combined_reasoning, valid=True)


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

    try:
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
    except Exception as e:
        print(f"Error during question verification: {e}")
        return VerificationResult(
            reasoning=f"Verification failed: {str(e)}", valid=False
        )