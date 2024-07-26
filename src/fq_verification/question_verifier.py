import os
from dotenv import load_dotenv
from common.datatypes import ForecastingQuestion, VerificationResult
from common.utils import write_jsonl_async_from_str
from common.llm_utils import answer
from common.path_utils import get_data_path
from datetime import datetime

load_dotenv()
write_verification = os.getenv("WRITE_VERIFICATION", "False") == "True"
print(
    f"The setting for WRITE_VERIFICATION in fq_body_generator.py: {write_verification}"
)

# From Halawi et al.
verify_forecasting_title_prompt = """
I'm trying to assess the quality of an old forecasting dataset. Here is a forecasting question from the dataset: {question}. Please flag questions that don't sound like binary forecasting questions by outputting "flag". If it sounds like a reasonable question, output valid is True.

Examples of strings that should be flagged:
- Will I finish my homework tonight?
- Metaculus party 2023
- Will Hell freeze over?
- Heads or tails
- Will this video reach 100k views by the EOD?

Examples of strings that should not be flagged:
- Will Megan Markle and Prince Harry have a baby by the end of the year?
- Will the Brain Preservation Foundation's Large Mammal preservation prize be won by Feb 9th, 2017?
- Will there be more novel new drugs approved by the FDA in 2016 than in 2015?

If a question is already resolved, that doesn't mean it should be flagged. When in doubt, mark it as "True".

Your response should take the following structure:
reasoning: {{ insert your concise thoughts here }}
valid: {{ True/False }}
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
- {{"resolution_criteria": "nini","background_info": "50% chance. Resolves NO if dismissed, dropped, or still in process.\n\nResolves YES if there is a settlement or judgement for either side.\n\nhttps://www.politico.com/news/2024/06/17/george-norcross-indictment-new-jersey-corruption-00163700\n\n[image]"}}
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


async def verify_title(question: ForecastingQuestion, **kwargs) -> VerificationResult:
    title_prompt = verify_forecasting_title_prompt.format(question=question.title)
    return await answer(title_prompt, response_model=VerificationResult, **kwargs)


async def verify_body(question: ForecastingQuestion, **kwargs) -> VerificationResult:
    body_prompt = verify_forecasting_body_prompt.format(body=question.body)
    return await answer(body_prompt, response_model=VerificationResult, **kwargs)


async def verify_full_question(
    question: ForecastingQuestion, **kwargs
) -> VerificationResult:
    full_prompt = verify_forecasting_question_prompt.format(
        title=question.title,
        body=question.body,
        resolution_date=question.resolution_date,
    )
    return await answer(full_prompt, response_model=VerificationResult, **kwargs)


async def verify_question_llm(
    question: ForecastingQuestion, current_date: datetime, **kwargs
) -> VerificationResult:
    # Verify the question title
    title_verification = await verify_title(question, **kwargs)
    if not title_verification.valid:
        return VerificationResult(
            reasoning=f"Title validation failed: {title_verification.reasoning}",
            valid=False,
        )

    # Verify the question body
    body_verification = await verify_body(question, **kwargs)
    if not body_verification.valid:
        return VerificationResult(
            reasoning=f"Body validation failed: {body_verification.reasoning}\nTitle was valid.",
            valid=False,
        )

    # Verify the full question (title, body, and resolution date)
    full_verification = await verify_full_question(question, **kwargs)
    if not full_verification.valid:
        return VerificationResult(
            reasoning=f"Full question validation failed: {full_verification.reasoning}\nTitle and body were valid individually.",
            valid=False,
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
