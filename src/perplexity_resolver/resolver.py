import time
import json
from common.llm_utils import answer_native
from common.path_utils import get_logs_path
from datetime import datetime
import asyncio
from typing import List
from .prompts import (
    perplexity_resolve_prompt,
    perplexity_resolve_example_1,
    perplexity_resolve_example_2,
    perplexity_resolve_example_3,
    perplexity_resolve_example_4,
    perplexity_resolve_example_5,
)
from .parse_resolve_output import parse_resolver_output
from common.datatypes import ResolverOutput


async def single_resolve(
    question_title: str,
    question_body: str,
    resolution_date: datetime,
    created_date: datetime | None,
    model: str,
) -> ResolverOutput:
    """
    Resolve a forecasting question using a single model.
    :param question_title: Title of the forecasting question
    :param question_body: Body of the forecasting question
    :param model: The model to use for the API call
    :return: The resolved answer as a ResolverOutput object
    """
    formatted_prompt = perplexity_resolve_prompt.format(
        example_1=perplexity_resolve_example_2,
        example_2=perplexity_resolve_example_1,
        example_3=perplexity_resolve_example_3,
        example_4=perplexity_resolve_example_4,
        example_5=perplexity_resolve_example_5,
        question_title=question_title,
        question_body=question_body,
        created_date=created_date,
        resolution_date=resolution_date,
    )
    try:
        response = await answer_native(formatted_prompt, model=model)
        assert isinstance(response, str)
        parsed_response = await parse_resolver_output(response, question_title)
        dump_file = get_logs_path() / "perplexity_resolver_logs.jsonl"
        with open(dump_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "question_title": question_title,
                        "question_body": question_body,
                        "model": model,
                        "response": response,
                        "parsed_response": str(parsed_response),
                    }
                )
                + "\n"
            )
        return parsed_response
    except Exception as e:
        return ResolverOutput(
            chain_of_thought=f"Error: {str(e)}", can_resolve_question=False, answer=None
        )


async def resolve_question(
    question_title: str,
    question_body: str,
    resolution_date: str,
    created_date: str,
    models: List[str] = [
        "perplexity/llama-3.1-sonar-huge-128k-online",
        "perplexity/llama-3.1-sonar-large-128k-online",
    ],
    n: int = 2,
) -> ResolverOutput:
    """
    Resolve a forecasting question using multiple models and combining their outputs.
    :param question_title: Title of the forecasting question
    :param question_body: Body of the forecasting question
    :param models: List of models to use for the API calls
    :param n: Number of calls to make for each model
    :return: The combined resolved answer as a ResolverOutput object
    """
    t0 = time.time()

    print(f"resolve_question: {question_title}")

    tasks = [
        single_resolve(
            question_title, question_body, resolution_date, created_date, model
        )
        for model in models
        for _ in range(n)
    ]
    results = await asyncio.gather(*tasks)

    combined_output = combine_outputs(results)
    print(f"Time taken: {time.time() - t0:.2f} seconds")
    return combined_output


def combine_outputs(outputs: List[ResolverOutput]) -> ResolverOutput:
    """
    Combine multiple ResolverOutput objects into a single output.
    - can_resolve_question is true if true for the majority of outputs
    - combined_answer is calculated based on outputs with non-None answers
    """
    total_outputs = len(outputs)
    assert total_outputs > 0
    resolvable_count = sum(1 for output in outputs if output.can_resolve_question)

    # can_resolve_question is true if true for the majority
    can_resolve = resolvable_count >= total_outputs / 2

    # Calculate combined_answer based on outputs with non-None answers
    valid_answers = [output.answer for output in outputs if output.answer is not None]
    total_valid_answers = len(valid_answers)

    if can_resolve and total_valid_answers > 0:
        # TODO: unsure why both of these checks are needed
        positive_answers = sum(1 for answer in valid_answers if answer is True)
        combined_answer = (positive_answers / total_valid_answers) > 0.5
    else:
        combined_answer = None

    combined_chain_of_thought = "\n\n".join(
        output.chain_of_thought for output in outputs
    )

    return ResolverOutput(
        chain_of_thought=combined_chain_of_thought,
        can_resolve_question=can_resolve,
        answer=combined_answer,
    )
