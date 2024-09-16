from common.llm_utils import answer
from .prompts import (
    perplexity_resolve_prompt,
    perplexity_resolve_example_1,
    perplexity_resolve_example_2,
    perplexity_resolve_example_3,
)
from .parse_resolve_output import parse_resolver_output
from common.datatypes import ResolverOutput, PlainText


async def resolve_question(
    question_title: str,
    question_body: str,
    model: str = "perplexity/llama-3.1-sonar-huge-128k-online",
) -> ResolverOutput:
    """
    Resolve a forecasting question using the Perplexity API.

    :param question: ForecastingQuestion object containing the question details
    :param model: The model to use for the Perplexity API call
    :return: The resolved answer as a string
    """
    formatted_prompt = perplexity_resolve_prompt.format(
        example_1=perplexity_resolve_example_2,
        example_2=perplexity_resolve_example_1,
        example_3=perplexity_resolve_example_3,
        question_title=question_title,
        question_body=question_body,
    )

    try:
        response = await answer(formatted_prompt, model=model)
        if isinstance(response, PlainText):
            response = response.text
        return await parse_resolver_output(response, question_title)
    except Exception as e:
        return f"Error: {str(e)}"
