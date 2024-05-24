# Standard library imports
import asyncio
import logging
import time

# Related third-party imports

from common.llm_utils import (
    get_provider,
    get_embedding_sync,
    get_embeddings_sync,
    query_api_chat_native,
    query_api_chat_sync_native,
)


from utils import string_utils


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_response_from_model(
    model_name,
    prompt,
    system_prompt="",
    max_tokens=2000,
    temperature=0.8,
    wait_time=30,
):
    """
    Make an API call to the specified model and retry on failure after a
    specified wait time.

    Args:
        model_name (str): Name of the model to use (such as "gpt-4").
        prompt (str): Fully specififed prompt to use for the API call.
        system_prompt (str, optional): Prompt to use for system prompt.
        max_tokens (int, optional): Maximum number of tokens to generate.
        temperature (float, optional): Sampling temperature.
        wait_time (int, optional): Time to wait before retrying, in seconds.
    """
    model_source = get_provider(model_name)
    call_messages = []

    # original Danny Halawi code only uses system prompt for OpenAI
    if system_prompt and model_source == "openai":
        call_messages.append({"role": "system", "content": system_prompt})
    call_messages.append({"role": "user", "content": prompt})

    if model_source == "anthropic" and max_tokens > 4096:
        max_tokens = 4096

    return query_api_chat_sync_native(
        model=model_name,
        messages=call_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def get_async_response(
    prompt,
    model_name="gpt-3.5-turbo-1106",
    temperature=0.0,
    max_tokens=4000,
):
    """
    Asynchronously get a response from the OpenAI API.

    Args:
        prompt (str): Fully specififed prompt to use for the API call.
        model_name (str, optional): Name of the model to use (such as "gpt-3.5-turbo").
        temperature (float, optional): Sampling temperature.
        max_tokens (int, optional): Maximum number of tokens to sample.

    Returns:
        str: Response string from the API call (not the dictionary).
    """
    model_source = get_provider(model_name)

    call_messages = [{"role": "user", "content": prompt}]

    if model_source == "anthropic" and max_tokens > 4096:
        max_tokens = 4096

    while True:
        try:
            break
        except (Exception, BaseException) as e:
            logger.info(f"Exception, erorr message: {e}")
            logger.info("Waiting for 30 seconds before retrying...")
            time.sleep(30)

    if model_source == "together":
        return await asyncio.to_thread(
            query_api_chat_sync_native,
            model=model_name,
            messages=call_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        return await query_api_chat_native(
            model=model_name,
            messages=call_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def get_openai_embedding(texts, model="text-embedding-3-large"):
    """
    Query OpenAI's text embedding model to get the embedding of the given text.

    Args:
        texts (list of str): List of texts to embed.

    Returns:
        list of Embedding objects: List of embeddings, where embedding[i].embedding is a list of floats.
    """
    texts = [text.replace("\n", " ") for text in texts]
    while True:
        try:
            if len(texts) == 1:
                embedding = get_embedding_sync(texts[0], embedding_model=model)
            else:
                embedding = get_embeddings_sync(texts, embedding_model=model)
                return embedding
        except Exception as e:
            logger.info(f"erorr message: {e}")
            logger.info("Waiting for 30 seconds before retrying...")
            time.sleep(30)
            continue


async def async_make_forecast(
    question,
    background_info,
    resolution_criteria,
    dates,
    retrieved_info,
    reasoning_prompt_templates,
    model_name="gpt-4-1106-preview",
    temperature=1.0,
    return_prompt=False,
):
    """
    Asynchronously make forecasts using the given information.

    Args:
        question (str): Question to ask the model.
        background_info (str): Background information to provide to the model.
        resolution_criteria (str): Resolution criteria to provide to the model.
        dates (str): Dates to provide to the model.
        retrieved_info (str): Retrieved information to provide to the model.
        reasoning_prompt_templates (list of str): List of reasoning prompt templates to use.
        model_name (str, optional): Name of the model to use (such as "gpt-4-1106-preview").
        temperature (float, optional): Sampling temperature.
        return_prompt (bool, optional): Whether to return the full prompt or not.

    Returns:
        list of str: List of forecasts and reasonings from the model.
    """
    assert (
        len(reasoning_prompt_templates) > 0
    ), "No reasoning prompt templates provided."
    reasoning_full_prompts = []
    for reasoning_prompt_template in reasoning_prompt_templates:
        template, fields = reasoning_prompt_template
        reasoning_full_prompts.append(
            string_utils.get_prompt(
                template,
                fields,
                question=question,
                retrieved_info=retrieved_info,
                background=background_info,
                resolution_criteria=resolution_criteria,
                dates=dates,
            )
        )
    # Get all reasonings from the model
    reasoning_tasks = [
        get_async_response(
            prompt,
            model_name=model_name,
            temperature=temperature,
        )
        for prompt in reasoning_full_prompts
    ]
    # a list of strings
    all_reasonings = await asyncio.gather(*reasoning_tasks)
    logger.info(
        "Finished {} base reasonings generated by {}".format(
            len(reasoning_full_prompts), model_name
        )
    )
    if return_prompt:
        return all_reasonings, reasoning_full_prompts
    return all_reasonings
