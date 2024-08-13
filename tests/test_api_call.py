import pytest
import os
from common.datatypes import PlainText
from common.perscache import register_model_for_cache
from common.llm_utils import (
    answer,
    answer_sync,
)  # Adjust the import based on your script's structure
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str
    age: int


register_model_for_cache(UserInfo)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o-mini",
        "meta-llama/llama-3-8b-instruct:nitro",
        "anthropic/claude-3.5-sonnet",
    ],
)
async def test_answer_real_api(model):
    print("Testing the OpenRouter API, + OpenAI API if key is set")
    prompt = "John Doe is 25 years old."

    # Save the original value of the environment variable
    original_use_openrouter = os.getenv("USE_OPENROUTER", "False")

    try:
        print(f"Testing model: {model}")
        # Set the environment variable based on the model
        if model.startswith("meta-llama"):
            os.environ["USE_OPENROUTER"] = "True"
        elif model.startswith("gpt"):
            if (
                os.getenv("OPENAI_API_KEY", None) is not None
                and original_use_openrouter == "False"
            ):
                print(
                    "OPENAI_API_KEY is set, OpenRouter API is not, setting USE_OPENROUTER to False"
                )
                os.environ["USE_OPENROUTER"] = "False"
            else:
                print("Using OpenRouter API for the OpenAI test call")
                os.environ["USE_OPENROUTER"] = "True"

        response = await answer(prompt, model=model, response_model=UserInfo)
        print(response)
        assert response is not None
        # assert response is of type UserInfo
        assert isinstance(response, UserInfo)
        assert response.name == "John Doe"
        assert response.age == 25
    finally:
        os.environ["USE_OPENROUTER"] = original_use_openrouter


# for other models and OpenRouter, we assert it fails
# so we know if it changes
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "mistralai/mistral-7b-instruct",
        "microsoft/wizardlm-2-8x22b",
    ],
)
async def test_answer_fails_openrouter(model):
    prompt = "John Doe is 25 years old."
    original_use_openrouter = os.getenv("USE_OPENROUTER", "False")
    try:
        os.environ["USE_OPENROUTER"] = "True"
        # we want an error here. if we get an error, it's ok. if the request goes through, fail
        with pytest.raises(Exception):
            response = await answer(prompt, model=model, response_model=UserInfo)
    finally:
        os.environ["USE_OPENROUTER"] = original_use_openrouter


def test_answer_sync():
    example_prompt = "Generate a sample forecasting question"

    result = answer_sync(
        prompt=example_prompt,
        preface=None,
        model="gpt-4o-mini-2024-07-18",
        response_model=PlainText,
    )

    assert isinstance(result, PlainText) and len(result.text) > 0
    print(f"Generated question: {result.text}")


@pytest.mark.asyncio
async def test_answer_sync_async():
    example_prompt = "Generate a sample forecasting question"
    result = await answer(
        prompt=example_prompt,
        preface=None,
        model="gpt-4o-mini-2024-07-18",
        response_model=PlainText,
    )
    assert isinstance(result, PlainText) and len(result.text) > 0
    print(f"Generated question: {result.text}")
