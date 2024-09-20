import pytest
import os
from common.datatypes import PlainText
from common.perscache import register_model_for_cache
from common.llm_utils import (
    answer,
    answer_sync,
    answer_native,
    answer_native_sync,
    query_api_chat_with_parsing,
)  # Adjust the import based on your script's structure
from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    name: str
    age: int


register_model_for_cache(UserInfo)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "mistralai/mistral-large",
        "microsoft/wizardlm-2-8x22b",
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
        if model.startswith("gpt"):
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
        elif model.startswith("anthropic"):
            if (
                os.getenv("ANTHROPIC_KEY", None) is not None
                and original_use_openrouter == "False"
            ):
                print(
                    "ANTHROPIC_KEY is set, OpenRouter API is not, setting USE_OPENROUTER to False"
                )
                os.environ["USE_OPENROUTER"] = "False"
            else:
                print("Using OpenRouter API for the Anthropic test call")
                os.environ["USE_OPENROUTER"] = "True"
        else:
            os.environ["USE_OPENROUTER"] = "True"
            # check if openrouter_api_key is set
            if not os.getenv("OPENROUTER_API_KEY", None):
                print(
                    f"OPENROUTER_API_KEY is not available, the test on model {model} will not run"
                )
                return

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
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        # this is empty now until we find some model we know doesn't work
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
async def test_answer_async():
    example_prompt = "Generate a sample forecasting question"
    result = await answer(
        prompt=example_prompt,
        preface=None,
        model="gpt-4o-mini-2024-07-18",
        response_model=PlainText,
    )
    assert isinstance(result, PlainText) and len(result.text) > 0
    print(f"Generated question: {result.text}")


def test_answer_native_sync():
    example_prompt = "Generate a sample forecasting question"
    result = answer_native_sync(
        prompt=example_prompt,
        preface=None,
        model="gpt-4o-mini-2024-07-18",
    )
    assert isinstance(result, str) and len(result) > 0
    print(f"Generated question: {result}")


@pytest.mark.asyncio
async def test_answer_native():
    example_prompt = "Generate a sample forecasting question"
    result = await answer_native(
        prompt=example_prompt,
        preface=None,
        model="gpt-4o-mini-2024-07-18",
    )
    assert isinstance(result, str) and len(result) > 0
    print(f"Generated question: {result}")


class TestResponse(BaseModel):
    name: str = Field(..., description="The name of the person")
    age: int = Field(..., description="The age of the person")


register_model_for_cache(TestResponse)


def test_openai_json_strict(monkeypatch):
    # Set up the environment
    # remember the original value of OPENAI_JSON_STRICT
    original_openai_json_strict = os.getenv("OPENAI_JSON_STRICT", "False")
    monkeypatch.setenv("OPENAI_JSON_STRICT", "True")
    monkeypatch.setenv(
        "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")
    )  # Ensure API key is set

    # Define the prompt
    prompt = "Generate a person with a name and age."

    # Test with strict JSON mode
    response = answer_sync(
        prompt=prompt, model="gpt-4o-mini-2024-07-18", response_model=TestResponse
    )
    assert isinstance(response, TestResponse)
    assert isinstance(response.name, str)
    assert isinstance(response.age, int)

    response = answer_sync(
        prompt=prompt, model="gpt-4o-mini-2024-07-18", response_model=PlainText
    )
    assert isinstance(response, PlainText)
    assert isinstance(response.text, str)

    # restore the original value of OPENAI_JSON_STRICT
    os.environ["OPENAI_JSON_STRICT"] = original_openai_json_strict


class IntResponseModel(BaseModel):
    reasoning: str
    answer_polynomial: int


register_model_for_cache(IntResponseModel)


# New tests for query_api_chat_with_parsing
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "meta-llama/llama-3-405b-instruct",
        "anthropic/claude-3.5-sonnet",
    ],
)
async def test_query_api_chat_with_parsing(model):
    original_use_openrouter = os.getenv("USE_OPENROUTER", "False")
    os.environ["USE_OPENROUTER"] = "True"

    messages = [
        {
            "role": "user",
            "content": "Using Fermat's theorem, find the remainder of 3^47 when it is divided by 23. Think step by step.",
        }
    ]

    response_model = IntResponseModel

    response = await query_api_chat_with_parsing(
        messages=messages,
        response_model=response_model,
        model=model,
        parsing_model="gpt-4o-mini-2024-07-18",
    )
    assert response is not None
    print(response)
    assert isinstance(response, response_model)
    assert response.answer_polynomial == (3**47) % 23
    os.environ["USE_OPENROUTER"] = original_use_openrouter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "intended_provider"),
    [
        ("gpt-4o-mini-2024-07-18", "openai"),
        ("gpt-4o-mini-2024-07-18", "openai_strict"),
        ("o1-mini-2024-09-12", "openai_o1"),
    ],
)
async def test_openai_instructor(model, intended_provider):
    """
    Test all ways we interact with the OpenAI API using Instructor.
    """
    prompt = "John Doe is 25 years old."
    response_model = UserInfo

    # Save the original USE_OPENROUTER setting
    original_use_openrouter = os.getenv("USE_OPENROUTER", "False")
    os.environ["USE_OPENROUTER"] = "False"
    original_openai_json_strict = os.getenv("OPENAI_JSON_STRICT", "False")

    try:
        match intended_provider:
            case "openai":
                pass
            case "openai_strict":
                # This will work even if OPENAI_JSON_STRICT raises an error at the start of llm_utils.py, because we don't reload the llm_utils module in this test
                os.environ["OPENAI_JSON_STRICT"] = "True"
            case "openai_o1":
                if os.getenv("ALLOW_OPENAI_O1", "False") == "True":
                    pass
                else:
                    print(
                        "OPENAI_O1 is not allowed, skipping the test for model: {model}"
                    )
                    return
            case _:
                raise ValueError(f"Invalid provider: {intended_provider}")

        response = await answer(prompt, model=model, response_model=response_model)
        assert response is not None
        assert isinstance(response, UserInfo)
        assert response.name == "John Doe"
        assert response.age == 25
    finally:
        # Restore the original USE_OPENROUTER setting
        os.environ["USE_OPENROUTER"] = original_use_openrouter
        os.environ["OPENAI_JSON_STRICT"] = original_openai_json_strict
