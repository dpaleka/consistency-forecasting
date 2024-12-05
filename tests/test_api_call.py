import pytest
import os
import vcr
import re
import asyncio
from contextlib import asynccontextmanager
from openai import APIStatusError
from common.datatypes import PlainText
from common.perscache import register_model_for_cache
from common.llm_utils import (
    answer,
    answer_sync,
    answer_native,
    answer_native_sync,
    query_api_chat_with_parsing,
)
from pydantic import BaseModel, Field

def sanitize_filename(s):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', s)

# Create an async context manager wrapper for VCR
@asynccontextmanager
async def async_vcr(cassette_name):
    with vcr.use_cassette(cassette_name):
        yield

# Configure VCR
vcr = vcr.VCR(
    cassette_library_dir='fixtures/vcr_cassettes',
    record_mode='once',
    match_on=['uri', 'method', 'body'],
    filter_headers=['authorization', 'x-api-key'],
    filter_query_parameters=['api_key', 'token'],
)

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
    cassette_name = f'fixtures/vcr_cassettes/test_answer_real_api_{sanitize_filename(model)}.yaml'
    async with async_vcr(cassette_name):
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
                # Check if OPENROUTER_API_KEY is set
                if not os.getenv("OPENROUTER_API_KEY", None):
                    print(
                        f"OPENROUTER_API_KEY is not available, the test on model {model} will not run"
                    )
                    return

            response = await answer(prompt, model=model, response_model=UserInfo)
            print(response)
            assert response is not None
            # Assert response is of type UserInfo
            assert isinstance(response, UserInfo)
            assert response.name == "John Doe"
            assert response.age == 25
        finally:
            os.environ["USE_OPENROUTER"] = original_use_openrouter

@vcr.use_cassette()
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
@vcr.use_cassette()
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

@vcr.use_cassette()
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
@vcr.use_cassette()
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

@vcr.use_cassette()
def test_openai_json_strict(monkeypatch):
    # Set up the environment
    # Remember the original value of OPENAI_JSON_STRICT
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

    # Restore the original value of OPENAI_JSON_STRICT
    os.environ["OPENAI_JSON_STRICT"] = original_openai_json_strict

class IntResponseModel(BaseModel):
    reasoning: str
    answer_polynomial: int

register_model_for_cache(IntResponseModel)

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        pytest.param("meta-llama/llama-3-405b-instruct", marks=pytest.mark.flaky(reruns=2)),
        "anthropic/claude-3.5-sonnet",
    ],
)
async def test_query_api_chat_with_parsing(model):
    cassette_name = f'fixtures/vcr_cassettes/test_query_api_chat_with_parsing_{sanitize_filename(model)}.yaml'
    original_use_openrouter = os.getenv("USE_OPENROUTER", "False")
    os.environ["USE_OPENROUTER"] = "True"

    messages = [
        {
            "role": "user",
            "content": "Using Fermat's theorem, find the remainder of 3^47 when it is divided by 23. Think step by step.",
        }
    ]

    response_model = IntResponseModel
    max_retries = 3
    initial_delay = 1

    try:
        for attempt in range(max_retries):
            try:
                # VCR context inside the retry loop
                async with async_vcr(cassette_name):
                    response = await query_api_chat_with_parsing(
                        messages=messages,
                        response_model=response_model,
                        model=model,
                        parsing_model="gpt-4o-mini-2024-07-18",
                    )
                    
                    assert response is not None
                    print(response)
                    assert isinstance(response, response_model)
                    assert response.answer_polynomial == pow(3, 47, 23)
                    break  # Success, exit retry loop
                    
            except APIStatusError as e:
                if e.status_code == 408:  # Timeout error
                    if attempt == max_retries - 1:
                        raise  # Re-raise if this was our last attempt
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Request timed out. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    raise  # Re-raise if it's not a timeout error
    finally:
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
    cassette_name = f'fixtures/vcr_cassettes/test_openai_instructor_{sanitize_filename(model)}_{intended_provider}.yaml'
    async with async_vcr(cassette_name):
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
            if intended_provider == "openai":
                pass
            elif intended_provider == "openai_strict":
                os.environ["OPENAI_JSON_STRICT"] = "True"
            elif intended_provider == "openai_o1":
                if os.getenv("ALLOW_OPENAI_O1", "False") != "True":
                    print(
                        f"OPENAI_O1 is not allowed, skipping the test for model: {model}"
                    )
                    return
            else:
                raise ValueError(f"Invalid provider: {intended_provider}")

            response = await answer(prompt, model=model, response_model=response_model)
            assert response is not None
            assert isinstance(response, UserInfo)
            assert response.name == "John Doe"
            assert response.age == 25
        finally:
            # Restore the original environment variables
            os.environ["USE_OPENROUTER"] = original_use_openrouter
            os.environ["OPENAI_JSON_STRICT"] = original_openai_json_strict
