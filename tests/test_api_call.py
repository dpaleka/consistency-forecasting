import pytest
import os
from common.llm_utils import (
    answer,
)  # Adjust the import based on your script's structure
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str
    age: int


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["gpt-3.5-turbo", "meta-llama/llama-3-8b-instruct:nitro", "claude-3-opus-20240229"],
)
async def test_answer_real_api(model):
    prompt = "John Doe is 25 years old."

    # Save the original value of the environment variable
    original_use_openrouter = os.getenv("USE_OPENROUTER", "False")

    try:
        # Set the environment variable based on the model
        if "meta-llama" in model:
            os.environ["USE_OPENROUTER"] = "True"
        else:
            os.environ["USE_OPENROUTER"] = "False"

        print(f"Testing model: {model}")
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
        "anthropic/claude-3-haiku",
        "mistralai/mistral-7b-instruct",
        "microsoft/wizardlm-2-8x22b",
    ],
)
async def test_answer_fails_openrouter(model):
    prompt = "John Doe is 25 years old."
    original_use_openrouter = os.getenv("USE_OPENROUTER", "False")
    try:
        os.environ["USE_OPENROUTER"] = "True"
        # we want an error here. if we get an error, it's ok. otherwise, fail
        with pytest.raises(Exception):
            response = await answer(prompt, model=model, response_model=UserInfo)
    finally:
        os.environ["USE_OPENROUTER"] = original_use_openrouter