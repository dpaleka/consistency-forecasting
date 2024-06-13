import pytest
import asyncio
import os
from common.llm_utils import answer  # Adjust the import based on your script's structure
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

@pytest.mark.asyncio
@pytest.mark.parametrize("model", [
    "gpt-3.5-turbo",
    "meta-llama/llama-3-8b-instruct:nitro",
    "claude-3-opus-20240229"
])
async def test_answer_real_api(model):
    # Define your inputs
    prompt = "John Doe is 25 years old."

    # Save the original value of the environment variable
    original_use_openrouter = os.getenv("USE_OPENROUTER","False")

    try:
        # Set the environment variable based on the model
        if "meta-llama" in model:
            os.environ["USE_OPENROUTER"] = "True"
        else:
            os.environ["USE_OPENROUTER"] = "False"

        print(f"Testing model: {model}")
        response = await answer(prompt, model=model, response_model=UserInfo)  # Assuming answer takes model and preface as arguments
        print(response)
        assert response is not None
        # assert response is of type UserInfo
        assert isinstance(response, UserInfo)
        assert response.name == "John Doe"
        assert response.age == 25
    finally:
        os.environ["USE_OPENROUTER"] = original_use_openrouter
