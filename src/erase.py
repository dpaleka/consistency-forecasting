from openai import OpenAI
import instructor
from typing import Iterable
from pydantic import BaseModel, Field, ConfigDict
import os
from dotenv import load_dotenv

load_dotenv(override=False)

api_key = os.getenv("OPENAI_API_KEY")

class UserInfo(BaseModel):
    name: str
    age: int


# Patch the OpenAI client
client = instructor.from_openai(OpenAI(api_key=api_key))    

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)

print(user_info.name)
#> John Doe
print(user_info.age)
#> 30