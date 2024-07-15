from openai import OpenAI
from os import getenv
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel

load_dotenv(override=False)
api_key = getenv("OPENAI_API_KEY")

# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int



client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
)
client = instructor.from_openai(client)
#client = instructor.from_openai(OpenAI())

user_info = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)

print(user_info.name)
#> John Doe
print(user_info.age)