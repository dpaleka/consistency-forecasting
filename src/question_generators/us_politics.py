# From https://arxiv.org/pdf/2310.20111
# %%
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv()


# %%
from pathlib import Path
import json
from common.datatypes import ForecastingQuestion, ForecastingQuestions


def load_questions(path: str) -> list[ForecastingQuestion]:
    with open(path, "r") as f:
        jsonl_content = f.read()
    return [
        ForecastingQuestion(**json.loads(jline)) for jline in jsonl_content.splitlines()
    ]


DATA_PATH = Path(__file__).parent.parent / "data"
question_examples = load_questions(
    DATA_PATH / "fq" / "synthetic" / "politics_qs_2_formatted.jsonl"
)
question_examples


# %%
instruction = """
    - You are creating {num_examples} examples that follow the format of the example(s) provided, but with different content.
    - The created examples **must** all be about US politics.
    - The created examples **must** all have different answers.
    - The output **must** be in unnumbered JSON format.
"""
instruction.format(num_examples=5)


# %%
from common.llm_utils import query_api_chat_sync, get_client

MODEL = "gpt-3.5-turbo"
client, _ = get_client(MODEL, use_async=False)


def generate_questions(
    num_questions: int = 5, examples: list[dict] = None
) -> list[str]:
    examples = examples or []
    messages = [
        {"role": "user", "content": instruction.format(num_examples=num_questions)},
        {"role": "user", "content": str(examples)},
    ]
    response = query_api_chat_sync(
        model=MODEL, messages=messages, response_model=ForecastingQuestions
    )
    return response


# %%
response = generate_questions(5, [e.model_dump_json() for e in question_examples])
response


# %%
def write_questions(questions: list[ForecastingQuestion], file_name: str):
    with open(DATA_PATH / "fq" / "synthetic" / file_name, "w") as f:
        for q in questions:
            f.write(f"{q.model_dump_json()}\n")


# %%
def generate_questions_batched(
    questions_per_batch: int = 1, num_batches: int = 1
) -> list[ForecastingQuestion]:
    questions_json = [e.model_dump_json() for e in question_examples]
    questions = []
    for _ in range(num_batches):
        try:
            response = generate_questions(questions_per_batch, questions_json)
        except Exception as e:
            print(e)
            break
        questions_json += [q.model_dump_json() for q in response.questions]
        questions += response.questions
    return questions


# %%
questions = generate_questions_batched(questions_per_batch=5, num_batches=20)
questions

# %%
write_questions(questions, "politics_qs_3.jsonl")


# %%
questions = load_questions(DATA_PATH / "fq" / "synthetic" / "politics_qs_3.jsonl")

# %%
from collections import defaultdict

grouped_questions = defaultdict(list)
for q in questions:
    grouped_questions[q.title].append(q)

questions = [grouped_questions[k][0] for k in grouped_questions]
write_questions(questions, "politics_qs_3_deduped.jsonl")
