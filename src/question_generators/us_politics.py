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
from common.llm_utils import query_api_chat_sync, get_client_pydantic

MODEL = "gpt-3.5-turbo"
client, _ = get_client_pydantic(MODEL, use_async=False)


def generate_questions(num_questions: int = 5, examples: list[str] = None) -> list[str]:
    examples = examples or []
    messages = [
        {"role": "user", "content": instruction.format(num_examples=num_questions)},
        {"role": "user", "content": "\n".join(examples)},
    ]
    response = query_api_chat_sync(
        model=MODEL, messages=messages, response_model=ForecastingQuestions
    )
    return response


# %%
question_examples_json = [e.model_dump_json(exclude=["id"]) for e in question_examples]
print(f"\033[91m{question_examples_json=}\033[0m\n")

# %%
response = generate_questions(5, question_examples_json)
# black bold
json_response = response.model_dump_json()
print(f"\n\033[1;30m{json_response}\033[0m\n")


# %%
def write_questions(questions: list[ForecastingQuestion], file_name: str):
    with open(DATA_PATH / "fq" / "synthetic" / file_name, "w") as f:
        for q in questions:
            f.write(f"{q.model_dump_json()}\n")


# %%
def generate_questions_batched(
    questions_per_batch: int = 1, num_batches: int = 1
) -> list[ForecastingQuestion]:
    questions_json = [e.model_dump_json(exclude=["id"]) for e in question_examples]
    questions = []
    for _ in range(num_batches):
        try:
            response = generate_questions(questions_per_batch, questions_json)
        except Exception as e:
            print(e)
            break
        questions_json += [
            q.model_dump_json(exclude=["id"]) for q in response.questions
        ]
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
# Remove duplicates with cosine similarity of embeddings
from scipy.spatial.distance import cosine


def cosine_similarity(
    query_embedding: list[float], embeddings: list[list[float]]
) -> list[list]:
    distances = [cosine(query_embedding, embedding) for embedding in embeddings]
    return distances


# %%
EMBEDDING_MODEL = "text-embedding-3-small"
async_client, _ = get_client_pydantic(MODEL, use_async=True)


async def get_embedding(text: str) -> list[float]:
    response = await async_client.client.embeddings.create(
        input=text, model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


# %%
import asyncio


async def get_distances(embeddings):
    embeddings = await asyncio.gather(*[get_embedding(q.title) for q in questions])
    distances = [cosine_similarity(e, embeddings) for e in embeddings]
    return distances


async def deduplicate(questions):
    distances = await get_distances(questions)
    ids_to_remove = set()

    for i, dist in enumerate(distances):
        # Find indices of similar questions based on a threshold distance
        similar_indices = {
            j for j, distance in enumerate(dist) if i != j and distance < 0.1
        }
        # Only update ids_to_remove if i is not already set to be removed
        if i not in ids_to_remove:
            ids_to_remove.update(similar_indices)

    # Filter questions by removing those with indices in ids_to_remove
    filtered_questions = [
        question for idx, question in enumerate(questions) if idx not in ids_to_remove
    ]
    return filtered_questions


# deduped_questions = await deduplicate(questions)
# write_questions(deduped_questions, "politics_qs_3_deduped.jsonl")
