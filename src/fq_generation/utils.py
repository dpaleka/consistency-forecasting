import asyncio
from scipy.spatial.distance import cosine
from common.llm_utils import get_embedding, query_api_chat_sync
from common.datatypes import ForecastingQuestion, ForecastingQuestions

from pydantic import BaseModel, validator


class InstructionForQuestions(BaseModel):
    prompt: str

    @validator("prompt")
    def validate_placeholder(cls, value):
        if "{num_questions}" not in value:
            raise ValueError(
                "The string must contain the '{num_questions}' placeholder."
            )
        try:
            value.format(num_questions=5)
        except KeyError:
            raise ValueError(
                "The string must be formattable with 'num_questions' as an integer."
            )
        return value


def cosine_similarity(
    query_embedding: list[float], embeddings: list[list[float]]
) -> list[list[float]]:
    distances = [cosine(query_embedding, embedding) for embedding in embeddings]
    return distances


async def get_distances(questions, embedding_model: str = "text-embedding-3-small"):
    embeddings = await asyncio.gather(
        *[get_embedding(q.title, embedding_model=embedding_model) for q in questions]
    )
    distances = [cosine_similarity(e, embeddings) for e in embeddings]
    return distances


async def get_similar_questions(
    old_questions,
    new_questions,
    threshold: float = 0.1,
    embedding_model: str = "text-embedding-3-small",
):
    old_embeddings = await asyncio.gather(
        *[
            get_embedding(q.title, embedding_model=embedding_model)
            for q in old_questions
        ]
    )
    new_embeddings = await asyncio.gather(
        *[
            get_embedding(q.title, embedding_model=embedding_model)
            for q in new_questions
        ]
    )
    distances = [cosine_similarity(e, old_embeddings) for e in new_embeddings]
    similar_questions = [
        [(i, sim)] for d in distances for i, sim in enumerate(d) if sim < threshold
    ]
    return similar_questions


async def deduplicate(questions, embedding_model: str = "text-embedding-3-small"):
    print(f"In deduplicate, type of questions is {type(questions[0])}")
    distances = await get_distances(questions, embedding_model=embedding_model)
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


def generate_questions(
    model: str,
    instruction: InstructionForQuestions,
    num_questions: int = 5,
    examples: list[str] = None,
) -> list[ForecastingQuestion]:
    examples = examples or []
    messages = [
        {
            "role": "system",
            "content": instruction.prompt.format(num_questions=num_questions),
        },
        {"role": "user", "content": "\n".join(examples)},
    ]
    response = query_api_chat_sync(
        model=model, messages=messages, response_model=ForecastingQuestions
    )
    return response


async def generate_questions_async(
    model: str,
    instruction: InstructionForQuestions,
    num_questions: int = 5,
    examples: list[str] = None,
):
    raise NotImplementedError


def generate_questions_batched(
    model: str,
    instruction: InstructionForQuestions,
    question_examples: list[ForecastingQuestion],
    questions_per_batch: int = 1,
    num_batches: int = 1,
    verbose: bool = False,
) -> list[ForecastingQuestion]:
    questions_json = [e.model_dump_json(exclude=["id"]) for e in question_examples]
    questions = []
    for batch_id in range(num_batches):
        if verbose:
            print(f"Generating batch {batch_id + 1} of {num_batches}")
        try:
            response = generate_questions(
                model, instruction, questions_per_batch, questions_json
            )
            if verbose:
                print(f"Generated {len(response.questions)} questions")
                print(response.questions)
        except Exception as e:
            print(e)
            break
        questions_json += [
            q.model_dump_json(exclude=["id"]) for q in response.questions
        ]
        questions += response.questions
    return questions
