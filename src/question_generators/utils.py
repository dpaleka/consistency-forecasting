import asyncio
from scipy.spatial.distance import cosine
from common.llm_utils import get_embedding, query_api_chat_sync
from common.datatypes import ForecastingQuestion


def cosine_similarity(
    query_embedding: list[float], embeddings: list[list[float]]
) -> list[list[float]]:
    distances = [cosine(query_embedding, embedding) for embedding in embeddings]
    return distances


async def get_distances(questions):
    embeddings = await asyncio.gather(*[get_embedding(q.title) for q in questions])
    distances = [cosine_similarity(e, embeddings) for e in embeddings]
    return distances


async def get_similar_questions(old_questions, new_questions, threshold: float = 0.1):
    old_embeddings = await asyncio.gather(
        *[get_embedding(q.title) for q in old_questions]
    )
    new_embeddings = await asyncio.gather(
        *[get_embedding(q.title) for q in new_questions]
    )
    distances = [cosine_similarity(e, old_embeddings) for e in new_embeddings]
    similar_questions = [
        [(i, sim)] for d in distances for i, sim in enumerate(d) if sim < threshold
    ]
    return similar_questions


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


def generate_questions(
    model: str, prompt: str, examples: list[str] = None
) -> list[ForecastingQuestion]:
    examples = examples or []
    messages = [
        {"role": "user", "content": prompt},
        {"role": "user", "content": "\n".join(examples)},
    ]
    response = query_api_chat_sync(
        model=model, messages=messages, response_model=list[ForecastingQuestion]
    )
    return response


def generate_questions_batched(
    question_examples: list[ForecastingQuestion],
    questions_per_batch: int = 1,
    num_batches: int = 1,
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
