# LLM Utils
# Run tests in this file with python -m dpy.llm_utils

# %%
import os
from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv

from .perscache import (
    Cache,
    JSONSerializer,
    RedisStorage,
    LocalFileStorage,
    ValueWrapperDictInspectArgs,
)  # If no redis, use LocalFileStorage

CACHE_FLAGS = ["NO_CACHE", "NO_READ_CACHE", "NO_WRITE_CACHE", "LOCAL_CACHE"]
cache = Cache(
    serializer=JSONSerializer(),
    storage=LocalFileStorage()
    if os.getenv("LOCAL_CACHE")
    else RedisStorage(namespace="llm_utils"),
    value_wrapper=ValueWrapperDictInspectArgs(),
)

FLAGS = CACHE_FLAGS + ["SINGLE_THREAD"]

client = None
load_dotenv(override=False)


def is_openai(model: str) -> bool:
    keywords = [
        "gpt-4",
        "gpt-3.5-turbo",
        "ada",
        "babbage",
        "curie",
        "davinci",
        "instruct",
    ]
    return any(keyword in model for keyword in keywords)


def is_llama2_tokenized(model: str) -> bool:
    keywords = ["Llama-2", "pythia"]
    return any(keyword in model for keyword in keywords)


# run with export OPENAI_LOG=debug in case you need to debug
def init_client(model: str):
    global client
    if is_openai(model):
        api_key = os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=api_key)
    else:
        raise NotImplementedError("Only OpenAI is supported for now")


@cache
async def query_api_chat(model: str, messages: list[dict[str, str]], **kwargs) -> dict:
    global client
    if client is None:
        init_client(model)
    response = await client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    response_text = response.choices[0].message.content
    print("Text:", messages[1]["content"][:30], "\nResponse:", response_text[:30])
    return response_text


@cache
async def query_api_text(model: str, text: str, **kwargs) -> str:
    global client
    if client is None:
        init_client(model)
    print("Querying API with text:", text[:30])
    response = await client.completions.create(model=model, prompt=text, **kwargs)
    response_text = response.choices[0].text
    print("Text:", text[:30], "\nResponse:", response_text[:30])
    return response_text


async def parallelized_call(
    func: callable,
    data=list[str],
    max_concurrent_queries: int = 100,
) -> list[dict]:
    """
    Run func in parallel on the given data.
    func will usually be a partial which uses query_api or whatever in some way.

    Example usage:
        partial_eval_method = functools.partial(eval_method, model=model, **kwargs)
        results = await parallelized_call(partial_eval_method, [format_post(d) for d in data])
    """
    print(f"Running {func} on {len(data)} datapoints")

    if os.getenv("SINGLE_THREAD"):
        return [await func(text=d) for d in data]

    sem = asyncio.Semaphore(max_concurrent_queries)

    async def call_func(sem, func, datapoint):
        async with sem:
            return await func(text=datapoint)

    tasks = [call_func(sem, func, d) for d in data]
    return await asyncio.gather(*tasks)


# %%
def get_all_cached_requests():
    all_cached = cache.storage.get_all(namespace="llm_utils")
    for key, value in all_cached.items():
        print(key, value.decode("utf-8"))
        break
    # connect to redis for any other operations
    import redis
    from .perscache import REDIS_CONFIG_DEFAULT

    r = redis.StrictRedis(**REDIS_CONFIG_DEFAULT)
    keys = r.keys("llm_utils:*")
    print(keys)


if __name__ == "__main__":
    get_all_cached_requests()


# %%
