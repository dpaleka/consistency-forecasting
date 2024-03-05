# LLM Utils
# Run tests in this file with python -m dpy.llm_utils

# %%
import os
from openai import AsyncOpenAI, OpenAI
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
import asyncio
from dotenv import load_dotenv
from typing import Union, Tuple
from mistralai.models.chat_completion import ChatMessage

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

def singleton_constructor(get_instance_func):
    instances = {}
    def wrapper(*args, **kwargs):
        if get_instance_func not in instances:
            instances[get_instance_func] = get_instance_func(*args, **kwargs)
        return instances[get_instance_func]
    return wrapper

@singleton_constructor
def get_async_openai_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return AsyncOpenAI(api_key=api_key)

@singleton_constructor
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

@singleton_constructor
def get_mistral_async_client() -> MistralAsyncClient:
    api_key = os.getenv("MISTRAL_API_KEY")
    return MistralAsyncClient(api_key=api_key)

@singleton_constructor
def get_mistral_client() -> MistralClient:
    api_key = os.getenv("MISTRAL_API_KEY")
    return MistralClient(api_key=api_key)

@singleton_constructor
def get_togetherai_client() -> OpenAI:
    url = "https://api.together.xyz/v1"
    api_key = os.getenv("TOGETHER_API_KEY")
    return OpenAI(api_key=api_key, base_url=url)


def is_openai(model: str) -> bool:
    keywords = [
        "gpt-4",
        "gpt-3.5-turbo",
        "ada",
        "babbage",
        "curie",
        "davinci",
        "instruct",
        "openai",
        "opem-ai"
    ]
    return any(keyword in model for keyword in keywords)

def is_mistral(model: str) -> bool:
    if model.startswith("mistral"):
        return True

def is_togetherai(model: str) -> bool:
    keywords = ["together","llama","phi","orca"]
    return any(keyword in model for keyword in keywords)

def get_client(model: str, use_async=True) -> Tuple[Union[AsyncOpenAI, OpenAI, MistralAsyncClient, MistralClient], str]:
    if is_openai(model):
        return (get_async_openai_client() if use_async else get_openai_client(), "openai")
    elif is_mistral(model):
        api_key = os.getenv("MISTRAL_API_KEY")
        return (get_mistral_async_client() if use_async else get_mistral_client(), "mistral")
    elif is_togetherai(model):
        if use_async:
            raise NotImplementedError("Only synchronous calls are supported for TogetherAI")
        url = "https://api.together.xyz/v1"
        api_key = os.getenv("TOGETHER_API_KEY")
        return (get_togetherai_client(), "togetherai") 
    else:
        raise NotImplementedError(f"Model {model} is not supported for now")

def is_llama2_tokenized(model: str) -> bool:
    keywords = ["Llama-2", "pythia"]
    return any(keyword in model for keyword in keywords)

def _mistral_message_transform(messages):
    mistral_messages = []
    for message in messages:
        mistral_message = ChatMessage(
            role=message["role"], content=message["content"])
        mistral_messages.append(mistral_message)
    return mistral_messages


@cache
async def query_api_chat(model: str, messages: list[dict[str, str]], verbose=False, **kwargs) -> dict:
    client, client_name = get_client(model, use_async=True)
    if client_name == "mistral":
        messages = _mistral_message_transform(messages)
        response = await client.chat(model=model, messages=messages, **kwargs)
    else:
        response = await client.chat.completions.create(model=model, messages=messages, **kwargs)
    response_text = response.choices[0].message.content
    if verbose:
        print("Text:", messages[1]["content"][:30], "\nResponse:", response_text[:30])
    return response_text

def query_api_chat_sync(model: str, messages: list[dict[str, str]], verbose=False, **kwargs) -> dict:
    client, client_name = get_client(model, use_async=False)
    if client_name == "mistral":
        messages = _mistral_message_transform(messages)
        response = client.chat(model=model, messages=messages, **kwargs)
    else:
        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    response_text = response.choices[0].message.content
    if verbose:
        print("Text:", messages[-1]["content"], "\nResponse:", response_text)
    return response_text

@cache
async def query_api_text(model: str, text: str, verbose=False, **kwargs) -> str:
    client, client_name = get_client(model, use_async=True)
    if verbose:
        print("Querying API with text:", text[:30])
    response = await client.completions.create(model=model, prompt=text, **kwargs)
    response_text = response.choices[0].text
    if verbose:
        print("Text:", text[:30], "\nResponse:", response_text[:30])
    return response_text

def query_api_text_sync(model: str, text: str, verbose=False, **kwargs) -> str:
    client, client_name = get_client(model, use_async=False)
    if verbose:
        print("Querying API with text:", text[:30])
    response = client.completions.create(model=model, prompt=text, **kwargs)
    response_text = response.choices[0].text
    if verbose:
        print("Text:", text, "\nResponse:", response_text)
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
            return await func(datapoint)

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
