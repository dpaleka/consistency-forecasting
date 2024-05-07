# LLM Utils
# Run tests in this file with python -m dpy.llm_utils

# %%
import os
from openai import AsyncOpenAI, OpenAI
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
import instructor
import asyncio
from pydantic import BaseModel
from dataclasses import dataclass
from dotenv import load_dotenv
from mistralai.models.chat_completion import ChatMessage
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers

from .datatypes import PlainText
from .path_utils import get_src_path


from .perscache import (
    Cache,
    JSONPydanticResponseSerializer,
    RedisStorage,
    LocalFileStorage,
    ValueWrapperDictInspectArgs,
)  # If no redis, use LocalFileStorage

CACHE_FLAGS = ["NO_CACHE", "NO_READ_CACHE", "NO_WRITE_CACHE", "LOCAL_CACHE"]

pydantic_cache = Cache(
    serializer=JSONPydanticResponseSerializer(),
    storage=(
        LocalFileStorage(location=get_src_path().parent / os.getenv("LOCAL_CACHE"))
        if os.getenv("LOCAL_CACHE")
        else RedisStorage(namespace="llm_utils")
    ),
    value_wrapper=ValueWrapperDictInspectArgs(),
)

FLAGS = CACHE_FLAGS + ["SINGLE_THREAD"]

client = None
load_dotenv(override=True)


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
    _client = AsyncOpenAI(api_key=api_key)
    client = instructor.from_openai(_client)
    return client


@singleton_constructor
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    _client = OpenAI(api_key=api_key)
    client = instructor.from_openai(_client)
    return client


@singleton_constructor
def get_async_openrouter_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    client = instructor.from_openai(_client)
    return client


@singleton_constructor
def get_openrouter_client() -> OpenAI:
    print("\033[Calling models through OpenRouter\033[0m")
    _client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    client = instructor.from_openai(_client)
    return client


@singleton_constructor
def get_mistral_async_client() -> MistralAsyncClient:
    api_key = os.getenv("MISTRAL_API_KEY")
    _client = MistralAsyncClient(api_key=api_key)
    client = instructor.from_openai(
        create=_client.chat, mode=instructor.Mode.MISTRAL_TOOLS
    )
    return client


@singleton_constructor
def get_mistral_client() -> MistralClient:
    api_key = os.getenv("MISTRAL_API_KEY")
    _client = MistralClient(api_key=api_key)
    client = instructor.from_openai(
        create=_client.chat, mode=instructor.Mode.MISTRAL_TOOLS
    )
    return client


@singleton_constructor
def get_togetherai_client() -> OpenAI:
    url = "https://api.together.xyz/v1"
    api_key = os.getenv("TOGETHER_API_KEY")
    _client = OpenAI(api_key=api_key, base_url=url)
    client = instructor.from_openai(_client)
    return client


@singleton_constructor
def get_huggingface_local_client(hf_repo) -> transformers.pipeline:
    hf_model_path = os.path.join(os.getenv("HF_MODELS_DIR"), hf_repo)
    if not os.path.exists(hf_model_path):
        snapshot_download(hf_repo, local_dir=hf_model_path)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_path)
    pipeline = transformers.pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048
    )
    return pipeline


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
        "open-ai",
    ]
    return any(keyword in model for keyword in keywords)


def is_mistral(model: str) -> bool:
    if model.startswith("mistral"):
        return True


def is_togetherai(model: str) -> bool:
    keywords = ["together", "llama", "phi", "orca"]
    return any(keyword in model for keyword in keywords)


def is_huggingface_local(model: str) -> bool:
    keywords = ["huggingface", "hf"]
    return any(keyword in model for keyword in keywords)


def get_client(
    model: str, use_async=True
) -> tuple[AsyncOpenAI | OpenAI | MistralAsyncClient | MistralClient, str]:
    if os.getenv("USE_OPENROUTER"):
        return (
            get_async_openrouter_client() if use_async else get_openrouter_client(),
            "openrouter",
        )
    elif is_openai(model):
        return (
            get_async_openai_client() if use_async else get_openai_client(),
            "openai",
        )
    elif is_mistral(model):
        api_key = os.getenv("MISTRAL_API_KEY")
        return (
            get_mistral_async_client() if use_async else get_mistral_client(),
            "mistral",
        )
    elif is_togetherai(model):
        if use_async:
            raise NotImplementedError(
                "Only synchronous calls are supported for TogetherAI"
            )
        url = "https://api.together.xyz/v1"
        api_key = os.getenv("TOGETHER_API_KEY")
        return (get_togetherai_client(), "togetherai")
    elif is_huggingface_local(model):
        assert model.startswith("hf:")
        model = model.split("hf:")[1]
        return (get_huggingface_local_client(model), "huggingface_local")
    else:
        raise NotImplementedError(f"Model {model} is not supported for now")


def is_llama2_tokenized(model: str) -> bool:
    keywords = ["Llama-2", "pythia"]
    return any(keyword in model for keyword in keywords)


def _mistral_message_transform(messages):
    mistral_messages = []
    for message in messages:
        mistral_message = ChatMessage(role=message["role"], content=message["content"])
        mistral_messages.append(mistral_message)
    return mistral_messages


@pydantic_cache
async def query_api_chat(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> BaseModel:
    """
    Query the API (through instructor.Instructor) with the given messages.

    Order of precedence for model:
    1. `model` argument
    2. `model` in `kwargs`
    3. Default model
    """
    default_options = {
        "model": "gpt-4-1106-preview",
        "response_model": PlainText,
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]
    client, client_name = get_client(options["model"], use_async=True)
    if client_name == "mistral":
        messages = _mistral_message_transform(messages)

    if options.get("n", 1) != 1:
        raise NotImplementedError("Multiple queries not supported yet")

    response = await client.chat.completions.create(
        messages=messages,
        **options,
    )
    if verbose:
        print(f"...\nText: {messages[-1]['content']}\nResponse: {response}")
    return response


@pydantic_cache
def query_api_chat_sync(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> BaseModel:
    if verbose:
        print("Messages array being sent to OpenAI API:", messages, "\n")

    default_options = {
        "model": "gpt-4-1106-preview",
        "response_model": PlainText,
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]
    client, client_name = get_client(options["model"], use_async=False)
    if client_name == "mistral":
        messages = _mistral_message_transform(messages)

    if options.get("n", 1) != 1:
        raise NotImplementedError("Multiple structured queries not supported yet")

    response = client.chat.completions.create(
        messages=messages,
        **options,
    )
    if verbose:
        print(f"...\nText: {messages[-1]['content']}\nResponse: {response}")
    return response


@dataclass
class Example:
    user: str | BaseModel
    assistant: str | BaseModel


def prepare_messages(
    prompt: str, preface: str | None = None, examples: list[Example] | None = None
) -> list[dict[str, str]]:
    preface = preface or "You are a helpful assistant."
    examples = examples or []
    messages = [{"role": "system", "content": preface}]
    for example in examples:
        if isinstance(example.user, BaseModel):
            example.user = example.user.model_dump_json()
        if isinstance(example.assistant, BaseModel):
            example.assistant = example.assistant.model_dump_json()
        messages.append({"role": "user", "content": example.user})
        # Convert assistant's response to string if it's not already
        assistant_content = (
            str(example.assistant)
            if isinstance(example.assistant, (float, int))
            else example.assistant
        )
        messages.append({"role": "assistant", "content": assistant_content})
    if isinstance(prompt, BaseModel):
        prompt = prompt.model_dump_json()
    messages.append({"role": "user", "content": prompt})
    return messages


async def answer(
    prompt: str,
    preface: str | None = None,
    examples: list[Example] | None = None,
    **kwargs,
) -> BaseModel:
    messages = prepare_messages(prompt, preface, examples)
    options = {
        "model": "gpt-4-1106-preview",
        "temperature": 0.0,
        "response_model": PlainText,
    } | kwargs
    return await query_api_chat(messages=messages, **options)


def answer_sync(
    prompt: str,
    preface: str | None = None,
    examples: list[Example] | None = None,
    **kwargs,
) -> BaseModel:
    messages = prepare_messages(prompt, preface, examples)
    options = {
        "model": "gpt-4-1106-preview",
        "temperature": 0.0,
        "response_model": PlainText,
    } | kwargs
    return query_api_chat_sync(messages=messages, **options)


@pydantic_cache
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


def query_hf_text(model: str, text: str, verbose=False, **kwargs) -> str:
    client, client_name = get_client(model, use_async=False)
    if verbose:
        print("Querying Huggingface with text:", text[:30])
    response_text = client(text)
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


async def get_embedding(
    text: str,
    embedding_model: str = "text-embedding-3-small",
    model: str = "gpt-3.5-turbo",
) -> list[float]:
    # model is largely ignored because we currently can't use the same model for both the embedding and the completion
    client, _ = get_client(model, use_async=True)
    response = await client.client.embeddings.create(input=text, model=embedding_model)
    return response.data[0].embedding


# %%
def get_all_cached_requests():
    all_cached = pydantic_cache.storage.get_all(namespace="llm_utils")
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
