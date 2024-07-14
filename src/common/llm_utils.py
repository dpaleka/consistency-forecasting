# LLM Utils
# Run tests in this file with python -m dpy.llm_utils

# %%
import os
from typing import Coroutine, Optional, List
from openai import AsyncOpenAI, OpenAI
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
import instructor
from instructor.client import Instructor
from instructor.mode import Mode
import asyncio
from pydantic import BaseModel
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dotenv import load_dotenv, dotenv_values
from mistralai.models.chat_completion import ChatMessage
from anthropic import AsyncAnthropic, Anthropic
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers

from .datatypes import PlainText
from .path_utils import get_src_path, get_root_path


from .perscache import (
    Cache,
    JSONPydanticResponseSerializer,
    JSONSerializer,
    PickleSerializer,
    RedisStorage,
    LocalFileStorage,
    ValueWrapperDictInspectArgs,
)  # If no redis, use LocalFileStorage


CACHE_FLAGS = ["NO_CACHE", "NO_READ_CACHE", "NO_WRITE_CACHE", "LOCAL_CACHE"]
print(f"LOCAL_CACHE: {os.getenv('LOCAL_CACHE')}")

load_dotenv(override=False)

# We override all keys and tokens (bc those could have been set globally in the user's system). Other flags stay if they are set.
env_path = get_root_path() / ".env"
env_vars = dotenv_values(env_path)
KEYS = [k for k in env_vars.keys() if "KEY" in k or "TOKEN" in k]
override_env_vars = {k: v for k, v in env_vars.items() if k in KEYS}
os.environ.update(override_env_vars)

max_concurrent_queries = int(os.getenv("MAX_CONCURRENT_QUERIES", 100))
print(f"max_concurrent_queries set for global semaphore: {max_concurrent_queries}")
global_llm_semaphore = asyncio.Semaphore(max_concurrent_queries)

pydantic_cache = Cache(
    serializer=JSONPydanticResponseSerializer(),
    storage=(
        LocalFileStorage(location=get_src_path().parent / os.getenv("LOCAL_CACHE"))
        if os.getenv("LOCAL_CACHE")
        else RedisStorage(namespace="llm_utils")
    ),
    value_wrapper=ValueWrapperDictInspectArgs(),
)

embeddings_cache = Cache(
    serializer=PickleSerializer(),
    storage=(
        LocalFileStorage(location=get_src_path().parent / os.getenv("LOCAL_CACHE"))
        if os.getenv("LOCAL_CACHE")
        else RedisStorage(namespace="llm_utils")
    ),
    value_wrapper=ValueWrapperDictInspectArgs(),
)

text_cache = Cache(
    serializer=JSONSerializer(),
    storage=(
        LocalFileStorage(location=get_src_path().parent / os.getenv("LOCAL_CACHE"))
        if os.getenv("LOCAL_CACHE")
        else RedisStorage(namespace="llm_utils")
    ),
    value_wrapper=ValueWrapperDictInspectArgs(),
)

embeddings_cache = Cache(
    serializer=PickleSerializer(),
    storage=(
        LocalFileStorage(location=get_src_path().parent / os.getenv("LOCAL_CACHE"))
        if os.getenv("LOCAL_CACHE")
        else RedisStorage(namespace="llm_utils")
    ),
    value_wrapper=ValueWrapperDictInspectArgs(),
)

FLAGS = CACHE_FLAGS + ["SINGLE_THREAD"] + ["VERBOSE"]


client = None
PROVIDERS = ["openai", "mistral", "anthropic", "togetherai", "huggingface_local"]


def singleton_constructor(get_instance_func):
    instances = {}

    def wrapper(*args, **kwargs):
        if get_instance_func not in instances:
            instances[get_instance_func] = get_instance_func(*args, **kwargs)
        return instances[get_instance_func]

    return wrapper


@singleton_constructor
def get_async_openai_client_pydantic() -> Instructor:
    api_key = os.getenv("OPENAI_API_KEY")
    _client = AsyncOpenAI(api_key=api_key)
    return instructor.from_openai(_client)


@singleton_constructor
def get_async_openai_client_native() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return AsyncOpenAI(api_key=api_key)


@singleton_constructor
def get_openai_client_pydantic() -> Instructor:
    api_key = os.getenv("OPENAI_API_KEY")
    _client = OpenAI(api_key=api_key)
    return instructor.from_openai(_client)


@singleton_constructor
def get_openai_client_native() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


@singleton_constructor
def get_async_openrouter_client_pydantic(**kwargs) -> Instructor:
    print(
        "Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website."
    )
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return instructor.from_openai(_client, mode=Mode.MD_JSON, **kwargs)


@singleton_constructor
def get_async_openrouter_client_native() -> AsyncOpenAI:
    print("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


@singleton_constructor
def get_openrouter_client_pydantic(**kwargs) -> Instructor:
    print(
        "Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website."
    )
    print("Calling models through OpenRouter")
    _client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    print(f"OPENROUTER_API_KEY: {os.getenv('OPENROUTER_API_KEY')}")
    return instructor.from_openai(_client, mode=Mode.MD_TOOLS, **kwargs)


@singleton_constructor
def get_openrouter_client_native() -> OpenAI:
    print("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


@singleton_constructor
def get_mistral_async_client_pydantic() -> Instructor:
    api_key = os.getenv("MISTRAL_API_KEY")
    _client = MistralAsyncClient(api_key=api_key)
    return instructor.from_openai(_client, mode=instructor.Mode.MISTRAL_TOOLS)


@singleton_constructor
def get_mistral_async_client_native() -> MistralAsyncClient:
    api_key = os.getenv("MISTRAL_API_KEY")
    return MistralAsyncClient(api_key=api_key)


@singleton_constructor
def get_mistral_client_pydantic() -> Instructor:
    api_key = os.getenv("MISTRAL_API_KEY")
    _client = MistralClient(api_key=api_key)
    return instructor.from_openai(_client, mode=instructor.Mode.MISTRAL_TOOLS)


@singleton_constructor
def get_mistral_client_native() -> MistralClient:
    api_key = os.getenv("MISTRAL_API_KEY")
    return MistralClient(api_key=api_key)


@singleton_constructor
def get_anthropic_async_client_pydantic() -> Instructor:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    _client = AsyncAnthropic(api_key=api_key)
    return instructor.from_anthropic(_client, mode=instructor.Mode.ANTHROPIC_JSON)


@singleton_constructor
def get_anthropic_async_client_native() -> AsyncAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return AsyncAnthropic(api_key=api_key)


@singleton_constructor
def get_anthropic_client_pydantic() -> Instructor:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    _client = Anthropic(api_key=api_key)
    return instructor.from_anthropic(_client, mode=instructor.Mode.ANTHROPIC_JSON)


@singleton_constructor
def get_anthropic_client_native() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return Anthropic(api_key=api_key)


@singleton_constructor
def get_togetherai_client_native() -> OpenAI:
    url = "https://api.together.xyz/v1"
    api_key = os.getenv("TOGETHER_API_KEY")
    return OpenAI(api_key=api_key, base_url=url)


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
        "ft:gpt",
        "gpt-4o",
        "gpt-4",
        "gpt-3.5",
        "babbage",
        "davinci",
        "openai",
        "open-ai",
    ]
    return any(keyword in model for keyword in keywords)


def is_mistral(model: str) -> bool:
    if model.startswith("mistral"):
        return True


def is_anthropic(model: str) -> bool:
    keywords = ["anthropic", "claude"]
    return any(keyword in model for keyword in keywords)


def is_togetherai(model: str) -> bool:
    keywords = ["together", "llama", "phi", "orca", "Hermes", "Yi"]
    return any(keyword in model for keyword in keywords)


def is_huggingface_local(model: str) -> bool:
    keywords = ["huggingface", "hf"]
    return any(keyword in model for keyword in keywords)


def get_provider(model: str) -> str:
    if is_openai(model):
        return "openai"
    elif is_mistral(model):
        return "mistral"
    elif is_anthropic(model):
        return "anthropic"
    elif is_togetherai(model):
        return "togetherai"
    elif is_huggingface_local(model):
        return "huggingface_local"
    else:
        raise NotImplementedError(f"Model {model} is not supported for now")


def is_model_name_valid(model: str) -> bool:
    if len(model) > 40:
        return False  # Model name is too long, probably a mistake
    try:
        return get_provider(model) is not None
    except NotImplementedError:
        return False


def get_client_pydantic(model: str, use_async=True) -> tuple[Instructor, str]:
    provider = get_provider(model)
    if provider == "togetherai" and "nitro" not in model:
        raise NotImplementedError(
            "Most models on TogetherAI API, and the same models on OpenRouter API too, do not support function calling / JSON output mode. So, no Pydantic outputs for now. The exception seem to be Nitro-hosted models on OpenRouter."
        )

    use_openrouter = (
        os.getenv("USE_OPENROUTER") and os.getenv("USE_OPENROUTER") != "False"
    )
    if use_openrouter:
        kwargs = {}
        if provider == "mistral":
            # https://python.useinstructor.com/hub/mistral/
            print(
                "Only some Mistral endpoints have `response_format` on OpenRouter. If you encounter errors, please check on the OpenRouter website."
            )
            kwargs["mode"] = instructor.Mode.MISTRAL_TOOLS
        elif provider == "anthropic":
            raise NotImplementedError(
                "Anthropic over OpenRouter does not work as of June 4 2024"
            )
        client = (
            get_async_openrouter_client_pydantic(**kwargs)
            if use_async
            else get_openrouter_client_pydantic(**kwargs)
        )
    elif provider == "openai":
        client = (
            get_async_openai_client_pydantic()
            if use_async
            else get_openai_client_pydantic()
        )
    elif provider == "mistral":
        client = (
            get_mistral_async_client_pydantic()
            if use_async
            else get_mistral_client_pydantic()
        )
    elif provider == "anthropic":
        client = (
            get_anthropic_async_client_pydantic()
            if use_async
            else get_anthropic_client_pydantic()
        )
    else:
        raise NotImplementedError(f"Model {model} is not supported for now")

    return client, provider


def get_client_native(
    model: str, use_async=True
) -> tuple[AsyncOpenAI | OpenAI | MistralAsyncClient | MistralClient, str]:
    provider = get_provider(model)

    if os.getenv("USE_OPENROUTER"):
        client = (
            get_async_openrouter_client_native()
            if use_async
            else get_openrouter_client_native()
        )
    elif provider == "openai":
        client = (
            get_async_openai_client_native()
            if use_async
            else get_openai_client_native()
        )
    elif provider == "mistral":
        client = (
            get_mistral_async_client_native()
            if use_async
            else get_mistral_client_native()
        )
    elif provider == "togetherai":
        if use_async:
            raise NotImplementedError(
                "Only synchronous calls are supported for TogetherAI"
            )
        client = get_togetherai_client_native()
    elif provider == "huggingface_local":
        assert model.startswith("hf:")
        model = model.split("hf:")[1]
        client = get_huggingface_local_client(model)
    else:
        raise NotImplementedError(f"Model {model} is not supported for now")

    return client, provider


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
    if not os.getenv("NO_CACHE"):
        assert (
            kwargs.get("response_model", -1) is not None
        ), "Cannot pass response_model=None if caching is enabled"

    default_options = {
        "model": "gpt-4o-2024-05-13",
        "response_model": PlainText,
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]
    client, client_name = get_client_pydantic(options["model"], use_async=True)
    if options.get("n", 1) != 1:
        raise NotImplementedError("Multiple queries not supported yet")

    call_messages = (
        _mistral_message_transform(messages) if client_name == "mistral" else messages
    )

    if client_name == "anthropic":
        options["max_tokens"] = options.get("max_tokens", 1024)

    print(
        options,
        f"Approx num tokens: {len(''.join([m['content'] for m in messages])) // 3}",
    )

    response = await client.chat.completions.create(
        messages=call_messages,
        **options,
    )
    if verbose or os.getenv("VERBOSE") == "True":
        print(f"...\nText: {messages[-1]['content']}\nResponse: {response}")
    return response


@text_cache
async def query_api_chat_native(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> str:
    default_options = {
        "model": "gpt-4o-2024-05-13",
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]

    client, client_name = get_client_native(options["model"], use_async=True)
    call_messages = (
        _mistral_message_transform(messages) if client_name == "mistral" else messages
    )

    print(
        options,
        f"Approx num tokens: {len(''.join([m['content'] for m in messages])) // 3}",
    )
    if client_name == "mistral" and not os.getenv("USE_OPENROUTER"):
        response = await client.chat(
            messages=call_messages,
            **options,
        )
    else:
        response = await client.chat.completions.create(
            messages=call_messages,
            **options,
        )

    text_response = response.choices[0].message.content

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"...\nText: {messages[-1]['content']}\nResponse: {text_response}\n")

    return text_response


@pydantic_cache
def query_api_chat_sync(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> BaseModel:
    if not os.getenv("NO_CACHE"):
        assert (
            kwargs.get("response_model", -1) is not None
        ), "Cannot pass response_model=None if caching is enabled"

    default_options = {
        "model": "gpt-4o-2024-05-13",
        "response_model": PlainText,
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]
    client, client_name = get_client_pydantic(options["model"], use_async=False)
    if options.get("n", 1) != 1:
        raise NotImplementedError("Multiple structured queries not supported yet")

    call_messages = (
        _mistral_message_transform(messages) if client_name == "mistral" else messages
    )

    if client_name == "anthropic":
        options["max_tokens"] = options.get("max_tokens", 1024)

    print(
        options,
        f"Approx num tokens: {len(''.join([m['content'] for m in messages])) // 3}",
    )

    response = client.chat.completions.create(
        messages=call_messages,
        **options,
    )

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"...\nText: {messages[-1]['content']}\nResponse: {response}")
    return response


@text_cache
def query_api_chat_sync_native(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> str:
    default_options = {
        "model": "gpt-4o-2024-05-13",
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]
    client, client_name = get_client_native(options["model"], use_async=False)
    call_messages = (
        _mistral_message_transform(messages) if client_name == "mistral" else messages
    )

    print(
        options,
        f"Approx num tokens: {len(''.join([m['content'] for m in messages])) // 3}",
    )

    if client_name == "mistral" and not os.getenv("USE_OPENROUTER"):
        response = client.chat(
            messages=call_messages,
            **options,
        )
    else:
        response = client.chat.completions.create(
            messages=call_messages,
            **options,
        )

    text_response = response.choices[0].message.content

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"...\nText: {messages[-1]['content']}\nResponse: {text_response}")

    return text_response


@dataclass_json
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


def prepare_messages_alt(
    prompt: str, preface: str | None = None, examples: list[Example] | None = None
) -> list[dict[str, str]]:
    sys_preface = "You are a helpful assistant."
    messages = [{"role": "system", "content": sys_preface}]
    examples = examples or []
    if not preface:
        preface = ""
    for example in examples:
        if isinstance(example.user, BaseModel):
            example.user = example.user.model_dump_json()
        if isinstance(example.assistant, BaseModel):
            example.assistant = example.assistant.model_dump_json()
        messages.append({"role": "user", "content": example.user})
        example.user = preface + "\n\n" + example.user
        # Convert assistant's response to string if it's not already
        assistant_content = (
            str(example.assistant)
            if isinstance(example.assistant, (float, int))
            else example.assistant
        )
        messages.append({"role": "assistant", "content": assistant_content})
    if isinstance(prompt, BaseModel):
        prompt = prompt.model_dump_json()
    prompt = preface + "\n\n" + prompt
    messages.append({"role": "user", "content": prompt})
    return messages


async def answer(
    prompt: str,
    preface: Optional[str] = None,
    examples: Optional[List[Example]] = None,
    prepare_messages_func=prepare_messages,
    **kwargs,
) -> BaseModel:
    assert not is_model_name_valid(
        str(prompt)
    ), "Are you sure you want to pass the model name as a prompt?"
    messages = prepare_messages_func(prompt, preface, examples)
    default_options = {
        "model": "gpt-4o",
        "temperature": 0.5,
        "response_model": PlainText,
    }
    options = default_options | kwargs  # override defaults with kwargs

    print(f"options: {options}")
    print(f"messages: {messages}")
    async with global_llm_semaphore:
        return await query_api_chat(messages=messages, **options)


def answer_sync(
    prompt: str,
    preface: str | None = None,
    examples: list[Example] | None = None,
    prepare_messages_func=prepare_messages,
    **kwargs,
) -> BaseModel:
    assert not is_model_name_valid(
        str(prompt)
    ), "Are you sure you want to pass the model name as a prompt?"
    messages = prepare_messages_func(prompt, preface, examples)
    options = {
        "model": "gpt-4o-2024-05-13",
        "temperature": 0.5,
        "response_model": PlainText,
    } | kwargs
    return query_api_chat_sync(messages=messages, **options)


@pydantic_cache
async def query_api_text(model: str, text: str, verbose=False, **kwargs) -> str:
    client, client_name = get_client_pydantic(model, use_async=True)
    response = await client.completions.create(model=model, prompt=text, **kwargs)
    response_text = response.choices[0].text
    if verbose or os.getenv("VERBOSE") == "True":
        print("Text:", text[:30], "\nResponse:", response_text[:30])
    return response_text


def query_api_text_sync(model: str, text: str, verbose=False, **kwargs) -> str:
    client, client_name = get_client_pydantic(model, use_async=False)
    response = client.completions.create(model=model, prompt=text, **kwargs)
    response_text = response.choices[0].text
    if verbose or os.getenv("VERBOSE") == "True":
        print("Text:", text, "\nResponse:", response_text)
    return response_text


def query_hf_text(model: str, text: str, verbose=False, **kwargs) -> str:
    client, client_name = get_client_pydantic(model, use_async=False)
    response_text = client(text)
    if verbose or os.getenv("VERBOSE") == "True":
        print("Text:", text, "\nResponse:", response_text)

    return response_text


async def parallelized_call(
    func: Coroutine,
    data: list[str],
    max_concurrent_queries: int = 100,
) -> list[any]:
    """
    Run async func in parallel on the given data.
    func will usually be a partial which uses query_api or whatever in some way.

    Example usage:
        partial_eval_method = functools.partial(eval_method, model=model, **kwargs)
        results = await parallelized_call(partial_eval_method, [format_post(d) for d in data])
    """

    if os.getenv("SINGLE_THREAD"):
        print(f"Running {func} on {len(data)} datapoints sequentially")
        return [await func(d) for d in data]

    max_concurrent_queries = min(
        max_concurrent_queries,
        int(os.getenv("MAX_CONCURRENT_QUERIES", max_concurrent_queries)),
    )

    print(
        f"Running {func} on {len(data)} datapoints with {max_concurrent_queries} concurrent queries"
    )

    # Create a local semaphore
    local_semaphore = asyncio.Semaphore(max_concurrent_queries)

    async def call_func(sem, func, datapoint):
        async with sem:
            return await func(datapoint)

    print("Calling call_func")
    tasks = [call_func(local_semaphore, func, d) for d in data]
    return await asyncio.gather(*tasks)


@embeddings_cache
async def get_embedding(
    text: str,
    embedding_model: str = "text-embedding-3-small",
    model: str = "gpt-3.5-turbo",
) -> list[float]:
    # model is largely ignored because we currently can't use the same model for both the embedding and the completion
    client, _ = get_client_pydantic(model, use_async=True)
    response = await client.client.embeddings.create(input=text, model=embedding_model)
    return response.data[0].embedding


@embeddings_cache
def get_embeddings_sync(
    texts: list[str],
    embedding_model: str = "text-embedding-3-small",
    model: str = "gpt-3.5-turbo",
) -> list[list[float]]:
    # model is largely ignored because we currently can't use the same model for both the embedding and the completion
    client, _ = get_client_pydantic(model, use_async=False)
    response = client.client.embeddings.create(input=texts, model=embedding_model)
    return [e.embedding for e in response.data]


@embeddings_cache
def get_embedding_sync(
    text: str,
    embedding_model: str = "text-embedding-3-small",
    model: str = "gpt-3.5-turbo",
) -> list[float]:
    return get_embeddings_sync([text], embedding_model, model)[0]


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
