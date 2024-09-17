# LLM Utils
# Run tests in this file with python -m common.llm_utils

# %%
import os
import logging
from typing import Coroutine, Optional, List
from openai import AsyncOpenAI, OpenAI
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
import logfire
from costly import CostlyResponse, costly
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from .datatypes import (
    PlainText,
    Prob,
    ForecastingQuestion,
    ForecastingQuestion_stripped,
)
from .path_utils import get_src_path, get_root_path, get_data_path
from .perplexity_client import AsyncPerplexityClient, SyncPerplexityClient


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

load_dotenv(override=False, dotenv_path=get_root_path() / ".env")

# We override all keys and tokens (bc those could have been set globally in the user's system). Other flags stay if they are set.
env_path = get_root_path() / ".env"
env_vars = dotenv_values(env_path)
KEYS = [k for k in env_vars.keys() if "KEY" in k or "TOKEN" in k]
override_env_vars = {k: v for k, v in env_vars.items() if k in KEYS}
os.environ.update(override_env_vars)

max_concurrent_queries = int(os.getenv("MAX_CONCURRENT_QUERIES", 25))
print(f"max_concurrent_queries set for global semaphore: {max_concurrent_queries}")


## All logging settings here
if os.getenv("USE_LOGFIRE") == "True":
    print("Setting up Pydantic Logfire")

    def scrubbing_callback(m: logfire.ScrubMatch):
        """
        Need to disable some security measures of logfire.
        Those trigges depending on whether some substrings like "auth" are present as param *values*;
        and our param values are *prompts* and such, so no need to scrub them.
        """
        if m.pattern_match.group(0) == "auth":
            return m.value

    logfire.configure(
        pydantic_plugin=logfire.PydanticPlugin(record="all"),
        scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback),
    )

if os.getenv("LOGGING_DEBUG") == "True":
    print("Setting logging level to DEBUG")
    logging.basicConfig(level=logging.DEBUG, force=True)


def reset_global_semaphore():
    """
    Use if your code uses asyncio.run()
    """
    global global_llm_semaphore
    global_llm_semaphore = asyncio.Semaphore(max_concurrent_queries)
    print(
        f"Resetting global semaphore, max concurrent queries: {max_concurrent_queries}"
    )


reset_global_semaphore()


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

FLAGS = CACHE_FLAGS + ["SINGLE_THREAD"] + ["VERBOSE", "LOGGING_DEBUG", "USE_LOGFIRE"]


client = None
PROVIDERS = ["openai", "mistral", "anthropic", "togetherai", "huggingface_local"]


class LLM_Simulator(LLM_Simulator_Faker):
    fqs_path = get_data_path() / "fq" / "real" / "test_formatted.jsonl"

    @staticmethod
    def pick_random_fq(file_path: str, strip=False):
        import random

        with open(file_path, "r") as file:
            lines = file.readlines()
        random_line = random.choice(lines)
        fq = ForecastingQuestion.model_validate_json(random_line)
        if strip:
            fq = fq.cast_stripped()
        return fq

    @classmethod
    def _fake_custom(cls, t: type):
        if issubclass(t, Prob):
            import random

            return t(prob=random.random())
        elif issubclass(t, ForecastingQuestion):
            return cls.pick_random_fq(cls.fqs_path, strip=False)
        elif issubclass(t, ForecastingQuestion_stripped):
            return cls.pick_random_fq(cls.fqs_path, strip=True)
        else:
            raise NotImplementedError(f"{t} is not a known custom type")


def singleton_constructor(get_instance_func):
    instances = {}

    def wrapper(*args, **kwargs):
        if get_instance_func not in instances:
            instances[get_instance_func] = get_instance_func(*args, **kwargs)
        return instances[get_instance_func]

    return wrapper


@singleton_constructor
def get_async_perplexity_client() -> AsyncPerplexityClient:
    load_dotenv()
    api_key = os.getenv("PERPLEXITY_API_TOKEN")
    if not api_key:
        raise ValueError("PERPLEXITY_API_TOKEN not found in environment variables")
    client = AsyncPerplexityClient(api_key)
    # If you have a logging/instrumentation library like logfire, you can add it here
    # logfire.instrument_perplexity(client)
    return client


@singleton_constructor
def get_sync_perplexity_client() -> SyncPerplexityClient:
    load_dotenv()
    api_key = os.getenv("PERPLEXITY_API_TOKEN")
    if not api_key:
        raise ValueError("PERPLEXITY_API_TOKEN not found in environment variables")
    client = SyncPerplexityClient(api_key)
    # If you have a logging/instrumentation library like logfire, you can add it here
    # logfire.instrument_perplexity(client)
    return client


@singleton_constructor
def get_async_openai_client_pydantic() -> Instructor:
    api_key = os.getenv("OPENAI_API_KEY")
    _client = AsyncOpenAI(api_key=api_key)
    logfire.instrument_openai(_client)
    mode = (
        Mode.TOOLS_STRICT if os.getenv("OPENAI_JSON_STRICT") == "True" else Mode.TOOLS
    )
    return instructor.from_openai(_client, mode=mode)


@singleton_constructor
def get_async_openai_client_native() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)
    logfire.instrument_openai(client)
    return client


@singleton_constructor
def get_openai_client_pydantic() -> Instructor:
    api_key = os.getenv("OPENAI_API_KEY")
    _client = OpenAI(api_key=api_key)
    logfire.instrument_openai(_client)
    mode = (
        Mode.TOOLS_STRICT if os.getenv("OPENAI_JSON_STRICT") == "True" else Mode.TOOLS
    )
    return instructor.from_openai(_client, mode=mode)


@singleton_constructor
def get_openai_client_native() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    logfire.instrument_openai(client)
    return client


@singleton_constructor
def get_async_openrouter_client_pydantic(**kwargs) -> Instructor:
    print(
        "Only some OpenRouter endpoints will work. If you encounter errors, please check on the OpenRouter website."
    )
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.MD_JSON, **kwargs)


@singleton_constructor
def get_async_openrouter_client_native() -> AsyncOpenAI:
    print("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    logfire.instrument_openai(client)
    return client


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
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.TOOLS, **kwargs)


@singleton_constructor
def get_openrouter_client_native() -> OpenAI:
    print("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    logfire.instrument_openai(client)
    return client


@singleton_constructor
def get_anthropic_async_client_pydantic() -> Instructor:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    _client = AsyncAnthropic(api_key=api_key)
    # As of 27 Aug 2024, cannot setup logfire for anthropic client, because of version mismatch.
    return instructor.from_anthropic(_client, mode=instructor.Mode.ANTHROPIC_JSON)


@singleton_constructor
def get_anthropic_async_client_native() -> AsyncAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    _client = AsyncAnthropic(api_key=api_key)
    # As of 27 Aug 2024, cannot setup logfire for anthropic client, because of version mismatch.
    return _client


@singleton_constructor
def get_anthropic_client_pydantic() -> Instructor:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    _client = Anthropic(api_key=api_key)
    # As of 27 Aug 2024, cannot setup logfire for anthropic client, because of version mismatch.
    return instructor.from_anthropic(_client, mode=instructor.Mode.ANTHROPIC_JSON)


@singleton_constructor
def get_anthropic_client_native() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    _client = Anthropic(api_key=api_key)
    # As of 27 Aug 2024, cannot setup logfire for anthropic client, because of version mismatch.
    return _client


@singleton_constructor
def get_togetherai_client_native() -> OpenAI:
    url = "https://api.together.xyz/v1"
    api_key = os.getenv("TOGETHER_API_KEY")
    _client = OpenAI(api_key=api_key, base_url=url)
    logfire.instrument_openai(_client)
    return _client


def is_openai(model: str) -> bool:
    keywords = [
        "ft:gpt",
        "o1",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-3.5",
        "babbage",
        "davinci",
        "openai",
        "open-ai",
    ]
    return any(keyword in model for keyword in keywords)


def is_perplexity_ai(model: str) -> bool:
    keywords = ["perplexity", "sonar"]
    return any(keyword.lower() in model.lower() for keyword in keywords)


def is_togetherai(model: str) -> bool:
    keywords = ["together", "llama", "phi", "orca", "Hermes", "Yi"]
    return any(keyword in model for keyword in keywords)


def is_anthropic(model: str) -> bool:
    keywords = ["anthropic", "claude"]
    return any(keyword in model for keyword in keywords)


def is_huggingface_local(model: str) -> bool:
    keywords = ["huggingface", "hf"]
    return any(keyword in model for keyword in keywords)


def get_provider(model: str) -> str:
    if os.getenv("USE_OPENROUTER") and os.getenv("USE_OPENROUTER") != "False":
        return "openrouter"
    elif is_openai(model):
        print(
            f"Using OpenAI provider for model {model}, key {os.getenv('OPENAI_API_KEY')}"
        )
        return "openai"
    elif is_perplexity_ai(model):
        return "perplexity"
    elif is_anthropic(model):
        return "anthropic"
    elif is_togetherai(model):
        return "togetherai"
    elif is_huggingface_local(model):
        return "huggingface_local"
    else:
        print(
            f"Model {model} is not supported with a provider; USE_OPENROUTER should be True"
        )
        assert False


def get_client_pydantic(model: str, use_async=True) -> tuple[Instructor, str, str]:
    provider = get_provider(model)
    final_model_name = model

    if provider == "openrouter":
        print(f"Using {provider} provider for model {model}")
        kwargs = {}
        client = (
            get_async_openrouter_client_pydantic(**kwargs)
            if use_async
            else get_openrouter_client_pydantic(**kwargs)
        )
    else:
        print(f"Using {provider} provider for model {model}")
        if provider == "openai":
            final_model_name = model.replace("openai/", "")
            client = (
                get_async_openai_client_pydantic()
                if use_async
                else get_openai_client_pydantic()
            )
        elif provider == "anthropic":
            final_model_name = model.replace("anthropic/", "")
            if final_model_name in ANTHROPIC_DEFAULT_MODEL_NAME_MAP:
                final_model_name = ANTHROPIC_DEFAULT_MODEL_NAME_MAP[final_model_name]
            client = (
                get_anthropic_async_client_pydantic()
                if use_async
                else get_anthropic_client_pydantic()
            )
        else:
            raise NotImplementedError(
                f"Model {model} Pydantic client is not supported for now outside of OpenRouter"
            )

    return client, provider, final_model_name


def get_client_native(
    model: str, use_async=True
) -> tuple[AsyncOpenAI | OpenAI | AsyncAnthropic | Anthropic, str, str]:
    provider = get_provider(model)
    final_model_name = model

    if provider == "openrouter":
        client = (
            get_async_openrouter_client_native()
            if use_async
            else get_openrouter_client_native()
        )
    else:
        print(f"Using {provider} provider for model {model}")
        if provider == "openai":
            final_model_name = model.replace("openai/", "")
            client = (
                get_async_openai_client_native()
                if use_async
                else get_openai_client_native()
            )
        elif provider == "anthropic":
            final_model_name = model.replace("anthropic/", "")
            if final_model_name in ANTHROPIC_DEFAULT_MODEL_NAME_MAP:
                final_model_name = ANTHROPIC_DEFAULT_MODEL_NAME_MAP[final_model_name]
            client = (
                get_anthropic_async_client_native()
                if use_async
                else get_anthropic_client_native()
            )
        elif provider == "togetherai":
            if use_async:
                raise NotImplementedError(
                    "Only synchronous calls are supported for TogetherAI"
                )
            client = get_togetherai_client_native()
        elif provider == "perplexity":
            if use_async:
                client = get_async_perplexity_client()
            else:
                client = get_sync_perplexity_client()
        else:
            raise NotImplementedError(f"Model {model} is not supported for now")

    return client, provider, final_model_name


def is_llama2_tokenized(model: str) -> bool:
    keywords = ["Llama-2", "pythia"]
    return any(keyword in model for keyword in keywords)


def _mistral_message_transform(messages):
    mistral_messages = []
    for message in messages:
        mistral_message = ChatMessage(role=message["role"], content=message["content"])
        mistral_messages.append(mistral_message)
    return mistral_messages


def _o1_message_params_transform(messages, options):
    o1_messages = []
    if messages[0]["role"] == "system":
        o1_messages.append({"role": "user", "content": messages[0]["content"]})
        o1_messages.append(
            {"role": "assistant", "content": "System message acknowledged"}
        )
        o1_messages.extend(messages[1:])
    else:
        o1_messages.extend(messages)

    options["temperature"] = 1
    return o1_messages, options


def supports_system_message(model: str, client_name: str) -> bool:
    """
    There might be other models that don't support system messages; check if there is an error when running the code.
    """
    if "o1" in model:
        return False
    return True


ANTHROPIC_DEFAULT_MODEL_NAME_MAP = {
    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}


@pydantic_cache
@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_chat", extract_args=True)
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
        "model": "gpt-4o-mini-2024-07-18",
        "response_model": PlainText,
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]
    client, client_name, final_model_name = get_client_pydantic(
        options["model"], use_async=True
    )
    options["model"] = final_model_name
    if options.get("n", 1) != 1:
        raise NotImplementedError("Multiple queries not supported yet")

    call_messages = (
        _mistral_message_transform(messages) if client_name == "mistral" else messages
    )
    call_messages, options = (
        _o1_message_params_transform(call_messages, options)
        if not supports_system_message(options["model"], client_name)
        else (call_messages, options)
    )

    if client_name == "anthropic":
        options["max_tokens"] = options.get("max_tokens", 1024)

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"{options=}, {len(messages)=}")

    response, completion = await client.chat.completions.create_with_completion(
        messages=call_messages,
        **options,
    )
    # print(f"Completion: {completion}")

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"...\nText: {messages[-1]['content']}\nResponse: {response}")
    return CostlyResponse(
        output=response,
        cost_info={
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        },
    )


@text_cache
@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_chat_native", extract_args=True)
async def query_api_chat_native(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> str:
    default_options = {
        "model": "gpt-4o-mini-2024-07-18",
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]

    client, client_name, final_model_name = get_client_native(
        options["model"], use_async=True
    )
    options["model"] = final_model_name
    call_messages = (
        _mistral_message_transform(messages) if client_name == "mistral" else messages
    )
    call_messages, options = (
        _o1_message_params_transform(call_messages, options)
        if not supports_system_message(options["model"], client_name)
        else (call_messages, options)
    )

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"{options=}, {len(messages)=}")

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

    return CostlyResponse(
        output=text_response,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )


@pydantic_cache
@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_chat_sync", extract_args=True)
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
        "model": "gpt-4o-mini-2024-07-18",
        "response_model": PlainText,
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]
    client, client_name, final_model_name = get_client_pydantic(
        options["model"], use_async=False
    )
    options["model"] = final_model_name
    if options.get("n", 1) != 1:
        raise NotImplementedError("Multiple structured queries not supported yet")

    call_messages = (
        _mistral_message_transform(messages) if client_name == "mistral" else messages
    )
    call_messages, options = (
        _o1_message_params_transform(call_messages, options)
        if not supports_system_message(options["model"], client_name)
        else (call_messages, options)
    )

    if client_name == "anthropic":
        options["max_tokens"] = options.get("max_tokens", 1024)

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"{options=}, {len(messages)=}")

    response, completion = client.chat.completions.create_with_completion(
        messages=call_messages,
        **options,
    )
    # print(f"Completion: {completion}")

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"...\nText: {messages[-1]['content']}\nResponse: {response}")
    return CostlyResponse(
        output=response,
        cost_info={
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        },
    )


@text_cache
@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_chat_sync_native", extract_args=True)
def query_api_chat_sync_native(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> str:
    default_options = {
        "model": "gpt-4o-mini-2024-07-18",
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]
    client, client_name, final_model_name = get_client_native(
        options["model"], use_async=False
    )
    options["model"] = final_model_name
    call_messages = (
        _mistral_message_transform(messages) if client_name == "mistral" else messages
    )
    call_messages, options = (
        _o1_message_params_transform(call_messages, options)
        if not supports_system_message(options["model"], client_name)
        else (call_messages, options)
    )

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"{options=}, {len(messages)=}")

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

    return CostlyResponse(
        output=text_response,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )


@dataclass_json
@dataclass
class Example:
    user: str | BaseModel
    assistant: str | BaseModel


def serialize_if_pydantic(obj: str | BaseModel) -> str:
    """
    Idempotent function to convert a BaseModel to a string.
    If the object is already a string, it returns the object as is.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    return obj


def prepare_messages(
    prompt: str | BaseModel | None,
    preface: str | None = None,
    examples: list[Example] | None = None,
) -> list[dict[str, str]]:
    preface = preface or "You are a helpful assistant."
    examples = examples or []
    messages = [{"role": "system", "content": preface}]
    for example in examples:
        example.user = serialize_if_pydantic(example.user)
        example.assistant = serialize_if_pydantic(example.assistant)
        messages.append({"role": "user", "content": example.user})
        # Convert assistant's response to string if it's not already
        assistant_content = (
            str(example.assistant)
            if isinstance(example.assistant, (float, int))
            else example.assistant
        )
        messages.append({"role": "assistant", "content": assistant_content})
    if prompt is not None:
        prompt = serialize_if_pydantic(prompt)
        messages.append({"role": "user", "content": prompt})
    return messages


def prepare_messages_alt(
    prompt: str | BaseModel | None,
    preface: str | None = None,
    examples: list[Example] | None = None,
) -> list[dict[str, str]]:
    sys_preface = "You are a helpful assistant."
    messages = [{"role": "system", "content": sys_preface}]
    examples = examples or []
    if not preface:
        preface = ""
    for example in examples:
        example.user = serialize_if_pydantic(example.user)
        example.assistant = serialize_if_pydantic(example.assistant)
        messages.append({"role": "user", "content": example.user})
        example.user = preface + "\n\n" + example.user
        # Convert assistant's response to string if it's not already
        assistant_content = (
            str(example.assistant)
            if isinstance(example.assistant, (float, int))
            else example.assistant
        )
        messages.append({"role": "assistant", "content": assistant_content})
    if prompt is not None:
        prompt = serialize_if_pydantic(prompt)
        prompt = preface + "\n\n" + prompt
        messages.append({"role": "user", "content": prompt})
    return messages


@logfire.instrument("answer", extract_args=True)
async def answer(
    prompt: str,
    preface: Optional[str] = None,
    examples: Optional[List[Example]] = None,
    prepare_messages_func=prepare_messages,
    with_parsing: bool = False,
    **kwargs,
) -> BaseModel:
    messages = prepare_messages_func(prompt, preface, examples)
    default_options = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.5,
        "response_model": PlainText,
    }
    options = default_options | kwargs  # override defaults with kwargs

    if os.getenv("VERBOSE") == "True":
        print(f"{options=}, {len(messages)=}")

    async with global_llm_semaphore:
        if with_parsing:
            return await query_api_chat_with_parsing(messages=messages, **options)
        else:
            return await query_api_chat(messages=messages, **options)


@logfire.instrument("answer_sync", extract_args=True)
def answer_sync(
    prompt: str,
    preface: str | None = None,
    examples: list[Example] | None = None,
    prepare_messages_func=prepare_messages,
    with_parsing: bool = False,
    **kwargs,
) -> BaseModel:
    messages = prepare_messages_func(prompt, preface, examples)
    options = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.5,
        "response_model": PlainText,
    } | kwargs
    if with_parsing:
        return query_api_chat_sync_with_parsing(messages=messages, **options)
    else:
        return query_api_chat_sync(messages=messages, **options)


async def answer_messages(
    messages: List[dict[str, str] | dict[str, BaseModel]],
    **kwargs,
) -> BaseModel:
    default_options = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.5,
        "response_model": PlainText,
    }
    options = default_options | kwargs  # override defaults with kwargs

    for message in messages:
        assert (
            isinstance(message, dict) and "content" in message
        ), "Messages must be dictionaries with a 'content' key"
        message["content"] = serialize_if_pydantic(message["content"])

    print(f"options: {options}")
    print(f"messages: {messages}")
    async with global_llm_semaphore:
        return await query_api_chat(messages=messages, **options)


def answer_messages_sync(
    messages: List[dict[str, str] | dict[str, BaseModel]],
    **kwargs,
) -> BaseModel:
    for message in messages:
        assert (
            isinstance(message, dict) and "content" in message
        ), "Messages must be dictionaries with a 'content' key"
        message["content"] = serialize_if_pydantic(message["content"])

    options = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.5,
        "response_model": PlainText,
    } | kwargs

    return query_api_chat_sync(messages=messages, **options)


@pydantic_cache
@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_text", extract_args=True)
async def query_api_text(model: str, text: str, verbose=False, **kwargs) -> str:
    client, client_name = get_client_pydantic(model, use_async=True)
    response, completion = await client.completions.create_with_completion(
        model=model, prompt=text, **kwargs
    )
    response_text = response.choices[0].text
    if verbose or os.getenv("VERBOSE") == "True":
        print("Text:", text[:30], "\nResponse:", response_text[:30])
    return CostlyResponse(
        output=response_text,
        cost_info={
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        },
    )


@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_text_sync", extract_args=True)
def query_api_text_sync(model: str, text: str, verbose=False, **kwargs) -> str:
    client, client_name = get_client_pydantic(model, use_async=False)
    response, completion = client.completions.create_with_completion(
        model=model, prompt=text, **kwargs
    )
    response_text = response.choices[0].text
    if verbose or os.getenv("VERBOSE") == "True":
        print("Text:", text, "\nResponse:", response_text)
    return CostlyResponse(
        output=response_text,
        cost_info={
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        },
    )


@logfire.instrument("query_parse_last_response_into_format", extract_args=True)
async def query_parse_last_response_into_format(
    messages: list[dict[str, str]],
    response_model: BaseModel,
    verbose: bool = False,
    model: str | None = None,
    **kwargs,
) -> BaseModel:
    parsing_messages = messages + [
        {
            "role": "user",
            "content": (
                "Now parse the latest response into the specified Pydantic model:\n\n"
                f"{response_model.model_fields=}"
            ),
        },
    ]

    parsed_response = await query_api_chat(
        messages=parsing_messages,
        response_model=response_model,
        verbose=verbose,
        model=model,
        **kwargs,
    )

    return parsed_response


@logfire.instrument("query_parse_last_response_into_format_sync", extract_args=True)
def query_parse_last_response_into_format_sync(
    messages: list[dict[str, str]],
    response_model: BaseModel,
    verbose: bool = False,
    model: str | None = None,
    **kwargs,
) -> BaseModel:
    parsing_messages = messages + [
        {
            "role": "user",
            "content": (
                "Now parse the latest response into the specified Pydantic model:\n\n"
                f"{response_model.model_fields=}"
            ),
        },
    ]

    response = query_api_chat_sync(
        messages=parsing_messages,
        response_model=response_model,
        verbose=verbose,
        model=model,
        **kwargs,
    )
    return response


def system_message_addition_for_parsing(response_model: BaseModel) -> str:
    return f"""\
Note: unless explicitly stated in the prompt, do not worry about the exact formatting of the output.
There will be an extra step that will summarize your output into the final answer format.
For context, the final answer format is described by the following Pydantic model:
{response_model.model_fields=}\n
Again, just try to answer the question as best as you can, with all the necessary information; the output will be cleaned up in the final step.
"""


@logfire.instrument("query_api_chat_with_parsing", extract_args=True)
async def query_api_chat_with_parsing(
    messages: list[dict[str, str]],
    response_model: BaseModel,
    verbose: bool = False,
    model: str | None = None,
    parsing_model: str | None = None,
    **kwargs,
) -> BaseModel:
    """
    Runs a native call using the specified model, then parses the output into the desired Pydantic model.
    """
    system_message_addition = system_message_addition_for_parsing(response_model)
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_message_addition}] + messages
    else:
        messages[0]["content"] += "\n\n" + system_message_addition

    native_output: str = await query_api_chat_native(
        messages=messages,
        verbose=verbose,
        model=model,
        **kwargs,
    )

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"Native output: {native_output}")

    messages.append({"role": "assistant", "content": native_output})

    parsed_response = await query_parse_last_response_into_format(
        messages=messages,
        response_model=response_model,
        verbose=verbose,
        model=parsing_model,
        **kwargs,
    )
    if verbose or os.getenv("VERBOSE") == "True":
        print(f"Parsed response: {parsed_response}")

    return parsed_response


@logfire.instrument("query_api_chat_sync_with_parsing", extract_args=True)
def query_api_chat_sync_with_parsing(
    messages: list[dict[str, str]],
    response_model: BaseModel,
    verbose: bool = False,
    model: str | None = None,
    parsing_model: str | None = None,
    **kwargs,
) -> BaseModel:
    """
    Runs a native call using the specified model, then parses the output into the desired Pydantic model.
    """
    system_message_addition = system_message_addition_for_parsing(response_model)

    system_message_addition = system_message_addition_for_parsing(response_model)

    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_message_addition}] + messages
    else:
        messages[0]["content"] += "\n\n" + system_message_addition

    native_output: str = query_api_chat_sync_native(
        messages=messages, verbose=verbose, model=model, **kwargs
    )
    if verbose or os.getenv("VERBOSE") == "True":
        print(f"Native output: {native_output}")

    messages.append({"role": "assistant", "content": native_output})
    parsed_response = query_parse_last_response_into_format_sync(
        messages=messages,
        response_model=response_model,
        verbose=verbose,
        model=parsing_model,
        **kwargs,
    )
    if verbose or os.getenv("VERBOSE") == "True":
        print(f"Parsed response: {parsed_response}")
    return parsed_response


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

    local_semaphore = asyncio.Semaphore(max_concurrent_queries)

    async def call_func(sem, func, datapoint):
        async with sem:
            return await func(datapoint)

    tasks = [call_func(local_semaphore, func, d) for d in data]
    return await asyncio.gather(*tasks)


@embeddings_cache
async def get_embedding(
    text: str,
    embedding_model: str = "text-embedding-3-small",
    model: str = "gpt-4o-mini-2024-07-18",
) -> list[float]:
    # model is largely ignored because we currently can't use the same model for both the embedding and the completion
    client, _, _ = get_client_pydantic(model, use_async=True)
    response = await client.client.embeddings.create(input=text, model=embedding_model)
    return response.data[0].embedding


@embeddings_cache
def get_embeddings_sync(
    texts: list[str],
    embedding_model: str = "text-embedding-3-small",
    model: str = "gpt-4o-mini-2024-07-18",
) -> list[list[float]]:
    # model is largely ignored because we currently can't use the same model for both the embedding and the completion
    client, _, _ = get_client_pydantic(model, use_async=False)
    response = client.client.embeddings.create(input=texts, model=embedding_model)
    return [e.embedding for e in response.data]


@embeddings_cache
def get_embedding_sync(
    text: str,
    embedding_model: str = "text-embedding-3-small",
    model: str = "gpt-4o-mini-2024-07-18",
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
