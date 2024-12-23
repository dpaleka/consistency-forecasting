# Python utils
This library provides reusable Python utilities and helper functions. While initially developed for the consistency forecasting project, it aims to be general-purpose and reusable across projects.
Project-specific code should be placed in the relevant project directories rather than this common library.

This code is licensed under the terms specified in [LICENSE](LICENSE).

## .env
Create an .env file with your OPENAI_API_KEY, and other secrets, if necessary. Place it in the root of the project, similar to [.env.example](../../.env.example).

## OpenRouter
Most models can alternatively be called in OpenAI-compliant format through [OpenRouter](https://openrouter.ai/).
To use OpenRouter, set the environment variable `USE_OPENROUTER=True` and provide your OpenRouter API key in the `OPENROUTER_API_KEY` environment variable. Note: this will route all model calls (incl. to OpenAI and Anthropic models) through OpenRouter.

## LLM API requests
For standard queries, prefer using methods in the `llm_utils.py` module over dealing with the `openai` package directly.
Most likely, the only methods you're going to need in the first weeks are:
- `query_api_chat_sync`: takes `model : str` and `messages : list[dict[str, str]]` in OpenAI chat format, and queries the corresponding model. Returns the `response_text : str`. It behaves like a normal function. Is cached by default, set `NO_CACHE=True` if you don't want to cache.
Optionally, set a Pydantic `response_model` to force the output to conform to some schema.
- `query_api_chat`: async version of the above. Is cached by default, set `NO_CACHE=True` if you don't want to cache.
- `parallelized_call`: run some async `func` over `data: list[str]`, "in parallel". Usually the `func` will create `messages` and then call `query_api_chat`; but you can provide an arbitrary `func` you implemented. Use when just running a for loop is too slow for you.

If you need a complex LLM request (e.g. logprobs), and you think it's not an one-off, implement another function in `llm_utils.py` and use that.

## Logging all LLM API calls
- To see logs of every query `instructor` sends to LLM APIs, set `LOGGING_DEBUG=True`. 
This has a zillion side effects on what gets printed.
As of Aug 2024, there is no way to avoid this when using `instructor`. See https://github.com/jxnl/instructor/pull/911 or https://github.com/jxnl/instructor/issues/767.
Luckily, `DEBUG` is the most verbose logging level, so every use of `logging` in other parts of the code will still print what it prints without this flag.
- To see every query and response to the LLM APIs, use the flag `VERBOSE=True` in `.env` or when calling the script. Other code (not in this folder) may use the `VERBOSE` flag to print details about its logic as well.

### Pydantic Logfire
If you use [Logfire](https://pydantic.dev/logfire), you can set `USE_LOGFIRE=True` in `.env` or when calling the script.
Please follow the Logfire setup instructions from the official documentation beforehand, and run a minimal working example outside the project to make sure it works on your machine.

To log some function to Logfire (even outside this folder), make a decorator around it:
```python
import logfire
import common.llm_utils # to make sure logfire is configured
@logfire.instrument("query_api_chat", extract_args=True)
def func(args):
    ...
```

## async
By default, prefer use async versions of LLM calls to external providers such as OpenAI.
This is because (1) it often happens that iteration speed is bottlenecked on waiting for many requests done sequentially;
(2) using `asyncio.Semaphore` is *so much better* than messing with threads and parallelism.
If you're not familiar with async/await, read the Python documentation on coroutines.

This means you need to run `await` when calling coroutines (`async def` functions), and define the functions that call coroutines as `async def` as well.
Moreover, if the code is running as a Python script (and not a Jupyter notebook or similar), you need to call `asyncio.run()` somewhere in the code, otherwise you'll get an error.
If you want to run your code both as a script and as a Jupyter/iPython notebook, you can use [`nest_asyncio`](https://github.com/erdewit/nest_asyncio).


## Caching requests
To ignore caching completely, set the environment variable `NO_CACHE` to `True`.
Essentially, run `export NO_CACHE=True` in your shell before running anything, 
or do `NO_CACHE=True python3 your_script.py`, or `os.environ['NO_CACHE'] = 'True'` in your Jupyter notebook.

Whenever you want to cache a new `BaseModel` response, add it to [`perscache.py`](perscache.py).

If you want to cache requests:
By default, try caching in a local folder. Set `LOCAL_CACHE={cache_folder}`, e.g. `LOCAL_CACHE=.cache/`.

If `NO_CACHE=False` and `LOCAL_CACHE=False` or is not set, 
we default to using Redis. 
This should work out of the box once you [install Redis](https://redis.io/docs/install/install-redis/).

### Advanced caching settings
- `NO_READ_CACHE=True` will not read from cache, but still write to it.
- `NO_WRITE_CACHE=True` will not write to cache, but still read from it.

In local caching, all the calls are stored in files with names like `{function_name}-{hash_of_args}.json`.


### Interactions of caching and logging
Caveat if you use Pydantic Logfire: if the function has a cache decorator and the result of a function call is retrieved from cache, Logfire will not display it properly.
Some basic stats like time taken, arguments, etc. will still be shown, but Logfire will not show the output of the function.
As of Aug 2024, it is unclear how to fix this.
