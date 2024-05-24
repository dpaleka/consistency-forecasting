# Python utils library, specialized for the consistency forecasting project
Feel free to add your own utils if you don't like the setup. You can also create PRs to modify the existing ones.
Try not to add code that's very specific to the consistency forecasting project; in most cases a better place for that code is in the relevant directory of that part of the project.

The licensing of this code is governed by [LICENSE](LICENSE).

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
- `parallelized_request`: run some async `func` over `data: list[str]`, "in parallel". Usually the `func` will create `messages` and then call `query_api_chat`; but you can provide an arbitrary `func` you implemented. Use when just running a for loop is too slow for you.

If you need a complex LLM request (e.g. logprobs), and you think it's not an one-off, implement another function in `llm_utils.py` and use that.

### Logging all LLM API calls
To see every query and response to the LLM APIs, use the flag `VERBOSE=True` in `.env` or when calling the script.

## async
By default, prefer use async versions of LLM calls to external providers such as OpenAI.
This is because (1) it often happens that iteration speed is bottlenecked on waiting for many requests done sequentially;
(2) using `asyncio.Semaphore` is *so much better* than messing with threads and parallelism.
If you're not familiar with async/await, read the Python documentation on coroutines.

This means you need to run `await` when calling coroutines (`async def` functions), and define the functions that call coroutines as `async def` as well.
Moreover, if the code is running as a Python script (and not a Jupyter notebook or similar), you need to call `asyncio.run()` somewhere in the code, otherwise you'll get an error.
If you want to run your code both as a script and as a Jupyter/iPython notebook, you can use [`nest_asyncio`](https://github.com/erdewit/nest_asyncio).

## caching requests
To ignore caching completely, set the environment variable `NO_CACHE` to `True`.
Essentially, run `export NO_CACHE=True` in your shell before running anything, 
or do `NO_CACHE=True python3 your_script.py`, or `os.environ['NO_CACHE'] = 'True'` in your Jupyter notebook.

Whenever you want to cache a new `BaseModel` response, add it to [`perscache.py`](perscache.py).

Default caching uses Redis, and should work out of the box once you [install Redis](https://redis.io/docs/install/install-redis/).
If you don't want to use Redis, but you still want to cache, use `LOCAL_CACHE={cache_folder}`, e.g. `LOCAL_CACHE=.cache/`.
