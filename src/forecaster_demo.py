# %%
import sys
from common.path_utils import get_src_path, get_data_path

sys.path.append(str(get_src_path()))

import os

os.environ["LOCAL_CACHE"] = ".forecaster_cache"


from common.datatypes import ForecastingQuestion
import json

# llm_forecasting imports


# %% [markdown]
# ### Load Data

# %%
data = []
with open(get_data_path() / "other/forecaster_testing_q.jsonl", "r") as file:
    for line in file:
        data.append(json.loads(line))

# %%
sample_question = data[1]
print(sample_question["title"])


# %%
fq = ForecastingQuestion(**sample_question)

# %% [markdown]
# ### Testing "Advanced Forecaster"

# %%

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  # configure root logger

# %%
from forecasters.advanced_forecaster import AdvancedForecaster

af = AdvancedForecaster(
    MAX_WORDS_NEWSCATCHER=5,
    MAX_WORDS_GNEWS=8,
    SEARCH_QUERY_MODEL_NAME="gpt-4o-2024-05-13",
    SUMMARIZATION_MODEL_NAME="gpt-4o-2024-05-13",
    BASE_REASONING_MODEL_NAMES=["gpt-4o-2024-05-13", "gpt-4o-2024-05-13"],
    RANKING_MODEL_NAME="gpt-4o-2024-05-13",
    AGGREGATION_MODEL_NAME="gpt-4o-2024-05-13",
)

# %%
import asyncio


if os.getenv("IPYKERNEL_CELL_NAME", None) is None:
    final_prob = asyncio.run(af.call_async_full(sentence=fq))
else:
    final_prob = await af.call_async_full(sentence=fq)  # noqa

# %%
print("Final LLM probability", final_prob)

# %% [markdown]
# Now we test the two procedures that make up AdvancedForecaster: retrieval and reasoning.
#
