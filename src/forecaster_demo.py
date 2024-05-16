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
with open(get_data_path() / "fq/real/questions_cleaned_formatted.jsonl", "r") as file:
    for line in file:
        data.append(json.loads(line))

# %%
sample_question = data[0]
print(sample_question["title"])


# %%
fq = ForecastingQuestion(**sample_question)

# %% [markdown]
# ### Testing "Advanced Forecaster"

# %%
from forecasters.advanced_forecaster import AdvancedForecaster

af = AdvancedForecaster(
    MAX_WORDS_NEWSCATCHER=5,
    MAX_WORDS_GNEWS=8,
    BASE_REASONING_MODEL_NAMES=["gpt-3.5-turbo-1106", "gpt-3.5-turbo-1106"],
)

# %%
import asyncio


final_prob = asyncio.run(af.call_async(sentence=fq))
# final_prob = await af.call_async(sentence=fq)

# %%
print("Final LLM probability", final_prob)

# %% [markdown]
# Now we test the two procedures that make up AdvancedForecaster: retrieval and reasoning.
#
