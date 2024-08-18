import pytest
from pydantic import BaseModel
from common.cost_estimator import CostItem, CostEstimator
from common.llm_utils import query_api_chat, query_api_chat_sync
from common.datatypes import ForecastingQuestion

PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "Give an example of a USER in the specification provided.",
    },
]
PROMPT2 = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "Give an example of a ForecastingQuestion in the specification provided.",
    },
]


class USER(BaseModel):
    name: str
    age: int
    location: str


PARAMS = [
    {
        "prompt": PROMPT,
        "model": "gpt-4o-mini",
        "response_model": USER,
    },
    {
        "prompt": PROMPT,
        "model": "gpt-4o",
        "response_model": USER,
    },
    {
        "prompt": PROMPT2,
        "model": "gpt-4o-mini",
        "response_model": ForecastingQuestion,
    },
]


@pytest.mark.parametrize("params", PARAMS)
def test_estimate_contains_exact(params):

    prompt, model, response_model = (
        params["prompt"],
        params["model"],
        params.get("response_model", None),
    )

    ce_sim = CostEstimator()
    ce_real = CostEstimator()
    x = query_api_chat_sync(
        prompt,
        model=model,
        response_model=response_model,
        simulate=True,
        cost_estimation={"cost_estimator": ce_sim},
        verbose=True,
    )
    y = query_api_chat_sync(
        prompt,
        model=model,
        response_model=response_model,
        simulate=False,
        cost_estimation={"cost_estimator": ce_real},
        verbose=True,
    )
    print("\n---")
    print("Estimated input tokens:", ce_sim.input_tokens)
    print("Real input tokens:", ce_real.input_tokens)
    print("Estimated output tokens:", ce_sim.output_tokens_range)
    print("Real output tokens:", ce_real.output_tokens_range)
    print("Estimated cost:", ce_sim.cost_range)
    print("Real cost:", ce_real.cost_range)
    print("Estimated time:", ce_sim.time_range)
    print("Real time:", ce_real.time_range)

    failures = []
    if not (
        ce_sim.input_tokens * 0.8 <= ce_real.input_tokens <= ce_sim.input_tokens * 1.2
    ):
        failures.append(
            f"input tokens real {ce_real.input_tokens} not within "
            f"20\% of estimate {ce_sim.input_tokens}"
        )
    if not (ce_sim.output_tokens_range[0] <= ce_real.output_tokens_range[0]):
        failures.append(
            f"output tokens estimate {ce_sim.output_tokens_range} too "
            f"high; real is {ce_real.output_tokens_range}"
        )
    if not (ce_sim.output_tokens_range[1] >= ce_real.output_tokens_range[1]):
        failures.append(
            f"output tokens estimate {ce_sim.output_tokens_range} too "
            f"low; real is {ce_real.output_tokens_range}"
        )
    if not (ce_sim.cost_range[0] <= ce_real.cost_range[0]):
        failures.append(
            f"cost estimate {ce_sim.cost_range} too high; real is "
            f"{ce_real.cost_range}"
        )
    if not (ce_sim.cost_range[1] >= ce_real.cost_range[1]):
        failures.append(
            f"cost estimate {ce_sim.cost_range} too low; real is "
            f"{ce_real.cost_range}"
        )
    if not (ce_sim.time_range[0] <= ce_real.time_range[0]):
        failures.append(
            f"time estimate {ce_sim.time_range} too high; real is "
            f"{ce_real.time_range}"
        )
    if not (ce_sim.time_range[1] >= ce_real.time_range[1]):
        failures.append(
            f"time estimate {ce_sim.time_range} too low; real is "
            f"{ce_real.time_range}"
        )
    assert not failures, "\n".join(failures)
