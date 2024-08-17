import pytest
from pydantic import BaseModel
from common.cost_estimator import CostItem, CostEstimator
from common.llm_utils import query_api_chat, query_api_chat_sync

PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "Give an example of a USER in the specification provided.",
    }
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
]


@pytest.mark.parametrize("params", PARAMS)
def test_estimate_contains_exact(params, request):

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
    )
    y = query_api_chat_sync(
        prompt,
        model=model,
        response_model=response_model,
        simulate=False,
        cost_estimation={"cost_estimator": ce_real},
    )
    print("\n---")
    print("Estimated cost:", ce_sim.cost_range)
    print("Real cost:", ce_real.cost_range)
    print("Estimated time:", ce_sim.time_range)
    print("Real time:", ce_real.time_range)
    assert ce_sim.cost_range[0] <= ce_real.cost_range[0]
    assert ce_sim.cost_range[1] >= ce_real.cost_range[1]
    assert ce_sim.time_range[0] <= ce_real.time_range[0]
    assert ce_sim.time_range[1] >= ce_real.time_range[1]
