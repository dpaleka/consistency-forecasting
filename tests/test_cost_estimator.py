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
    )
    y = query_api_chat_sync(
        prompt,
        model=model,
        response_model=response_model,
        simulate=False,
        cost_estimation={"cost_estimator": ce_real},
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

    checks = [
        {
            "check": ce_sim.input_tokens * 0.8
            <= ce_real.input_tokens
            <= ce_sim.input_tokens * 1.2,
            "error": "input tokens estimate not in range",
        },
        {
            "check": ce_sim.output_tokens_range[0] <= ce_real.output_tokens_range[0],
            "error": "output tokens estimate too high",
        },
        {
            "check": ce_sim.output_tokens_range[1] >= ce_real.output_tokens_range[1],
            "error": "output tokens estimate too low",
        },
        {
            "check": ce_sim.cost_range[0] <= ce_real.cost_range[0],
            "error": "cost estimate too high",
        },
        {
            "check": ce_sim.cost_range[1] >= ce_real.cost_range[1],
            "error": "cost estimate too low",
        },
        {
            "check": ce_sim.time_range[0] <= ce_real.time_range[0],
            "error": "time estimate too high",
        },
        {
            "check": ce_sim.time_range[1] >= ce_real.time_range[1],
            "error": "time estimate too low",
        },
    ]

    failures = [c for c in checks if not c["check"]]
    assert not failures, [f["error"] for f in failures]