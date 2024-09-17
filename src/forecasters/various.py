from forecasters.forecaster import Forecaster
from typing import Any
from common.datatypes import ForecastingQuestion, Forecast, Prob_cot
from common.llm_utils import (
    query_api_chat_native,
    query_parse_last_response_into_format,
    Example,
)
from common.utils import normalize_date_format
import asyncio


class BaselineForecaster(Forecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        return Forecast(prob=0.5, metadata={})

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        return asyncio.run(self.call_async(fq, **kwargs))

    def dump_config(self) -> dict[str, Any]:
        return {}


CHEATING_MODEL_EXAMPLE_QUESTION = ForecastingQuestion(
    title="Who will be the Democratic nominee for the 2020 US Presidential Election? (Hillary Clinton)",
    body="This question will resolve as **Yes** for the candidate below who is selected by the Democratic National Convention as the nominee for the 2020 US Presidential Election.  All other candidates will resolve as **No**.  This question is not restricted to the candidates currently below; other options may be added in the future.\n\nFor this question, it is not relevant who recieves the Democratic nomination on the day of the 2020 US election, it is solely determined by who is selected by the delegates of the [Democratic National Convention](https://en.wikipedia.org/wiki/Democratic_National_Convention).",
    resolution_date=normalize_date_format("2020-08-01 04:00:00"),
    created_date=normalize_date_format("2019-02-22 23:40:52"),
    question_type="binary",
    data_source="metaculus",
)
CHEATING_MODEL_EXAMPLE_ANSWER: str = """
Let's rephrase the resolution criteria: 
- The question resolves as **Yes** if Hillary Clinton is selected as the Democratic nominee for the 2020 US Presidential Election;
- The question resolves as **No** otherwise.

My search says Joe Biden was selected as the Democratic nominee for the 2020 US Presidential Election.
Thus, the question resolves as **No**, hence my final output is 0.
"""


def render_question_to_prompt(fq: ForecastingQuestion | dict) -> str:
    if isinstance(fq, ForecastingQuestion):
        fq_dict = fq.to_dict_forecast_mode()
    else:
        fq_dict = fq
    return f"""
Resolution date: {fq_dict["resolution_date"]}

Title:
{fq_dict["title"]}

Body:
{fq_dict["body"]}


Created date: {fq_dict["created_date"]} (if this is None, ignore it)
"""


today_date = "15 September 2024"

CHEATING_MODEL_PROMPT = f"""
You are acting as an administrator for a forecasting platform.
Given the full data of a forecasting question, you must determine whether the question resolves YES or NO.
If YES, your final output should be 1. If NO, your final output should be 0.

Today's date is {today_date} or later. You can search for events that have happened before this date.
The question has resolved before your knowledge cutoff date; if there is evidence of a certain event happening, you should be able to find it.
Please do not make up events that have not been observed. 

If you are really unsure on the resolution of the question, output a probability between 0 and 1 based on your reasoning.
This should happen very rarely, about once in 50 questions.

Think everything through step-by-step before coming to a conclusion.
"""


class PerplexityAnswerForecaster(Forecaster):
    def __init__(
        self,
        model: str = "perplexity/llama-3.1-sonar-huge-128k-online",
        examples: list[Example] | None = None,
        prompt_template: str | None = None,
        parsing_model: str = "gpt-4o-mini-2024-07-18",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            model.startswith("perplexity/") and "online" in model
        ), "Model must start with perplexity/ and be an online model"
        self.model = model
        self.examples = examples or [
            Example(CHEATING_MODEL_EXAMPLE_QUESTION, CHEATING_MODEL_EXAMPLE_ANSWER)
        ]
        self.prompt = prompt_template or CHEATING_MODEL_PROMPT
        self.parsing_model = parsing_model

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        prompt = self.prompt
        for example in self.examples:
            prompt += "\n\nExample question:\n" + render_question_to_prompt(
                example.user
            )
            prompt += "\n\nExample answer:\n" + example.assistant
        prompt += "\n\nQuestion:\n" + render_question_to_prompt(fq)

        messages = [{"role": "user", "content": prompt}]

        native_output = await query_api_chat_native(
            messages, model=self.model, **kwargs
        )
        messages += [{"role": "assistant", "content": native_output}]
        response = await query_parse_last_response_into_format(
            messages, Prob_cot, model=self.parsing_model
        )
        return Forecast(
            prob=response.prob, metadata={"chain_of_thought": response.chain_of_thought}
        )

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        return asyncio.run(self.call_async(fq, **kwargs))

    def dump_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "parsing_model": self.parsing_model,
        }
