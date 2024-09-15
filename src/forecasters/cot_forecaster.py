from .forecaster import Forecaster
from common.datatypes import (
    ForecastingQuestion_stripped,
    ForecastingQuestion,
    Forecast,
    Prob_cot,
)
from common.llm_utils import answer, answer_sync, Example

COT_FORECASTER_PREFACE = (
    "You are an informed and well-calibrated forecaster. I need you to give me "
    "your best probability estimate for the following sentence or question resolving YES. "
    "I want you to first provide a reasoning for your answer, and then give me the probability. "
    "Your answer should be in the format: 'Reasoning: [your reasoning here] Probability: [float between 0 and 1]'"
)


class CoT_Forecaster(Forecaster):
    def __init__(
        self, model: str, preface: str | None = None, examples: list | None = None
    ):
        self.model = model
        self.preface = preface or COT_FORECASTER_PREFACE
        self.examples = examples

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        print(f"LLM API request: {fq.to_str_forecast_mode()}...")
        response = answer_sync(
            model=self.model,
            prompt=fq.to_str_forecast_mode(),
            preface=self.preface,
            examples=self.examples,
            response_model=Prob_cot,
            **kwargs,
        )
        print(f"LLM API response: {response}")
        return Forecast(
            prob=response.prob, metadata={"chain_of_thought": response.chain_of_thought}
        )

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        print(f"LLM API request: {fq.to_str_forecast_mode()}...")
        response = await answer(
            model=self.model,
            prompt=fq.to_str_forecast_mode(),
            preface=self.preface,
            examples=self.examples,
            response_model=Prob_cot,
            **kwargs,
        )
        print(f"LLM API response: {response}")
        return Forecast(
            prob=response.prob, metadata={"chain_of_thought": response.chain_of_thought}
        )

    def dump_config(self):
        return {
            "model": self.model,
            "preface": self.preface,
            "examples": [
                {"user": e.user.model_dump_json(), "assistant": e.assistant}
                for e in self.examples
            ],
        }

    @classmethod
    def load_config(cls, config):
        return cls(
            model=config["model"],
            preface=config["preface"],
            examples=[
                Example(
                    user=ForecastingQuestion_stripped.model_validate_json(e["user"]),
                    assistant=e["assistant"],
                )
                for e in config["examples"]
            ],
        )


COT_FORECASTER_EXAMPLE: Example = Example(
    user=ForecastingQuestion_stripped(
        title="Will Manhattan have a skyscraper a mile tall by 2030?",
        body=(
            "Resolves YES if at any point before 2030, there is at least "
            "one building in the NYC Borough of Manhattan (based on current "
            "geographic boundaries) that is at least a mile tall."
        ),
    ),
    assistant="Reasoning: As of 2021, there are no skyscrapers a mile tall. There are also "
    "no plans to build any mile tall skyscraper in New York. The tallest building "
    "currently under construction in Manhattan is only about a quarter of a mile tall. "
    "Given the technical challenges, enormous costs, and lack of current plans, it's "
    "highly unlikely that a mile-high skyscraper will be built in Manhattan by 2030. "
    "However, there's always a small chance of rapid technological advancements or "
    "unforeseen circumstances. Probability: 0.03",
)


class CoT_ForecasterWithExamples(CoT_Forecaster):
    def __init__(self, model: str, preface: str = None, examples: list = None):
        super().__init__(model, preface, examples)
        self.examples = examples or [COT_FORECASTER_EXAMPLE]


class CoT_ForecasterTextBeforeParsing(CoT_Forecaster):
    def __init__(
        self,
        model: str,
        preface: str = None,
        examples: list = None,
        parsing_model: str = "gpt-4o-mini-2024-07-18",
    ):
        super().__init__(model=model, preface=preface, examples=examples)
        self.parsing_model = parsing_model

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        print(f"LLM API request: {fq.to_str_forecast_mode()}...")
        response = answer_sync(
            model=self.model,
            prompt=fq.to_str_forecast_mode(),
            preface=self.preface,
            examples=self.examples,
            response_model=Prob_cot,
            with_parsing=True,
            parsing_model=self.parsing_model,
            **kwargs,
        )
        print(f"LLM API response: {response}")
        return Forecast(
            prob=response.prob, metadata={"chain_of_thought": response.chain_of_thought}
        )

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        print(f"LLM API request: {fq.to_str_forecast_mode()}...")
        response = await answer(
            model=self.model,
            prompt=fq.to_str_forecast_mode(),
            preface=self.preface,
            examples=self.examples,
            response_model=Prob_cot,
            with_parsing=True,
            parsing_model=self.parsing_model,
            **kwargs,
        )
        print(f"LLM API response: {response}")
        return Forecast(
            prob=response.prob, metadata={"chain_of_thought": response.chain_of_thought}
        )
