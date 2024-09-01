from .forecaster import Forecaster
from common.datatypes import (
    ForecastingQuestion_stripped,
    ForecastingQuestion,
    Forecast,
    Prob_cot,
)
from common.llm_utils import answer, answer_sync, Example


class COT_Forecaster(Forecaster):
    def __init__(self, preface: str = None, examples: list = None):
        self.preface = preface or (
            "You are an informed and well-calibrated forecaster. I need you to give me "
            "your best probability estimate for the following sentence or question resolving YES. "
            "I want you to first provide a reasoning for your answer, and then give me the probability. "
            "Your answer should be in the format: 'Reasoning: [your reasoning here] Probability: [float between 0 and 1]'"
        )

        self.examples = examples or [
            Example(
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
        ]

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        print(f"LLM API request: {fq.to_str_forecast_mode()}...")
        response = answer_sync(
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
            "preface": self.preface,
            "examples": [
                {"user": e.user.model_dump_json(), "assistant": e.assistant}
                for e in self.examples
            ],
        }

    @classmethod
    def load_config(cls, config):
        return cls(
            preface=config["preface"],
            examples=[
                Example(
                    user=ForecastingQuestion_stripped.model_validate_json(e["user"]),
                    assistant=e["assistant"],
                )
                for e in config["examples"]
            ],
        )
