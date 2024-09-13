from .forecaster import Forecaster
from common.datatypes import (
    ForecastingQuestion_stripped,
    ForecastingQuestion,
    Forecast,
    Prob,
)
from common.llm_utils import answer, answer_sync, Example


class BasicForecaster(Forecaster):
    def __init__(self, model: str, preface: str = None, examples: list = None):
        self.model = model
        self.preface = preface or (
            "You are an informed and well-calibrated forecaster. I need you to give me "
            "your best probability estimate for the following sentence or question resolving YES. "
            "Your answer should be a float between 0 and 1, with nothing else in your response."
        )
        self.examples = examples or []

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        print("AAA")
        print(f"LLM API request: {fq.to_str_forecast_mode()}...")
        response = answer_sync(
            prompt=fq.to_str_forecast_mode(),
            preface=self.preface,
            examples=self.examples,
            response_model=Prob,
            model=self.model,
            **kwargs,
        )
        print(f"LLM API response: {response}")
        return Forecast(prob=response.prob, metadata=None)

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> float:
        print(f"LLM API request: {fq.to_str_forecast_mode()}...")
        response = await answer(
            prompt=fq.to_str_forecast_mode(),
            preface=self.preface,
            examples=self.examples,
            response_model=Prob,
            model=self.model,
            **kwargs,
        )
        print(f"LLM API response: {response}")
        return Forecast(prob=response.prob, metadata=None)

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
            preface=config["preface"],
            examples=[
                Example(
                    user=ForecastingQuestion_stripped.load_model_json(e["user"]),
                    assistant=e["assistant"],
                )
                for e in config["examples"]
            ],
        )


class BasicForecasterWithExamples(BasicForecaster):
    def __init__(self, preface: str = None, examples: list = None):
        super().__init__(preface=preface)
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
                assistant="0.03",
            )
        ]
