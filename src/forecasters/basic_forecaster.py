from forecasters.forecaster import Forecaster
from common.datatypes import (
    ForecastingQuestion_stripped,
    ForecastingQuestion,
    Forecast,
    Prob,
)
from common.llm_utils import answer, answer_sync, Example
from common.utils import make_json_serializable


BASIC_FORECASTER_PREFACE = (
    "You are an informed and well-calibrated forecaster. I need you to give me "
    "your best probability estimate for the following sentence or question resolving YES. "
    "Your answer should be a float between 0 and 1, with nothing else in your response."
)


class BasicForecaster(Forecaster):
    def __init__(self, model: str, preface: str = None, examples: list = None):
        self.model = model
        self.preface = preface or BASIC_FORECASTER_PREFACE
        self.examples = examples or []

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
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
            "examples": make_json_serializable(self.examples),
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
    def __init__(self, model: str, preface: str = None, examples: list = None):
        super().__init__(model=model, preface=preface)
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


class BasicForecasterTextBeforeParsing(BasicForecaster):
    def __init__(
        self,
        model: str,
        preface: str | None = None,
        examples: list | None = None,
        parsing_model: str = "gpt-4o-mini-2024-07-18",
    ):
        super().__init__(model=model, preface=preface, examples=examples)
        self.parsing_model = parsing_model

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        print(f"LLM API request: {fq.to_str_forecast_mode()}...")
        response = answer_sync(
            prompt=fq.to_str_forecast_mode(),
            preface=self.preface,
            examples=self.examples,
            response_model=Prob,
            model=self.model,
            with_parsing=True,
            parsing_model=self.parsing_model,
            **kwargs,
        )
        print(f"LLM API response: {response}")
        return Forecast(prob=response.prob, metadata=None)

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        print(f"LLM API request: {fq.to_str_forecast_mode()}...")
        response = await answer(
            prompt=fq.to_str_forecast_mode(),
            preface=self.preface,
            examples=self.examples,
            response_model=Prob,
            model=self.model,
            with_parsing=True,
            parsing_model=self.parsing_model,
            **kwargs,
        )
        print(f"LLM API response: {response}")
        return Forecast(prob=response.prob, metadata=None)

    def dump_config(self):
        config = super().dump_config()
        config["parsing_model"] = self.parsing_model
        return config

    @classmethod
    def load_config(cls, config):
        return cls(
            model=config["model"],
            preface=config["preface"],
            examples=[
                Example(
                    user=ForecastingQuestion_stripped.load_model_json(e["user"]),
                    assistant=e["assistant"],
                )
                for e in config["examples"]
            ],
            parsing_model=config.get("parsing_model", "gpt-4o-mini-2024-07-18"),
        )
