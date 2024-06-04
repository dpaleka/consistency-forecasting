from .forecaster import Forecaster
from common.datatypes import ForecastingQuestion_stripped, ForecastingQuestion
from common.llm_utils import answer, answer_sync, Example


class BasicForecaster(Forecaster):
    def __init__(self, preface: str = None, examples: list = None):
        self.preface = preface or (
            "You are an informed and well-calibrated forecaster. I need you to give me "
            "your best probability estimate for the following sentence or question resolving YES. "
            "Your answer should be a float between 0 and 1, with nothing else in your response."
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
                assistant="0.03",
            )
        ]

    def call(self, sentence: ForecastingQuestion, **kwargs) -> float:
        # Log the request details being sent to the OpenAI API
        print("Sending the following request to the LLM API:")
        print(f"Prompt: {sentence.__str__()}")
        print(f"Preface: {self.preface}")
        print(f"Examples: {self.examples}")
        print(f"Response Model: {sentence.expected_answer_type()}")
        response = answer_sync(
            prompt=sentence.__str__(),
            preface=self.preface,
            examples=self.examples,
            response_model=sentence.expected_answer_type(),
            **kwargs,
        )
        # Log the response from the OpenAI API
        print(f"Received the following response from OpenAI API: {response}")
        return response.prob

    async def call_async(self, sentence: ForecastingQuestion, **kwargs) -> float:
        # Log the request details being sent to the OpenAI API
        print("Sending the following request to the LLM API:")
        print(f"Prompt: {sentence.__str__()}")
        print(f"Preface: {self.preface}")
        print(f"Examples: {self.examples}")
        print(f"Response Model: {sentence.expected_answer_type()}")
        response = await answer(
            prompt=sentence.__str__(),
            preface=self.preface,
            examples=self.examples,
            response_model=sentence.expected_answer_type(),
            **kwargs,
        )
        # Log the response from the OpenAI API
        print(f"Received the following response from OpenAI API: {response}")
        return response.prob

    def dump_config(self):
        return {
            "preface": self.preface,
            "examples": [
                {"user": e.user.model_dump_json(), "assistant": e.assistant}
                for e in self.examples
            ],
        }
