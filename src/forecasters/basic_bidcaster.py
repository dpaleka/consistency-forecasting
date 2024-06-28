from .forecaster import Forecaster
from .basic_forecaster import BasicForecaster
from common.datatypes import (
    ForecastingQuestion_stripped,
    ForecastingQuestion,
    ForecastingQuestion_with_subsidy,
    BiddingQuestion,
    InformationPiece,
)
from common.llm_utils import answer, answer_sync, Example


class BasicBidcaster(Forecaster):
    preface_bidding_default = (
        "You are a well-informed and well-calibrated forecaster. `goal_questions` is "
        "the list of Forecasting Questions that you intend to trade on prediction markets, "
        "given along with the logarithmic market subsidy on each of these questions (i.e. "
        "you will earn a score of market_subsidy * (log(p') - log(p)) where p' is your "
        "forecast for the true outcome of that question, and p was the market forecast "
        "before you placed your trade). Your goal is to maximize your expected score."
        "\n\n"
        "But first before you do so, I will offer you the chance to get the answers to "
        "some related questions given in `offered_information`. You have no inherent "
        "interest in these questions (you are not trading on any prediction markets for "
        "these questions), but you can use the information to inform your trading decisions "
        "on the `goal_questions`."
        "\n\n"
        "Now, tell me your *bid* for the offered_information, i.e. how much are you willing to "
        "pay to get the answers to these questions. Remember: your goal is only to maximize "
        "your expected score on the `goal_questions`, so you should estimate how much knowing "
        "the answers to these questions will help you in that goal. In particular, the market "
        "subsidies on each of the `goal_questions` are given to you, so you can use that to "
        "inform your bid."
        "\n\n"
        "Another way of thinking about your task is that you want to estimate and give me the "
        "weighted (by market subsidy) sum of the *mutual informations* of the offered_information "
        "with the goal_question: sum_i market_subsidy_i * MI(offered_information, goal_question_i)."
    )

    examples_bidding_default = [
        Example(
            user=BiddingQuestion(
                offered_information=[
                    InformationPiece(
                        title="Will Donald Trump be the GOP presidential nominee in 2024?",
                        body=(
                            "Resolves YES if Donald Trump is the GOP presidential nominee for the "
                            "2024 US presidential election, as determined by the Republican National "
                            "Committee or equivalent body."
                        ),
                        question_type="binary",
                    )
                ],
                goal_questions=[
                    ForecastingQuestion_with_subsidy(
                        fq=ForecastingQuestion_stripped(
                            title="Will Donald Trump be elected US president in 2024?",
                            body=(
                                "Resolves YES if Donald Trump is elected as the President of the United "
                                "States in the 2024 US presidential election, as determined by the US "
                                "Congress or equivalent body."
                            ),
                            question_type="binary",
                        ),
                        market_subsidy=1.0,
                    )
                ],
            ),
            assistant=0.342,  # to reproduce, estimate P(nominee)=0.5, P(elected|nominee)=0.7
        ),
        Example(
            user=BiddingQuestion(
                offered_information=[
                    InformationPiece(
                        title="Will it rain in London on July 4, 2024?",
                        body=(
                            "Resolves YES if it rains in London on July 4, 2024, as determined by the "
                            "UK Met Office."
                        ),
                        question_type="binary",
                    )
                ],
                goal_questions=[
                    ForecastingQuestion_with_subsidy(
                        fq=ForecastingQuestion_stripped(
                            title="Will Donald Trump be elected US president in 2024?",
                            body=(
                                "Resolves YES if Donald Trump is elected as the President of the United "
                                "States in the 2024 US presidential election, as determined by the US "
                                "Congress or equivalent body."
                            ),
                            question_type="binary",
                        ),
                        market_subsidy=2.4,
                    )
                ],
            ),
            assistant=0.0,
        ),
    ]

    def __init__(
        self,
        preface_bidding: str = None,
        examples_bidding: list = None,
        preface_forecasting: str = None,
        examples_forecasting: list = None,
    ):
        self.forecaster = BasicForecaster(preface_forecasting, examples_forecasting)
        self.preface_bidding = preface_bidding or self.preface_bidding_default
        self.examples_bidding = examples_bidding or self.examples_bidding_default

    def call_forecasting(self, sentence: ForecastingQuestion, **kwargs) -> float:
        return self.forecaster.call(sentence, **kwargs)

    async def call_forecasting_async(
        self, sentence: ForecastingQuestion, **kwargs
    ) -> float:
        return await self.forecaster.call_async(sentence, **kwargs)

    def call_bidding(self, sentence: BiddingQuestion, **kwargs) -> float:
        # Log the request details being sent to the OpenAI API
        print("Sending the following request to the LLM API:")
        print(f"Prompt: {sentence.__str__()}")
        print(f"Preface: {self.preface_bidding}")
        print(f"Examples: {self.examples_bidding}")
        print(f"Response Model: {sentence.expected_answer_type()}")

        response = answer_sync(
            prompt=sentence.__str__(),
            preface=self.preface_bidding,
            examples=self.examples_bidding,
            response_model=sentence.expected_answer_type(mode="default"),
            **kwargs,
        )

        return response.bid

    async def call_bidding_async(self, sentence: BiddingQuestion, **kwargs) -> float:
        # Log the request details being sent to the OpenAI API
        print("Sending the following request to the LLM API:")
        print(f"Prompt: {sentence.__str__()}")
        print(f"Preface: {self.preface_bidding}")
        print(f"Examples: {self.examples_bidding}")
        print(f"Response Model: {sentence.expected_answer_type()}")

        response = await answer(
            prompt=sentence.__str__(),
            preface=self.preface_bidding,
            examples=self.examples_bidding,
            response_model=sentence.expected_answer_type(mode="default"),
            **kwargs,
        )

        return response.bid

    def call(self, sentence: ForecastingQuestion | BiddingQuestion, **kwargs) -> float:
        if isinstance(sentence, ForecastingQuestion):
            return self.call_forecasting(sentence, **kwargs)
        elif isinstance(sentence, BiddingQuestion):
            return self.call_bidding(sentence, **kwargs)
        else:
            raise ValueError(f"Unsupported datatype: {type(sentence)}")

    async def call_async(
        self, sentence: ForecastingQuestion | BiddingQuestion, **kwargs
    ) -> float:
        if isinstance(sentence, ForecastingQuestion):
            return await self.call_forecasting_async(sentence, **kwargs)
        elif isinstance(sentence, BiddingQuestion):
            return await self.call_bidding_async(sentence, **kwargs)
        else:
            raise ValueError(f"Unsupported datatype: {type(sentence)}")

    # def dump_config(self):
    #     return {
    #         "preface_bidding": self.preface_bidding,
    #         "examples_bidding": [
    #             {"user": e.user.model_dump_json(), "assistant": e.assistant}
    #             for e in self.examples_bidding
    #         ],
    #         "forecaster": self.forecaster.dump_config()
    #     }
