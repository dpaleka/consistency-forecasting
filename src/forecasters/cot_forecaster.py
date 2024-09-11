from .forecaster import Forecaster
from common.datatypes import (
    ForecastingQuestion,
    ForecastingQuestion_stripped,
    Prob_cot,
    Prob,
    reasoning_field,
    PlainText,
    Forecast,
)
from common.llm_utils import (
    answer,
    answer_sync,
    answer_messages,
    answer_messages_sync,
    Example,
)
from datetime import datetime


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


class CoT_multistep_Forecaster(Forecaster):
    def __init__(
        self, preface: str = "You are a helpful assistant", examples: list[str] = []
    ):
        # TODO define nice preface and examples
        self.preface = preface
        self.examples = examples

        """
        self.first_user_suffix = "\n\nPlease provide a step-by-step analysis to arrive at a probability estimate. We will do this in {self.steps} steps."
        self.intermezzos = [
            "Thank you. Now, let's continue with step {step + 1} of our analysis."
            for step in range(self.steps - 1)
        ]
        """

    def call(
        self,
        user_prompts_lists: list[str],
        examples=None,
        include_metadata=False,
        **kwargs,
    ) -> Forecast:
        self.steps = len(user_prompts_lists)
        assert self.steps >= 1, "Must have at least one step"

        messages = [
            {"role": "system", "content": self.preface},
            # {
            #    "role": "user",
            #    "content": f"Question: {sentence.__str__()}{self.first_user_suffix}",
            # },
        ]

        num_examples = 0
        ## comment this out for examples
        # examples = None
        if examples:
            for e in examples:
                messages.extend(e)
                num_examples += 1

        else:
            print(
                "\n\nIgnoring examples!!! Try debugging by uncommenting examples=None and using a simpler list of examples \n\n"
            )

        step_formats = {
            **{i: PlainText for i in range(self.steps - 1)},
            self.steps - 1: Prob,
        }

        for step_idx in range(self.steps):
            new_msg = {"role": "user", "content": user_prompts_lists[step_idx]}
            ##ad new user input
            messages.append(new_msg)

            # response = await answer_messages(
            response = answer_messages_sync(
                messages=messages,
                # examples=self.examples,
                response_model=step_formats[step_idx],
                **kwargs,
            )
            ## add llm output to stream
            messages.append({"role": "assistant", "content": reasoning_field(response)})

        ## this includes examples if it's there!  It's just all messages
        result = {
            "chain_of_thought": "\n\n".join(
                [m["content"] for m in messages if m["role"] == "assistant"]
            ),
            "response": response,
        }

        self.result = result
        if include_metadata:
            result["metadata"] = {
                "model": kwargs.get("model", "default_model"),
                "timestamp": datetime.now().isoformat(),
                "user_prompts": user_prompts_lists,
                "chain_of_thought": result["chain_of_thought"],
                "steps": self.steps,
            }

            return Forecast(prob=result["response"].prob, metadata=result["metadata"])

        return Forecast(prob=result["response"].prob)

    async def call_async(
        self,
        user_prompts_lists: list[str],
        examples=None,
        include_metadata=False,
        **kwargs,
    ) -> Forecast:
        self.steps = len(user_prompts_lists)
        assert self.steps >= 1, "Must have at least one step"

        messages = [
            {"role": "system", "content": self.preface},
            # {
            #    "role": "user",
            #    "content": f"Question: {sentence.__str__()}{self.first_user_suffix}",
            # },
        ]

        num_examples = 0
        ## comment this out for examples
        # examples = None
        if examples:
            for e in examples:
                messages.extend(e)
                num_examples += 1

        else:
            print(
                "\n\nIgnoring examples!!! Try debugging by uncommenting examples=None and using a simpler list of examples \n\n"
            )

        step_formats = {
            **{i: PlainText for i in range(self.steps - 1)},
            self.steps - 1: Prob,
        }

        for step_idx in range(self.steps):
            new_msg = {"role": "user", "content": user_prompts_lists[step_idx]}
            ##ad new user input
            messages.append(new_msg)

            # response = await answer_messages(
            response = await answer_messages(
                messages=messages,
                # examples=self.examples,
                response_model=step_formats[step_idx],
                **kwargs,
            )
            ## add llm output to stream
            messages.append({"role": "assistant", "content": reasoning_field(response)})

        ## this includes examples if it's there!  It's just all messages
        result = {
            "chain_of_thought": "\n\n".join(
                [m["content"] for m in messages if m["role"] == "assistant"]
            ),
            "response": response,
        }

        if include_metadata:
            result["metadata"] = {
                "model": kwargs.get("model", "default_model"),
                "timestamp": datetime.now().isoformat(),
                "user_prompts": user_prompts_lists,
                "steps": self.steps,
            }

        self.result = result
        if include_metadata:
            result["metadata"] = {
                "model": kwargs.get("model", "default_model"),
                "timestamp": datetime.now().isoformat(),
                "user_prompts": user_prompts_lists,
                "chain_of_thought": result["chain_of_thought"],
                "steps": self.steps,
            }

            return Forecast(prob=result["response"].prob, metadata=result["metadata"])

        return Forecast(prob=result["response"].prob)

    def dump_config(self):
        return {
            "preface": self.preface,
            "examples": [
                {
                    "user": e.user.model_dump_json(),
                    "assistant": e.assistant.model_dump_json(),
                }
                for e in self.examples
            ],
        }

    @classmethod
    def load_config(cls, config):
        return cls(**config)
