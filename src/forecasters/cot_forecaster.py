from .forecaster import Forecaster
from common.datatypes import (
    ForecastingQuestion,
    ForecastingQuestion_stripped,
    Prob_cot,
    Prob,
    reasoning_field,
    PlainText,
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
    def __init__(self, preface: str = None, examples: list[Example] = None):
        self.preface = preface or (
            "You are an informed and well-calibrated forecaster. I need you to give me "
            "your best probability estimate for the following sentence or question resolving YES. "
            "I want you to first provide a reasoning for your answer, and then give me the probability. "
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
                assistant=Prob_cot(
                    chain_of_thought=(
                        "As of 2021, there are no skyscrapers a mile tall. There are also "
                        "no plans to build any mile tall skyscraper in new york. The probability "
                        "is: 0.03"
                    ),
                    prob=0.03,
                ),
            )
        ]

    def call(self, sentence: ForecastingQuestion, **kwargs) -> tuple[float, str]:
        response = answer_sync(
            prompt=sentence.__str__(),
            preface=self.preface,
            examples=self.examples,
            response_model=sentence.expected_answer_type(mode="cot"),
            **kwargs,
        )
        return response.prob, response.chain_of_thought

    async def call_async(
        self, sentence: ForecastingQuestion, include_metadata=False, **kwargs
    ) -> tuple[float, str]:
        response = await answer(
            prompt=sentence.__str__(),
            preface=self.preface,
            examples=self.examples,
            response_model=sentence.expected_answer_type(mode="cot"),
            **kwargs,
        )
        return response.prob, response.chain_of_thought

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


class CoT_multistep_Forecaster:
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
    ) -> float:
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

        print(f"{len(messages)=} after examples")
        print(f"messages[-1]: {messages[-1]}")
        print(f"{len(user_prompts_lists)=}")

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

            """print("\n\n\nADAMXXXXXXX")
            print(f"{len(messages)=}")
            print(f"messages[-2]: {messages[-2]}")
            print("")
            print(f"messages[-1]: {messages[-1]}")
            print("\n\n")"""

        ## this includes examples if it's there!  It's just all messages
        result = {
            "chain_of_thought": "\n\n".join(
                [m["content"] for m in messages if m["role"] == "assistant"]
            ),
            "prob": response.prob,
        }

        if include_metadata:
            result["metadata"] = {
                "model": kwargs.get("model", "default_model"),
                "timestamp": datetime.now().isoformat(),
                "user_prompts": user_prompts_lists,
                "steps": self.steps,
            }

        self.result = result

        return result["prob"]

    async def call_async(
        self,
        user_prompts_lists: list[str],
        examples=None,
        include_metadata=False,
        **kwargs,
    ) -> float:
        messages = [
            {"role": "system", "content": self.preface},
            # {
            #    "role": "user",
            #    "content": f"Question: {sentence.__str__()}{self.first_user_suffix}",
            # },
        ]

        if examples:
            for e in examples:
                messages.extend(e)

        for s in range(self.steps):
            new_msg = {"role": "user", "content": user_prompts_lists[s]}
            ##ad new user input
            messages.append(new_msg)

            response = await answer_messages(
                # response = answer_messages_sync(
                messages=messages,
                # examples=self.examples,
                response_model=Prob_cot,
                **kwargs,
            )
            ## add llm output to stream
            messages.append({"role": "assistant", "content": response.chain_of_thought})

        result = {
            "chain_of_thought": "\n\n".join(
                [m["content"] for m in messages if m["role"] == "assistant"]
            ),
            "prob": response.prob,
        }

        if include_metadata:
            result["metadata"] = {
                "model": kwargs.get("model", "default_model"),
                "timestamp": datetime.now().isoformat(),
                "user_prompts": user_prompts_lists,
                "steps": self.steps,
            }

        self.result = result
        return result["prob"]

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
