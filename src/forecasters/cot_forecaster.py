from forecasters.forecaster import Forecaster
from datetime import datetime
from common.utils import make_json_serializable
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
    Example,
    query_api_chat_native,
    query_api_chat,
    answer_messages_sync,
    answer_messages,
)
import asyncio

COT_FORECASTER_PREFACE = (
    "You are an informed and well-calibrated forecaster. I need you to give me "
    "your best probability estimate for the following question resolving YES. "
    "If you think it is likely the question resolves YES, the probability should be large; "
    "if you think it is unlikely the question resolves NO, the probability should be small. "
    "I want you to first provide a detailed reasoning for your answer, and then give me the probability. "
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
            "examples": make_json_serializable(self.examples),
        }


class CoT_multistep_Forecaster(Forecaster):
    def __init__(
        self,
        model,
        preface: str = "You are a helpful assistant",
        examples: list[str] = [],
    ):
        # TODO define nice preface and examples
        self.model = model
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
                model=self.model,
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
                "model": kwargs.get("model", self.model),
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
            "model": self.model,
            "preface": self.preface,
            "examples": make_json_serializable(self.examples),
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


class CoT_ForecasterAskThenParse(Forecaster):
    def __init__(
        self,
        model: str,
        preface: str = None,
        parsing_model: str = "gpt-4o-mini-2024-07-18",
    ):
        self.model = model
        self.preface = preface or COT_FORECASTER_PREFACE
        self.parsing_model = parsing_model

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        native_response = await query_api_chat_native(
            model=self.model,
            messages=[
                {"role": "system", "content": self.preface},
                {"role": "user", "content": fq.to_str_forecast_mode()},
            ],
            **kwargs,
        )
        parsed_response = await query_api_chat(
            response_model=Prob_cot,
            model=self.parsing_model,
            messages=[
                {
                    "role": "system",
                    "content": "Parse the user's message into the provided output format. ",
                },
                {"role": "user", "content": native_response},
            ],
            **kwargs,
        )
        return Forecast(
            prob=parsed_response.prob,
            metadata={"chain_of_thought": parsed_response.chain_of_thought},
        )

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        return asyncio.run(self.call_async(fq, **kwargs))

    def dump_config(self):
        return {
            "model": self.model,
            "preface": self.preface,
            "parsing_model": self.parsing_model,
        }

    @classmethod
    def load_config(cls, config):
        return cls(
            model=config["model"],
            preface=config["preface"],
            parsing_model=config["parsing_model"],
        )
