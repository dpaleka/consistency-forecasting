# Path: static_checks/Base.py
import jsonlines
from common.utils import shallow_dict
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Type, Self, Any, Optional
from pydantic import BaseModel, create_model, field_validator
from common.utils import write_jsonl_async_from_str
from common.llm_utils import answer, answer_sync, Example, parallelized_call
from common.datatypes import ForecastingQuestion, ForecastingQuestion_stripped, Prob
from forecasters import Forecaster


class Checker(ABC):

    def __init__(self, tolerance=0.1, path=None):
        self.tolerance = tolerance
        if path is None:
            self.path = f"src/data/{self.__class__.__name__}.jsonl"
        else:
            self.path = path

    @property
    @abstractmethod
    def TupleFormat(self) -> Type[BaseModel]:
        pass

    @abstractmethod
    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        pass

    @abstractmethod
    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        pass

    async def instantiate_and_write(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ):
        result = await self.instantiate(base_sentences, **kwargs)
        await write_jsonl_async_from_str(
            self.path, [result.model_dump_json()], append=True
        )

    async def instantiate_and_write_many(
        self, base_sentencess: list[dict[str, ForecastingQuestion]], **kwargs
    ):
        _instantiate_and_write = lambda base_sentences: self.instantiate_and_write(
            base_sentences, **kwargs
        )
        await parallelized_call(_instantiate_and_write, base_sentencess)

    @abstractmethod
    def violation(self, answers: dict[str, Prob]) -> float:
        pass

    def check(self, answers: dict[str, Any]) -> bool:
        return self.violation(answers) < self.tolerance

    def elicit_and_violation(
        self, forecaster: Forecaster, sentences: "Self.TupleFormat", **kwargs
    ) -> float:
        return self.violation(forecaster.elicit(sentences, **kwargs))

    def elicit_and_check(
        self, forecaster: Forecaster, sentences: "Self.TupleFormat", **kwargs
    ) -> bool:
        return self.check(forecaster.elicit(sentences, **kwargs))

    def test(self, forecaster: Forecaster, **kwargs):
        for line in jsonlines.open(self.path):
            print("START")
            print(f"line: {line}")
            line_obj = self.TupleFormat.model_validate(line)
            answers = forecaster.elicit(line_obj, **kwargs)
            print(answers)
            if not all(answers.values()):
                print("ERROR: Some answers are None!")
                continue
            loss = self.violation(answers)
            res_bool = self.check(answers)
            res = {True: "Passed", False: "Failed"}[res_bool]
            print(f"Violation: {loss}")
            print(f"Check result: {res}")
            print("")


class MiniInstantiator(ABC):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def BaseSentenceFormat(self) -> Type[BaseModel]:
        pass

    @property
    def BaseSentenceFormat_stripped(self) -> Type[BaseModel]:
        return create_model(
            "BaseSentenceFormat_stripped",
            **{
                k: (ForecastingQuestion_stripped, ...)
                for k in self.BaseSentenceFormat.model_fields
            },
        )

    @property
    @abstractmethod
    def OutputFormat(self) -> Type[BaseModel]:
        pass

    @property
    def OutputFormat_stripped(self) -> Type[BaseModel]:
        return create_model(
            "OutputFormat_stripped",
            **{
                k: (ForecastingQuestion_stripped, ...)
                for k in self.OutputFormat.model_fields
            },
        )

    def title_body_sync_(
        self, base_sentences: "Self.BaseSentenceFormat_stripped", **kwargs
    ) -> "Self.OutputFormat_stripped":
        return answer_sync(
            prompt=base_sentences.model_dump_json(),
            preface=self.preface,
            examples=self.examples,
            response_model=self.OutputFormat_stripped,
            **kwargs,
        )

    async def title_body_(
        self, base_sentences: "Self.BaseSentenceFormat_stripped", **kwargs
    ) -> "Self.OutputFormat_stripped":
        return await answer(
            prompt=base_sentences.model_dump_json(),
            preface=self.preface,
            examples=self.examples,
            response_model=self.OutputFormat_stripped,
            **kwargs,
        )

    def title_body_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.OutputFormat_stripped":
        based_sentences = self.BaseSentenceFormat_stripped(
            **{k: v.cast_stripped() for k, v in base_sentences.items()}
        )
        return self.title_body_sync_(based_sentences, **kwargs)

    async def title_body(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.OutputFormat_stripped":
        based_sentences = self.BaseSentenceFormat_stripped(
            **{k: v.cast_stripped() for k, v in base_sentences.items()}
        )
        return await self.title_body_(based_sentences, **kwargs)

    def resolution_date(
        self, base_sentences: dict[str, ForecastingQuestion]
    ) -> datetime:
        return max([base_sentences[key].resolution_date for key in base_sentences])

    def question_type(self, base_sentences: dict[str, ForecastingQuestion]) -> str:
        return base_sentences[
            list(base_sentences.keys())[0]
        ].question_type  # default to the first key

    def data_source(self, base_sentences: dict[str, ForecastingQuestion]) -> str:
        return "synthetic_inst"

    @abstractmethod
    def resolution_(self, resolutions: dict[str, bool]) -> Optional[dict[str, bool]]:
        """Basically just the MiniInstantiator's logic. E.g. return {'not_P': not resolutions['P']}"""
        pass

    def resolution(
        self, base_sentences: dict[str, ForecastingQuestion]
    ) -> Optional[dict[str, bool]]:
        resolutions = {key: base_sentences[key].resolution for key in base_sentences}
        if all([res is not None for res in resolutions.values()]):
            return self.resolution_(resolutions)
        return None

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.OutputFormat":
        title_body = self.title_body_sync(base_sentences, **kwargs)
        resolutions = self.resolution(base_sentences)
        return self.OutputFormat(
            **{
                k: v.cast_FQ(
                    resolution_date=self.resolution_date(base_sentences),
                    question_type=self.question_type(base_sentences),
                    data_source=self.data_source(base_sentences),
                    resolution=resolutions["k"] if resolutions is not None else None,
                )
                for k, v in shallow_dict(title_body).items()
            }
        )

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.OutputFormat":
        title_body = await self.title_body(base_sentences, **kwargs)
        resolutions = self.resolution(base_sentences)
        return self.OutputFormat(
            **{
                k: v.cast_FQ(
                    resolution_date=self.resolution_date(base_sentences),
                    question_type=self.question_type(base_sentences),
                    data_source=self.data_source(base_sentences),
                    resolution=resolutions["k"] if resolutions is not None else None,
                )
                for k, v in shallow_dict(title_body).items()
            }
        )


class Trivial(MiniInstantiator):

    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion

    class OutputFormat(BaseModel):
        P: ForecastingQuestion

    def title_body_sync_(
        self, base_sentences: "Self.BaseSentenceFormat", **kwargs
    ) -> "Self.OutputFormat_stripped":
        return self.OutputFormat_stripped(P=base_sentences.P.cast_stripped())

    async def title_body_(
        self, base_sentences: "Self.BaseSentenceFormat", **kwargs
    ) -> ForecastingQuestion_stripped:
        return self.OutputFormat_stripped(P=base_sentences.P.cast_stripped())

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, Optional[bool]]:
        return resolutions


class Neg(MiniInstantiator):

    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion

        @field_validator("P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    class OutputFormat(BaseModel):
        not_P: ForecastingQuestion

    def __init__(self):

        self.preface = (
            "You are a helpful assistant. I will give you a forecasting question with Yes/No "
            "answer. You should then give me the NEGATION of the question, i.e. the question that "
            "would be answered YES if the original question would be answered NO, and vice "
            "versa. Demorgan's laws should be followed with and/or negation. Avoid using the word "
            "'won't'."
        )

        self.examples = [
            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if the spot price of Bitcoin against USD is more than "
                            "100,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    )
                ).model_dump_json(),
                assistant=self.OutputFormat_stripped(
                    not_P=ForecastingQuestion_stripped(
                        title="Will the price of Bitcoin be less than or equal to $100,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if the spot price of Bitcoin against USD is less than or equal to "
                            "100,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    )
                ).model_dump_json(),
            )
        ]

        super().__init__()

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool] | None:
        return {"not_P": not resolutions["P"]}


class And(MiniInstantiator):

    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion

        @field_validator("P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

        @field_validator("Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    class OutputFormat(BaseModel):
        P_and_Q: ForecastingQuestion

    def __init__(self):

        self.preface = (
            "You are a helpful assistant. I will give you two forecasting questions with Yes/No "
            "answers. You should then give me the question that would be answered YES if both "
            "questions would be answered YES, and NO otherwise. Avoid a very blatant AND "
            "construction; instead try to integrate the two questions in a more natural way "
            "while achieving the correct meaning."
        )

        self.examples = [
            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if the spot price of Bitcoin against USD is more than "
                            "100,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    ),
                    Q=ForecastingQuestion_stripped(
                        title="Will the price of Ethereum be above $10,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if the spot price of Ethereum against USD is more than "
                            "10,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    ),
                ).model_dump_json(),
                assistant=self.OutputFormat_stripped(
                    P_and_Q=ForecastingQuestion_stripped(
                        title="Will the prices of Bitcoin and Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025?",
                        body=(
                            "Resolves YES if the spot price of Bitcoin against USD is more than 100,000 "
                            "AND the spot price of Ethereum against USD is more than 10,000 on 1st January "
                            "2025. Resolves NO otherwise."
                        ),
                    )
                ).model_dump_json(),
            )
        ]

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool] | None:
        return {"P_and_Q": resolutions["P"] and resolutions["Q"]}


class Or(MiniInstantiator):

    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion

        @field_validator("P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

        @field_validator("Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    class OutputFormat(BaseModel):
        P_or_Q: ForecastingQuestion

    def __init__(self):

        self.preface = (
            "You are a helpful assistant. I will give you two forecasting questions with Yes/No "
            "answers. You should then give me the question that would be answered YES if either "
            "question would be answered YES, and NO otherwise. Avoid a very blatant OR "
            "construction; instead try to integrate the two questions in a more natural way "
            "while achieving the correct meaning."
        )

        self.examples = [
            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will Jeb Bush be the President of the US in January 2032?",
                        body=(
                            "Resolves YES if Jeb Bush is the President of the US in January 2032. "
                            "Resolves NO otherwise."
                        ),
                    ),
                    Q=ForecastingQuestion_stripped(
                        title="Will Jeb Bush be the President of the US in January 2036?",
                        body=(
                            "Resolves YES if Jeb Bush is the President of the US in January 2036. "
                        ),
                    ),
                ).model_dump_json(),
                assistant=self.OutputFormat_stripped(
                    P_or_Q=ForecastingQuestion_stripped(
                        title="Will Jeb Bush be the President of the US in January 2032 or 2036?",
                        body=(
                            "Resolves YES if Jeb Bush is the President of the US in either January 2032 "
                            "or January 2036 (or both). Resolves NO otherwise."
                        ),
                    )
                ).model_dump_json(),
            ),
            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will Joe Biden be elected president in the 2024 presidential election?",
                        body=(
                            "Resolves YES if Joe Biden is elected president in the 2024 presidential "
                            "election. Resolves NO otherwise."
                        ),
                    ),
                    Q=ForecastingQuestion_stripped(
                        title="Will the price of Ethereum be above $10,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if the spot price of Ethereum against USD is more than "
                            "10,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    ),
                ).model_dump_json(),
                assistant=self.OutputFormat_stripped(
                    P_or_Q=ForecastingQuestion_stripped(
                        title="Will either of the following occur: "
                        "(a) Joe Biden is elected president in the 2024 presidential election, or "
                        "(b) the price of Ethereum is above $10,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if either of the said events occur (or both). Resolves NO "
                            "otherwise."
                        ),
                    )
                ).model_dump_json(),
            ),
        ]

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool] | None:
        return {"P_or_Q": resolutions["P"] or resolutions["Q"]}


class Paraphrase(MiniInstantiator):

    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion

        @field_validator("P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    class OutputFormat(BaseModel):
        para_P: ForecastingQuestion

    def __init__(self):

        self.preface = (
            "You are a helpful assistant. I will give you a forecasting question with Yes/No "
            "answer. You should then give me a paraphrased version of the question that "
            "expresses the same underlying concept. The question should be as different as "
            "possible from the original question, while still meaning the exact same thing. "
            "Use synonyms, etc. "
        )

        self.examples = [
            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if the price of Bitcoin on 1st January 2025 is more than "
                            "$100,000. Resolves NO otherwise."
                        ),
                    )
                ).model_dump_json(),
                assistant=self.OutputFormat_stripped(
                    para_P=ForecastingQuestion_stripped(
                        title="On 1st January 2025, will one Bitcoin be worth more than 100,000 US Dollars?",
                        body=(
                            "Resolves YES if the spot price of Bitcoin against USD is higher than "
                            "100,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    )
                ).model_dump_json(),
            )
        ]

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool] | None:
        return {"para_P": resolutions["P"]}


class Conditional(MiniInstantiator):

    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion

        @field_validator("P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

        @field_validator("Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    class OutputFormat(BaseModel):
        Q_given_P: ForecastingQuestion

    def __init__(self):

        preface = (
            "You are a helpful assistant. I will give you two forecasting questions P and Q with Yes/No "
            "answers. You should then give me a question that expresses their *conditional* expression"
            "i.e. 'GIVEN that P is true, then is Q true?'"
        )

        examples = [
            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if the spot price of Bitcoin against USD is more than "
                            "100,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    ),
                    Q=ForecastingQuestion_stripped(
                        title="Will the price of Ethereum be above $10,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if the spot price of Ethereum against USD is more than "
                            "10,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    ),
                ).model_dump_json(),
                assistant=self.OutputFormat_stripped(
                    Q_given_P=ForecastingQuestion_stripped(
                        title=(
                            "Given that on 1st January 2025, the price of Bitcoin will be above $100,000, "
                            "will the price of Ethereum be above $10,000 on the same date?"
                        ),
                        body=(
                            "Resolves N/A if the price of Bitcoin is not above $100,000 on 1st January 2025. "
                            "If the condition is met (if the price of Bitcoin is above $100,000 on 1st Jan 2025 "
                            ", then resolves YES if the spot price of Ethereum against USD is more than 10,000 "
                            "and NO if it's not on 1st Jan 2025."
                        ),
                    )
                ).model_dump_json(),
            )
        ]
