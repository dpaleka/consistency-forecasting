# Path: static_checks/Base.py
from common.utils import shallow_dict
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Type, Any, Optional, Self  # noqa
from pydantic import BaseModel, create_model, field_validator
from common.utils import write_jsonl_async_from_str  # noqa
from common.llm_utils import answer, answer_sync, Example, parallelized_call  # noqa
from common.datatypes import (
    ForecastingQuestion,
    ForecastingQuestion_stripped,
)  # noqa
from question_generators import question_formatter


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
            prompt=base_sentences,
            preface=self.preface,
            examples=self.examples,
            response_model=self.OutputFormat_stripped,
            **kwargs,
        )

    async def title_body_(
        self, base_sentences: "Self.BaseSentenceFormat_stripped", **kwargs
    ) -> "Self.OutputFormat_stripped":
        return await answer(
            prompt=base_sentences,
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
    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
        """Basically just the MiniInstantiator's logic. E.g. return {'not_P': not resolutions['P']}"""
        pass

    def resolution(
        self, base_sentences: dict[str, ForecastingQuestion]
    ) -> dict[str, bool | None]:
        resolutions = {key: base_sentences[key].resolution for key in base_sentences}
        if all([res is not None for res in resolutions.values()]):
            return self.resolution_(resolutions)
        return {k: None for k in self.OutputFormat.model_fields}

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.OutputFormat":
        title_body = self.title_body_sync(base_sentences, **kwargs)
        return self.OutputFormat(
            **{
                k: v.cast_FQ(
                    resolution_date=self.resolution_date(base_sentences),
                    question_type=self.question_type(base_sentences),
                    data_source=self.data_source(base_sentences),
                    resolution=self.resolution(base_sentences)[k],
                )
                for k, v in shallow_dict(title_body).items()
            }
        )

    async def instantiate(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        validate_before=False,
        n_validation=3,
        **kwargs,
    ) -> "Self.OutputFormat":
        if validate_before:
            for i in range(n_validation):
                title_body = await self.title_body(base_sentences, **kwargs)
                sd = shallow_dict(title_body)
                fqs = {k: None for k in sd.keys()}
                valid = {k: False for k in sd.keys()}
                for k, v in sd.items():
                    fqs[k] = v.cast_FQ(
                        resolution_date=self.resolution_date(base_sentences),
                        question_type=self.question_type(base_sentences),
                        data_source=self.data_source(base_sentences),
                        resolution=self.resolution(base_sentences)[k],
                    )
                    validate_result = await question_formatter.validate_question(fqs[k])
                    valid[k] = validate_result.valid
                if all([res is not None for res in fqs.values()]):
                    break
            return self.OutputFormat(**fqs)
        else:
            title_body = await self.title_body(base_sentences, **kwargs)
            return self.OutputFormat(
                **{
                    k: v.cast_FQ(
                        resolution_date=self.resolution_date(base_sentences),
                        question_type=self.question_type(base_sentences),
                        data_source=self.data_source(base_sentences),
                        resolution=self.resolution(base_sentences)[k],
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
    ) -> "Self.OutputFormat_stripped":
        return self.OutputFormat_stripped(P=base_sentences.P.cast_stripped())

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
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
            "'won't'  If applicable the different parts of the question should be negated one to one. " 
            "For example the new title should be an negation of the original title.  Body questions should be negations"
            "of the original body questions.  Statements / background information can be kept the same."
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
                ),
                assistant=self.OutputFormat_stripped(
                    not_P=ForecastingQuestion_stripped(
                        title="Will the price of Bitcoin be less than or equal to $100,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if the spot price of Bitcoin against USD is less than or equal to "
                            "100,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    )
                ),
            ),
            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will we reach the island of stability by 2050?",
                        body=(
                            "Resolution Criteria\nSince the synthesis of neptunium in 1940, we have been continually expanding the periodic table by creating new elements. Regrettably, as atoms have become bigger, they also have become less stable, the last few elements to be created having a half-life of less than a second.\nYet it is theorized that at some point, stability of new elements might start increasing again, creating an island of stability. There are certain \"magic numbers\" of protons that offer the chance of higher stability; 114, 120 and 126 are magic numbers. We have yet to reach elements 120 and 126 and there might still be more stable isotopes of element 114 that have not yet been created.\nIt is asked:\nWill we create an isotope of an element that has more than 110 protons and that has a half-life of at least one day (86,400 seconds) prior to 2050?\nIn order for the question to resolve positive the half-life of the isotope must be verified by an independent scientific team to be greater than one day prior to 2050.\n"
                        ),
                    )
                ),
                assistant=self.OutputFormat_stripped(
                    not_P=ForecastingQuestion_stripped(
                        title="Will we not reach the island of stability by 2050?",
                        body=(
                            "Resolution Criteria\nSince the synthesis of neptunium in 1940, we have been continually expanding the periodic table by creating new elements. Regrettably, as atoms have become bigger, they also have become less stable, the last few elements to be created having a half-life of less than a second.\nYet it is theorized that at some point, stability of new elements might start increasing again, creating an island of stability. There are certain \"magic numbers\" of protons that offer the chance of higher stability; 114, 120 and 126 are magic numbers. We have yet to reach elements 120 and 126 and there might still be more stable isotopes of element 114 that have not yet been created.\nIt is asked:\nWill we not create an isotope of an element that has more than 110 protons and that has a half-life of at least one day (86,400 seconds) prior to 2050?\nIn order for the question to resolve positive there must not be a half-life of an isotope that has been verified by an independent scientific team to be greater than one day prior to 2050.\n"
                        ),
                    )
                ),
            ),

            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will the prices of Bitcoin and Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025?",
                        body=(
                            "Resolves YES if both of these events happen. "
                            "a.) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 AND "
                            "b.) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
                assistant=self.OutputFormat_stripped(
                    not_P=ForecastingQuestion_stripped(
                        title="Will the prices of Bitcoin or Ethereum not exceed $100,000 or $10,000 respectively on 1st January 2025?",
                        body=(
                            "Resolves YES if either of these events happen. "
                            "a.) the spot price of Bitcoin against USD is not more than 100,000 on 1st January 2025 OR "
                            "b.) the spot price of Ethereum against USD is not more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
            ),
            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will the prices of Bitcoin or Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025?",
                        body=(
                            "Resolves YES if either one of these events happen."
                            "a.) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 OR"
                            "b.) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025"
                            "Resolves NO otherwise."
                        ),
                    )
                ),
                assistant=self.OutputFormat_stripped(
                    not_P=ForecastingQuestion_stripped(
                        title="Will the prices of Bitcoin and Ethereum not exceed $100,000 or $10,000 respectively on 1st January 2025?",
                        body=(
                            "Resolves YES if both of these events happen. "
                            "a.) the spot price of Bitcoin against USD is not more than 100,000 on 1st January 2025 AND "
                            "b.) the spot price of Ethereum against USD is not more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
            ),
        ]

        super().__init__()

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
        return {"not_P": not resolutions["P"]}


class And(MiniInstantiator):
    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion

        @field_validator("P")
        def check_question_type(cls, value):  # noqa
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

        @field_validator("Q")
        def check_question_type(cls, value):  # noqa
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    class OutputFormat(BaseModel):
        P_and_Q: ForecastingQuestion

    def __init__(self):
        self.preface = (
            "You are a helpful assistant. I will give you two forecasting questions with Yes/No "
            "answers. You should then give me the question that would be answered YES if both "
            "questions would be answered YES, and NO otherwise. Make sure your response is "
            "as clear as possible, since the words 'and' and 'or' are used quite ambiguously "
            "in natural language. When the questions allow a simple rephrasing (e.g. using words "
            "like 'respectively' or 'either'), go for it."
            "Additionally, if the existing question is already a logical combination of "
            "two or more questions, you just add the additional question to the current combination."
            "For example if P is (A and B), and given Q, then P and Q are (A and B and Q)."
            "If P is (A or B), and given Q, then P and Q are ((A or B) and Q)."
            "Also we should output the final question in as unambiguous phrasing as possible."
            "All information in all sections from the two original questions should be retained. "
            "For example information, in the body or resolution criteria should be kept "
            "in the new question.  Do not remove any information of the original questions."
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
                ),
                assistant=self.OutputFormat_stripped(
                    P_and_Q=ForecastingQuestion_stripped(
                        title="Will the prices of Bitcoin and Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025?",
                        body=(
                            "Resolves YES if both of these events happen. "
                            "a.) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 AND "
                            "b.) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
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
                ),
                assistant=self.OutputFormat_stripped(
                    P_and_Q=ForecastingQuestion_stripped(
                        title=(
                            "Will Joe Biden be elected president in the 2024 presidential election AND "
                            "the price of Ethereum be above $10,000 on 1st January 2025?"
                        ),
                        body=(
                            "Resolves YES if Joe Biden is elected president in the 2024 presidential "
                            "election AND the spot price of Ethereum against USD is more than 10,000 on 1st "
                            "January 2025. Resolves NO otherwise."
                        ),
                    )
                ),
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
                        title="Will the prices of Bitcoin and Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025?",
                        body=(
                            "Resolves YES if both of these events happen. "
                            "a.) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 AND "
                            "b.) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    ),
                ),
                assistant=self.OutputFormat_stripped(
                    P_and_Q=ForecastingQuestion_stripped(
                        title=(
                            "Will Joe Biden be elected president in the 2024 presidential election AND "
                            "the prices of Bitcoin and Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025?"
                        ),
                        body=(
                            "Resolves YES if both of these events happen. "
                            "a.) Joe Biden is elected president in the 2024 presidential election AND "
                            "b.) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 AND the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
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
                        title="Will the prices of Bitcoin or Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025?",
                        body=(
                            "Resolves YES if either of these events happen. "
                            "a.) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 OR "
                            "b.) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    ),
                ),
                assistant=self.OutputFormat_stripped(
                    P_and_Q=ForecastingQuestion_stripped(
                        title=(
                            "Will Joe Biden be elected president in the 2024 presidential election AND "
                            "the prices of Bitcoin or Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025?"
                        ),
                        body=(
                            "Resolves YES if both of these events happen. "
                            "a.) Joe Biden is elected president in the 2024 presidential election AND "
                            "b.) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 OR the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
            ),
        ]

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
        return {"P_and_Q": resolutions["P"] and resolutions["Q"]}


class Or(MiniInstantiator):
    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion

        @field_validator("P")
        def check_question_type(cls, value):  # noqa
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

        @field_validator("Q")
        def check_question_type(cls, value):  # noqa
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    class OutputFormat(BaseModel):
        P_or_Q: ForecastingQuestion

    def __init__(self):
        self.preface = (
            "You are a helpful assistant. I will give you two forecasting questions with Yes/No "
            "answers. You should then give me the question that would be answered YES if either "
            "question would be answered YES, and NO otherwise. Make sure your response is as clear "
            "as possible, since the words 'and' and 'or' are used quite ambiguously in natural language."
            "When the questions allow a simple rephrasing (e.g. using words like 'respectively' or "
            "'either'), go for it."
            "Additionally, if the existing question is already a logical combination of "
            "two or more questions, you just add the additional question to the current combination."
            "For example if P is (A and B), and given Q, then P or Q is (A and B or Q)."
            "If P is (A or B), and given Q, then P or Q is (A or B or Q)."
            "Also we should output the final question in as unambiguous phrasing as possible."
            "All information in all sections from the two original questions should be retained. "
            "For example information, in the body or resolution criteria should be kept "
            "in the new question.  Do not remove any information of the original questions."
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
                        title="Will the prices of Bitcoin and Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025?",
                        body=(
                            "Resolves YES if both of these events happen. "
                            "a.) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 AND "
                            "b.) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    ),
                ),
                assistant=self.OutputFormat_stripped(
                    P_or_Q=ForecastingQuestion_stripped(
                        title="Will Jeb Bush be the President of the US in January 2032 or the prices of Bitcoin and Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025??",
                        body=(
                            "Resolves YES if either of these events happen. "
                            "a.) Jeb Bush is the President of the US in January 2032 AND "
                            "b.) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 OR the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
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
                ),
                assistant=self.OutputFormat_stripped(
                    P_or_Q=ForecastingQuestion_stripped(
                        title="Will either Joe Biden be elected president in the 2024 presidential election or the price of Ethereum be above $10,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if either of these events occur (or both). "
                            "a.) Joe Biden is elected president in the 2024 presidential election OR "
                            "b.) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
            ),
        ]

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
        return {"P_or_Q": resolutions["P"] or resolutions["Q"]}

    def resolution(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
        # a bit more complicated than for other instantiators
        if True in resolutions:
            return {"P_or_Q": True}
        elif None in resolutions:
            return {"P_or_Q": None}
        return {"P_or_Q": False}


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
                ),
                assistant=self.OutputFormat_stripped(
                    para_P=ForecastingQuestion_stripped(
                        title="On 1st January 2025, will one Bitcoin be worth more than 100,000 US Dollars?",
                        body=(
                            "Resolves YES if the spot price of Bitcoin against USD is higher than "
                            "100,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    )
                ),
            )
        ]

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
        return {"para_P": resolutions["P"]}


class Conditional(MiniInstantiator):
    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion

        @field_validator("P")
        def check_question_type(cls, value):  # noqa
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

        @field_validator("Q")
        def check_question_type(cls, value):  # noqa
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    class OutputFormat(BaseModel):
        Q_given_P: ForecastingQuestion

    def __init__(self):
        self.preface = (
            "You are a helpful assistant. I will give you two forecasting questions P and Q with Yes/No "
            "answers. You should then give me a question that expresses their *conditional* expression"
            "i.e. 'GIVEN that P is true, then is Q true?'"
            "Either P and Q can already be a composite question."
            "All information in all sections from the two original questions should be retained. "
            "For example information, in the body or resolution criteria should be kept "
            "in the new question.  Do not remove any information of the original questions."
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
                ),
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
                ),
            )
        ]

    def question_type(self, base_sentences: dict[str, ForecastingQuestion]) -> str:
        return "conditional_binary"

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
        return {"Q_given_P": resolutions["Q"] if resolutions["P"] else None}


class Consequence(MiniInstantiator):
    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion

        @field_validator("P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    class OutputFormat(BaseModel):
        cons_P: ForecastingQuestion

    def __init__(self):
        self.preface = (
            "You are a helpful assistant. I will give you a forecasting question P with Yes/No "
            "answer. You should then give me a question that expresses any logical consequence "
            "of P. That is, a question that is necessarily true if P is true. If P is false, "
            "the question could be either true or false. Make sure it is a perfect logical consequence. "
            "It should also be very obvious that if P is true, then cons_P must be true.  It should be impossible for cons_P to occur without P also having occured. "
            "P is in a way a stronger refinement of cons_P. "
            "In logical notation we say P -> cons_P. "
            "The relationship must be strict. "
            "cons_P MUST be something that logically / mathematically is inevitable if P occurs. "
            "cons_P also cannot just be the same question or rephrease of P. "
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
                ),
                assistant=self.OutputFormat_stripped(
                    cons_P=ForecastingQuestion_stripped(
                        title="Will the price of Bitcoin be above $70,000 on 1st January 2025?",
                        body=(
                            "Resolves YES if the spot price of Bitcoin against USD is more than "
                            "70,000 on 1st January 2025. Resolves NO otherwise."
                        ),
                    )
                ),
            ),
            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will SpaceX manage to land a person on Mars before 2100?",
                        body=(
                            "Resolves YES if the company SpaceX builds the rocketship that carries a person to Mars before Jan 1, 2100 as defined by that individual touching the surface of the planet. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
                assistant=self.OutputFormat_stripped(
                    cons_P=ForecastingQuestion_stripped(
                        title="Will a human land on Mars before 2100?",
                        body=(
                            "Resolves YES if a member of the human species manages to touch the surface of Mars before Jan 1, 2100.  "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
            ),

            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will a peace treaty be declared between Russia and Ukraine before 2025?",
                        body=(
                            "Resolves YES if a peace treaty is declared between Russia and Ukraine before Jan 1, 2025 regarding the current war. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
                assistant=self.OutputFormat_stripped(
                    cons_P=ForecastingQuestion_stripped(
                        title="Will a ceasefire be declared between Russia and Ukraine before 2025?",
                        body=(
                            "Resolves YES if a ceasefire is declared between Russia and Ukraine before Jan 1, 2025.  "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
            ),

            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will Shinzo Abe be elected US president in 2024?",
                        body=(
                            "Resolves YES if Shinzo Abe is elected US president in 2024. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
                assistant=self.OutputFormat_stripped(
                    cons_P=ForecastingQuestion_stripped(
                        title=(
                            "Will someone who is not a natural-born US citizen be elected US "
                            "president in 2024?"
                        ),
                        body=(
                            "Resolves YES if someone who is not a natural-born US citizen is "
                            "elected US president in 2024. Resolves NO otherwise."
                        ),
                    )
                ),
            ),

            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will a human being land on Mars by 2200?",
                        body=(
                            "Resolves YES if a human being lands on Mars before Jan 1, 2200. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
                assistant=self.OutputFormat_stripped(
                    cons_P=ForecastingQuestion_stripped(
                        title="Will a human being land on Mars by 2100?",
                        body=(
                            "Resolves YES if a human being lands on Mars before Jan 1, 2100. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
            ),

            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will Vladimir Putin be assassinated by Jan 1, 2030?",
                        body=(
                            "Resolves YES if a professional autopsy determines that Vladimir Putin, the current president of Russia, is dead before Jan 1, 2030, and the cause is most likely by an assassination attempt. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
                assistant=self.OutputFormat_stripped(
                    cons_P=ForecastingQuestion_stripped(
                        title="Will Vladimir Putin be dead by Jan 1, 2030?",
                        body=(
                            "Resolves YES if a professional autopsy determines that Vladimir Putin, the current president of Russia, is dead before Jan 1, 2030. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
            ),


            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will Sam Bankman-Fried, ex-FTX CEO, serve at least 20 years of his prison sentence?",
                        body=(
                            "Resolves YES Sam Bankman-Fried is incarcerated for 20 years or more. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
                assistant=self.OutputFormat_stripped(
                    cons_P=ForecastingQuestion_stripped(
                        title="Will Sam Bankman-Fried, ex-FTX CEO, serve at least 15 years of his prison sentence?",
                        body=(
                            "Resolves YES Sam Bankman-Fried is incarcerated for 15 years or more. "
                            "Resolves NO otherwise."
                        ),
                    )
                ),
            ),            
        ]

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
        return {"cons_P": resolutions["P"]}
