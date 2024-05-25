# Path: static_checks/Base.py
import os
from dotenv import load_dotenv
from common.utils import shallow_dict
from datetime import datetime
from dateutil.tz import UTC
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

load_dotenv()
verify_before_instantion = os.getenv("VERIFY_BEFORE_INSTANTIATION", "False") == "True"
use_examples = os.getenv("USE_EXAMPLES", "False") == "True"

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
        self,
        base_sentences: "Self.BaseSentenceFormat_stripped",
        **kwargs,
    ) -> "Self.OutputFormat_stripped":
        if use_examples:
            examples = self.examples
        else:
            examples = None
        return answer_sync(
            prompt=base_sentences,
            preface=self.preface,
            examples=examples,
            response_model=self.OutputFormat_stripped,
            **kwargs,
        )

    async def title_body_(
        self,
        base_sentences: "Self.BaseSentenceFormat_stripped",
        **kwargs,
    ) -> "Self.OutputFormat_stripped":
        if use_examples:
            examples = self.examples
        else:
            examples = None
        return await answer(
            prompt=base_sentences,
            preface=self.preface,
            examples=examples,
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
        # HACK -- set the timezone to UTC if it's not set.
        # Ideally this should be fixed in the base data.
        dates = []
        for key in base_sentences:
            dt = base_sentences[key].resolution_date
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            dates.append(dt)
        return max(dates)

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
        self,
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> "Self.OutputFormat":
        title_body = self.title_body_sync(
            base_sentences, **kwargs
        )
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
        n_verify=3,
        **kwargs,
    ) -> "Self.OutputFormat":
        if verify_before_instantion:
            for i in range(n_verify):
                title_body = await self.title_body(
                    base_sentences, **kwargs
                )
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
                    validate_result = await question_formatter.verify_question(
                        fqs[k], **kwargs
                    )
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
            "'won't'. If applicable, the different parts of the question should be negated one to one. "
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
                            'Resolution Criteria\nSince the synthesis of neptunium in 1940, we have been continually expanding the periodic table by creating new elements. Regrettably, as atoms have become bigger, they also have become less stable, the last few elements to be created having a half-life of less than a second.\nYet it is theorized that at some point, stability of new elements might start increasing again, creating an island of stability. There are certain "magic numbers" of protons that offer the chance of higher stability; 114, 120 and 126 are magic numbers. We have yet to reach elements 120 and 126 and there might still be more stable isotopes of element 114 that have not yet been created.\nIt is asked:\nWill we create an isotope of an element that has more than 110 protons and that has a half-life of at least one day (86,400 seconds) prior to 2050?\nIn order for the question to resolve positive the half-life of the isotope must be verified by an independent scientific team to be greater than one day prior to 2050.\n'
                        ),
                    )
                ),
                assistant=self.OutputFormat_stripped(
                    not_P=ForecastingQuestion_stripped(
                        title="Will we not reach the island of stability by 2050?",
                        body=(
                            'Resolution Criteria\nSince the synthesis of neptunium in 1940, we have been continually expanding the periodic table by creating new elements. Regrettably, as atoms have become bigger, they also have become less stable, the last few elements to be created having a half-life of less than a second.\nYet it is theorized that at some point, stability of new elements might start increasing again, creating an island of stability. There are certain "magic numbers" of protons that offer the chance of higher stability; 114, 120 and 126 are magic numbers. We have yet to reach elements 120 and 126 and there might still be more stable isotopes of element 114 that have not yet been created.\nIt is asked:\nWill we not create an isotope of an element that has more than 110 protons and that has a half-life of at least one day (86,400 seconds) prior to 2050?\nIn order for the question to resolve positive there must not be a half-life of an isotope that has been verified by an independent scientific team to be greater than one day prior to 2050.\n'
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
            "You are a helpful assistant."
            "I will give you two forecasting questions with Yes/No answers. "
            "You should then give me the logical AND of these two questions, i.e. "
            "the question that would be answered YES if BOTH questions are answered YES, "
            "and NO otherwise. "
            "Notes:\n\n"
            " - Your response should be as clear as possible, since the words 'and' and 'or' "
            "are used ambiguously in natural language. For example, 'Will P happen and will Q "
            "happen? is usually confusing, as it sounds like you are asking two questions. "
            "Instead, if there is any chance of confusion, you should give me something like: "
            "Will both of the following occur: (a) and P (b) Q?\n\n"
            " - When the questions allow for a simple rephrasing or factorization "
            "(e.g. using words like 'respectively', 'both' or 'either'), go for it.\n"
            " - If one or both of the given questions is already a logical combination of questions, "
            "join them in the most natural way possible. E.g. \n"
            "    - combine ((P1 AND P2) AND Q) how you would combine (P1 AND P2 AND Q)\n"
            "    - ((P1 OR P2) AND Q) might have to be combined as something like: "
            "Will BOTH of the following occur: (1) EITHER of the following occurs: (a) P1 OR (b) P2 "
            "(2) Q. Unless a more natural formulation exists.\n"
            " - Most importantly: make sure you retain ALL the information in the question bodies from "
            "BOTH base questions! You cannot discard a single relevant detail. "
            "All this is for an experiment to test the logical consistency of forecasters: "
            "The combined question you give will be handed to the forecasters without having seen the "
            "base questions, so it is critical that all the information in the base questions be included "
            "in your logical combination; the resolution criterion for each component should be neatly and "
            "clearly provided. "
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
            "You are a helpful assistant."
            "I will give you two forecasting questions with Yes/No answers. "
            "You should then give me the logical OR of these two questions, i.e. "
            "the question that would be answered YES if EITHER question is answered YES, "
            "and NO otherwise. "
            "Notes:\n\n"
            " - Your response should be as clear as possible, since the words 'and' and 'or' "
            "are used ambiguously in natural language. For example, 'Will P happen or will Q "
            "happen? is usually confusing, as it sounds like you are asking which of the two "
            "will happen (whereas you're actually seeking a YES/NO answer on whether either of "
            "the two will happen). "
            "Instead, if there is any chance of confusion, you should give me something like: "
            "Will either of the following occur: (a) P (b) Q?\n\n"
            " - When the questions allow for a simple rephrasing or factorization "
            "(e.g. using words like 'respectively', 'both' or 'either'), go for it.\n"
            " - If one or both of the given questions is already a logical combination of questions, "
            "join them in the most natural way possible. E.g. \n"
            "    - combine ((P1 OR P2) OR Q) how you would combine (P1 OR P2 OR Q)\n"
            "    - ((P1 AND P2) OR Q) might have to be combined as something like: "
            "Will EITHER of the following occur: (1) BOTH of the following occur: (a) P1 AND (b) P2 "
            "(2) Q. Unless a more natural formulation exists.\n"
            " - Most importantly: make sure you retain ALL the information in the question bodies from "
            "BOTH base questions! You cannot discard a single relevant detail. "
            "All this is for an experiment to test the logical consistency of forecasters: "
            "The combined question you give will be handed to the forecasters without having seen the "
            "base questions, so it is critical that all the information in the base questions be included "
            "in your logical combination; the resolution criterion for each component should be neatly and "
            "clearly provided. "
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
            "Use synonyms, etc. Make sure to retain all the information in the question title "
            "and body! This is very important."
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
            "You are a helpful assistant."
            "I will give you two forecasting questions P and Q with Yes/No answers. "
            "You should then give me the conditional expression of these two questions, i.e. "
            "'GIVEN P is true, then is Q true?' P is the condition, Q is the outcome we are "
            "interested in. "
            "Notes:\n\n"
            " - Your response should be as clear as possible. If writing the whole thing in a "
            "single sentence becomes too long and cumbersome, you can write something like: "
            "Suppose the following is true: P. Then is Q true?\n"
            " - If the question allows for a simple rephrasing or factorization, go for it.\n"
            " - If Q is already a conditional expression, you can just combine the conditions. "
            "I.e. (Given P then (Given P2 then Q)) can just be written as (Given P AND P2 then Q). "
            "Again if P and P2 is too long and cumbersome, you might want to write something like: "
            "Suppose the following is true: P. Then is Q true? And as always if a more natural "
            "formulation exists, always go for it.\n"
            " - Most importantly: make sure you retain all relevant information in the question bodies. "
            "of BOTH base questions. You cannot leave out a single relevant detail. "
            "All this is for an experiment to test the logical consistency of forecasters: "
            "The conditional question you give will be handed to the forecasters without having seen the "
            "base questions, so it is critical that all the information in the base questions be included "
            "in your conditional expression; the resolution criterion for each component should be neatly and "
            "clearly provided. "
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
            "You are a helpful assistant."
            "I will give you a forecasting questions P with Yes/No answer. "
            "You should then give me a guaranteed logical consequence of P. "
            "Notes:\n\n"
            " - Make sure that your output is truly a guaranteed logical consequence of P. "
            "Look at the resolution criteria for P and make sure your output question will "
            "necessarily resolve True if the resolution criteria for P are met.\n"
            " - Most importantly: if there is any background information (not resolution criteria) "
            "in the question body, make sure you retain all relevant information in the question body "
            "of the output you give. We will be giving these questions to a population sample to see "
            "if they give consistent answers (like giving a higher probability to 'Is Kelly a bank-'"
            "teller?' than 'Is Kelly a bank-teller active in the feminist movement?') so it is critical "
            "that any information that might inform someone's probability estimate to your output question "
            "is clearly included."
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
