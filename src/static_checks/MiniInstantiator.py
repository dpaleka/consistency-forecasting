# Path: static_checks/Base.py
import os
from dotenv import load_dotenv
from common.utils import shallow_dict
from datetime import datetime
from dateutil.tz import UTC
from abc import ABC, abstractmethod
from enum import Enum
from typing import Type, Any, Optional, Self, List, List  # noqa
from pydantic import BaseModel, field_validator
from common.utils import write_jsonl_async_from_str  # noqa
from common.llm_utils import (
    answer,
    answer_sync,
    Example,
    prepare_messages_alt,
)
from common.datatypes import (
    ForecastingQuestion,
    ForecastingQuestion_stripped,
    InformationPiece,
)
from common.perscache import register_models_for_cache
from question_generators import question_formatter

load_dotenv()
verify_before_instantion = os.getenv("VERIFY_BEFORE_INSTANTIATION", "False") == "True"
use_examples = os.getenv("USE_EXAMPLES", "False") == "True"


class MiniInstantiator(ABC):
    use_examples_here = use_examples

    def __init__(self):
        pass

    @property
    @abstractmethod
    def BaseSentenceFormat(self) -> Type[BaseModel]:
        pass

    @property
    def BaseSentenceFormat_stripped(self) -> Type[BaseModel]:
        # return create_model(
        #     "BaseSentenceFormat_stripped",
        #     **{
        #         k: (ForecastingQuestion_stripped, ...)
        #         for k in self.BaseSentenceFormat.model_fields
        #     },
        # )
        pass

    @property
    @abstractmethod
    def OutputFormat(self) -> Type[BaseModel]:
        pass

    @property
    def OutputFormat_stripped(self) -> Type[BaseModel]:
        # return create_model(
        #     "OutputFormat_stripped",
        #     **{
        #         k: (ForecastingQuestion_stripped, ...)
        #         for k in self.OutputFormat.model_fields
        #     },
        # )
        pass

    def title_body_sync_(
        self,
        base_sentences: "Self.BaseSentenceFormat_stripped",
        **kwargs,
    ) -> "Self.OutputFormat_stripped":
        if self.use_examples_here:
            examples = self.examples
        else:
            examples = None
        return answer_sync(
            prompt=base_sentences,
            preface=self.preface,
            examples=examples,
            prepare_messages_func=prepare_messages_alt,
            response_model=self.OutputFormat_stripped,
            **kwargs,
        )

    async def title_body_(
        self,
        base_sentences: "Self.BaseSentenceFormat_stripped",
        **kwargs,
    ) -> "Self.OutputFormat_stripped":
        if self.use_examples_here:
            examples = self.examples
        else:
            examples = None
        return await answer(
            prompt=base_sentences,
            preface=self.preface,
            examples=examples,
            prepare_messages_func=prepare_messages_alt,
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
    ) -> "Self.OutputFormat" | List["Self.OutputFormat"]:
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
        n_verify=3,
        **kwargs,
    ) -> "Self.OutputFormat" | List["Self.OutputFormat"]:
        if verify_before_instantion:
            for i in range(n_verify):
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

    class BaseSentenceFormat_stripped(BaseModel):
        P: ForecastingQuestion_stripped

    class OutputFormat(BaseModel):
        P: ForecastingQuestion

    class OutputFormat_stripped(BaseModel):
        P: ForecastingQuestion_stripped

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

    class BaseSentenceFormat_stripped(BaseModel):
        P: ForecastingQuestion_stripped

    class OutputFormat(BaseModel):
        not_P: ForecastingQuestion

    class OutputFormat_stripped(BaseModel):
        not_P: ForecastingQuestion_stripped

    def __init__(self):
        self.preface = (
            "You are a helpful assistant. I will give you a forecasting question with Yes/No "
            "answer. You should then give me the NEGATION of the question, i.e. the question that "
            "would be answered YES if the original question would be answered NO, and vice "
            "versa. Notes to keep in mind:\n"
            "- When negating compound questions, be careful to apply de Morgan's law correctly. "
            "The negation of (P OR Q) is (NOT P AND NOT Q). "
            "The negation of (P AND Q) is (NOT P OR NOT Q).\n"
            "- In a conditional question, do not negate the condition. E.g. 'If it rains tomorrow, "
            "will the temperature be less than 15 degrees Celsius?' should be negated to 'If it rains "
            "tomorrow, will the temperature be 15 degrees Celsius or more?'.\n"
            "- The negation of 'Will there be proof/evidence/reports of X?' is usually "
            "'Will there be no proof/evidence/reports of X?', not "
            "'Will there be proof/evidence/reports of not X?.' "
            "This also applies to the resolution criteria: "
            "'Resolves YES if there is proof/evidence reports of X' should be negated to "
            "'Resolves YES if there is no proof/evidence/reports of X' "
            "unless explicit resolution criteria are given for the original question to resolve NO, "
            "in which case those explicit criteria should be given as the resolution criteria for the negated "
            "question.\n"
            "- If applicable, the different parts of the question should be negated one to one. "
            "For example the new title should be an negation of the original title. Body questions "
            "should be negations of the original body questions. Statements / background information "
            "would generally be kept the same.\n"
            "- Pay attention to correctly negate existential and universal quantifiers. For example, "
            "the negation of 'Will Elon Musk's net worth, at any point before 2050, exceed 1 trillion "
            "USD?' is 'Will Elon Musk's net worth never exceed 1 trillion USD before 2050?', and NOT "
            "'Will Elon Musk's net worth, at any point before 2050, not exceed 1 trillion USD?'.\n"
            "- Make sure you retain ALL the information in the question title and body! You cannot "
            "discard a single relevant detail.\n"
            " - One type of question you may be given is a single choice from a multiple choice question. For example, "
            "you may be given 'Which of these countries will legalize human cloning by 2030? (Japan)'. "
            "This is asking if Japan will recognize and legalize human cloning by 2030. "
            "You can negate this normally, i.e. 'Which of these countries will not legalize human cloning by 2030? (Japan)'. "
            "Such a question may also itself be a logical combination -- e.g. "
            "'Which of these countries will legalize human cloning by 2030? (UK, France, or Germany) "
            "is asking if any either of the UK, France, or Germany will legalize human cloning by 2030. "
            "Make sure to correctly negate such a combination with de Morgan's law."
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

    class BaseSentenceFormat_stripped(BaseModel):
        P: ForecastingQuestion_stripped
        Q: ForecastingQuestion_stripped

    class OutputFormat(BaseModel):
        P_and_Q: ForecastingQuestion

    class OutputFormat_stripped(BaseModel):
        P_and_Q: ForecastingQuestion_stripped

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
            " - Be careful when combining conditional expressions (which often have words like 'given' and 'if'). "
            "'(Given A then P) AND (Given B then Q) "
            "should be combined as 'Given (A AND B) then (P AND Q)'. \n"
            " - When only one of the questions is conditional, the combined question should be conditional. "
            "i.e. '(Given A then P) AND Q' should be combined as 'Given A then (P AND Q)'.\n"
            " - Most importantly: make sure you retain ALL the information in the question bodies from "
            "BOTH base questions! You cannot discard a single relevant detail. "
            "All this is for an experiment to test the logical consistency of forecasters: "
            "The combined question you give will be handed to the forecasters without having seen the "
            "base questions, so it is critical that all the information in the base questions be included "
            "in your logical combination; the resolution criterion for each component should be neatly and "
            "clearly provided.\n"
            " - Also, make sure that the title is self-sufficient independent of the body, i.e. "
            "is a question that can be meaningfully answered without looking at the body. So you CANNOT "
            "give me a question title like 'Is the following true?' or 'What will happen if the following happens?'\n"
            " - One type of question you may be given is a single choice from a multiple choice question. For example, "
            "you may be given 'Which of these countries will legalize human cloning by 2030? (Japan)'. "
            "This is asking if Japan will recognize and legalize human cloning by 2030. Such a question may also "
            "itself be a logical combination -- e.g. "
            "'Which of these countries will legalize human cloning by 2030? (UK, France, or Germany) "
            "is asking if any either of the UK, France, or Germany will legalize human cloning by 2030. "
            "Make sure to correctly combine such combinations as previously described."
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

    class BaseSentenceFormat_stripped(BaseModel):
        P: ForecastingQuestion_stripped
        Q: ForecastingQuestion_stripped

    class OutputFormat(BaseModel):
        P_or_Q: ForecastingQuestion

    class OutputFormat_stripped(BaseModel):
        P_or_Q: ForecastingQuestion_stripped

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
            " - Be careful when combining conditional expressions (which often have words like 'given' and 'if'). "
            "'(Given A then P) OR (Given B then Q) "
            "should be combined as is, rather than messing up the conditions. E.g. a phrasing like "
            "'Will either of the following occur given their respective conditions: (a) Given A then P? "
            "(b) Given B then Q?' is good.\n"
            " - This also applies when only one of the questions is conditional. Like 'P OR (Given A then Q)'"
            "should be phrased as something like: "
            "'Will either of the following occur given their respective conditions are met? "
            "(a) P (b) Given A, then Q?'.\n"
            " - Most importantly: make sure you retain ALL the information in the question bodies from "
            "BOTH base questions! You cannot discard a single relevant detail. "
            "All this is for an experiment to test the logical consistency of forecasters: "
            "The combined question you give will be handed to the forecasters without having seen the "
            "base questions, so it is critical that all the information in the base questions be included "
            "in your logical combination; the resolution criterion for each component should be neatly and "
            "clearly provided.\n"
            "- Also, make sure that the title is self-sufficient independent of the body, i.e. "
            "is a question that can be meaningfully answered without looking at the body. So you CANNOT "
            "give me a question title like 'Is the following true?' or 'What will happen if the following happens?'\n"
            " - One type of question you may be given is a single choice from a multiple choice question. For example, "
            "you may be given 'Which of these countries will legalize human cloning by 2030? (Japan)'. "
            "This is asking if Japan will recognize and legalize human cloning by 2030. Such a question may also "
            "itself be a logical combination -- e.g. "
            "'Which of these countries will legalize human cloning by 2030? (UK, France, or Germany) "
            "is asking if any either of the UK, France, or Germany will legalize human cloning by 2030. "
            "Make sure to correctly combine such combinations as previously described."
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

    class BaseSentenceFormat_stripped(BaseModel):
        P: ForecastingQuestion_stripped

    class OutputFormat(BaseModel):
        para_P: ForecastingQuestion

    class OutputFormat_stripped(BaseModel):
        para_P: ForecastingQuestion_stripped

    def __init__(self):
        self.preface = (
            "You are a helpful assistant. I will give you a forecasting question with Yes/No "
            "answer. You should then give me a paraphrased version of the question that "
            "expresses the same underlying concept. The question should be as different as "
            "possible from the original question, while still meaning the exact same thing. "
            "Use synonyms, etc. Notes:\n\n"
            " - Make sure to retain all information in the question title and body! "
            "Paraphrase, but don't discard the essential meaning. This is very important.\n"
            " - Something to note: you cannot rephrase P to 'Do you think that P?', 'Is P likely?' "
            "or 'Will there be evidence of P?'. These are not paraphrases, they mean different "
            "things (something may be likely and not happen, etc.)."
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
    use_examples_here = True

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

    class BaseSentenceFormat_stripped(BaseModel):
        P: ForecastingQuestion_stripped
        Q: ForecastingQuestion_stripped

    class OutputFormat(BaseModel):
        Q_given_P: ForecastingQuestion

    class OutputFormat_stripped(BaseModel):
        Q_given_P: ForecastingQuestion_stripped

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
            "clearly provided.\n"
            "- Also, make sure that the title is more-or-less self-sufficient independent of the body. "
            "It's OK if some detailed criteria/nuances/background info/ambiguity resolution are only in "
            "the body, but the title should be basically a well-formed question on its own.\n"
            " - One type of question you may be given is a single choice from a multiple choice question. For example, "
            "you may be given 'Which of these countries will legalize human cloning by 2030? (Japan)'. "
            "This is asking if Japan will recognize and legalize human cloning by 2030. Such a question may also "
            "itself be a logical combination -- e.g. "
            "'Which of these countries will legalize human cloning by 2030? (UK, France, or Germany) "
            "is asking if any either of the UK, France, or Germany will legalize human cloning by 2030."
        )

        self.examples = [
            Example(
                user=self.BaseSentenceFormat_stripped(
                    P=ForecastingQuestion_stripped(
                        title="Will Kristaps Porzingis miss at least one playoff game due to injury?",
                        body=(
                            "The sportswriter Dan Devine recently wrote an article titled "
                            "''Why Kristaps Porziņģis is the key to the Celtics' title hopes'':\n\n"
                            "With the addition of Porzingis’ varied scoring game, a Celtics attack "
                            "that was below league-average in fourth-quarter scoring last season has "
                            "now jumped up to eighth. A group that finished 11th last season in scoring "
                            "efficiency in the “clutch” — when the score’s within five points in the last "
                            "five minutes — is up to fifth.\n\n"
                            "However, Porzingis has an extensive injury history, and many fear that he won't be "
                            "available for the playoffs. Will Kristaps Porzingis miss at least one playoff game due to injury?\n\n"
                            "Resolution criteria: Kristaps Porzingis needs to miss at least one full game, and he needs to "
                            "show up in the Celtics injury report for it to count."
                        ),
                    ),
                    Q=ForecastingQuestion_stripped(
                        title="Will the Celtics win the NBA title?",
                        body=(
                            "Resolves YES if the Boston Celtics win the 2023-2024 NBA championship, "
                            "and NO if they do not win the title.."
                        ),
                    ),
                ),
                assistant=self.OutputFormat_stripped(
                    Q_given_P=ForecastingQuestion_stripped(
                        title=(
                            "Conditional on Kristaps Porzingis missing at least one playoff game due to injury, "
                            "will the Celtics win the NBA title?"
                        ),
                        body=(
                            "The sportswriter Dan Devine recently wrote an article titled "
                            "''Why Kristaps Porziņģis is the key to the Celtics' title hopes'':\n\n"
                            "With the addition of Porzingis’ varied scoring game, a Celtics attack "
                            "that was below league-average in fourth-quarter scoring last season has "
                            "now jumped up to eighth. A group that finished 11th last season in scoring "
                            "efficiency in the “clutch” — when the score’s within five points in the last "
                            "five minutes — is up to fifth.\n\n"
                            "However, Porzingis has an extensive injury history, and many fear that he won't be "
                            "available for the playoffs. How much impact impact would a Kristaps Porzingis injury have? \n\n"
                            "Criterion for condition: Kristaps Porzingis needs to miss at least one full game, and he needs to "
                            "show up in the Celtics injury report for it to count.\n"
                            "Resolution criteria for the outcome: Resolves YES if Boston Celtics win the 2023-2024 "
                            "NBA championship. Resolves NO otherwise.\n"
                            "If the condition is not met, resolves N/A."
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

    class BaseSentenceFormat_stripped(BaseModel):
        P: ForecastingQuestion_stripped

    class ConsequenceType(str, Enum):
        quantity = "quantity"
        time = "time"
        misc = "misc"
        none = "none"

    class ClassifyOutput(BaseModel):
        consequence_type: List["Consequence.ConsequenceType"]

    class InstantiateOutput(BaseModel):
        title: str
        body: str
        resolution_date: datetime

    class OutputFormat(BaseModel):
        cons_P: ForecastingQuestion

    class OutputFormat_stripped(BaseModel):
        cons_P: ForecastingQuestion_stripped

    def __init__(self):
        self.consequence_type_prompt = (
            "You are a helpful assistant. I want you to assist me with a task. \n"
            "We need to classify a forecasting question P with a Yes/No answer "
            "into one or more of the following categories: (1) _quantity_, (2) _time_, (3) _misc_ or (4) _none_. "
            "We will do this depending on which questions can be found to be logical consequences of P. "
            "A question Q is a _consequence_ of P if a positive answer to P implies a positive answer to Q. "
            "*Quantity:*\n"
            "A question Q can be a consequence of P due to _quantity monotonicity_. Example: "
            "P: Will China win more than 10 gold medals in the 2026 Winter Olympics? "
            "Q: Will China win more than 8 medals in the 2026 Winter Olympics? "
            "We will classify P as _quantity-monotonic_ if we can find a question Q that is a consequence of P "
            "due to quantity monotonicity. "
            "Examples:\n"
            "P: Will the Republican nominee get more than 60\% of the vote in the 2028 presidential election? "
            "P: Will the record for 100m sprint will be lower than 9.30 seconds by 2030"
            "*Time:*\n"
            "A question Q can be a consequence of P due to _time monotonicity_. Example: "
            "P: Will the price of Bitcoin reach a peak of 100k at any point before 2027? "
            "Q: Will the price of Bitcoin reach a peak of 100k at any point before 2028? "
            "We will classify P as _time-monotonic_ if we can find a question Q that is a consequence of P "
            "due to time monotonicity. "
            "Examples:\n"
            "P: Will a Swiss person set foot on Mars before 2085? "
            "P: Will Catalonia have a referendum on independence before 2030?"
            "*Misc:*\n"
            "A question Q can be a consequence of P due to other reasons. Example: "
            "P: Will a Swiss person set foot on Mars before 2085? "
            "Q: Will any human set foot on Mars before 2085?\n "
            "P: Will solar energy account for 50\% or more of the US energy production in 2030?\n"
            "Q: Will renewables (solar, wind, hydro) account for 50\% or more of the US energy production in 2030?\n"
            "We will classify P as _misc_ if we can find a question Q that is a consequence of P "
            "due to other reasons, including but not limited to:\n"
            "(i)  Generalization of subjects: If P involves a specific subject or group, Q can generalize this to a broader category, as in the Swiss person -> any human example above.\n"
            "(ii) Inclusion of additional scenarios: If P involves a specific scenario, Q can include this scenario, as in the renewables -> solar, wind, hydro example above.\n"
            "*None:*\n"
            "Finally there are cases where it is not clear if there is a consequence between the two questions. "
            "Classify P as _none_ if it is not clear how to find a question Q that is a straightforward consequence of P. "
            "Examples:\n"
            "P: Will Rihanna release a new album in 2025?\n"
            "P: Will there be a snowstorm in Toronto on New Year's Eve?\n"
            "P: Will Germany win or draw against France in their next World Cup match?\n"
            "\n\n"
            "Your task now is to classify the following question into one or more of the categories: quantity, time, misc or none. "
            "P: {P}"
        )

        self.quantity_instantiator_prompt = (
            "You are a helpful assistant. I want you to assist me with a task. \n"
            "For a given P I want to find a question Q that is a consequence of P due to quantity monotonicity.\n"
            "A question Q is a consequence of P if a positive answer to P implies a positive answer to Q.\n"
            "Both P and Q will have Yes/No answers. Both P and Q will have the following fields:\n"
            "Title: The question statement.\n"
            "Body: A longer description of the conditions and the resolution criteria.\n"
            "Resolution date: The date by which the question should be resolved.\n"
            "Q should be a consequence of P due to quantity monotonicity, so the resolution date should be the same.\n"
            "Change the body as little as possible, but make sure that the question is a consequence of P due to quantity monotonicity.\n"
            "Examples:\n"
            "P:\n"
            "title: Will China win more than 10 gold medals in the 2026 Winter Olympics?\n"
            "body: Resolves YES if China wins more than 10 gold medals in the 2026 Winter Olympics. Resolves NO otherwise.\n"
            "resolution date: 2026-03-01 00:00:00\n"
            "Q:\n"
            "title: Will China win more than 8 medals in the 2026 Winter Olympics?\n"
            "body: Resolves YES if China wins more than 8 medals in the 2026 Winter Olympics. Resolves NO otherwise.\n"
            "resolution date: 2026-03-01 00:00:00\n"
            "\n"
            "P:\n"
            "title: Will the global sales of electric vehicles exceed 5 million units in 2025?\n"
            "body: Resolves YES if the global sales of electric vehicles exceed 5 million units in 2025. Resolves NO otherwise.\n"
            "resolution date: 2025-12-31 00:00:00\n"
            "Q:\n"
            "title: Will the global sales of electric vehicles exceed 4 million units in 2025?\n"
            "body: Resolves YES if the global sales of electric vehicles exceed 4 million units in 2025. Resolves NO otherwise.\n"
            "resolution date: 2025-12-31 00:00:00\n"
            "\n"
            "P:\n"
            "title: Will the average temperature of the Earth increase by less than 1.5 degrees Celsius by 2050?\n"
            "body: Resolves YES if the average temperature of the Earth increases by less than 1.5 degrees Celsius by 2050. Resolves NO otherwise.\n"
            "resolution date: 2050-12-31 00:00:00\n"
            "Q:\n"
            "title: Will the average temperature of the Earth increase by less than 2 degrees Celsius by 2050?\n"
            "body: Resolves YES if the average temperature of the Earth increases by less than 2 degrees Celsius by 2050. Resolves NO otherwise.\n"
            "resolution date: 2050-12-31 00:00:00\n"
            "\n\n"
            "P:\n"
            "title: {title}\n"
            "body: {body}\n"
            "resolution date: {resolution_date}\n"
        )

        self.time_instantiator_prompt = (
            "You are a helpful assistant. I want you to assist me with a task. \n"
            "For a given P I want to find a question Q that is a consequence of P due to time monotonicity.\n"
            "A question Q is a consequence of P if a positive answer to P implies a positive answer to Q.\n"
            "Both P and Q will have Yes/No answers. Both P and Q will have the following fields:\n"
            "Title: The question statement.\n"
            "Body: A longer description of the conditions and the resolution criteria.\n"
            "Resolution date: The date by which the question should be resolved.\n"
            "Q should be a consequence of P due to time monotonicity, so the resolution date should not be the same.\n"
            "It should be a later date if P being true at that date implies Q being true at a later date.\n"
            "It should be an earlier date if P being true at that date implies Q being true at an earlier date.\n"
            "Change the body as little as possible, but make sure that the question is a consequence of P due to time monotonicity.\n"
            "Examples:\n"
            "P:\n"
            "title: Will the price of Bitcoin reach a peak of 100k at any point before 2027?\n"
            "body: Resolves YES if the price of Bitcoin reaches a peak of 100k at any point before 2027. Resolves NO otherwise.\n"
            "resolution date: 2027-01-01 00:00:00\n"
            "Q:\n"
            "title: Will the price of Bitcoin reach a peak of 100k at any point before 2028?\n"
            "body: Resolves YES if the price of Bitcoin reaches a peak of 100k at any point before 2028. Resolves NO otherwise.\n"
            "resolution date: 2027-01-01 00:00:00\n"
            "\n"
            "P:\n"
            "title: Will a Swiss person set foot on Mars before 2085?\n"
            "body: Resolves YES if a Swiss person sets foot on Mars before 2085. Resolves NO otherwise.\n"
            "resolution date: 2085-01-01 00:00:00\n"
            "Q:\n"
            "title: Will a Swiss person set foot on Mars before 2086?\n"
            "body: Resolves YES if a Swiss person sets foot on Mars before 2086.\n"
            "resolution date: 2086-01-01 00:00:00\n"
            "P:\n"
            "title: Will the Golden Bay Bridge remain standing in 2095?\n"
            "body: The resolution of this question will be based on credible reports and sources available on January 1, 2095."
            " Sources may include, but are not limited to, news articles, government publications, and official announcements "
            "from the agency responsible for the bridge's maintenance.\n"
            "If there is ambiguity or conflicting reports, a preponderance of evidence standard will be used to determine the outcome."
            "resolution date: 2095-01-01 00:00:00\n"
            "Q:\n"
            "title: Will the Golden Bay Bridge remain standing in 2090?\n"
            "body: The resolution of this question will be based on credible reports and sources available on January 1, 2090."
            " Sources may include, but are not limited to, news articles, government publications, and official announcements "
            "from the agency responsible for the bridge's maintenance.\n"
            "If there is ambiguity or conflicting reports, a preponderance of evidence standard will be used to determine the outcome."
            "resolution date: 2090-01-01 00:00:00\n"
            "\n\n"
            "P:\n"
            "title: {title}\n"
            "body: {body}\n"
            "resolution date: {resolution_date}\n"
        )

        self.misc_instantiator_prompt = (
            "You are a helpful assistant. I want you to assist me with a task. \n"
            "For a given P I want to find a question Q that is a consequence of P due to reasons other than time and quantity monotonicity.\n"
            "A question Q is a consequence of P if a positive answer to P implies a positive answer to Q.\n"
            "Both P and Q will have Yes/No answers. Both P and Q will have the following fields:\n"
            "Title: The question statement.\n"
            "Body: A longer description of the conditions and the resolution criteria.\n"
            "Resolution date: The date by which the question should be resolved.\n"
            "The resolution date should be the same as P.\n"
            "Change the body as little as possible, but make sure that the question is a consequence of P due to other reasons.\n"
            "Examples:\n"
            "P:\n"
            "title: Will a Swiss person set foot on Mars before 2085?\n"
            "body: Resolves YES if a Swiss person sets foot on Mars before 2085. Resolves NO otherwise.\n"
            "resolution date: 2085-01-01 00:00:00\n"
            "Q:\n"
            "title: Will any human set foot on Mars before 2085?\n"
            "body: Resolves YES if any human sets foot on Mars before 2085. Resolves NO otherwise.\n"
            "resolution date: 2085-01-01 00:00:00\n"
            "\n"
            "P:\n"
            "title: Will the Golden Bay Bridge remain standing in 2095?\n"
            "body: The resolution of this question will be based on credible reports and sources available on January 1, 2095. "
            "Sources may include, but are not limited to, news articles, government publications, and official announcements "
            "from the agency responsible for the bridge's maintenance. If there is ambiguity or conflicting reports, a preponderance "
            "of evidence standard will be used to determine the outcome.\n"
            "resolution date: 2095-01-01 00:00:00\n"
            "Q:\n"
            "title: Will the Golden Bay Bridge remain operational in 2095?\n"
            "body: The resolution of this question will be based on credible reports and sources available on January 1, 2095. "
            "Sources may include, but are not limited to, news articles, government publications, and official announcements "
            "from the agency responsible for the bridge's maintenance. If there is ambiguity or conflicting reports, a preponderance "
            "of evidence standard will be used to determine the outcome.\n"
            "resolution date: 2095-01-01 00:00:00\n"
            "\n"
            "P:\n"
            "title: Will a new species of mammal be discovered by 2050?\n"
            "body: The resolution of this question will be based on credible scientific reports and publications available by December 31, 2050. "
            "Sources may include, but are not limited to, peer-reviewed journals, announcements from scientific organizations, and official records "
            "of new species discovery. If there is ambiguity or conflicting reports, a preponderance of evidence standard will be used to determine the outcome.\n"
            "resolution date: 2050-12-31 00:00:00\n"
            "Q:\n"
            "title: Will a new species of mammal be discovered in the Amazon rainforest by 2050?\n"
            "body: The resolution of this question will be based on credible scientific reports and publications available by December 31, 2050. "
            "Sources may include, but are not limited to, peer-reviewed journals, announcements from scientific organizations, and official records "
            "of new species discovery. If there is ambiguity or conflicting reports, a preponderance of evidence standard will be used to determine the outcome.\n"
            "resolution date: 2050-12-31 00:00:00\n"
            "\n\n"
            "P:\n"
            "title: {title}\n"
            "body: {body}\n"
            "resolution_date: {resolution_date}\n"
        )

    async def instantiate(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        n_verify=3,
        **kwargs,
    ) -> "Self.OutputFormat":
        p = base_sentences["P"]
        consequence_types = await self._classify_consequence(p.title)
        instantiation_results = []
        if self.ConsequenceType.none in consequence_types.consequence_type:
            return instantiation_results
        for consequence_type in consequence_types.consequence_type:
            if consequence_type == self.ConsequenceType.quantity:
                instantiation_results.append(await self._instantiate(p, "quantity"))
            elif consequence_type == self.ConsequenceType.time:
                instantiation_results.append(await self._instantiate(p, "time"))
            else:
                instantiation_results.append(await self._instantiate(p, "misc"))
        return instantiation_results

    def instantiate_sync(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        n_verify=3,
        **kwargs,
    ) -> "Self.OutputFormat":
        p = base_sentences["P"]
        consequence_types = self._classify_consequence_sync(p.title)
        instantiation_results = []
        if self.ConsequenceType.none in consequence_types.consequence_type:
            return instantiation_results
        for consequence_type in consequence_types.consequence_type:
            if consequence_type == self.ConsequenceType.quantity:
                instantiation_results.append(self._instantiate_sync(p, "quantity"))
            elif consequence_type == self.ConsequenceType.time:
                instantiation_results.append(self._instantiate_sync(p, "time"))
            else:
                instantiation_results.append(self._instantiate_sync(p, "misc"))
        return instantiation_results

    async def _classify_consequence(self, p: str) -> "Self.ClassifyOutput":
        prompt = self.consequence_type_prompt.format(P=p)
        consequence_types = await answer(prompt, response_model=self.ClassifyOutput)
        return consequence_types

    def _classify_consequence_sync(self, p: str) -> "Self.ClassifyOutput":
        prompt = self.consequence_type_prompt.format(P=p)
        consequence_types = answer_sync(prompt, response_model=self.ClassifyOutput)
        return consequence_types

    async def _instantiate(
        self, p: ForecastingQuestion, consequence_type: str
    ) -> "Self.OutputFormat":
        if consequence_type == "quantity":
            prompt = self.quantity_instantiator_prompt.format(
                title=p.title,
                body=p.body,
                resolution_date=p.resolution_date,
            )
        elif consequence_type == "time":
            prompt = self.time_instantiator_prompt.format(
                title=p.title,
                body=p.body,
                resolution_date=p.resolution_date,
            )
        else:
            prompt = self.misc_instantiator_prompt.format(
                title=p.title,
                body=p.body,
                resolution_date=p.resolution_date,
            )
        return self._get_output_format(
            p, await answer(prompt, response_model=self.InstantiateOutput)
        )

    def _instantiate_sync(
        self, p: ForecastingQuestion, consequence_type: str
    ) -> "Self.OutputFormat":
        if consequence_type == "quantity":
            prompt = self.quantity_instantiator_prompt.format(
                title=p.title,
                body=p.body,
                resolution_date=p.resolution_date,
            )
        elif consequence_type == "time":
            prompt = self.time_instantiator_prompt.format(
                title=p.title,
                body=p.body,
                resolution_date=p.resolution_date,
            )
        else:
            prompt = self.misc_instantiator_prompt.format(
                title=p.title,
                body=p.body,
                resolution_date=p.resolution_date,
            )
        return self._get_output_format(
            p, answer_sync(prompt, response_model=self.InstantiateOutput)
        )

    def _get_output_format(
        self, p: ForecastingQuestion, instantiate_output: "Self.InstantiateOutput"
    ) -> "Self.OutputFormat":
        forecasting_question = ForecastingQuestion(
            title=instantiate_output.title,
            body=instantiate_output.body,
            resolution_date=instantiate_output.resolution_date,
            question_type=p.question_type,
            metadata={**p.metadata, "consequence_type": "quantity"},
        )
        return self.OutputFormat(cons_P=forecasting_question)

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
        return {"cons_P": resolutions["P"]}


class RelevantInfo(MiniInstantiator):
    """Given some forecasting question(s), generates relevant question(s)
    whose answer a forecaster might want to know in order to answer the
    original question(s)."""

    class BaseSentenceFormat(BaseModel):
        X: list[ForecastingQuestion]

    class BaseSentenceFormat_stripped(BaseModel):
        X: list[ForecastingQuestion_stripped]

    class OutputFormat(BaseModel):
        K: list[InformationPiece]

    class OutputFormat_stripped(BaseModel):
        K: list[InformationPiece]

    def __init__(self):
        self.preface = (
            "You are a superforecaster. I will give you one or more forecasting questions. "
            "You need to think of what pieces of information would be {rel} useful to you in "
            "answering these questions / what questions you would like the answer to, "
            "that you expect to be {rel} useful in informing your probability estimate for the "
            "original forecasting question(s).\n\n"
            "These pieces of information should be specific answerable questions with YES/NO "
            "answers, whose answers can be definitively known with some research, well before "
            "the original forecasting question resolves."
        )
        self.preface_relevance_high = self.preface.format(rel="most")
        self.preface_relevance_mid = self.preface.format(rel="somewhat")
        self.preface_relevance_low = self.preface.format(
            rel="not at all (completely irrelevant)"
        )
        self.examples = [
            {
                "example": Example(
                    user=self.BaseSentenceFormat_stripped(
                        X=[
                            ForecastingQuestion_stripped(
                                title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
                                body=(
                                    "Resolves YES if the price of Bitcoin on 1st January 2025 is more than "
                                    "$100,000. Resolves NO otherwise."
                                ),
                            )
                        ]
                    ),
                    assistant=self.OutputFormat_stripped(
                        K=[
                            InformationPiece(
                                title="Has a bill been proposed in the US Congress to ban Bitcoin?",
                                body=(
                                    "Resolves YES if any member of the US Congress has introduced a bill to "
                                    "ban Bitcoin as of 31 March 2024. "
                                ),
                                question_type="binary",
                            ),
                            InformationPiece(
                                title="Has a major financial institution announced plans to accept Bitcoin?",
                                body=(
                                    "Resolves YES if any major financial institution has announced plans to "
                                    "accept Bitcoin as of 31 March 2024. "
                                ),
                                question_type="binary",
                            ),
                        ]
                    ),
                ),
                "relevance": 8,
            },
            {
                "example": Example(
                    user=self.BaseSentenceFormat_stripped(
                        X=[
                            ForecastingQuestion_stripped(
                                title="Will a Republican win the 2028 US presidential election?",
                                body=(
                                    "Resolves YES if the Republican nominee in the 2028 presidential election "
                                    "gets more than 60% of the vote. Resolves NO otherwise."
                                ),
                            )
                        ]
                    ),
                    assistant=self.OutputFormat_stripped(
                        K=[
                            InformationPiece(
                                title="Will the Democratic nominee get more than 60% of the vote in the 2028 presidential election?",
                                body=(
                                    "Resolves YES if the Democratic nominee in the 2028 presidential election "
                                    "gets more than 60% of the vote. Resolves NO otherwise."
                                ),
                                question_type="binary",
                            ),
                            InformationPiece(
                                title="Will the Libertarian nominee get more than 60% of the vote in the 2028 presidential election?",
                                body=(
                                    "Resolves YES if the Libertarian nominee in the 2028 presidential election "
                                    "gets more than 60% of the vote. Resolves NO otherwise."
                                ),
                                question_type="binary",
                            ),
                        ]
                    ),
                ),
                "relevance": 8,
            },
            {
                "example": Example(
                    user=self.BaseSentenceFormat_stripped(
                        X=[
                            ForecastingQuestion_stripped(
                                title="Will Donald Trump be convicted of a crime by 2025?",
                                body=(
                                    "Resolves YES if Donald Trump is convicted of a crime by 2025. "
                                ),
                            ),
                        ]
                    ),
                    assistant=self.OutputFormat_stripped(
                        K=[
                            InformationPiece(
                                title="Of the jury members in the trial of Donald Trump, are there at least 3 who are registered Democrats?",
                                body=(
                                    "Resolves YES if there are at least 3 jury members who are registered Democrats. "
                                ),
                                question_type="binary",
                            ),
                            InformationPiece(
                                title="Is the judge in the trial of Donald Trump a registered Democrat?",
                                body=(
                                    "Resolves YES if the judge is a registered Democrat. "
                                ),
                                question_type="binary",
                            ),
                        ]
                    ),
                ),
                "relevance": 8,
            },
            {
                "example": Example(
                    user=self.BaseSentenceFormat_stripped(
                        X=[
                            ForecastingQuestion_stripped(
                                title="Will there be a trillionaire by 2050?",
                                body=(
                                    "Resolves YES if at any point before 2050, there is a person whose net worth "
                                    "exceeds $1 trillion according to Forbes. Resolves NO otherwise."
                                ),
                            ),
                        ]
                    ),
                    assistant=self.OutputFormat_stripped(
                        K=[
                            InformationPiece(
                                title="Is Elon Musk considering running for president?",
                                body=(
                                    "Resolves YES if Elon Musk is considering running for president "
                                    "at any time in his lifetime."
                                ),
                                question_type="binary",
                            ),
                            InformationPiece(
                                title="Is socialism popular among 18-24 year olds in the US?",
                                body=(
                                    "Resolves YES if at least 50 percent of 18-24 year olds in the US have a positive "
                                    "view of socialism according to a reputable poll."
                                ),
                                question_type="binary",
                            ),
                        ]
                    ),
                ),
                "relevance": 5,
            },
            {
                "example": Example(
                    user=self.BaseSentenceFormat_stripped(
                        X=[
                            ForecastingQuestion_stripped(
                                title="Will Joe Biden win the 2024 US presidential election?",
                                body=(
                                    "Resolves YES if Joe Biden wins the 2024 US presidential election. "
                                    "Resolves NO otherwise."
                                ),
                            ),
                        ],
                    ),
                    assistant=self.OutputFormat_stripped(
                        K=[
                            InformationPiece(
                                title="Is the US GDP growth rate in Q3 of FY 2023 at least 2 percent?",
                                body=(
                                    "Resolves YES if the US GDP growth rate in Q3 of FY 2023 is at least 2 percent."
                                ),
                                question_type="binary",
                            ),
                        ],
                    ),
                ),
                "relevance": 5,
            },
            {
                "example": Example(
                    user=self.BaseSentenceFormat_stripped(
                        X=[
                            ForecastingQuestion_stripped(
                                title="Will Joe Biden win the 2024 US presidential election?",
                                body=(
                                    "Resolves YES if Joe Biden wins the 2024 US presidential election. "
                                    "Resolves NO otherwise."
                                ),
                            ),
                        ],
                    ),
                    assistant=self.OutputFormat_stripped(
                        K=[
                            InformationPiece(
                                title="Did it rain in Tokyo on 3 July 2024?",
                                body=(
                                    "Resolves YES if it rained in Tokyo on 3 July 2024."
                                ),
                                question_type="binary",
                            ),
                        ],
                    ),
                ),
                "relevance": 1,
            },
        ]
        self.examples_relevance_high = [
            example["example"] for example in self.examples if example["relevance"] >= 7
        ]
        self.examples_relevance_mid = [
            example["example"]
            for example in self.examples
            if 4 <= example["relevance"] < 7
        ]
        self.examples_relevance_low = [
            example["example"] for example in self.examples if example["relevance"] < 4
        ]

    def instantiate_sync(
        self,
        base_sentences: list[ForecastingQuestion],
        relevance="high",
        **kwargs,
    ) -> "Self.OutputFormat":
        preface = (
            self.preface_relevance_high
            if relevance == "high"
            else (
                self.preface_relevance_mid
                if relevance == "mid"
                else self.preface_relevance_low
            )
        )
        if self.use_examples_here:
            examples = (
                self.examples_relevance_high
                if relevance == "high"
                else (
                    self.examples_relevance_mid
                    if relevance == "mid"
                    else self.examples_relevance_low
                )
            )
        else:
            examples = None
        based_sentences = self.BaseSentenceFormat_stripped(
            X=[x.cast_stripped() for x in base_sentences]
        )
        return answer_sync(
            prompt=based_sentences,
            preface=preface,
            examples=examples,
            prepare_messages_func=prepare_messages_alt,
            response_model=self.OutputFormat_stripped,
            **kwargs,
        )

    async def instantiate(
        self,
        base_sentences: list[ForecastingQuestion],
        relevance="high",
        **kwargs,
    ) -> "Self.OutputFormat":
        preface = (
            self.preface_relevance_high
            if relevance == "high"
            else (
                self.preface_relevance_mid
                if relevance == "mid"
                else self.preface_relevance_low
            )
        )
        if self.use_examples_here:
            examples = (
                self.examples_relevance_high
                if relevance == "high"
                else (
                    self.examples_relevance_mid
                    if relevance == "mid"
                    else self.examples_relevance_low
                )
            )
        else:
            examples = None
        based_sentences = self.BaseSentenceFormat_stripped(
            X=[x.cast_stripped() for x in base_sentences]
        )
        return await answer(
            prompt=based_sentences,
            preface=preface,
            examples=examples,
            prepare_messages_func=prepare_messages_alt,
            response_model=self.OutputFormat,
            **kwargs,
        )


register_models_for_cache([Consequence.ClassifyOutput, Consequence.InstantiateOutput])

for instantiator in [
    Trivial,
    Neg,
    And,
    Or,
    Paraphrase,
    Conditional,
    Consequence,
    RelevantInfo,
]:
    register_models_for_cache(
        [
            instantiator.BaseSentenceFormat,
            instantiator.BaseSentenceFormat_stripped,
            instantiator.OutputFormat,
            instantiator.OutputFormat_stripped,
        ]
    )
