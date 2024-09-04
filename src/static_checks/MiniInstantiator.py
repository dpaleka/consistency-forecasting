# Path: static_checks/Base.py
import os
from dotenv import load_dotenv
from common.utils import shallow_dict
from datetime import datetime
from dateutil.tz import UTC
from abc import ABC, abstractmethod
from enum import Enum
from typing import Type, Any, Optional, Self, List, List, Union  # noqa
from pydantic import BaseModel, field_validator
from common.llm_utils import (
    answer,
    answer_sync,
    Example,
    prepare_messages_alt,
)
from common.utils import (
    write_jsonl_async_from_str,
    write_jsonl_from_str,
)
from common.path_utils import get_data_path
from common.datatypes import (
    ForecastingQuestion,
    ForecastingQuestion_stripped,
    VerificationResult,
)
from common.perscache import register_models_for_cache
from fq_verification.question_verifier import verify_full_question
from .checker_prompts import (
    neg_verification_prompt,
    and_verification_prompt,
    or_verification_prompt,
    conditional_verification_prompt,
    consequence_verification_prompt,
    consequence_quantity_verification_prompt,
    consequence_time_verification_prompt,
    paraphrase_verification_prompt,
)

TUPLE_VERIFY_SEED = 32

load_dotenv()
verify_before_instantion = os.getenv("VERIFY_BEFORE_INSTANTIATION", "False") == "True"
write_verification = os.getenv("WRITE_VERIFICATION", "False") == "True"
use_examples = os.getenv("USE_EXAMPLES", "False") == "True"
verify_length = os.getenv("VERIFY_LENGTH", "False") == "True"


async def write_verification_result(tuple_type, generated_tuple, verification):
    filename = get_data_path() / "verification/tuple_verifications.jsonl"
    verification_jsonl = generated_tuple.model_dump_json()
    verification_jsonl = (
        verification_jsonl[:-1]
        + f', "valid": "{verification.valid}", "reasoning": "{verification.reasoning}"'
        + f', "tuple_type":"{tuple_type}"'
        + "}"
    )
    await write_jsonl_async_from_str(filename, [verification_jsonl], append=True)


def write_verification_result_sync(tuple_type, generated_tuple, verification):
    filename = get_data_path() / "verification/tuple.jsonl"
    verification_jsonl = generated_tuple.model_dump_json()
    verification_jsonl = (
        verification_jsonl[:-1]
        + f', "valid": "{verification.valid}", "reasoning": "{verification.reasoning}"'
        + f", tuple_type:{tuple_type}"
        + "}"
    )
    write_jsonl_from_str(filename, [verification_jsonl], append=True)


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

    def to_base_sentence_format_stripped(
        self, base_sentences: dict[str, ForecastingQuestion]
    ) -> "Self.BaseSentenceFormat_stripped":
        try:
            return self.BaseSentenceFormat_stripped(
                **{k: v.cast_stripped() for k, v in base_sentences.items()}
            )
        except Exception as e:
            raise ValueError(f"Failed to cast to BaseSentenceFormat_stripped: {str(e)}")

    def title_body_sync_(
        self,
        base_sentences: "Self.BaseSentenceFormat_stripped",
        **kwargs,
    ) -> "Self.OutputFormat_stripped":
        examples = self.examples if self.use_examples_here else None
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
        examples = self.examples if self.use_examples_here else None
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
        base_sentences = self.to_base_sentence_format_stripped(base_sentences)
        return self.title_body_sync_(base_sentences, **kwargs)

    async def title_body(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.OutputFormat_stripped":
        base_sentences = self.to_base_sentence_format_stripped(base_sentences)
        return await self.title_body_(base_sentences, **kwargs)

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
        n_verification: int = 3,
        **kwargs,
    ) -> Union["Self.OutputFormat", List["Self.OutputFormat"]]:
        if verify_before_instantion:
            for i in range(n_verification):
                print("Base sentences:", base_sentences)
                print(f"Instantiation attempt {i}")
                output = self._instantiate_sync(base_sentences, **kwargs)
                based_sentences = self.to_base_sentence_format_stripped(base_sentences)
                print(f"Verifying output: {output}")
                verification = self.verify_sync(output, based_sentences)
                print("Verification result:", verification)
                if verification.valid:
                    return output
            print("All attempts failed")
            return []
        else:
            return self._instantiate_sync(base_sentences, **kwargs)

    def _instantiate_sync(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
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
        n_verification: int = 3,
        **kwargs,
    ) -> Union["Self.OutputFormat", List["Self.OutputFormat"]]:
        if verify_before_instantion:
            for i in range(n_verification):
                print("Base sentences:", base_sentences)
                print(f"Instantiating with instantiator {self.__class__.__name__}")
                print(f"Instantiation attempt {i}")
                output = await self._instantiate(base_sentences, **kwargs)
                based_sentences = self.to_base_sentence_format_stripped(base_sentences)
                print(f"Verifying output: {output}")
                verification = await self.verify(output, based_sentences)
                print("Verification result:", verification)
                if verification.valid:
                    return output
            print("All attempts failed")
            return []
        else:
            return await self._instantiate(base_sentences, **kwargs)

    async def _instantiate(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> "Self.OutputFormat":
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

    def verify_length(
        self, output: "Self.OutputFormat", base_sentences: "Self.BaseSentenceFormat"
    ) -> VerificationResult:
        return VerificationResult(
            valid=True, reasoning="Verify length not implemented, valid by default"
        )

    def _verification_prompt(
        self, output: "Self.OutputFormat", base_sentences: "Self.BaseSentenceFormat"
    ) -> VerificationResult:
        return ""

    def verify_sync(
        self,
        output: Optional["Self.OutputFormat"],
        base_sentences: "Self.BaseSentenceFormat",
        **kwargs,
    ) -> VerificationResult:
        if output is None:
            return VerificationResult(valid=False, reasoning="Output is None")

        prompt = self._verification_prompt(output, base_sentences)
        if verify_length and not self.verify_length(output, base_sentences):
            return VerificationResult(
                valid=False, reasoning="Length of combined question body is too short"
            )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("and", output, verification)
        return verification

    async def verify(
        self,
        output: Optional["Self.OutputFormat"],
        base_sentences: "Self.BaseSentenceFormat",
        seed: int = TUPLE_VERIFY_SEED,
        temperature: float = 0.0,
        **kwargs,
    ) -> VerificationResult:
        if output is None:
            return VerificationResult(valid=False, reasoning="Output is None")

        # Check if any attribute of output has type ForecastingQuestion
        forecasting_question = None
        for attr_name, attr_value in output.__dict__.items():
            if isinstance(attr_value, ForecastingQuestion):
                forecasting_question = attr_value
                break

        # If a ForecastingQuestion is found, call verify_full_question on it
        if forecasting_question:
            full_question_verification = await verify_full_question(
                forecasting_question, **kwargs
            )
            if not full_question_verification.valid:
                return full_question_verification

        prompt = self._verification_prompt(output, base_sentences)
        if verify_length and not self.verify_length(output, base_sentences):
            return VerificationResult(
                valid=False, reasoning="Length of combined question body is too short"
            )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("and", output, verification)
        return verification


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

    def verify_sync(
        self, output: "Self.OutputFormat", base_sentences, **kwargs
    ) -> VerificationResult:
        return VerificationResult(
            valid=True, reasoning="Trivial Instantiation always valid"
        )

    async def verify(
        self,
        output: "Self.OutputFormat",
        base_sentences,
        seed: int = TUPLE_VERIFY_SEED,
        temperature: float = 0.0,
        **kwargs,
    ) -> VerificationResult:
        return VerificationResult(
            valid=True, reasoning="Trivial Instantiation always valid"
        )


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

    def verify_length(
        self,
        output: "Self.OutputFormat",
        base_sentences: "Self.BaseSentenceFormat",
        **kwargs,
    ) -> bool:
        return len(output.not_P.body) > 0.8 * len(base_sentences.P.body)

    def _verification_prompt(
        self, output: "Self.OutputFormat", base_sentences: "Self.BaseSentenceFormat"
    ) -> str:
        return neg_verification_prompt.format(
            P_title=base_sentences.P.title,
            P_body=base_sentences.P.body,
            not_P_title=output.not_P.title,
            not_P_body=output.not_P.body,
        )


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

    def verify_length(
        self,
        output: "Self.OutputFormat",
        base_sentences: "Self.BaseSentenceFormat",
        **kwargs,
    ) -> bool:
        return len(output.P_and_Q.body) > 1.4 * max(
            len(base_sentences.P.body), len(base_sentences.Q.body)
        )

    def _verification_prompt(
        self, output: "Self.OutputFormat", base_sentences: "Self.BaseSentenceFormat"
    ) -> str:
        return and_verification_prompt.format(
            P_title=base_sentences.P.title,
            P_body=base_sentences.P.body,
            Q_title=base_sentences.Q.title,
            Q_body=base_sentences.Q.body,
            R_title=output.P_and_Q.title,
            R_body=output.P_and_Q.body,
        )


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

    def verify_length(
        self,
        output: "Self.OutputFormat",
        base_sentences: "Self.BaseSentenceFormat",
        **kwargs,
    ) -> bool:
        return len(output.P_or_Q.body) > 1.4 * max(
            len(base_sentences.P.body), len(base_sentences.Q.body)
        )

    def _verification_prompt(
        self, output: "Self.OutputFormat", base_sentences: "Self.BaseSentenceFormat"
    ) -> str:
        return or_verification_prompt.format(
            P_title=base_sentences.P.title,
            P_body=base_sentences.P.body,
            Q_title=base_sentences.Q.title,
            Q_body=base_sentences.Q.body,
            R_title=output.P_or_Q.title,
            R_body=output.P_or_Q.body,
        )

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

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return len(generated_tuple.para_P.body) > 0.65 * len(base_sentences.P.body)

    def _verification_prompt(
        self, output: "Self.OutputFormat", base_sentences: "Self.BaseSentenceFormat"
    ) -> str:
        return paraphrase_verification_prompt.format(
            P_title=base_sentences.P.title,
            P_body=base_sentences.P.body,
            para_P_title=output.para_P.title,
            para_P_body=output.para_P.body,
        )


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

    def verify_length(
        self,
        output: "Self.OutputFormat",
        base_sentences: "Self.BaseSentenceFormat",
        **kwargs,
    ) -> bool:
        return len(output.Q_given_P.body) > 1.4 * max(
            len(base_sentences.P.body), len(base_sentences.Q.body)
        )

    def _verification_prompt(
        self, output: "Self.OutputFormat", base_sentences: "Self.BaseSentenceFormat"
    ) -> str:
        return conditional_verification_prompt.format(
            P_title=base_sentences.P.title,
            P_body=base_sentences.P.body,
            Q_title=base_sentences.Q.title,
            Q_body=base_sentences.Q.body,
            Q_given_P_title=output.Q_given_P.title,
            Q_given_P_body=output.Q_given_P.body,
        )


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
            "To help, we also provide precise resolution criteria (body) as well as the title.\n"
            "P:\n"
            "title: {title}\n"
            "body: {body}\n\n"
            "Recall that you need to classify P into one or more of the categories: quantity, time, misc or none."
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
        n_verification: int = 3,
        **kwargs,
    ) -> "Self.OutputFormat":
        p = base_sentences["P"]
        consequence_types = await self._classify_consequence(p)
        instantiation_results = []
        if self.ConsequenceType.none in consequence_types.consequence_type:
            return instantiation_results
        for consequence_type in consequence_types.consequence_type:
            if consequence_type == self.ConsequenceType.quantity:
                instantiation_results += (
                    await self._instantiate_by_type_with_verification(
                        p, "quantity", n_verification=n_verification
                    )
                )
            elif consequence_type == self.ConsequenceType.time:
                instantiation_results += (
                    await self._instantiate_by_type_with_verification(
                        p, "time", n_verification=n_verification
                    )
                )
            else:
                instantiation_results += (
                    await self._instantiate_by_type_with_verification(
                        p, "misc", n_verification=n_verification
                    )
                )
        return instantiation_results

    def instantiate_sync(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        n_verification: int = 3,
        **kwargs,
    ) -> "Self.OutputFormat":
        p = base_sentences["P"]
        consequence_types = self._classify_consequence_sync(base_sentences["P"])
        instantiation_results = []
        if self.ConsequenceType.none in consequence_types.consequence_type:
            return instantiation_results
        for consequence_type in consequence_types.consequence_type:
            if consequence_type == self.ConsequenceType.quantity:
                instantiation_results += (
                    self._instantiate_sync_by_type_with_verification(
                        p, "quantity", n_verification=n_verification
                    )
                )
            elif consequence_type == self.ConsequenceType.time:
                instantiation_results += (
                    self._instantiate_sync_by_type_with_verification(
                        p, "time", n_verification=n_verification
                    )
                )
            else:
                instantiation_results += (
                    self._instantiate_sync_by_type_with_verification(
                        p, "misc", n_verification=n_verification
                    )
                )
        return instantiation_results

    def _instantiate_sync_by_type_with_verification(
        self,
        base_sentences: ForecastingQuestion,
        consequence_type: str,
        n_verification: int = 3,
        **kwargs,
    ) -> Union["Self.OutputFormat", List["Self.OutputFormat"]]:
        if verify_before_instantion:
            for _ in range(n_verification):
                output = self._instantiate_sync_by_type(
                    base_sentences, consequence_type=consequence_type, **kwargs
                )
                if self.verify_sync(output, base_sentences).valid:
                    return [output]
            return []
        else:
            return [
                self._instantiate_sync_by_type(
                    base_sentences, consequence_type=consequence_type, **kwargs
                )
            ]

    async def _instantiate_by_type_with_verification(
        self,
        base_sentences: ForecastingQuestion,
        consequence_type: str,
        n_verification: int = 3,
        **kwargs,
    ) -> Union["Self.OutputFormat", List["Self.OutputFormat"]]:
        if verify_before_instantion:
            for _ in range(n_verification):
                output = await self._instantiate_by_type(
                    base_sentences, consequence_type=consequence_type, **kwargs
                )
                verification = await self.verify(output, base_sentences)
                if verification.valid:
                    return [output]
            return []
        else:
            return await [
                self._instantiate_by_type(
                    base_sentences, consequence_type=consequence_type, **kwargs
                )
            ]

    async def _classify_consequence(
        self, p: ForecastingQuestion
    ) -> "Self.ClassifyOutput":
        prompt = self.consequence_type_prompt.format(
            title=p.title,
            body=p.body,
        )
        consequence_types = await answer(prompt, response_model=self.ClassifyOutput)
        return consequence_types

    def _classify_consequence_sync(
        self, p: ForecastingQuestion
    ) -> "Self.ClassifyOutput":
        prompt = self.consequence_type_prompt.format(
            title=p.title,
            body=p.body,
        )
        consequence_types = answer_sync(prompt, response_model=self.ClassifyOutput)
        return consequence_types

    async def _instantiate_by_type(
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

    def _instantiate_sync_by_type(
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
            metadata=({**p.metadata} if p.metadata else {}).update(
                {"consequence_type": "quantity"}
            ),
        )
        return self.OutputFormat(cons_P=forecasting_question)

    def _verification_prompt(
        self, output: "Self.OutputFormat", base_sentences: "Self.BaseSentenceFormat"
    ) -> str:
        consequence_type = base_sentences.metadata.get("consequence_type", "misc")

        common_format_args = {
            "P_title": base_sentences.title,
            "P_body": base_sentences.body,
            "Q_title": output.cons_P.title,
            "Q_body": output.cons_P.body,
        }

        if consequence_type == "quantity":
            return consequence_quantity_verification_prompt.format(**common_format_args)
        elif consequence_type == "time":
            time_format_args = {
                **common_format_args,
                "P_resolution_date": base_sentences.resolution_date,
                "Q_resolution_date": output.cons_P.resolution_date,
            }
            return consequence_time_verification_prompt.format(**time_format_args)
        else:
            return consequence_verification_prompt.format(**common_format_args)

    def resolution_(self, resolutions: dict[str, bool]) -> dict[str, bool | None]:
        return {"cons_P": resolutions["P"]}


register_models_for_cache([Consequence.ClassifyOutput, Consequence.InstantiateOutput])

for instantiator in [Trivial, Neg, And, Or, Paraphrase, Conditional, Consequence]:
    register_models_for_cache(
        [
            instantiator.BaseSentenceFormat,
            instantiator.BaseSentenceFormat_stripped,
            instantiator.OutputFormat,
            instantiator.OutputFormat_stripped,
        ]
    )
