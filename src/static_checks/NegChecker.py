from common.llm_utils import answer_sync, answer, Example
from common.datatypes import *
from .BaseChecker import BaseChecker
from pydantic import BaseModel, field_validator
from typing import Self


class NegChecker(BaseChecker):
    """Where f(x) is the forecaster,
    R(x1, x2)       :=  x2 == ¬x1
    S(f(x1), f(x2)) :=  f(x1) + f(x2) = 1
    """

    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion

        @field_validator("P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        not_P: ForecastingQuestion

        @field_validator("P", "not_P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    preface = (
        "You are a helpful assistant. I will give you a forecasting question with Yes/No "
        "answer. You should then give me the NEGATION of the question, i.e. the question that "
        "would be answered YES if the original question would be answered NO, and vice "
        "versa. Demorgan's laws should be followed with and/or negation. Avoid using the word "
        "'won't'."
    )

    examples = [
        Example(
            user=ForecastingQuestion_simple(
                title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
                body=(
                    "Resolves YES if the spot price of Bitcoin against USD is more than "
                    "100,000 on 1st January 2025. Resolves NO otherwise."
                ),
            ).model_dump_json(),
            assistant=ForecastingQuestion_simple(
                title="Will the price of Bitcoin be less than or equal to $100,000 on 1st January 2025?",
                body=(
                    "Resolves YES if the spot price of Bitcoin against USD is more than "
                    "100,000 on 1st January 2025. Resolves NO otherwise."
                ),
            ).model_dump_json(),
        )
    ]

    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)

    def instantiate_sync(
        self, base_sentences: "Self.BaseSentenceFormat", **kwargs
    ) -> "Self.TupleFormat":
        response = answer_sync(
            prompt=base_sentences.model_dump_json(indent=4),
            preface=self.preface,
            examples=self.examples,
            response_model=ForecastingQuestion_simple,
            **kwargs
        )
        response_FQ = response.cast_FQ(
            resolution_date=base_sentences.P.resolution_date,
            question_type=base_sentences.P.question_type,
            data_source="synthetic_inst",
            resolution=(
                not base_sentences.P.resolution
                if base_sentences.P.resolution is not None
                else None
            ),
        )
        return self.TupleFormat(P=base_sentences.P, not_P=response_FQ)

    async def instantiate(
        self, base_sentences: "Self.BaseSentenceFormat", **kwargs
    ) -> "Self.TupleFormat":
        response = await answer(
            prompt=base_sentences.model_dump_json(indent=4),
            preface=self.preface,
            examples=self.examples,
            response_model=ForecastingQuestion_simple,
            **kwargs
        )
        response_FQ = response.cast_FQ(
            resolution_date=base_sentences.P.resolution_date,
            question_type=base_sentences.P.question_type,
            data_source="synthetic_inst",
            resolution=(
                not base_sentences.P.resolution
                if base_sentences.P.resolution is not None
                else None
            ),
        )
        return self.TupleFormat(P=base_sentences.P, not_P=response_FQ)

    def violation(self, answers: dict[str, Prob]) -> float:
        return abs(answers["P"] + answers["not_P"] - 1)
