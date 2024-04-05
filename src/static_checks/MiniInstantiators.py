from abc import ABC, abstractmethod
from typing import Type, Optional, Self
from pydantic import BaseModel, create_model, field_validator
from common.datatypes import ForecastingQuestion, ForecastingQuestion_stripped
from common.llm_utils import answer, answer_sync, Example
from datetime import datetime

class MiniInstantiator(ABC):

    # @property
    # @abstractmethod
    # def bs_format(self) -> dict[str, str]:
    #     pass  # e.g. {'P' : 'binary', 'Q' : 'numerical', 'R' : 'binary'}

    # class BaseSentenceFormat(BaseModel):
    #     pass

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

    @abstractmethod
    def title_body_sync_(
        self, base_sentences: "Self.BaseSentenceFormat_stripped", **kwargs
    ) -> ForecastingQuestion_stripped:
        pass

    @abstractmethod
    async def title_body_(
        self, base_sentences: "Self.BaseSentenceFormat_stripped", **kwargs
    ) -> ForecastingQuestion_stripped:
        pass

    def title_body_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> ForecastingQuestion_stripped:
        based_sentences = self.BaseSentenceFormat_stripped(
            **{k: v.cast_stripped() for k, v in base_sentences.items()}
        )
        return self.title_body_sync_(based_sentences, **kwargs)

    async def title_body(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> ForecastingQuestion_stripped:
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
    def resolution_(self, resolutions: dict[str, bool]) -> Optional[bool]:
        """Basically just the MiniInstantiator's logic. E.g. return not resolutions['P']"""
        pass

    def resolution(
        self, base_sentences: dict[str, ForecastingQuestion]
    ) -> Optional[bool]:
        resolutions = {key: base_sentences[key].resolution for key in base_sentences}
        if all([res is not None for res in resolutions.values()]):
            return self.resolution_(resolutions)
        return None

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> ForecastingQuestion:
        title_body = self.title_body_sync(base_sentences, **kwargs)
        return ForecastingQuestion(
            title=title_body.title,
            body=title_body.body,
            resolution_date=self.resolution_date(base_sentences),
            question_type=self.question_type(base_sentences),
            data_source=self.data_source(base_sentences),
            resolution=self.resolution(base_sentences),
        )

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> ForecastingQuestion:
        title_body = await self.title_body(base_sentences, **kwargs)
        return ForecastingQuestion(
            title=title_body.title,
            body=title_body.body,
            resolution_date=self.resolution_date(base_sentences),
            question_type=self.question_type(base_sentences),
            data_source=self.data_source(base_sentences),
            resolution=self.resolution(base_sentences),
        )

class Trivial(MiniInstantiator):

    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion

    def title_body_sync_(
        self, base_sentences: "Self.BaseSentenceFormat", **kwargs
    ) -> ForecastingQuestion_stripped:
        return base_sentences.P.cast_stripped()

    async def title_body_(
        self, base_sentences: "Self.BaseSentenceFormat", **kwargs
    ) -> ForecastingQuestion_stripped:
        return base_sentences.P.cast_stripped()

    def resolution_(self, resolutions: dict[str, bool]) -> Optional[bool]:
        return resolutions["P"]

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

    preface = (
        "You are a helpful assistant. I will give you a forecasting question with Yes/No "
        "answer. You should then give me the NEGATION of the question, i.e. the question that "
        "would be answered YES if the original question would be answered NO, and vice "
        "versa. Demorgan's laws should be followed with and/or negation. Avoid using the word "
        "'won't'."
    )

    examples = [
        Example(
            user=BaseSentenceFormat_stripped(
                P=ForecastingQuestion_stripped(
                    title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
                    body=(
                        "Resolves YES if the spot price of Bitcoin against USD is more than "
                        "100,000 on 1st January 2025. Resolves NO otherwise."
                    ),
                )
            ).model_dump_json(),
            assistant=ForecastingQuestion_stripped(
                title="Will the price of Bitcoin be less than or equal to $100,000 on 1st January 2025?",
                body=(
                    "Resolves YES if the spot price of Bitcoin against USD is less than or equal to "
                    "100,000 on 1st January 2025. Resolves NO otherwise."
                ),
            ).model_dump_json(),
        )
    ]

    def __init__(self):
        super().__init__()

    def title_body_sync_(
        self, base_sentences: "Self.BaseSentenceFormat_stripped", **kwargs
    ) -> ForecastingQuestion_stripped:
        return answer_sync(
            prompt=base_sentences.model_dump_json(),
            preface=self.preface,
            examples=self.examples,
            response_model=ForecastingQuestion_stripped,
            **kwargs
        )

    async def title_body_(
        self, base_sentences: "Self.BaseSentenceFormat_stripped", **kwargs
    ) -> ForecastingQuestion_stripped:
        return await answer(
            prompt=base_sentences.model_dump_json(),
            preface=self.preface,
            examples=self.examples,
            response_model=ForecastingQuestion_stripped,
            **kwargs
        )

    def resolution_(self, resolutions: dict[str, bool]) -> bool | None:
        return not resolutions["P"]
