from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict
import asyncio
from dataclasses import dataclass


class Prob(float):
    def __new__(cls, value):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return super(Prob, cls).__new__(cls, value)


class QuestionType(str):
    def __new__(cls, value):
        if value not in ["binary", "conditional_binary"]:
            raise ValueError(
                "Question type must be one of 'binary', 'conditional_binary'."
            )
        return super(QuestionType, cls).__new__(cls, value)

    exp_answers = {"binary": Prob, "conditional_binary": Prob}

    def expected_answer_type(self) -> str:
        return self.exp_answers.get(self, None)


class Sentence:
    def __init__(
        self,
        id : str, # TODO: change to ID
        title: str, # aka "text"
        body: str, # aka "resolution_criteria"
        question_type: QuestionType,
        resolution_date: datetime | None,
        url: str | None,
        data_source: str | None, # e.g. synthetic, metaculus, manifold, predictit
        metadata: dict, # for example, topics : list[str]
        resolution: str | None, # some questions may already have been resolved
    ):
        self.id = id
        self.title = title
        self.body = body
        self.question_type = question_type
        self.resolution_date = resolution_date
        self.url = url
        self.data_source = data_source
        self.metadata = metadata
        self.resolution = resolution

    def __str__(self):
        return (
            f"TITLE: {self.title}\n"
            f"RESOLUTION DATE: {self.resolution_date}\n"
            f"DETAILS:\n{self.body}")


SentencesTemplate = Dict[str, Sentence]
ProbsTemplate = Dict[str, Prob]

class Forecaster(ABC):

    def elicit(self, sentences: SentencesTemplate) -> ProbsTemplate:
        return {k: self.call(v) for k, v in sentences.items()}

    async def elicit_async(self, sentences: SentencesTemplate) -> ProbsTemplate:
        keys, values = zip(*sentences.items())
        tasks = [self.call_async(v) for v in values]
        results = await asyncio.gather(*tasks)
        return {k: v for k, v in zip(keys, results)}

    @abstractmethod
    def call(self, sentence: str) -> Prob:
        pass

    @abstractmethod
    async def call_async(self, sentence: str) -> Prob:
        pass
