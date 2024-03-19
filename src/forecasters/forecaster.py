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
            raise ValueError("Question type must be one of 'binary', 'conditional_binary'.")
        return super(QuestionType, cls).__new__(cls, value)
    
    exp_answers = {
        "binary": Prob,
        "conditional_binary": Prob
    }
    
    def expected_answer_type(self) -> str:
        return self.exp_answers.get(self, None)

@dataclass
class SentenceTemplate(str):
    id: str # TODO: change to ID
    title: str # aka "text"
    body: str # aka "resolution_criteria"
    question_type: QuestionType
    resolution_date: datetime | None
    url: str | None
    data_source : str | None # usually one of “synthetic”, “metaculus”, “manifold”, “predictit”
    metadata : dict # for example, topics : list[str]
    resolution : str | None # some questions may already have been resolved
    
    

SentencesTemplate = Dict[str, SentenceTemplate]
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