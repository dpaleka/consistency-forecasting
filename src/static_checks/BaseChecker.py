# Path: static_checks/Base.py
import jsonlines
from abc import ABC, abstractmethod
from typing import Type
from pydantic import BaseModel
from common.utils import write_jsonl_async
from common.llm_utils import parallelized_call
from common.datatypes import *
from forecasters import Forecaster


class BaseChecker(ABC):
    def __init__(self, tolerance=0.1, path=None):
        self.tolerance = tolerance
        if path is None:
            self.path = f"src/data/{self.__class__.__name__}.jsonl"
        else:
            self.path = path

    @abstractmethod
    def instantiate_sync(
        self, base_sentences : Type[BaseModel], **kwargs
    ) -> ForecastingQuestionTuple:
        pass

    @abstractmethod
    async def instantiate(
        self, base_sentences : Type[BaseModel], **kwargs
    ) -> ForecastingQuestionTuple:
        pass

    @abstractmethod
    def violation(self, answers: Type[BaseModel]) -> float:
        pass

    async def instantiate_and_write(self, base_sentences: Type[BaseModel], **kwargs):
        result = await self.instantiate(base_sentences, **kwargs)
        
        result_serial = {k: v.to_dict() for k, v in result.items()} # serialize ForecastingQuestions into dicts
        if kwargs.get("verbose", True):
            print(f"Writing tuple to {self.path}: {result_serial}")
        await write_jsonl_async(self.path, [result_serial], append=True)

    async def instantiate_and_write_many(
        self, base_sentencess: list[list[ForecastingQuestion]], **kwargs
    ):
        _instantiate_and_write = lambda base_sentences: self.instantiate_and_write(
            *base_sentences, **kwargs
        )
        await parallelized_call(_instantiate_and_write, base_sentencess)

    def check(self, answers: ProbsTuple) -> bool:
        return self.violation(answers) < self.tolerance

    def elicit_and_violation(
        self, forecaster: Forecaster, sentences: ForecastingQuestionTuple
    ) -> float:
        return self.violation(forecaster.elicit(sentences))

    def elicit_and_check(
        self, forecaster: Forecaster, sentences: ForecastingQuestionTuple
    ) -> bool:
        return self.check(forecaster.elicit(sentences))

    def test(self, forecaster: Forecaster, **kwargs):
        for line in jsonlines.open(self.path):
            print("START")
            print(f"line: {line}")
            line_obj = {k: ForecastingQuestion.from_dict(v) for k, v in line.items()}
            answers = forecaster.elicit(line_obj)
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
