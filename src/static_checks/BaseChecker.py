# Path: static_checks/Base.py
import jsonlines
from abc import ABC, abstractmethod
from typing import Type, Self
from pydantic import BaseModel
from common.utils import write_jsonl_async_from_str
from common.llm_utils import parallelized_call
from common.datatypes import *
from forecasters import Forecaster


class BaseChecker(ABC):

    @property
    @abstractmethod
    def BaseSentenceFormat(self) -> Type[BaseModel]:
        pass

    @property
    @abstractmethod
    def TupleFormat(self) -> Type[BaseModel]:
        pass

    def __init__(self, tolerance=0.1, path=None):
        self.tolerance = tolerance
        if path is None:
            self.path = f"src/data/{self.__class__.__name__}.jsonl"
        else:
            self.path = path

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

    @abstractmethod
    def violation(self, answers: dict[str, Prob]) -> float:
        pass

    async def instantiate_and_write(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ):
        result = await self.instantiate(base_sentences, **kwargs)

        if kwargs.get("verbose", True):
            print(f"Writing tuple to {self.path}: {result}")
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

    def check(self, answers: dict[str, Prob]) -> bool:
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
