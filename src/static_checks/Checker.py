import jsonlines
from abc import ABC, abstractmethod
from typing import Type, Any, Self
from pydantic import BaseModel, field_validator
from common.datatypes import ForecastingQuestion, Prob
from common.utils import write_jsonl_async_from_str
from common.llm_utils import parallelized_call
from forecasters import Forecaster
from .MiniInstantiator import *

class Checker(ABC):

    def __init__(self, tolerance=0.1, path=None):
        self.tolerance = tolerance
        if path is None:
            self.path = f"src/data/{self.__class__.__name__}.jsonl"
        else:
            self.path = path

    @property
    @abstractmethod
    def TupleFormat(self) -> Type[BaseModel]:
        pass

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

    async def instantiate_and_write(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ):
        result = await self.instantiate(base_sentences, **kwargs)
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

    @abstractmethod
    def violation(self, answers: dict[str, Prob]) -> float:
        pass

    def check(self, answers: dict[str, Any]) -> bool:
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
            if any([a is None for a in answers.values()]):
                print("ERROR: Some answers are None!")
                continue
            loss = self.violation(answers)
            res_bool = self.check(answers)
            res = {True: "Passed", False: "Failed"}[res_bool]
            print(f"Violation: {loss}")
            print(f"Check result: {res}")
            print("")

class NegChecker(Checker):

    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        not_P: ForecastingQuestion

        @field_validator("P", "not_P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = Trivial().instantiate_sync(base_sentences, **kwargs)
        not_P = Neg().instantiate_sync(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, not_P=not_P.not_P)

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate(base_sentences, **kwargs)
        not_P = await Neg().instantiate(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, not_P=not_P.not_P)

    def violation(self, answers: dict[str, Prob]) -> float:
        return abs(answers["P"] + answers["not_P"] - 1)

class AndChecker(Checker):

    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion
        P_and_Q: ForecastingQuestion

        @field_validator("P", "Q", "P_and_Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = Trivial().instantiate_sync(base_sentences, **kwargs)
        Q = Trivial().instantiate_sync(base_sentences, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q)

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate(base_sentences, **kwargs)
        Q = await Trivial().instantiate(base_sentences, **kwargs)
        P_and_Q = await And().instantiate(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q)

    def violation(self, answers: dict[str, Prob]) -> float:
        return max(
            max(answers["P"] + answers["Q"] - 1, 0) - answers["P_and_Q"],
            answers["P_and_Q"] - min(answers["P"], answers["Q"]),
        )

class OrChecker(Checker):

    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion
        P_or_Q: ForecastingQuestion

        @field_validator("P", "Q", "P_or_Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_or_Q = Or().instantiate_sync(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, Q=Q.P, P_or_Q=P_or_Q.P_or_Q)

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, Q=Q.P, P_or_Q=P_or_Q.P_or_Q)

    def violation(self, answers: dict[str, Prob]) -> float:
        return max(
            max(answers["P"], answers["Q"]) - answers["P_or_Q"],
            answers["P_or_Q"] - min(1, answers["P"] + answers["Q"]),
        )

class AndOrChecker(Checker):
    
    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)
    
    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion
        P_and_Q: ForecastingQuestion
        P_or_Q: ForecastingQuestion
        
        @field_validator("P", "Q", "P_and_Q", "P_or_Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value
    
    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)
        P_or_Q = Or().instantiate_sync(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, P_or_Q=P_or_Q.P_or_Q)
    
    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = await And().instantiate(base_sentences, **kwargs)
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, P_or_Q=P_or_Q.P_or_Q)
    
    def violation(self, answers: dict[str, Prob]) -> float:
        return abs(answers["P"] + answers["Q"] - answers["P_and_Q"] - answers["P_or_Q"])


