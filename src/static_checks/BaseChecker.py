# Path: static_checks/Base.py
import jsonlines
from abc import ABC, abstractmethod
from typing import Type, Self, Any
from pydantic import BaseModel
from common.utils import write_jsonl_async_from_str
from common.llm_utils import parallelized_call
from common.datatypes import *
from forecasters import Forecaster


class MiniInstantiator(ABC):

    # @property
    # @abstractmethod
    # def bs_format(self) -> dict[str, str]:
    #     pass  # e.g. {'P' : 'binary', 'Q' : 'numerical', 'R' : 'binary'}

    # class BaseSentenceFormat(BaseModel):
    #     pass

    def __init__():
        pass
    
    @property
    @abstractmethod
    def BaseSentenceFormat(self) -> Type[BaseModel]:
        pass

    @abstractmethod
    def title_body_sync_(
        self, base_sentences: "Self.BaseSentenceFormat", **kwargs
    ) -> ForecastingQuestion_stripped:
        pass

    @abstractmethod
    async def title_body_(
        self, base_sentences: "Self.BaseSentenceFormat", **kwargs
    ) -> ForecastingQuestion_stripped:
        pass

    def title_body_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> ForecastingQuestion_stripped:
        based_sentences = self.BaseSentenceFormat(**base_sentences)
        return self.title_body_sync_(based_sentences, **kwargs)

    async def title_body(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> ForecastingQuestion_stripped:
        based_sentences = self.BaseSentenceFormat(**base_sentences)
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
    def resolution_(
        self, resolutions: dict[str, bool]
    ) -> Optional[bool]:
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
            resolution_date=self.resolution_date(base_sentences, **kwargs),
            question_type=self.question_type(base_sentences, **kwargs),
            data_source=self.data_source(base_sentences, **kwargs),
            resolution=self._resolution(base_sentences, **kwargs),
        )
    
    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> ForecastingQuestion:
        title_body = await self.title_body(base_sentences, **kwargs)
        return ForecastingQuestion(
            title=title_body.title,
            body=title_body.body,
            resolution_date=self.resolution_date(base_sentences, **kwargs),
            question_type=self.question_type(base_sentences, **kwargs),
            data_source=self.data_source(base_sentences, **kwargs),
            resolution=self._resolution(base_sentences, **kwargs),
        )

class Trivial(MiniInstantiator):
        
    class BaseSentenceFormat(BaseModel):
        P: ForecastingQuestion

    def title_body_sync_(
        self, base_sentences: "Self.BaseSentenceFormat", **kwargs
    ) -> ForecastingQuestion_stripped:
        return base_sentences.P.cast_simple()
    
    async def title_body_(
        self, base_sentences: "Self.BaseSentenceFormat", **kwargs
    ) -> ForecastingQuestion_stripped:
        return base_sentences.P.cast_simple()

    def resolution_(
        self, resolutions: dict[str, bool]
    ) -> Optional[bool]:
        return resolutions["P"]

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
            if not all(answers.values()):
                print("ERROR: Some answers are None!")
                continue
            loss = self.violation(answers)
            res_bool = self.check(answers)
            res = {True: "Passed", False: "Failed"}[res_bool]
            print(f"Violation: {loss}")
            print(f"Check result: {res}")
            print("")
    

# class BaseChecker(ABC):

#     @property
#     @abstractmethod
#     def BaseSentenceFormat(self) -> Type[BaseModel]:
#         pass

#     @property
#     @abstractmethod
#     def TupleFormat(self) -> Type[BaseModel]:
#         pass

#     def __init__(self, tolerance=0.1, path=None):
#         self.tolerance = tolerance
#         if path is None:
#             self.path = f"src/data/{self.__class__.__name__}.jsonl"
#         else:
#             self.path = path

#     @abstractmethod
#     def instantiate_sync(
#         self, base_sentences: dict[str, ForecastingQuestion], **kwargs
#     ) -> "Self.TupleFormat":
#         pass

#     @abstractmethod
#     async def instantiate(
#         self, base_sentences: dict[str, ForecastingQuestion], **kwargs
#     ) -> "Self.TupleFormat":
#         pass

#     @abstractmethod
#     def violation(self, answers: dict[str, Prob]) -> float:
#         pass

#     async def instantiate_and_write(
#         self, base_sentences: dict[str, ForecastingQuestion], **kwargs
#     ):
#         result = await self.instantiate(base_sentences, **kwargs)

#         if kwargs.get("verbose", True):
#             print(f"Writing tuple to {self.path}: {result}")
#         await write_jsonl_async_from_str(
#             self.path, [result.model_dump_json()], append=True
#         )

#     async def instantiate_and_write_many(
#         self, base_sentencess: list[dict[str, ForecastingQuestion]], **kwargs
#     ):
#         _instantiate_and_write = lambda base_sentences: self.instantiate_and_write(
#             base_sentences, **kwargs
#         )
#         await parallelized_call(_instantiate_and_write, base_sentencess)

#     def check(self, answers: dict[str, Prob]) -> bool:
#         return self.violation(answers) < self.tolerance

#     def elicit_and_violation(
#         self, forecaster: Forecaster, sentences: "Self.TupleFormat", **kwargs
#     ) -> float:
#         return self.violation(forecaster.elicit(sentences, **kwargs))

#     def elicit_and_check(
#         self, forecaster: Forecaster, sentences: "Self.TupleFormat", **kwargs
#     ) -> bool:
#         return self.check(forecaster.elicit(sentences, **kwargs))

#     def test(self, forecaster: Forecaster, **kwargs):
#         for line in jsonlines.open(self.path):
#             print("START")
#             print(f"line: {line}")
#             line_obj = self.TupleFormat.model_validate(line)
#             answers = forecaster.elicit(line_obj, **kwargs)
#             print(answers)
#             if not all(answers.values()):
#                 print("ERROR: Some answers are None!")
#                 continue
#             loss = self.violation(answers)
#             res_bool = self.check(answers)
#             res = {True: "Passed", False: "Failed"}[res_bool]
#             print(f"Violation: {loss}")
#             print(f"Check result: {res}")
#             print("")
