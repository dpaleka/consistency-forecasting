from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any
import functools
from common.datatypes import ForecastingQuestion
from common.utils import shallow_dict
from common.llm_utils import parallelized_call


class Forecaster(ABC):
    def elicit(
        self, sentences: BaseModel | dict[str, ForecastingQuestion], **kwargs
    ) -> dict[str, Any]:
        if isinstance(sentences, BaseModel):
            sentences = shallow_dict(sentences)
        return {k: self.call(v, **kwargs) for k, v in sentences.items()}

    async def elicit_async(
        self, sentences: BaseModel | dict[str, ForecastingQuestion], **kwargs
    ) -> dict[str, Any]:
        if isinstance(sentences, BaseModel):
            sentences = shallow_dict(sentences)
        list_kv = sentences.items()
        keys, questions = zip(*list_kv)
        call_func = functools.partial(self.call_async, **kwargs)
        results = await parallelized_call(call_func, questions)
        return {k: v for k, v in zip(keys, results)}

    @abstractmethod
    def call(self, sentence: ForecastingQuestion, include_metadata=False, **kwargs) -> Any:
        pass

    @abstractmethod
    async def call_async(self, sentence: ForecastingQuestion, include_metadata=False, **kwargs) -> Any:
        pass

    @abstractmethod
    def dump_config(self) -> dict[str, Any]:
        pass
