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
        return {k: self.call_full(v, **kwargs) for k, v in sentences.items()}

    async def elicit_async(
        self, sentences: BaseModel | dict[str, ForecastingQuestion], **kwargs
    ) -> dict[str, Any]:
        if isinstance(sentences, BaseModel):
            sentences = shallow_dict(sentences)
        list_kv = sentences.items()
        keys, questions = zip(*list_kv)
        call_func = functools.partial(self.call_async_full, **kwargs)
        results = await parallelized_call(call_func, questions)
        return {k: v for k, v in zip(keys, results)}

    def pre_call(self, sentence: ForecastingQuestion, **kwargs) -> ForecastingQuestion:
        sentence.resolution = None
        sentence.metadata = None
        return sentence

    def call_full(self, sentence: ForecastingQuestion, **kwargs) -> Any:
        sentence = self.pre_call(sentence, **kwargs)
        return self.call(sentence, **kwargs)

    def call_async_full(self, sentence: ForecastingQuestion, **kwargs) -> Any:
        sentence = self.pre_call(sentence, **kwargs)
        return self.call_async(sentence, **kwargs)

    @abstractmethod
    def call(
        self, sentence: ForecastingQuestion, include_metadata=False, **kwargs
    ) -> Any:
        pass

    @abstractmethod
    async def call_async(
        self, sentence: ForecastingQuestion, include_metadata=False, **kwargs
    ) -> Any:
        pass

    @abstractmethod
    def dump_config(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def load_config(cls, config: dict[str, Any]) -> "Forecaster":
        pass
