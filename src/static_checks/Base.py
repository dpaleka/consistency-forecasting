# Path: static_checks/Base.py
from abc import ABC, abstractmethod
from common.utils import write_jsonl_async
from common.llm_utils import parallelized_call
from forecasters import Forecaster, SentencesTemplate, ProbsTemplate

class BaseChecker(ABC):
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance

    @abstractmethod
    def instantiate(self, *base_sentences: str, **kwargs) -> SentencesTemplate:
        pass

    @abstractmethod
    async def instantiate_async(self, *base_sentences: str, **kwargs) -> SentencesTemplate:
        pass

    @abstractmethod
    def violation(self, answers: ProbsTemplate) -> float:
        pass
    
    # stack base sentences into a single prompt
    def stack(*base_sentences : str) -> str:
        return "\n".join(f"q{i}: {base_sentence}" for i, base_sentence in enumerate(base_sentences))
    
    async def instantiate_and_write(self, *base_sentences : list[str], path_out : str | None = None, **kwargs):
        if path_out is None:
            path_out = f"src/data/{self.__class__.__name__}.jsonl"
        result = await self.instantiate_async(*base_sentences, **kwargs)
        await write_jsonl_async(path_out, [result], append=True)
    
    async def instantiate_and_write_many(self, base_sentencess : list[list[str]], path_out : str | None = None, **kwargs):
        if path_out is None:
            path_out = f"src/data/{self.__class__.__name__}.jsonl"
        _instantiate_and_write = lambda base_sentences: self.instantiate_and_write(*base_sentences, path_out = path_out, **kwargs)
        await parallelized_call(_instantiate_and_write, base_sentencess)
    
    def check(self, answers: ProbsTemplate) -> bool:
        return self.violation(answers) < self.tolerance

    def elicit_and_violation(
        self, forecaster: Forecaster, sentences: SentencesTemplate
    ) -> float:
        return self.violation(forecaster.elicit(sentences))

    def elicit_and_check(
        self, forecaster: Forecaster, sentences: SentencesTemplate
    ) -> bool:
        return self.check(forecaster.elicit(sentences))