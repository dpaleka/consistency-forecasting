# Path: static_checks/Base.py
import jsonlines
from abc import ABC, abstractmethod
from common.utils import write_jsonl_async
from common.llm_utils import parallelized_call
from forecasters import Forecaster, SentencesTemplate, ProbsTemplate

class BaseChecker(ABC):
    def __init__(self, tolerance=0.1, path = None):
        self.tolerance = tolerance
        if path is None:
            self.path = f"src/data/{self.__class__.__name__}.jsonl"
        else:
            self.path = path

    @abstractmethod
    def instantiate_sync(self, *base_sentences: str, **kwargs) -> SentencesTemplate:
        pass

    @abstractmethod
    async def instantiate(self, *base_sentences: str, **kwargs) -> SentencesTemplate:
        pass

    @abstractmethod
    def violation(self, answers: ProbsTemplate) -> float:
        pass
    
    # stack base sentences into a single prompt
    def stack(*base_sentences : str) -> str:
        return "\n".join(f"q{i}: {base_sentence}" for i, base_sentence in enumerate(base_sentences))
    
    async def instantiate_and_write(self, *base_sentences : list[str], **kwargs):
        result = await self.instantiate(*base_sentences, **kwargs)
        await write_jsonl_async(self.path, [result], append=True)
    
    async def instantiate_and_write_many(self, base_sentencess : list[list[str]], **kwargs):
        _instantiate_and_write = lambda base_sentences: self.instantiate_and_write(*base_sentences, **kwargs)
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
    
    def test(self, forecaster : Forecaster, **kwargs):
        for line in jsonlines.open(self.path):
            print("START")
            print(f"line: {line}")
            answers = forecaster.elicit(line)
            print(answers)
            if not all(answers.values()):
                print("ERROR: Some answers are None!")
                continue
            loss = self.violation(answers)
            res_bool = self.check(answers)
            res = { True : "Passed", False : "Failed"}[res_bool]
            print(f"Violation: {loss}")
            print(f"Check result: {res}")
            print("")