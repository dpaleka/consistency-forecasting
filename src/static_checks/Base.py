# Path: static_checks/Base.py
from abc import ABC, abstractmethod

from common.forecasters import Forecaster, SentencesTemplate, ProbsTemplate, elicit


class BaseChecker(ABC):
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance

    @abstractmethod
    def instantiate(self, *base_sentences: str) -> SentencesTemplate:
        pass

    @abstractmethod
    def violation(self, answers: ProbsTemplate) -> float:
        pass

    def check(self, answers: ProbsTemplate) -> bool:
        return self.violation(answers) < self.tolerance

    def instantiate_and_elicit(
        self, forecaster: Forecaster, *base_sentences: str
    ) -> ProbsTemplate:
        return elicit(forecaster, self.instantiate(*base_sentences))

    def elicit_and_violation(
        self, forecaster: Forecaster, sentences: SentencesTemplate
    ) -> float:
        return self.violation(elicit(forecaster, sentences))

    def elicit_and_check(
        self, forecaster: Forecaster, sentences: SentencesTemplate
    ) -> bool:
        return self.check(elicit(forecaster, sentences))

    def instantiate_and_elicit_and_violation(
        self, forecaster: Forecaster, *base_sentences: str
    ) -> float:
        return self.violation(self.instantiate_and_elicit(forecaster, *base_sentences))

    def instantiate_and_elicit_and_check(
        self, forecaster: Forecaster, *base_sentences: str
    ) -> float:
        return self.check(self.instantiate_and_elicit(forecaster, *base_sentences))