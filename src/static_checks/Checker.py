import jsonlines
import numpy as np
from scipy.optimize import minimize
from itertools import product
from abc import ABC, abstractmethod
from typing import Type, Any, Self, Callable
from pydantic import BaseModel, field_validator
from common.datatypes import ForecastingQuestion, Prob
from common.utils import write_jsonl_async_from_str
from common.llm_utils import parallelized_call
from forecasters import Forecaster
from .MiniInstantiator import (
    Neg,
    Or,
    And,
    Trivial,
    Conditional,
    Paraphrase,
    Consequence,
)


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
        if result:
            await write_jsonl_async_from_str(
                self.path, [result.model_dump_json()], append=True
            )

    async def instantiate_and_write_many(
        self,
        base_sentencess: list[dict[str, ForecastingQuestion]],
        overwrite=False,
        **kwargs,
    ):
        if overwrite:
            with open(self.path, "w") as f:
                f.write("")
        _instantiate_and_write = lambda base_sentences: self.instantiate_and_write(
            base_sentences, **kwargs
        )
        await parallelized_call(_instantiate_and_write, base_sentencess)

    @abstractmethod
    def check_exact(self, answers: dict[str, Any]) -> bool:
        pass

    def arbitrage(
        self,
        outcome: dict[str, bool | None],
        answers: dict[str, Prob],
        arbitrageur_answers: dict[str, Prob],
        scoring: Callable[[Prob], float] = np.log,
    ) -> float:
        score = 0.0
        for question, answer in answers.items():
            if outcome[question] is None:
                continue
            elif outcome[question] == True:
                score += scoring(arbitrageur_answers[question]) - scoring(answer)
            elif outcome[question] == False:
                score += scoring(1 - arbitrageur_answers[question]) - scoring(
                    1 - answer
                )
        return score

    def max_min_arbitrage(
        self, answers: dict[str, Prob], scoring: Callable[[Prob], float] = np.log
    ) -> float:
        x = answers.keys()
        v = [True, False, None]
        outcomes = product(v, repeat=len(x))

        Omega = []
        for outcome in outcomes:
            outcome_dict = dict(zip(x, outcome))
            if self.check_exact(outcome_dict):
                Omega.append(outcome_dict)

        # actually this is -min_arbitrage because scipy minimize
        min_arbitrage = lambda arbitrageur_answers_list: -np.amin(
            [
                self.arbitrage(
                    outcome=outcom,
                    answers=answers,
                    arbitrageur_answers=dict(zip(x, arbitrageur_answers_list)),
                    scoring=scoring,
                )
                for outcom in Omega
            ]
        )

        # initial guess
        arbitrageur_answers_list_initial = [0.5] * len(x)

        # bounds
        bounds = [(0.001, 0.999)] * len(x)  # avoid log(0)

        result = minimize(
            min_arbitrage,
            arbitrageur_answers_list_initial,
            bounds=bounds,
            # options={"disp": True},
            # tol=1e-6,
        )

        arbitrage_argmax = dict(zip(x, result.x))
        arbitrage_max = -result.fun

        return arbitrage_argmax, arbitrage_max

    def violation_numerical(
        self, answers: dict[str, Prob], scoring: Callable[[Prob], float] = np.log
    ) -> float:
        return self.max_min_arbitrage(answers, scoring)[1]

    def violation(self, answers: dict[str, Any]) -> float:
        """Can be re-defined in subclass to use an exact calculation."""
        return self.violation_numerical(answers)

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

    def test(self, forecaster: Forecaster, **kwargs) -> list[dict[str, Any]]:
        results = []
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
            results.append(
                {
                    "line": line,
                    "violation": loss,
                    "check": res_bool,
                    "check_result": res,
                }
            )
        return results


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

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return abs(answers["P"] + answers["not_P"] - 1)

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P"] + answers["not_P"] == 1
        )


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

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return max(
    #         max(answers["P"] + answers["Q"] - 1, 0) - answers["P_and_Q"],
    #         answers["P_and_Q"] - min(answers["P"], answers["Q"]),
    #     )

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and max(answers["P"] + answers["Q"] - 1, 0) <= answers["P_and_Q"]
            and answers["P_and_Q"] <= min(answers["P"], answers["Q"])
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

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return max(
    #         max(answers["P"], answers["Q"]) - answers["P_or_Q"],
    #         answers["P_or_Q"] - min(1, answers["P"] + answers["Q"]),
    #     )

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and max(answers["P"], answers["Q"]) <= answers["P_or_Q"]
            and answers["P_or_Q"] <= min(1, answers["P"] + answers["Q"])
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
        return self.TupleFormat(
            P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, P_or_Q=P_or_Q.P_or_Q
        )

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = await And().instantiate(base_sentences, **kwargs)
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)
        return self.TupleFormat(
            P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, P_or_Q=P_or_Q.P_or_Q
        )

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return abs(answers["P"] + answers["Q"] - answers["P_and_Q"] - answers["P_or_Q"])

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P"] + answers["Q"] == answers["P_and_Q"] + answers["P_or_Q"]
        )


class ButChecker(Checker):

    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q_and_not_P: ForecastingQuestion
        P_or_Q: ForecastingQuestion

        @field_validator("P", "Q_and_not_P", "P_or_Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        not_P = Neg().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q_and_not_P = And().instantiate_sync(
            {"P": base_sentences["Q"], "Q": not_P.not_P}
        )
        P_or_Q = Or().instantiate_sync(base_sentences, **kwargs)
        return self.TupleFormat(
            P=P.P, Q_and_not_P=Q_and_not_P.P_and_Q, P_or_Q=P_or_Q.P_or_Q
        )

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        not_P = await Neg().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q_and_not_P = await And().instantiate(
            {"P": base_sentences["Q"], "Q": not_P.not_P}
        )
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)
        return self.TupleFormat(
            P=P.P, Q_and_not_P=Q_and_not_P.P_and_Q, P_or_Q=P_or_Q.P_or_Q
        )

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return abs(answers["P"] + answers["Q_and_not_P"] - answers["P_or_Q"])

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P"] + answers["Q_and_not_P"] == answers["P_or_Q"]
        )


class CondChecker(Checker):

    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q_given_P: ForecastingQuestion
        P_and_Q: ForecastingQuestion

        @field_validator("P", "P_and_Q")
        def check_question_type(cls, value):  # noqa
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

        @field_validator("Q_given_P")
        def check_question_type(cls, value):  # noqa
            if value.question_type != "conditional_binary":
                raise ValueError("Question type must be conditional binary")
            return value

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q_given_P = Conditional().instantiate_sync(base_sentences, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)
        return self.TupleFormat(
            P=P.P, Q_given_P=Q_given_P.Q_given_P, P_and_Q=P_and_Q.P_and_Q
        )

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q_given_P = await Conditional().instantiate(base_sentences, **kwargs)
        P_and_Q = await And().instantiate(base_sentences, **kwargs)
        return self.TupleFormat(
            P=P.P, Q_given_P=Q_given_P.Q_given_P, P_and_Q=P_and_Q.P_and_Q
        )

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return abs(answers["P"] * answers["Q_given_P"] - answers["P_and_Q"])

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            answers["P"] is not None
            and answers["P_and_Q"] is not None
            and (
                answers["Q_given_P"] is None
                or answers["P"] * answers["Q_given_P"] == answers["P_and_Q"]
            )
        )


class ConsequenceChecker(Checker):

    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        cons_P: ForecastingQuestion

        @field_validator("P", "cons_P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = Trivial().instantiate_sync(base_sentences, **kwargs)
        cons_P = Consequence().instantiate_sync(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, cons_P=cons_P.cons_P)

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate(base_sentences, **kwargs)
        cons_P = await Consequence().instantiate(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, cons_P=cons_P.cons_P)

    def violation(self, answers: dict[str, Prob]) -> float:
        return max(0.0, answers["P"] - answers["cons_P"])


class ParaphraseChecker(Checker):

    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        para_P: ForecastingQuestion

        @field_validator("P", "para_P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = Trivial().instantiate_sync(base_sentences, **kwargs)
        para_P = Paraphrase().instantiate_sync(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, para_P=para_P.para_P)

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate(base_sentences, **kwargs)
        para_P = await Paraphrase().instantiate(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, para_P=para_P.para_P)

    def violation(self, answers: dict[str, Prob]) -> float:
        return abs(answers["P"] - answers["para_P"])


class SymmetryAndChecker(Checker):

    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion
        P_and_Q: ForecastingQuestion
        Q_and_P: ForecastingQuestion

        @field_validator("P", "Q", "P_and_Q", "Q_and_P")
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
        Q_and_P = And().instantiate_sync(
            {"P": base_sentences["Q"], "Q": base_sentences["P"]}, **kwargs
        )
        return self.TupleFormat(
            P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, Q_and_P=Q_and_P.P_and_Q
        )

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = await And().instantiate(base_sentences, **kwargs)
        Q_and_P = await And().instantiate(
            {"P": base_sentences["Q"], "Q": base_sentences["P"]}, **kwargs
        )
        return self.TupleFormat(
            P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, Q_and_P=Q_and_P.P_and_Q
        )

    def violation(self, answers: dict[str, Prob]) -> float:
        return abs(answers["P_and_Q"] - answers["Q_and_P"])


class SymmetryOrChecker(Checker):

    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion
        P_or_Q: ForecastingQuestion
        Q_or_P: ForecastingQuestion

        @field_validator("P", "Q", "P_or_Q", "Q_or_P")
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
        Q_or_P = Or().instantiate_sync(
            {"P": base_sentences["Q"], "Q": base_sentences["P"]}, **kwargs
        )
        return self.TupleFormat(
            P=P.P, Q=Q.P, P_or_Q=P_or_Q.P_or_Q, Q_or_P=Q_or_P.P_or_Q
        )

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)
        Q_or_P = await Or().instantiate(
            {"P": base_sentences["Q"], "Q": base_sentences["P"]}, **kwargs
        )
        return self.TupleFormat(
            P=P.P, Q=Q.P, P_or_Q=P_or_Q.P_or_Q, Q_or_P=Q_or_P.P_or_Q
        )

    def violation(self, answers: dict[str, Prob]) -> float:
        return abs(answers["P_or_Q"] - answers["Q_or_P"])


class CondCondChecker(Checker):

    def __init__(self, tolerance=0.1, path=None):
        super().__init__(tolerance, path)

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q_given_P: ForecastingQuestion
        R_given_P_and_Q: ForecastingQuestion
        P_and_Q_and_R: ForecastingQuestion

        @field_validator("P", "P_and_Q_and_R")
        def check_question_type(cls, value):  # noqa
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

        @field_validator("Q_given_P", "R_given_P_and_Q")
        def check_question_type(cls, value):  # noqa
            if value.question_type != "conditional_binary":
                raise ValueError("Question type must be conditional binary")
            return value

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        base_sentences_PQ = {"P": base_sentences["P"], "Q": base_sentences["Q"]}

        P_obj = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        P = P_obj.P

        Q_given_P_obj = Conditional().instantiate_sync(base_sentences_PQ, **kwargs)
        Q_given_P = Q_given_P_obj.Q_given_P

        P_and_Q_obj = And().instantiate_sync(base_sentences_PQ, **kwargs)
        P_and_Q = P_and_Q_obj.P_and_Q

        R_given_P_and_Q_obj = Conditional().instantiate_sync(
            {"P": P_and_Q, "Q": base_sentences["R"]}, **kwargs
        )
        R_given_P_and_Q = R_given_P_and_Q_obj.Q_given_P

        P_and_Q_and_R_obj = And().instantiate_sync(
            {"P": P_and_Q, "Q": base_sentences["R"]}, **kwargs
        )
        P_and_Q_and_R = P_and_Q_and_R_obj.P_and_Q

        return self.TupleFormat(
            P=P,
            Q_given_P=Q_given_P,
            R_given_P_and_Q=R_given_P_and_Q,
            P_and_Q_and_R=P_and_Q_and_R,
        )

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        base_sentences_PQ = {"P": base_sentences["P"], "Q": base_sentences["Q"]}

        P_obj = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        P = P_obj.P

        Q_given_P_obj = await Conditional().instantiate(base_sentences_PQ, **kwargs)
        Q_given_P = Q_given_P_obj.Q_given_P

        P_and_Q_obj = await And().instantiate(base_sentences_PQ, **kwargs)
        P_and_Q = P_and_Q_obj.P_and_Q

        R_given_P_and_Q_obj = await Conditional().instantiate(
            {"P": P_and_Q, "Q": base_sentences["R"]}, **kwargs
        )
        R_given_P_and_Q = R_given_P_and_Q_obj.Q_given_P

        P_and_Q_and_R_obj = await And().instantiate(
            {"P": P_and_Q, "Q": base_sentences["R"]}, **kwargs
        )
        P_and_Q_and_R = P_and_Q_and_R_obj.P_and_Q

        return self.TupleFormat(
            P=P,
            Q_given_P=Q_given_P,
            R_given_P_and_Q=R_given_P_and_Q,
            P_and_Q_and_R=P_and_Q_and_R,
        )

    def violation(self, answers: dict[str, Prob]) -> float:
        return abs(
            answers["P"] * answers["Q_given_P"] * answers["R_given_P_and_Q"]
            - answers["P_and_Q_and_R"]
        )
