import jsonlines
import numpy as np
from numpy.random import random
from scipy.optimize import (
    minimize,
    basinhopping,
    differential_evolution,
    dual_annealing,
    shgo,
    brute,
)
from itertools import product
from abc import ABC, abstractmethod
from typing import Type, Any, Self, Callable
from pydantic import BaseModel, field_validator, create_model
from common.datatypes import ForecastingQuestion, Prob, ValidationResult
from common.utils import write_jsonl_async_from_str, update_recursive
from common.path_utils import get_data_path
from common.llm_utils import parallelized_call, answer, answer_sync
from .checker_prompts import (
    neg_validation_prompt,
    and_validation_prompt,
    or_validation_prompt,
    but_validation_prompt,
    conditional_validation_prompt,
    consequence_validation_prompt,
    paraphrase_validation_prompt,
)
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
    def __init__(self, tolerance=0.001, path=None):
        self.tolerance = tolerance
        if path is None:
            self.path = get_data_path() / "tuples" / f"{self.__class__.__name__}.jsonl"
        else:
            self.path = path

    @property
    @abstractmethod
    def TupleFormat(self) -> Type[BaseModel]:
        pass

    @property
    def TupleFormat_with_metadata(self) -> Type[BaseModel]:
        fields = {k: (ForecastingQuestion, ...) for k in self.TupleFormat.model_fields}
        fields = {**fields, "metadata": (dict, ...)}

        def make_validator(field_name: str, field_type: Any):
            @field_validator(field_name)
            def validate(cls, v):
                if v.question_type != field_type:
                    raise ValueError(f"{field_name}.question_type must be {field_type}")
                return v

            return validate

        validators = {
            k: make_validator(k, v) for k, v in fields.items() if k != "metadata"
        }

        model = create_model(
            "TupleFormat_with_metadata", **fields, __validators__=validators
        )

        return model

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

    def instantiate_sync_with_metadata(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        **kwargs,
    ) -> "Self.TupleFormat_with_metadata":
        """Instantiate with a metadata field that can store the base questions and other things.
        supplied_metadata is used to *recursively* update the metadata so you can surgically
        update nested fields."""
        result = self.instantiate_sync(base_sentences, **kwargs)
        if supplied_metadata is None:
            supplied_metadata = {}
        metadata = {"base_sentences": base_sentences}
        update_recursive(metadata, supplied_metadata)
        return self.TupleFormat_with_metadata(**result.dict(), metadata=metadata)

    async def instantiate_with_metadata(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        **kwargs,
    ) -> "Self.TupleFormat_with_metadata":
        """Instantiate with a metadata field that can store the base questions and other things.
        supplied_metadata is used to *recursively* update the metadata so you can surgically
        update nested fields."""
        result = await self.instantiate(base_sentences, **kwargs)
        if supplied_metadata is None:
            supplied_metadata = {}
        metadata = {"base_sentences": base_sentences}
        update_recursive(metadata, supplied_metadata)
        return self.TupleFormat_with_metadata(**result.dict(), metadata=metadata)

    @abstractmethod
    def validate_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        pass

    @abstractmethod
    async def validate(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        pass

    async def instantiate_and_write(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        **kwargs,
    ):
        result = await self.instantiate_with_metadata(
            base_sentences, supplied_metadata=supplied_metadata, **kwargs
        )
        # result = await self.instantiate(base_sentences, **kwargs)
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
        _instantiate_and_write = lambda base_sentences: self.instantiate_and_write(  # noqa
            base_sentences, **kwargs
        )
        # Added print statement to log the base sentences being processed
        print(f"Base sentences: {base_sentencess}")
        results = await parallelized_call(_instantiate_and_write, base_sentencess)
        # Added print statement to log the results of instantiation
        print(f"Results of instantiation: {results}")

    @abstractmethod
    def check_exact(self, answers: dict[str, Any]) -> bool:
        """Suffices to define this for answers: dict[str, bool], because
        it is only used to check if a given tuple of resolutions is a
        possible world."""
        pass

    def arbitrage(
        self,
        outcome: dict[str, bool | None],
        answers: dict[str, Prob],
        arbitrageur_answers: dict[str, Prob],
        scoring: Callable[[Prob], float] = np.log,
    ) -> float:
        """Arbitrage earned given a particular outcome, forcaster answers and
        arbitrageur_answers.
        """
        score = 0.0
        for qun, ans in answers.items():
            if outcome[qun] is None:
                continue
            elif outcome[qun] == True:  # noqa
                score += scoring(arbitrageur_answers[qun]) - scoring(ans)
            elif outcome[qun] == False:  # noqa
                score += scoring(1 - arbitrageur_answers[qun]) - scoring(1 - ans)
        return score

    def min_arbitrage(
        self,
        answers: dict[str, Prob],
        arbitrageur_answers: dict[str, Prob],
        scoring: Callable[[Prob], float] = np.log,
    ) -> float:
        """Minimum arbitrage earned regardless of outcome, given forcaster answers
        and arbitrageur_answers."""
        x = answers.keys()
        v = [True, False, None]
        outcomes = product(v, repeat=len(x))

        Omega = []
        for outcome in outcomes:
            outcome_dict = dict(zip(x, outcome))
            if self.check_exact(outcome_dict):
                Omega.append(outcome_dict)

        return np.amin(
            [
                self.arbitrage(
                    outcome=outcom,
                    answers=answers,
                    arbitrageur_answers=arbitrageur_answers,
                    scoring=scoring,
                )
                for outcom in Omega
            ]
        )

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        scoring: Callable[[Prob], float] = np.log,
        initial_guess: list[float] | str | None = None,
        method="shgo",
    ) -> float:
        """Finding the best arbitrageur_answers to maximize the guaranteed minimum
        arbitrage earned for some given forecaster answers.

        Args:
            answers (dict[str, Prob]): Forecaster answers.
            scoring (Callable[[Prob], float], optional): Scoring function. Defaults to np.log.
            initial_guess (list[float] | str | None, optional): Initial guess for the optimization. Defaults to None.
            method (str, optional): Optimization method. Options:
                Nelder-Mead, L-BFGS-B, trust-exact -- often unreliable, as they are local optimization
                basinhopping -- slow I think? at least for AndChecker, OrChecker, AndOrChecker
                brute -- some syntax error
                differential_evolution, shgo, dual_annealing -- working
            Defaults to "shgo".

        """

        x = answers.keys()

        fun_to_minimize = lambda arbitrageur_answers_list: -self.min_arbitrage(  # noqa
            answers, dict(zip(x, arbitrageur_answers_list)), scoring
        )

        if initial_guess is None:
            arbitrageur_answers_list_initial = [0.5] * len(x)
            # arbitrageur_answers_list_initial = [answers[question] + 0.1*random() for question in x]
        elif initial_guess == "answers":
            arbitrageur_answers_list_initial = [answers[question] for question in x]
        elif initial_guess == "answers_randomize":
            arbitrageur_answers_list_initial = [
                answers[question] + 0.1 * (0.5 - random()) for question in x
            ]
        elif initial_guess == "random":
            arbitrageur_answers_list_initial = [random() for _ in x]
        else:
            arbitrageur_answers_list_initial = initial_guess

        # bounds
        bounds = [(0.001, 0.999)] * len(x)  # avoid log(0)

        if method == "differential_evolution":
            result = differential_evolution(
                fun_to_minimize,
                bounds=bounds,
                disp=False,
            )
        elif method == "brute":
            result = brute(
                fun_to_minimize,
                ranges=bounds,
                disp=False,
            )
        elif method == "shgo":
            result = shgo(
                fun_to_minimize,
                bounds=bounds,
            )
        elif method == "dual_annealing":
            result = dual_annealing(
                fun_to_minimize,
                bounds=bounds,
            )
        elif method == "basinhopping":
            result = basinhopping(
                fun_to_minimize,
                arbitrageur_answers_list_initial,
                minimizer_kwargs={"bounds": bounds},
            )
        else:
            result = minimize(
                fun_to_minimize,
                arbitrageur_answers_list_initial,
                bounds=bounds,
                # options={"disp": True},
                method=method,
                # tol=1e-6,
            )

        arbitrage_argmax = dict(zip(x, result.x))
        arbitrage_max = -result.fun

        return arbitrage_argmax, arbitrage_max

    def violation(self, answers: dict[str, Any]) -> float:
        """Can be re-defined in subclass to use an exact calculation."""
        return self.max_min_arbitrage(answers)[1]

    def check(self, answers: dict[str, Any]) -> bool:
        return bool(self.violation(answers) < self.tolerance)

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
        log_path = (
            get_data_path()
            / "check_tuple_logs"
            / f"{self.__class__.__name__}_test_log.jsonl"
        )
        with jsonlines.open(log_path, mode="a") as writer:
            for line in jsonlines.open(self.path):
                print("START")
                print(f"line: {line}")
                metadata = line.pop(
                    "metadata", None
                )  # remove metadata before validation
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
                # get the probability into line for each question
                for question, prob in answers.items():
                    line[question]["elicited_prob"] = prob
                result = {
                    "line": line,
                    "violation": loss,
                    "check": res_bool,
                    "check_result": res,
                }
                results.append(result)
                writer.write(result)
        return results


class NegChecker(Checker):
    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        not_P: ForecastingQuestion

        @field_validator("P", "not_P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = neg_validation_prompt.format(
            P=generated_tuple.P, not_P=generated_tuple.not_P
        )
        return answer_sync(prompt, response_model=ValidationResult)

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = neg_validation_prompt.format(
            P=generated_tuple.P, not_P=generated_tuple.not_P
        )
        return await answer(prompt, response_model=ValidationResult)

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
    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion
        P_and_Q: ForecastingQuestion

        @field_validator("P", "Q", "P_and_Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = and_validation_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_and_Q=generated_tuple.P_and_Q
        )
        return answer_sync(prompt, response_model=ValidationResult)

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = and_validation_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_and_Q=generated_tuple.P_and_Q
        )
        return await answer(prompt, response_model=ValidationResult)

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q)

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
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
    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion
        P_or_Q: ForecastingQuestion

        @field_validator("P", "Q", "P_or_Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = or_validation_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_or_Q=generated_tuple.P_or_Q
        )
        return answer_sync(prompt, response_model=ValidationResult)

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = or_validation_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_or_Q=generated_tuple.P_or_Q
        )
        return await answer(prompt, response_model=ValidationResult)

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

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = or_validation_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_or_Q=generated_tuple.P_or_Q
        )
        or_validation_result = answer_sync(prompt, response_model=ValidationResult)
        prompt = and_validation_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_and_Q=generated_tuple.P_and_Q
        )
        and_validation_result = answer_sync(prompt, response_model=ValidationResult)
        return ValidationResult(
            valid=and_validation_result.valid and or_validation_result,
            reasoning="And reasoning:\n"
            + and_validation_result.reasoning
            + "\nOr reasoning:\n"
            + or_validation_result.reasoning,
        )

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = or_validation_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_or_Q=generated_tuple.P_or_Q
        )
        or_validation_result = answer(prompt, response_model=ValidationResult)
        prompt = and_validation_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_and_Q=generated_tuple.P_and_Q
        )
        and_validation_result = answer(prompt, response_model=ValidationResult)
        return ValidationResult(
            valid=and_validation_result.valid and or_validation_result,
            reasoning="And reasoning:\n"
            + and_validation_result.reasoning
            + "\nOr reasoning:\n"
            + or_validation_result.reasoning,
        )

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
    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q_and_not_P: ForecastingQuestion
        P_or_Q: ForecastingQuestion

        @field_validator("P", "Q_and_not_P", "P_or_Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = but_validation_prompt.format(
            P=generated_tuple.P,
            P_and_not_Q=generated_tuple.Q_and_not_P,
            Q=generated_tuple.P_or_Q,
        )
        return answer_sync(prompt, response_model=ValidationResult)

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = but_validation_prompt.format(
            P=generated_tuple.P,
            P_and_not_Q=generated_tuple.Q_and_not_P,
            Q=generated_tuple.P_or_Q,
        )
        return await answer(prompt, response_model=ValidationResult)

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

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = conditional_validation_prompt.format(
            P=generated_tuple.P,
            Q_given_P=generated_tuple.Q_given_P,
            P_and_Q=generated_tuple.P_and_Q,
        )
        return answer_sync(prompt, response_model=ValidationResult)

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = conditional_validation_prompt.format(
            P=generated_tuple.P,
            Q_given_P=generated_tuple.Q_given_P,
            P_and_Q=generated_tuple.P_and_Q,
        )
        return await answer(prompt, response_model=ValidationResult)

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
        return answers in [
            {"P": True, "Q_given_P": True, "P_and_Q": True},
            {"P": True, "Q_given_P": False, "P_and_Q": False},
            {"P": False, "Q_given_P": None, "P_and_Q": False},
        ]
        # return (
        #     all([a is not None for a in answers.values()])
        #     and answers["P"] * answers["Q_given_P"] == answers["P_and_Q"]
        # ) or (
        #     answers["P"] == False
        #     and answers["Q_given_P"] is None
        #     and answers["P_and_Q"] == False
        # )


class ConsequenceChecker(Checker):
    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        cons_P: ForecastingQuestion

        @field_validator("P", "cons_P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = consequence_validation_prompt.format(
            P=generated_tuple.P, cons_P=generated_tuple.cons_P
        )
        return answer_sync(prompt, response_model=ValidationResult)

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = consequence_validation_prompt.format(
            P=generated_tuple.P, cons_P=generated_tuple.cons_P
        )
        return await answer(prompt, response_model=ValidationResult)

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

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return max(0.0, answers["P"] - answers["cons_P"])

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P"] <= answers["cons_P"]
        )


class ParaphraseChecker(Checker):
    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        para_P: ForecastingQuestion

        @field_validator("P", "para_P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = paraphrase_validation_prompt.format(
            P=generated_tuple.P, para_P=generated_tuple.para_P
        )
        return answer_sync(prompt, response_model=ValidationResult)

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        prompt = paraphrase_validation_prompt.format(
            P=generated_tuple.P, para_P=generated_tuple.para_P
        )
        return await answer(prompt, response_model=ValidationResult)

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

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return abs(answers["P"] - answers["para_P"])

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P"] == answers["para_P"]
        )


class SymmetryAndChecker(Checker):
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

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        # TODO(Alejadnro): Implement this
        pass

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        # TODO(Alejadnro): Implement this
        pass

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

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return abs(answers["P_and_Q"] - answers["Q_and_P"])

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P_and_Q"] == answers["Q_and_P"]
        )


class SymmetryOrChecker(Checker):
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

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        # TODO(Alejadnro): Implement this
        pass

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        # TODO(Alejadnro): Implement this
        pass

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

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return abs(answers["P_or_Q"] - answers["Q_or_P"])

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P_or_Q"] == answers["Q_or_P"]
        )


class CondCondChecker(Checker):
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

    def validate_sync(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        # TODO(Alejadnro): Implement this
        pass

    async def validate(
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> ValidationResult:
        # TODO(Alejadnro): Implement this
        pass

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

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return abs(
    #         answers["P"] * answers["Q_given_P"] * answers["R_given_P_and_Q"]
    #         - answers["P_and_Q_and_R"]
    #     )

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return answers in [
            {
                "P": True,
                "Q_given_P": True,
                "R_given_P_and_Q": True,
                "P_and_Q_and_R": True,
            },
            {
                "P": True,
                "Q_given_P": True,
                "R_given_P_and_Q": False,
                "P_and_Q_and_R": False,
            },
            {
                "P": True,
                "Q_given_P": False,
                "R_given_P_and_Q": None,
                "P_and_Q_and_R": False,
            },
            {
                "P": False,
                "Q_given_P": None,
                "R_given_P_and_Q": None,
                "P_and_Q_and_R": False,
            },
        ]
