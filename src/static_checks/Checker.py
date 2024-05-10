import jsonlines
from dotenv import load_dotenv
import os
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
from common.datatypes import (
    ForecastingQuestion,
    Prob,
    VerificationResult,
)
from common.utils import (
    write_jsonl_async_from_str,
    update_recursive,
    write_jsonl_from_str,
)
from common.path_utils import get_data_path
from common.llm_utils import parallelized_call, answer, answer_sync
from .checker_prompts import (
    neg_verification_prompt,
    and_verification_prompt,
    or_verification_prompt,
    but_verification_prompt,
    conditional_verification_prompt,
    consequence_verification_prompt,
    paraphrase_verification_prompt,
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

load_dotenv()
write_verification = os.getenv("WRITE_VERIFICATION", "False") == "True"
verify_before_instantiation = (
    os.getenv("VERIFY_BEFORE_INSTANTIATION", "False") == "True"
)


async def write_verification_result(tuple_type, generated_tuple, verification):
    print("-----------")
    print("write cerification result")
    print("0-0-0-0-0-0-0-0-0-0-0-0-")
    filename = get_data_path() / "verification/tuple_verifications.jsonl"
    verification_jsonl = generated_tuple.model_dump_json()
    verification_jsonl = (
        verification_jsonl[:-1]
        + f', "valid": {verification.valid}, "reasoning": "{verification.reasoning}"'
        + f', "tuple_type":"{tuple_type}"'
        + "}"
    )
    await write_jsonl_async_from_str(filename, [verification_jsonl], append=True)


def write_verification_result_sync(tuple_type, generated_tuple, verification):
    filename = get_data_path() / "verification/tuple.jsonl"
    verification_jsonl = generated_tuple.model_dump_json()
    verification_jsonl = (
        verification_jsonl[:-1]
        + f', "valid": {verification.valid}, "reasoning": "{verification.reasoning}"'
        + f", tuple_type:{tuple_type}"
        + "}"
    )
    write_jsonl_from_str(filename, [verification_jsonl], append=True)


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

    def instantiate_with_verification_sync(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        n_verification=3,
        **kwargs,
    ) -> "Self.TupleFormat":
        """
        Synchronously instantiate and verify the tuple format with metadata multiple times, returning the instantiated
        object if verification succeeds within the given attempts.
        """
        if verify_before_instantiation:
            for _ in range(n_verification):
                instantiated_object = self.instantiate_sync(base_sentences, **kwargs)
                verification_result = self.verify_sync(instantiated_object, **kwargs)
                if verification_result.valid:
                    return instantiated_object
            return instantiated_object
        else:
            instantiated_object = self.instantiate_sync(
                base_sentences, supplied_metadata, **kwargs
            )
            return instantiated_object

    async def instantiate_with_verification(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        n_verification=3,
        **kwargs,
    ) -> "Self.TupleFormat":
        """
        If the global variable 'verify_before_instantiation' is True, this method
        will instantiate and verify the tuple format with metadata three times,
        returning the instantiated object and a list of verification results from each iteration.
        """
        if verify_before_instantiation:
            print(f"verifyin before instantiation {n_verification} times")
            for _ in range(n_verification):
                instantiated_object = await self.instantiate(base_sentences, **kwargs)
                verification_result = await self.verify(instantiated_object, **kwargs)
                print(f"verification result: {verification_result}")
                print(f"self = {self}")
                if verification_result.valid:
                    return instantiated_object
            return instantiated_object
        else:
            instantiated_object = await self.instantiate(
                base_sentences, supplied_metadata, **kwargs
            )
            return instantiated_object

    def instantiate_sync_with_metadata(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        **kwargs,
    ) -> "Self.TupleFormat_with_metadata":
        """Instantiate with a metadata field that can store the base questions and other things.
        supplied_metadata is used to *recursively* update the metadata so you can surgically
        update nested fields."""
        result = self.instantiate_with_verification_sync(base_sentences, **kwargs)
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
        result = await self.instantiate_with_verification(base_sentences, **kwargs)
        if supplied_metadata is None:
            supplied_metadata = {}
        metadata = {"base_sentences": base_sentences}
        update_recursive(metadata, supplied_metadata)
        return self.TupleFormat_with_metadata(**result.dict(), metadata=metadata)

    @abstractmethod
    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        pass

    @abstractmethod
    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
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

    def verify_sync(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = neg_verification_prompt.format(
            P=generated_tuple.P, not_P=generated_tuple.not_P
        )
        verification = answer_sync(prompt, response_model=VerificationResult,**kwargs)
        if write_verification:
            write_verification_result_sync("negation", generated_tuple, verification)
        return verification

    async def verify(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = neg_verification_prompt.format(
            P=generated_tuple.P, not_P=generated_tuple.not_P
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("negation", generated_tuple, verification)
        return verification

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

    def verify_sync(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = and_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_and_Q=generated_tuple.P_and_Q
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("and", generated_tuple, verification)
        return verification

    async def verify(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = and_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_and_Q=generated_tuple.P_and_Q
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("and", generated_tuple, verification)
        return verification

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

    def verify_sync(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = or_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_or_Q=generated_tuple.P_or_Q
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("or", generated_tuple, verification)
        return verification

    async def verify(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = or_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_or_Q=generated_tuple.P_or_Q
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("or", generated_tuple, verification)
        return verification

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

    def verify_sync(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = or_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_or_Q=generated_tuple.P_or_Q
        )
        or_verification_result = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        prompt = and_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_and_Q=generated_tuple.P_and_Q
        )
        and_verification_result = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        verification = VerificationResult(
            valid=and_verification_result.valid and or_verification_result,
            reasoning="And reasoning:\n"
            + and_verification_result.reasoning
            + "\nOr reasoning:\n"
            + or_verification_result.reasoning,
        )
        if write_verification:
            write_verification_result_sync("AndOr", generated_tuple, verification)
        return verification

    async def verify(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = or_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_or_Q=generated_tuple.P_or_Q
        )
        or_verification_result = await answer(prompt, response_model=VerificationResult, **kwargs)
        prompt = and_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_and_Q=generated_tuple.P_and_Q
        )
        and_verification_result = await answer(prompt, response_model=VerificationResult, **kwargs)
        print(f"type of and_verification_result: {type(and_verification_result)}")
        print(f"verification result for and: {and_verification_result}")
        verification = VerificationResult(
            valid=and_verification_result.valid and or_verification_result.valid,
            reasoning="And reasoning:\n"
            + and_verification_result.reasoning
            + "\nOr reasoning:\n"
            + or_verification_result.reasoning,
        )
        if write_verification:
            await write_verification_result("AndOr", generated_tuple, verification)
        return verification

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

    def verify_sync(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = but_verification_prompt.format(
            P=generated_tuple.P,
            P_and_not_Q=generated_tuple.Q_and_not_P,
            Q=generated_tuple.P_or_Q,
        )
        verification = answer_sync(prompt, response_model=VerificationResult,**kwargs)
        if write_verification:
            write_verification_result_sync("But", generated_tuple, verification)
        return verification

    async def verify(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = but_verification_prompt.format(
            P=generated_tuple.P,
            Q_and_not_P=generated_tuple.Q_and_not_P,
            P_or_Q=generated_tuple.P_or_Q,
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("But", generated_tuple, verification)
        return verification

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

    def verify_sync(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = conditional_verification_prompt.format(
            P=generated_tuple.P,
            Q_given_P=generated_tuple.Q_given_P,
            P_and_Q=generated_tuple.P_and_Q,
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("Conditional", generated_tuple, verification)
        return verification

    async def verify(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = conditional_verification_prompt.format(
            P=generated_tuple.P,
            Q_given_P=generated_tuple.Q_given_P,
            P_and_Q=generated_tuple.P_and_Q,
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result(
                "Conditional", generated_tuple, verification
            )
        return verification

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

    def verify_sync(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = consequence_verification_prompt.format(
            P=generated_tuple.P, cons_P=generated_tuple.cons_P
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("Consequence", generated_tuple, verification)
        return verification

    async def verify(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = consequence_verification_prompt.format(
            P=generated_tuple.P, cons_P=generated_tuple.cons_P
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result(
                "Consequence", generated_tuple, verification
            )
        return verification

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

    def verify_sync(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = paraphrase_verification_prompt.format(
            P=generated_tuple.P, para_P=generated_tuple.para_P
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("Paraphrase", generated_tuple, verification)
        return verification

    async def verify(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = paraphrase_verification_prompt.format(
            P=generated_tuple.P, para_P=generated_tuple.para_P
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("Paraphrase", generated_tuple, verification)
        return verification

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

    def verify_sync(self, generated_tuple: "Self.TupleFormat", **kwargs) -> VerificationResult:
        and_pq_prompt = and_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_and_Q=generated_tuple.P_and_Q
        )
        verification_pq = answer_sync(and_pq_prompt, response_model=VerificationResult,**kwargs)

        and_qp_prompt = and_verification_prompt.format(
            P=generated_tuple.Q, Q=generated_tuple.P, P_and_Q=generated_tuple.Q_and_P
        )
        verification_qp = answer_sync(and_qp_prompt, response_model=VerificationResult,**kwargs)

        valid = verification_pq.valid and verification_qp.valid
        reasoning = f"Symmetry And reasoning:\nP_and_Q reasoning:\n{verification_pq.reasoning}\n" \
                    f"Q_and_P reasoning:\n{verification_qp.reasoning}"

        verification_result = VerificationResult(valid=valid, reasoning=reasoning)
        
        if write_verification:
            write_verification_result_sync("symmetry_and", generated_tuple, verification_result)

        return verification_result

    async def verify(self, generated_tuple: "Self.TupleFormat", **kwargs) -> VerificationResult:
        and_pq_prompt = and_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_and_Q=generated_tuple.P_and_Q
        )
        verification_pq = await answer(and_pq_prompt, response_model=VerificationResult,**kwargs)

        and_qp_prompt = and_verification_prompt.format(
            P=generated_tuple.Q, Q=generated_tuple.P, P_and_Q=generated_tuple.Q_and_P
        )
        verification_qp = await answer(and_qp_prompt, response_model=VerificationResult, **kwargs)

        valid = verification_pq.valid and verification_qp.valid
        reasoning = f"Symmetry And reasoning:\nP_and_Q reasoning:\n{verification_pq.reasoning}\n" \
                    f"Q_and_P reasoning:\n{verification_qp.reasoning}"

        verification_result = VerificationResult(valid=valid, reasoning=reasoning)
        
        if write_verification:
            await write_verification_result("symmetry_and", generated_tuple, verification_result)

        return verification_result


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

    def verify_sync(self, generated_tuple: "Self.TupleFormat", **kwargs) -> VerificationResult:
        or_pq_prompt = or_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_or_Q=generated_tuple.P_or_Q
        )
        verification_pq = answer_sync(or_pq_prompt, response_model=VerificationResult, **kwargs)

        or_qp_prompt = or_verification_prompt.format(
            P=generated_tuple.Q, Q=generated_tuple.P, P_or_Q=generated_tuple.Q_or_P
        )
        verification_qp = answer_sync(or_qp_prompt, response_model=VerificationResult, **kwargs)

        valid = verification_pq.valid and verification_qp.valid
        reasoning = f"Symmetry Or reasoning:\nP_or_Q reasoning:\n{verification_pq.reasoning}\n" \
                    f"Q_or_P reasoning:\n{verification_qp.reasoning}"

        verification_result = VerificationResult(valid=valid, reasoning=reasoning)
        
        if write_verification:
            write_verification_result_sync("symmetry_or", generated_tuple, verification_result)

        return verification_result

    async def verify(self, generated_tuple: "Self.TupleFormat", **kwargs) -> VerificationResult:
        or_pq_prompt = or_verification_prompt.format(
            P=generated_tuple.P, Q=generated_tuple.Q, P_or_Q=generated_tuple.P_or_Q
        )
        verification_pq = await answer(or_pq_prompt, response_model=VerificationResult, **kwargs)

        or_qp_prompt = or_verification_prompt.format(
            P=generated_tuple.Q, Q=generated_tuple.P, P_or_Q=generated_tuple.Q_or_P
        )
        verification_qp = await answer(or_qp_prompt, response_model=VerificationResult, **kwargs)

        valid = verification_pq.valid and verification_qp.valid
        reasoning = f"Symmetry Or reasoning:\nP_or_Q reasoning:\n{verification_pq.reasoning}\n" \
                    f"Q_or_P reasoning:\n{verification_qp.reasoning}"

        verification_result = VerificationResult(valid=valid, reasoning=reasoning)
        
        if write_verification:
            await write_verification_result("symmetry_or", generated_tuple, verification_result)

        return verification_result


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

    def verify_sync(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        # TODO(Alejadnro): Implement this
        return VerificationResult(reasoning = "", valid = True)

    async def verify(self,
        generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        # TODO(Alejadnro): Implement this
        return VerificationResult(reasoning = "", valid = True)

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
