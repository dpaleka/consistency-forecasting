import jsonlines
import json
from dotenv import load_dotenv
import os
import functools
import numpy as np
from datetime import datetime
from numpy.random import random
from scipy.optimize import (
    minimize,
    basinhopping,
    differential_evolution,
    dual_annealing,
    shgo,
    brute,
    root,
)
from pathlib import Path
from itertools import product
from abc import ABC, abstractmethod
from typing import Type, Any, List, Self, Callable
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
    make_json_serializable,
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
    consequence_quantity_verification_prompt,
    consequence_time_verification_prompt,
    paraphrase_verification_prompt,
)
from .MiniInstantiator import (
    Neg,
    Or,
    And,
    Trivial,
    Conditional,
    Paraphrase,
    Consequence,
)
from forecasters import Forecaster

load_dotenv()
write_verification = os.getenv("WRITE_VERIFICATION", "False") == "True"
verify_before_instantiation = (
    os.getenv("VERIFY_BEFORE_INSTANTIATION", "False") == "True"
)
verify_length = os.getenv("VERIFY_LENGTH", "False") == "True"


async def write_verification_result(tuple_type, generated_tuple, verification):
    filename = get_data_path() / "verification/tuple_verifications.jsonl"
    verification_jsonl = generated_tuple.model_dump_json()
    verification_jsonl = (
        verification_jsonl[:-1]
        + f', "valid": "{verification.valid}", "reasoning": "{verification.reasoning}"'
        + f', "tuple_type":"{tuple_type}"'
        + "}"
    )
    await write_jsonl_async_from_str(filename, [verification_jsonl], append=True)


def write_verification_result_sync(tuple_type, generated_tuple, verification):
    filename = get_data_path() / "verification/tuple.jsonl"
    verification_jsonl = generated_tuple.model_dump_json()
    verification_jsonl = (
        verification_jsonl[:-1]
        + f', "valid": "{verification.valid}", "reasoning": "{verification.reasoning}"'
        + f", tuple_type:{tuple_type}"
        + "}"
    )
    write_jsonl_from_str(filename, [verification_jsonl], append=True)


class Checker(ABC):
    def __init__(self, default_tolerance=0.01, frequentist_hparams=None, path=None):
        self.default_tolerance = default_tolerance
        if frequentist_hparams is None:
            frequentist_hparams = {"sigma": 0.05, "gamma": 2.58, "beta": 1e-3}
        self.frequentist_hparams = frequentist_hparams
        self.name = self.__class__.__name__
        if path is None:
            self.path = get_data_path() / "tuples" / f"{self.name}.jsonl"
        else:
            self.path = path
        self.counter = 0  # number of tuples successfully instantiated

    def dump_config(self):
        return {
            "name": str(self.name),
            "default_tolerance": self.default_tolerance,
            "frequentist_hparams": self.frequentist_hparams,
            "path": str(self.path),
        }

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
    ) -> List["Self.TupleFormat"]:
        pass

    @abstractmethod
    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        pass

    def instantiate_with_verification_sync(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        n_verification=3,
        **kwargs,
    ) -> List[tuple["Self.TupleFormat", VerificationResult]]:
        """
        Synchronously instantiate and verify the tuple format with metadata multiple times,
        returning a list of instantiated objects and their verification results if verification
        succeeds within the given attempts.
        """
        verified_objects = []

        if verify_before_instantiation:
            for _ in range(n_verification):
                instantiated_objects = self.instantiate_sync(base_sentences, **kwargs)
                for instantiated_object in instantiated_objects:
                    verification_result = self.verify_sync(
                        instantiated_object, **kwargs
                    )
                    if verify_length:
                        length_check = self.verify_length(
                            instantiated_object, base_sentences, **kwargs
                        )
                    else:
                        length_check = True
                    if verification_result.valid and length_check:
                        self.counter += 1
                        verified_objects.append(
                            (instantiated_object, verification_result)
                        )
                if verified_objects:
                    return verified_objects
            return []
        else:
            instantiated_objects = self.instantiate_sync(
                base_sentences, supplied_metadata, **kwargs
            )
            for instantiated_object in instantiated_objects:
                self.counter += 1
                verified_objects.append((instantiated_object, None))
            return verified_objects

    async def instantiate_with_verification(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        n_verification=3,
        **kwargs,
    ) -> List[tuple["Self.TupleFormat", VerificationResult]]:
        """
        Asynchronously instantiate and verify the tuple format with metadata multiple times,
        returning a list of instantiated objects and their verification results if verification
        succeeds within the given attempts.
        """
        verified_objects = []

        if verify_before_instantiation:
            for _ in range(n_verification):
                instantiated_objects = await self.instantiate(base_sentences, **kwargs)
                for instantiated_object in instantiated_objects:
                    verification_result = await self.verify(
                        instantiated_object, **kwargs
                    )
                    if verify_length:
                        length_check = self.verify_length(
                            instantiated_object, base_sentences, **kwargs
                        )
                    else:
                        length_check = True
                    if verification_result.valid and length_check:
                        self.counter += 1
                        verified_objects.append(
                            (instantiated_object, verification_result)
                        )
                if verified_objects:
                    return verified_objects
            return []
        else:
            instantiated_objects = await self.instantiate(
                base_sentences, supplied_metadata, **kwargs
            )
            for instantiated_object in instantiated_objects:
                self.counter += 1
                verified_objects.append((instantiated_object, None))
            return verified_objects

    def instantiate_sync_with_metadata(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        **kwargs,
    ) -> List["Self.TupleFormat_with_metadata"]:
        """Instantiate with a metadata field that can store the base questions and other things.
        supplied_metadata is used to *recursively* update the metadata so you can surgically
        update nested fields."""
        if supplied_metadata is None:
            supplied_metadata = {}
        metadata = {"base_sentences": base_sentences}
        update_recursive(metadata, supplied_metadata)
        results = self.instantiate_with_verification_sync(base_sentences, **kwargs)

        instantiated_with_metadata = []
        for result in results:
            if verify_before_instantiation:
                instantiated_object, verification_result = result
                more_metadata = {"verification_result": verification_result.dict()}
                update_recursive(metadata, more_metadata)
            else:
                instantiated_object = result
            instantiated_with_metadata.append(
                self.TupleFormat_with_metadata(
                    **instantiated_object.dict(), metadata=metadata
                )
            )

        return instantiated_with_metadata

    async def instantiate_with_metadata(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        **kwargs,
    ) -> List["Self.TupleFormat_with_metadata"]:
        """Instantiate with a metadata field that can store the base questions and other things.
        supplied_metadata is used to *recursively* update the metadata so you can surgically
        update nested fields."""
        if supplied_metadata is None:
            supplied_metadata = {}
        metadata = {"base_sentences": base_sentences}
        update_recursive(metadata, supplied_metadata)
        results = await self.instantiate_with_verification(base_sentences, **kwargs)

        instantiated_with_metadata = []
        for result in results:
            if verify_before_instantiation:
                instantiated_object, verification_result = result
                more_metadata = {"verification_result": verification_result.dict()}
                update_recursive(metadata, more_metadata)
            else:
                instantiated_object = result
            instantiated_with_metadata.append(
                self.TupleFormat_with_metadata(
                    **instantiated_object.dict(), metadata=metadata
                )
            )

        return instantiated_with_metadata

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

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return True

    async def instantiate_and_write(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        **kwargs,
    ):
        results = await self.instantiate_with_metadata(
            base_sentences, supplied_metadata=supplied_metadata, **kwargs
        )
        if results:
            json_list = [result.model_dump_json() for result in results]
            await write_jsonl_async_from_str(self.path, json_list, append=True)

    async def instantiate_and_write_many(
        self,
        base_sentencess: (
            list[dict[str, ForecastingQuestion]]
            | list[tuple[dict[str, ForecastingQuestion], dict[str, Any]]]  # +metadata
        ),
        n_write: int = -1,
        overwrite=False,
        **kwargs,
    ):
        """
        Args:
            base_sentencess: list of base sentences, each of which is a dict of ForecastingQuestions
            n_write: maximum number of tuples to actually make (usually less than len(base_sentencess)
                because some will fail verification). If -1, will make as many as possible.
        """
        if overwrite:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

        def _instantiate_and_write(
            base_sentences: (
                dict[str, ForecastingQuestion]
                | tuple[dict[str, ForecastingQuestion], dict[str, Any]]
            ),
        ):
            if isinstance(base_sentences, tuple):
                base_sentences, supplied_metadata = base_sentences
            else:
                supplied_metadata = None
            return self.instantiate_and_write(
                base_sentences, supplied_metadata=supplied_metadata, **kwargs
            )

        # Added print statement to log the base sentences being processed
        # print(f"Base sentences: {base_sentencess}")
        bq_counter = 0  # number of base sentences processed
        while n_write == -1 or self.counter < n_write:
            counter_prev = self.counter
            results = await parallelized_call(
                _instantiate_and_write,
                base_sentencess[bq_counter : bq_counter + n_write - counter_prev],
                max_concurrent_queries=10,
            )
            bq_counter += n_write - counter_prev
            print(f"Counter: {self.counter}")
            print(f"BQ Counter: {bq_counter}")
        # # Added print statement to log the results of instantiation
        # print(f"Results of instantiation: {results}")

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
        scoring: dict[str, Callable[[Prob], float]] = np.log,
    ) -> float:
        """Arbitrage earned given a particular outcome, forcaster answers and
        arbitrageur_answers.
        """

        scoring = self.get_scoring(answers, scoring)

        score = 0.0
        for qun, ans in answers.items():
            if outcome[qun] is None:
                continue
            elif outcome[qun] == True:  # noqa
                score += scoring[qun](arbitrageur_answers[qun]) - scoring[qun](ans)
            elif outcome[qun] == False:  # noqa
                score += scoring[qun](1 - arbitrageur_answers[qun]) - scoring[qun](
                    1 - ans
                )
        return score

    @property
    def Omega(self):
        x = self.TupleFormat.model_fields
        v = [True, False, None]
        outcomes = product(v, repeat=len(x))

        Omega = []
        for outcome in outcomes:
            outcome_dict = dict(zip(x, outcome))
            if self.check_exact(outcome_dict):
                Omega.append(outcome_dict)

        return Omega

    def min_arbitrage(
        self,
        answers: dict[str, Prob],
        arbitrageur_answers: dict[str, Prob],
        scoring: dict[str, Callable[[Prob], float]] = np.log,
    ) -> float:
        """Minimum arbitrage earned regardless of outcome, given forcaster answers
        and arbitrageur_answers."""
        # x = answers.keys()
        # v = [True, False, None]
        # outcomes = product(v, repeat=len(x))

        # Omega = []
        # for outcome in outcomes:
        #     outcome_dict = dict(zip(x, outcome))
        #     if self.check_exact(outcome_dict):
        #         Omega.append(outcome_dict)

        return np.amin(
            [
                self.arbitrage(
                    outcome=outcom,
                    answers=answers,
                    arbitrageur_answers=arbitrageur_answers,
                    scoring=scoring,
                )
                for outcom in self.Omega
            ]
        )

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        scoring: dict[str, Callable[[Prob], float]] = np.log,
        initial_guess: list[float] | str | None = None,
        methods: tuple[str] = ("shgo", "differential_evolution"),
    ) -> float:
        """Finding the best arbitrageur_answers to maximize the guaranteed minimum
        arbitrage earned for some given forecaster answers.

        Args:
            answers (dict[str, Prob]): Forecaster answers.
            scoring (dict[str, Callable[[Prob], float]], optional): Scoring function. Defaults to np.log.
            initial_guess (list[float] | str | None, optional): Initial guess for the optimization. Defaults to None.
            methods (tuple[str], optional): Optimization method. Options:
                Nelder-Mead, L-BFGS-B, trust-exact -- often unreliable, as they are local optimization
                basinhopping -- slow I think? at least for AndChecker, OrChecker, AndOrChecker
                brute -- some syntax error
                shgo, differential_evolution, dual_annealing -- working. shgo takes negligible time but is unreliable
                    for small violations; differential_evolution takes much longer but is more reliable. dual_annealing
                    takes 2x the time as differential_evolution and doesn't seem to hold any advantage over it.
                root -- instead of maximizing min_arbitrage, it finds the values of arbitrageur_answers at which
                    arbitrage(outcome, answers, arbitrageur_answers) are all equal for all outcomes; then picks the
                    arbitrageur_answers for which this (equal) arbitrage is highest. Mostly broken though.
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

        maxes = []

        for method in methods:
            if method == "root":

                def funs_to_equate(arbitrageur_answers_list):
                    return [
                        self.arbitrage(
                            outcome,
                            answers,
                            dict(zip(x, arbitrageur_answers_list)),
                            scoring,
                        )
                        for outcome in self.Omega
                    ]

                def funs_to_zero(arbitrageur_answers_list):
                    return (
                        np.array(funs_to_equate(arbitrageur_answers_list))
                        - funs_to_equate(arbitrageur_answers_list)[0]
                    )

                solutions = root(
                    funs_to_zero,
                    arbitrageur_answers_list_initial,
                    method="lm",
                )
                if solutions.success:
                    arbitrage_argmax = dict(zip(x, solutions.x))
                    arbitrage_max = self.arbitrage(
                        self.Omega[0], answers, arbitrage_argmax, scoring
                    )
                    maxes.append(
                        {
                            "arbitrage_argmax": arbitrage_argmax,
                            "arbitrage_max": arbitrage_max,
                        }
                    )
                    # arbitrage_argmaxes = [dict(zip(x, sol)) for sol in solutions.x]
                    # results = {arbitr_argmax:
                    #     self.arbitrage(self.Omega[0], answers, arbitr_argmax, scoring)
                    #     for arbitr_argmax in arbitrage_argmaxes
                    # }
                    # arbitrage_argmax = max(results, key=results.get)
                    # arbitrage_max = results[arbitrage_argmax]
                    # return arbitrage_argmax, arbitrage_max
                else:
                    print(f"Root optimization failed: {solutions.message}")
            elif method == "differential_evolution":
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

            maxes.append(
                {"arbitrage_argmax": arbitrage_argmax, "arbitrage_max": arbitrage_max}
            )

        best_max = max(maxes, key=lambda x: x["arbitrage_max"])
        return best_max["arbitrage_argmax"], best_max["arbitrage_max"]

    def arbitrage_violation(self, answers: dict[str, Prob], **kwargs) -> float:
        try:
            return self.max_min_arbitrage(answers, **kwargs)[1]
        except ZeroDivisionError:
            return 123
        except Exception as e:
            print(f"Error in arbitrage_violation: {e}")
            return 148

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        raise NotImplementedError("Subclasses must implement this")

    def violation(
        self, answers: dict[str, Any], force_pos=True, metric="default"
    ) -> float:
        """Can be re-defined in subclass to use an exact calculation."""
        if metric == "default":
            v = self.arbitrage_violation(answers)
            if force_pos:
                v = max(0, v)
        elif metric == "frequentist":
            v = self.frequentist_violation(answers)
        else:
            raise ValueError(f"Metric {metric} not implemented")

        return v

    def check(self, answers: dict[str, Any], metric: str = "default") -> bool:
        if metric == "default":
            return bool(self.violation(answers) < self.default_tolerance)
        elif metric == "frequentist":
            return bool(
                self.frequentist_violation(answers)
                < self.frequentist_hparams["gamma"] * self.frequentist_hparams["sigma"]
            )
        else:
            raise ValueError(f"Metric {metric} not implemented")

    def elicit_and_violation(
        self, forecaster: Forecaster, sentences: "Self.TupleFormat", **kwargs
    ) -> float:
        return self.violation(forecaster.elicit(sentences, **kwargs))

    def elicit_and_check(
        self, forecaster: Forecaster, sentences: "Self.TupleFormat", **kwargs
    ) -> bool:
        return self.check(forecaster.elicit(sentences, **kwargs))

    async def elicit_and_violation_async(
        self, forecaster: Forecaster, sentences: "Self.TupleFormat", **kwargs
    ) -> float:
        return self.violation(await forecaster.elicit_async(sentences, **kwargs))

    async def elicit_and_check_async(
        self, forecaster: Forecaster, sentences: "Self.TupleFormat", **kwargs
    ) -> bool:
        return self.check(await forecaster.elicit_async(sentences, **kwargs))

    def get_line_obj(self, line: dict[str, Any]) -> "Self.TupleFormat":
        metadata = line.pop("metadata", None)
        line_obj = self.TupleFormat.model_validate(line)
        return line_obj

    def check_from_elicited_probs(
        self, answers: dict[str, Prob], metric: str = "default"
    ) -> dict[str, Any]:
        print(f"answers: {answers}\n")
        if any([a is None for a in answers.values()]):
            print("ERROR: Some answers are None!")
            return {"successful_elicitation": False}
        loss: float = self.violation(answers, metric=metric)
        res_bool: bool = self.check(answers, metric=metric)
        res: str = {True: "Passed", False: "Failed"}[res_bool]
        print(f"Violation: {loss}\nCheck result: {res}\n")
        return {
            "metric": metric,
            "violation": loss,
            "check": res_bool,
            "check_result": res,
            "successful_elicitation": True,
        }

    def check_all_from_elicited_probs(
        self, all_answers: list[dict[str, Prob]]
    ) -> list[dict[str, Any]]:
        results = []
        for answers in all_answers:
            result = self.check_from_elicited_probs(answers)
            results.append(result)
        return results

    def test_sync(
        self,
        forecaster: Forecaster,
        line_begin: int = 0,
        line_end: int = -1,
        do_check=True,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Args:
            [line_begin, line_end) : closed-open range of lines to check.
            If line_end = -1, defaults to the end of the file.

            do_check (bool): Whether to compute violation on the elicited probabilities.
        """
        results = []
        log_path = (
            get_data_path()
            / "check_tuple_logs"
            / f"{self.__class__.__name__}_test_log.jsonl"
        )
        if line_end != -1:
            assert (
                line_begin >= 0 and line_begin < line_end
            ), "We want a non-empty range"

        with jsonlines.open(log_path, mode="a") as writer:
            writer.write({"test_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            with open(self.path, "r", encoding="utf-8") as file:
                # data: list[dict[str, Any]] = [json.loads(line) for line in file]
                data = []
                for line in file:
                    print(line)
                    data.append(json.loads(line))
            if line_end >= 0:
                print(f"Limiting to lines {line_begin} to {line_end} of {self.path}")
                data = data[line_begin:line_end]

            for line in data:
                print(f"START\nline: {line}\n")
                line_obj: "Self.TupleFormat" = self.get_line_obj(line)
                answers_: dict[
                    str, tuple[Prob, dict] | Prob | None
                ] = forecaster.elicit(line_obj, include_metadata=True, **kwargs)
                answers = {
                    q: a[0] if isinstance(a, tuple) else a for q, a in answers_.items()
                }
                if do_check:
                    result_without_line: dict[
                        str, Any
                    ] = self.check_from_elicited_probs(line_obj, answers)
                else:
                    result_without_line = {}

                for question, prob in answers.items():
                    line[question]["elicited_prob"] = prob
                    if isinstance(answers_[question], tuple):
                        line[question]["elicitation_metadata"] = make_json_serializable(
                            answers_[question][1]
                        )

                result = {"line": line, **result_without_line}
                results.append(result)
                writer.write(result)

        return results

    async def test(
        self,
        forecaster: Forecaster,
        line_begin: int = 0,
        line_end: int = -1,
        do_check=True,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Args:
            [line_begin, line_end) : closed-open range of lines to check.
            If line_end = -1, defaults to the end of the file.

            do_check (bool): Whether to compute violation on the elicited probabilities.
        """
        results = []
        log_path = (
            get_data_path()
            / "check_tuple_logs"
            / f"{self.__class__.__name__}_test_log.jsonl"
        )
        if line_end != -1:
            assert (
                line_begin >= 0 and line_begin < line_end
            ), "We want a non-empty range"

        with jsonlines.open(log_path, mode="a") as writer:
            writer.write({"test_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            with open(self.path, "r", encoding="utf-8") as file:
                data = [json.loads(line) for line in file]
            if line_end >= 0:
                print(f"Limiting to lines {line_begin} to {line_end} of {self.path}")
                data = data[line_begin:line_end]

            validated_lines: list[BaseModel] = [
                self.get_line_obj(line) for line in data
            ]
            print(validated_lines)
            print("Starting async elicitation")
            elicit_func = functools.partial(
                forecaster.elicit_async, include_metadata=True, **kwargs
            )
            all_answers_ = await parallelized_call(
                elicit_func,
                validated_lines,
                max_concurrent_queries=10,
            )
            all_answers = [
                {q: a[0] if isinstance(a, tuple) else a for q, a in answers_.items()}
                for answers_ in all_answers_
            ]

            if do_check:
                print("Starting checking")
                results_without_line = self.check_all_from_elicited_probs(all_answers)
            else:
                results_without_line = [{} for _ in data]

            for line, answers_, answers, result_without_line in zip(
                data, all_answers_, all_answers, results_without_line
            ):
                for question, prob in answers.items():
                    line[question]["elicited_prob"] = prob
                    if isinstance(answers_[question], tuple):
                        line[question]["elicitation_metadata"] = make_json_serializable(
                            answers_[question][1]
                        )

                result = {"line": line, **result_without_line}

                results.append(result)
                writer.write(result)

        return results

    @classmethod
    def get_scoring(
        cls, answers: dict[str, Prob], scoring: Any, return_just_log_weights=False
    ) -> dict[str, Callable[[Prob], float]] | dict[str, float] | None:
        if isinstance(scoring, list):
            if len(scoring) < len(answers):
                scoring = scoring + [scoring[-1]] * (len(answers) - len(scoring))
            scoring = {q: scoring[i] for i, q in enumerate(answers.keys())}
        if not isinstance(scoring, dict):
            scoring = {q: scoring for q in answers.keys()}
        scoring_weights = {}
        scoring_functions = {}
        for key, scoring_item in scoring.items():
            if isinstance(scoring_item, (float, int)):
                scoring_weights[key] = scoring_item
                scoring_functions[key] = lambda x, sf=scoring_item: sf * np.log(
                    x
                )  # stupid HACK
            elif callable(scoring_item):
                scoring_functions[key] = scoring_item
                scoring_weights = None
            else:
                raise ValueError(f"Scoring function {scoring_item} not recognized")
        if return_just_log_weights:
            return scoring_weights
        return scoring_functions

    @classmethod
    def must_compute_arbitrage_numerically(
        cls, answers: dict[str, Prob], **kwargs
    ) -> bool:
        if kwargs:
            if len(kwargs) > 1 or "scoring" not in kwargs:
                return True  # there are kwargs that aren't just scoring weights
            else:
                scoring = kwargs["scoring"]
                scoring_weights = cls.get_scoring(
                    answers, scoring, return_just_log_weights=True
                )
                if scoring_weights is None:
                    return True
        return False


class NegChecker(Checker):
    num_base_questions = 1

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        not_P: ForecastingQuestion

        @field_validator("P", "not_P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = neg_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            not_P_title=generated_tuple.not_P.title,
            not_P_body=generated_tuple.not_P.body,
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("negation", generated_tuple, verification)
        return verification

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = neg_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            not_P_title=generated_tuple.not_P.title,
            not_P_body=generated_tuple.not_P.body,
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("negation", generated_tuple, verification)
        return verification

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return len(generated_tuple.not_P.body) > 0.8 * len(generated_tuple.P.body)

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync(base_sentences, **kwargs)
        not_P = Neg().instantiate_sync(base_sentences, **kwargs)
        return self.TupleFormat(P=P.P, not_P=not_P.not_P)

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate(base_sentences, **kwargs)
        not_P = await Neg().instantiate(base_sentences, **kwargs)
        return [self.TupleFormat(P=P.P, not_P=not_P.not_P)]

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        **kwargs,
    ) -> float:
        if self.must_compute_arbitrage_numerically(answers, **kwargs):
            return super().max_min_arbitrage(answers, **kwargs)
        weights = self.get_scoring(
            answers, kwargs.get("scoring", [1.0]), return_just_log_weights=True
        )
        W = sum(weights.values())

        answers_ = {"P": answers["P"], "implied_P": 1 - answers["not_P"]}
        weights_ = {"P": weights["P"], "implied_P": weights["not_P"]}

        logodds = (
            sum(
                [
                    weights_[q] * np.log(answers_[q] / (1 - answers_[q]))
                    for q in ["P", "implied_P"]
                ]
            )
            / W
        )
        p = 1 / (1 + np.exp(-logodds))
        v = W * np.log(p) - sum(
            [weights_[q] * np.log(answers_[q]) for q in ["P", "implied_P"]]
        )

        return {"P": p, "not_P": 1 - p}, v

    def frequentist_violation(
        self,
        answers: dict[str, Any],
    ) -> float:
        P, not_P = answers["P"], answers["not_P"]
        denom = (1 - P) * P + (1 - not_P) * not_P
        denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
        v = abs(P + not_P - 1) / denom
        return v

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P"] + answers["not_P"] == 1
        )


class AndChecker(Checker):
    num_base_questions = 2

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion
        P_and_Q: ForecastingQuestion

        @field_validator("P", "Q", "P_and_Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = and_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_and_Q.title,
            R_body=generated_tuple.P_and_Q.body,
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("and", generated_tuple, verification)
        return verification

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = and_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_and_Q.title,
            R_body=generated_tuple.P_and_Q.body,
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("and", generated_tuple, verification)
        return verification

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return len(generated_tuple.P_and_Q.body) > 1.4 * max(
            len(generated_tuple.P.body), len(generated_tuple.Q.body)
        )

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)
        return [self.TupleFormat(P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q)]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = await And().instantiate(base_sentences, **kwargs)
        return [self.TupleFormat(P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q)]

    def frequentist_violation(
        self, answers: dict[str, Any], sigma: float = 0.01, gamma: float = 3
    ) -> float:
        P, Q, P_and_Q = answers["P"], answers["Q"], answers["P_and_Q"]
        if P + Q - 1 <= P_and_Q:
            v_lhs = 0
        else:
            denom = P * (1 - P) + Q * (1 - Q) + P_and_Q * (1 - P_and_Q)
            denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
            v_lhs = (P + Q - 1 - P_and_Q) / denom

        R = min(P, Q)
        if R >= P_and_Q:
            v_rhs = 0
        else:
            denom = P_and_Q * (1 - P_and_Q) + R * (1 - R)
            denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
            v_rhs = (P_and_Q - R) / denom

        return max(v_lhs, v_rhs)

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and max(answers["P"] + answers["Q"] - 1, 0) <= answers["P_and_Q"]
            and answers["P_and_Q"] <= min(answers["P"], answers["Q"])
        )


class OrChecker(Checker):
    num_base_questions = 2

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion
        P_or_Q: ForecastingQuestion

        @field_validator("P", "Q", "P_or_Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = or_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_or_Q.title,
            R_body=generated_tuple.P_or_Q.body,
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("or", generated_tuple, verification)
        return verification

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = or_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_or_Q.title,
            R_body=generated_tuple.P_or_Q.body,
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("or", generated_tuple, verification)
        return verification

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return len(generated_tuple.P_or_Q.body) > 1.4 * max(
            len(generated_tuple.P.body), len(generated_tuple.Q.body)
        )

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_or_Q = Or().instantiate_sync(base_sentences, **kwargs)
        return [self.TupleFormat(P=P.P, Q=Q.P, P_or_Q=P_or_Q.P_or_Q)]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)
        return [self.TupleFormat(P=P.P, Q=Q.P, P_or_Q=P_or_Q.P_or_Q)]

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        # This is essentially the reverse of the AndChecker.frequentist_violation
        P, Q, P_or_Q = answers["P"], answers["Q"], answers["P_or_Q"]
        S = max(P, Q)
        if S <= P_or_Q:
            v_lhs = 0
        else:
            denom = S * (1 - S) + P_or_Q * (1 - P_or_Q)
            denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
            v_lhs = (S - P_or_Q) / denom

        if P + Q >= P_or_Q:
            v_rhs = 0
        else:
            denom = P * (1 - P) + Q * (1 - Q) + P_or_Q * (1 - P_or_Q)
            denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
            v_rhs = (P_or_Q - P - Q) / denom

        return max(v_lhs, v_rhs)

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and max(answers["P"], answers["Q"]) <= answers["P_or_Q"]
            and answers["P_or_Q"] <= min(1, answers["P"] + answers["Q"])
        )


class AndOrChecker(Checker):
    num_base_questions = 2

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

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = or_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_or_Q.title,
            R_body=generated_tuple.P_or_Q.body,
        )
        or_verification_result = answer_sync(
            prompt, response_model=VerificationResult, **kwargs
        )
        prompt = and_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_and_Q.title,
            R_body=generated_tuple.P_and_Q.body,
        )
        and_verification_result = answer_sync(
            prompt, response_model=VerificationResult, **kwargs
        )
        verification = VerificationResult(
            valid=and_verification_result.valid and or_verification_result.valid,
            reasoning="And reasoning:\\n"
            + and_verification_result.reasoning
            + "\\nOr reasoning:\\n"
            + or_verification_result.reasoning,
        )
        if write_verification:
            write_verification_result_sync("AndOr", generated_tuple, verification)
        return verification

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = or_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_or_Q.title,
            R_body=generated_tuple.P_or_Q.body,
        )
        or_verification_result = await answer(
            prompt, response_model=VerificationResult, **kwargs
        )
        prompt = and_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_and_Q.title,
            R_body=generated_tuple.P_and_Q.body,
        )
        and_verification_result = await answer(
            prompt, response_model=VerificationResult, **kwargs
        )
        verification = VerificationResult(
            valid=and_verification_result.valid and or_verification_result.valid,
            reasoning="And reasoning:\\n"
            + and_verification_result.reasoning
            + "\\nOr reasoning:\\n"
            + or_verification_result.reasoning,
        )
        if write_verification:
            await write_verification_result("AndOr", generated_tuple, verification)
        return verification

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return len(generated_tuple.P_or_Q.body) > 1.4 * max(
            len(generated_tuple.P.body), len(generated_tuple.Q.body)
        ) and len(generated_tuple.P_and_Q.body) > 1.4 * max(
            len(generated_tuple.P.body), len(generated_tuple.Q.body)
        )

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)
        P_or_Q = Or().instantiate_sync(base_sentences, **kwargs)
        return [
            self.TupleFormat(
                P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, P_or_Q=P_or_Q.P_or_Q
            )
        ]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = await And().instantiate(base_sentences, **kwargs)
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)
        return [
            self.TupleFormat(
                P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, P_or_Q=P_or_Q.P_or_Q
            )
        ]

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        P, Q, P_or_Q, P_and_Q = (
            answers["P"],
            answers["Q"],
            answers["P_or_Q"],
            answers["P_and_Q"],
        )

        denom = (
            P * (1 - P) + Q * (1 - Q) + P_or_Q * (1 - P_or_Q) + P_and_Q * (1 - P_and_Q)
        )
        denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
        v = abs(P + Q - P_or_Q - P_and_Q) / denom
        return v

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P"] + answers["Q"] == answers["P_and_Q"] + answers["P_or_Q"]
        )


class ButChecker(Checker):
    num_base_questions = 2

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q_and_not_P: ForecastingQuestion
        P_or_Q: ForecastingQuestion

        @field_validator("P", "Q_and_not_P", "P_or_Q")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = but_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            R_title=generated_tuple.Q_and_not_P.title,
            R_body=generated_tuple.Q_and_not_P.body,
            S_title=generated_tuple.P_or_Q.title,
            S_body=generated_tuple.P_or_Q.body,
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("But", generated_tuple, verification)
        return verification

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = but_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            R_title=generated_tuple.Q_and_not_P.title,
            R_body=generated_tuple.Q_and_not_P.body,
            S_title=generated_tuple.P_or_Q.title,
            S_body=generated_tuple.P_or_Q.body,
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("But", generated_tuple, verification)
        return verification

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return len(generated_tuple.Q_and_not_P.body) > 1.4 * max(
            len(base_sentences["P"].body), len(base_sentences["Q"].body)
        ) and len(generated_tuple.P_or_Q.body) > 1.4 * max(
            len(base_sentences["P"].body), len(base_sentences["Q"].body)
        )

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        not_P = Neg().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q_and_not_P = And().instantiate_sync(
            {"P": base_sentences["Q"], "Q": not_P.not_P}
        )
        P_or_Q = Or().instantiate_sync(base_sentences, **kwargs)
        return [
            self.TupleFormat(
                P=P.P, Q_and_not_P=Q_and_not_P.P_and_Q, P_or_Q=P_or_Q.P_or_Q
            )
        ]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        not_P = await Neg().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q_and_not_P = await And().instantiate(
            {"P": base_sentences["Q"], "Q": not_P.not_P}
        )
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)
        return [
            self.TupleFormat(
                P=P.P, Q_and_not_P=Q_and_not_P.P_and_Q, P_or_Q=P_or_Q.P_or_Q
            )
        ]

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        P, Q_and_not_P, P_or_Q = answers["P"], answers["Q_and_not_P"], answers["P_or_Q"]

        denom = P_or_Q * (1 - P_or_Q) + P * (1 - P) + Q_and_not_P * (1 - Q_and_not_P)
        denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
        v = abs(P_or_Q - (P + Q_and_not_P)) / denom
        return v

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P"] + answers["Q_and_not_P"] == answers["P_or_Q"]
        )


class CondChecker(Checker):
    num_base_questions = 2

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

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
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

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
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

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return len(generated_tuple.Q_given_P.body) > 1.4 * max(
            len(base_sentences["P"].body), len(base_sentences["Q"].body)
        ) and len(generated_tuple.P_and_Q.body) > 1.4 * max(
            len(base_sentences["P"].body), len(base_sentences["Q"].body)
        )

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q_given_P = Conditional().instantiate_sync(base_sentences, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)
        return [
            self.TupleFormat(
                P=P.P, Q_given_P=Q_given_P.Q_given_P, P_and_Q=P_and_Q.P_and_Q
            )
        ]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q_given_P = await Conditional().instantiate(base_sentences, **kwargs)
        P_and_Q = await And().instantiate(base_sentences, **kwargs)
        return [
            self.TupleFormat(
                P=P.P, Q_given_P=Q_given_P.Q_given_P, P_and_Q=P_and_Q.P_and_Q
            )
        ]

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        **kwargs,
    ) -> float:
        if kwargs:
            return super().max_min_arbitrage(answers, **kwargs)

        a = np.sqrt(
            (1 - answers["P"] * answers["Q_given_P"])
            / (answers["P"] * (1 - answers["Q_given_P"]))
        )
        b = np.sqrt(
            (1 - answers["Q_given_P"])
            * (1 - answers["P_and_Q"])
            / (answers["Q_given_P"] * answers["P_and_Q"])
        )

        p = (1 + b / a) / (1 + b * a)
        q = 1 / (1 + b / a)
        r = 1 / (1 + b * a)

        v = -2 * np.log(
            np.sqrt(answers["P"] * answers["Q_given_P"] * answers["P_and_Q"])
            + np.sqrt(
                (1 - answers["P"] * answers["Q_given_P"]) * (1 - answers["P_and_Q"])
            )
        )

        return {"P": p, "Q_given_P": q, "P_and_Q": r}, v

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        P, Q_given_P, P_and_Q = answers["P"], answers["Q_given_P"], answers["P_and_Q"]
        denom = P * Q_given_P * (
            Q_given_P * (1 - P) + P * (1 - Q_given_P)
        ) + P_and_Q * (1 - P_and_Q)
        denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
        v = abs(P * Q_given_P - P_and_Q) / denom
        return v

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
    num_base_questions = 1

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        cons_P: ForecastingQuestion

        @field_validator("P", "cons_P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = consequence_verification_prompt.format(
            P=generated_tuple.P, cons_P=generated_tuple.cons_P
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("Consequence", generated_tuple, verification)
        return verification

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        metadata = generated_tuple.cons_P.metadata
        consequence_type = metadata.get("consequence_type", None) if metadata else None
        if consequence_type == "quantity":
            prompt = consequence_quantity_verification_prompt.format(
                P_title=generated_tuple.P.title,
                P_body=generated_tuple.P.body,
                Q_title=generated_tuple.cons_P.title,
                Q_body=generated_tuple.cons_P.body,
            )
        elif consequence_type == "time":
            prompt = consequence_time_verification_prompt.format(
                P_title=generated_tuple.P.title,
                P_body=generated_tuple.P.body,
                Q_title=generated_tuple.cons_P.title,
                Q_body=generated_tuple.cons_P.body,
            )
        else:
            prompt = consequence_verification_prompt.format(
                P_title=generated_tuple.P.title,
                P_body=generated_tuple.P.body,
                Q_title=generated_tuple.cons_P.title,
                Q_body=generated_tuple.cons_P.body,
            )

        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result(
                "Consequence", generated_tuple, verification
            )
        return verification

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return True

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync(base_sentences, **kwargs)
        cons_P_list = Consequence().instantiate_sync(base_sentences, **kwargs)
        return [self.TupleFormat(P=P.P, cons_P=cons_P.cons_P) for cons_P in cons_P_list]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> "Self.TupleFormat":
        P = await Trivial().instantiate(base_sentences, **kwargs)
        cons_P_list = await Consequence().instantiate(base_sentences, **kwargs)
        return [self.TupleFormat(P=P.P, cons_P=cons_P.cons_P) for cons_P in cons_P_list]

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        **kwargs,
    ) -> float:
        if kwargs:
            return super().max_min_arbitrage(answers, **kwargs)
        if answers["P"] <= answers["cons_P"]:
            return answers, 0.0
        else:
            A = np.sqrt(answers["P"] * answers["cons_P"])
            B = np.sqrt((1 - answers["P"]) * (1 - answers["cons_P"]))
            p = A / (A + B)
            v = -2 * np.log(A + B)
            return {"P": p, "cons_P": p}, v

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return max(0.0, answers["P"] - answers["cons_P"])

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        P, cons_P = answers["P"], answers["cons_P"]
        if P <= cons_P:
            v = 0
        else:
            denom = P * (1 - P) + cons_P * (1 - cons_P)
            denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
            v = abs(P - cons_P) / denom
        return v

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P"] <= answers["cons_P"]
        )


class ParaphraseChecker(Checker):
    num_base_questions = 1

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        para_P: ForecastingQuestion

        @field_validator("P", "para_P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = paraphrase_verification_prompt.format(
            P=generated_tuple.P, para_P=generated_tuple.para_P
        )
        verification = answer_sync(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            write_verification_result_sync("Paraphrase", generated_tuple, verification)
        return verification

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        prompt = paraphrase_verification_prompt.format(
            P=generated_tuple.P, para_P=generated_tuple.para_P
        )
        verification = await answer(prompt, response_model=VerificationResult, **kwargs)
        if write_verification:
            await write_verification_result("Paraphrase", generated_tuple, verification)
        return verification

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return len(generated_tuple.para_P.body) > 0.65 * len(generated_tuple.P.body)

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync(base_sentences, **kwargs)
        para_P = Paraphrase().instantiate_sync(base_sentences, **kwargs)
        return [self.TupleFormat(P=P.P, para_P=para_P.para_P)]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate(base_sentences, **kwargs)
        para_P = await Paraphrase().instantiate(base_sentences, **kwargs)
        return [self.TupleFormat(P=P.P, para_P=para_P.para_P)]

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        **kwargs,
    ) -> float:
        if self.must_compute_arbitrage_numerically(answers, **kwargs):
            return super().max_min_arbitrage(answers, **kwargs)
        weights = self.get_scoring(
            answers, kwargs.get("scoring", [1.0]), return_just_log_weights=True
        )
        W = sum(weights.values())
        logodds = (
            sum(
                [
                    weights[q] * np.log(answers[q] / (1 - answers[q]))
                    for q in ["P", "para_P"]
                ]
            )
            / W
        )
        p = 1 / (1 + np.exp(-logodds))
        v = W * np.log(p) - sum(
            [weights[q] * np.log(answers[q]) for q in ["P", "para_P"]]
        )

        return {"P": p, "para_P": p}, v

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        P, para_P = answers["P"], answers["para_P"]
        denom = P * (1 - P) + para_P * (1 - para_P)
        denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
        v = abs(P - para_P) / denom
        return v

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P"] == answers["para_P"]
        )


class CondCondChecker(Checker):
    num_base_questions = 3

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

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        # TODO(Alejadnro): Implement this
        return VerificationResult(reasoning="", valid=True)

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        # TODO(Alejadnro): Implement this
        return VerificationResult(reasoning="", valid=True)

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
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

        return [
            self.TupleFormat(
                P=P,
                Q_given_P=Q_given_P,
                R_given_P_and_Q=R_given_P_and_Q,
                P_and_Q_and_R=P_and_Q_and_R,
            )
        ]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
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

        return [
            self.TupleFormat(
                P=P,
                Q_given_P=Q_given_P,
                R_given_P_and_Q=R_given_P_and_Q,
                P_and_Q_and_R=P_and_Q_and_R,
            )
        ]

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        P, Q_given_P, R_given_P_and_Q, P_and_Q_and_R = (
            answers["P"],
            answers["Q_given_P"],
            answers["R_given_P_and_Q"],
            answers["P_and_Q_and_R"],
        )
        denom = (
            P
            * Q_given_P
            * R_given_P_and_Q
            * (
                P * Q_given_P
                + Q_given_P * R_given_P_and_Q
                + P * Q_given_P * R_given_P_and_Q
            )
            - 3 * (P * Q_given_P * R_given_P_and_Q) ** 2
            + P_and_Q_and_R * (1 - P_and_Q_and_R)
        )
        denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
        v = abs(P * Q_given_P * R_given_P_and_Q - P_and_Q_and_R) / denom
        return v

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


"""
The following checks are deprecated. We should rethink how those integrate with Paraphrase.
"""


class SymmetryAndChecker(Checker):
    num_base_questions = 2

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

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        and_pq_prompt = and_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_and_Q.title,
            R_body=generated_tuple.P_and_Q.body,
        )
        verification_pq = answer_sync(
            and_pq_prompt, response_model=VerificationResult, **kwargs
        )

        and_qp_prompt = and_verification_prompt.format(
            P_title=generated_tuple.Q.title,
            P_body=generated_tuple.Q.body,
            Q_title=generated_tuple.P.title,
            Q_body=generated_tuple.P.body,
            R_title=generated_tuple.Q_and_P.title,
            R_body=generated_tuple.Q_and_P.body,
        )
        verification_qp = answer_sync(
            and_qp_prompt, response_model=VerificationResult, **kwargs
        )

        valid = verification_pq.valid and verification_qp.valid
        reasoning = (
            f"Symmetry And reasoning:\\nP_and_Q reasoning:\\n{verification_pq.reasoning}\\n"
            f"Q_and_P reasoning:\\n{verification_qp.reasoning}"
        )

        verification_result = VerificationResult(valid=valid, reasoning=reasoning)

        if write_verification:
            write_verification_result_sync(
                "symmetry_and", generated_tuple, verification_result
            )

        return verification_result

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        and_pq_prompt = and_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_and_Q.title,
            R_body=generated_tuple.P_and_Q.body,
        )
        verification_pq = await answer(
            and_pq_prompt, response_model=VerificationResult, **kwargs
        )
        and_qp_prompt = and_verification_prompt.format(
            P_title=generated_tuple.Q.title,
            P_body=generated_tuple.Q.body,
            Q_title=generated_tuple.P.title,
            Q_body=generated_tuple.P.body,
            R_title=generated_tuple.Q_and_P.title,
            R_body=generated_tuple.Q_and_P.body,
        )
        verification_qp = await answer(
            and_qp_prompt, response_model=VerificationResult, **kwargs
        )

        valid = verification_pq.valid and verification_qp.valid
        reasoning = (
            f"Symmetry And reasoning:\\nP_and_Q reasoning:\\n{verification_pq.reasoning}\\n"
            f"Q_and_P reasoning:\\n{verification_qp.reasoning}"
        )

        verification_result = VerificationResult(valid=valid, reasoning=reasoning)

        if write_verification:
            await write_verification_result(
                "symmetry_and", generated_tuple, verification_result
            )

        return verification_result

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return len(generated_tuple.P_and_Q.body) > 1.4 * max(
            len(base_sentences["P"].body), len(base_sentences["Q"].body)
        ) and len(generated_tuple.Q_and_P.body) > 1.4 * max(
            len(base_sentences["P"].body), len(base_sentences["Q"].body)
        )

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)
        Q_and_P = And().instantiate_sync(
            {"P": base_sentences["Q"], "Q": base_sentences["P"]}, **kwargs
        )
        return [
            self.TupleFormat(
                P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, Q_and_P=Q_and_P.P_and_Q
            )
        ]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = await And().instantiate(base_sentences, **kwargs)
        Q_and_P = await And().instantiate(
            {"P": base_sentences["Q"], "Q": base_sentences["P"]}, **kwargs
        )
        return [
            self.TupleFormat(
                P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, Q_and_P=Q_and_P.P_and_Q
            )
        ]

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return abs(answers["P_and_Q"] - answers["Q_and_P"])

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        raise NotImplementedError

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P_and_Q"] == answers["Q_and_P"]
        )


class SymmetryOrChecker(Checker):
    num_base_questions = 2

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

    def verify_sync(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        or_pq_prompt = or_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_or_Q.title,
            R_body=generated_tuple.P_or_Q.body,
        )
        verification_pq = answer_sync(
            or_pq_prompt, response_model=VerificationResult, **kwargs
        )
        or_qp_prompt = or_verification_prompt.format(
            P_title=generated_tuple.Q.title,
            P_body=generated_tuple.Q.body,
            Q_title=generated_tuple.P.title,
            Q_body=generated_tuple.P.body,
            R_title=generated_tuple.Q_or_P.title,
            R_body=generated_tuple.Q_or_P.body,
        )
        verification_qp = answer_sync(
            or_qp_prompt, response_model=VerificationResult, **kwargs
        )

        valid = verification_pq.valid and verification_qp.valid
        reasoning = (
            f"Symmetry Or reasoning:\\nP_or_Q reasoning:\\n{verification_pq.reasoning}\\n"
            f"Q_or_P reasoning:\\n{verification_qp.reasoning}"
        )

        verification_result = VerificationResult(valid=valid, reasoning=reasoning)

        if write_verification:
            write_verification_result_sync(
                "symmetry_or", generated_tuple, verification_result
            )

        return verification_result

    async def verify(
        self, generated_tuple: "Self.TupleFormat", **kwargs
    ) -> VerificationResult:
        or_pq_prompt = or_verification_prompt.format(
            P_title=generated_tuple.P.title,
            P_body=generated_tuple.P.body,
            Q_title=generated_tuple.Q.title,
            Q_body=generated_tuple.Q.body,
            R_title=generated_tuple.P_or_Q.title,
            R_body=generated_tuple.P_or_Q.body,
        )
        verification_pq = await answer(
            or_pq_prompt, response_model=VerificationResult, **kwargs
        )
        or_qp_prompt = or_verification_prompt.format(
            P_title=generated_tuple.Q.title,
            P_body=generated_tuple.Q.body,
            Q_title=generated_tuple.P.title,
            Q_body=generated_tuple.P.body,
            R_title=generated_tuple.Q_or_P.title,
            R_body=generated_tuple.Q_or_P.body,
        )
        verification_qp = await answer(
            or_qp_prompt, response_model=VerificationResult, **kwargs
        )

        valid = verification_pq.valid and verification_qp.valid
        reasoning = (
            f"Symmetry Or reasoning:\\nP_or_Q reasoning:\\n{verification_pq.reasoning}\\n"
            f"Q_or_P reasoning:\\n{verification_qp.reasoning}"
        )

        verification_result = VerificationResult(valid=valid, reasoning=reasoning)

        if write_verification:
            await write_verification_result(
                "symmetry_or", generated_tuple, verification_result
            )

        return verification_result

    def verify_length(
        self,
        generated_tuple: "Self.TupleFormat",
        base_sentences: dict[str, ForecastingQuestion],
        **kwargs,
    ) -> bool:
        return len(generated_tuple.P_or_Q.body) > 1.4 * max(
            len(base_sentences["P"].body), len(base_sentences["Q"].body)
        ) and len(generated_tuple.Q_or_P.body) > 1.4 * max(
            len(base_sentences["P"].body), len(base_sentences["Q"].body)
        )

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_or_Q = Or().instantiate_sync(base_sentences, **kwargs)
        Q_or_P = Or().instantiate_sync(
            {"P": base_sentences["Q"], "Q": base_sentences["P"]}, **kwargs
        )
        return [
            self.TupleFormat(P=P.P, Q=Q.P, P_or_Q=P_or_Q.P_or_Q, Q_or_P=Q_or_P.P_or_Q)
        ]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)
        Q_or_P = await Or().instantiate(
            {"P": base_sentences["Q"], "Q": base_sentences["P"]}, **kwargs
        )
        return [
            self.TupleFormat(P=P.P, Q=Q.P, P_or_Q=P_or_Q.P_or_Q, Q_or_P=Q_or_P.P_or_Q)
        ]

    # def violation(self, answers: dict[str, Prob]) -> float:
    #     return abs(answers["P_or_Q"] - answers["Q_or_P"])

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        raise NotImplementedError

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return (
            all([a is not None for a in answers.values()])
            and answers["P_or_Q"] == answers["Q_or_P"]
        )


checker_classes = [
    ("NegChecker", NegChecker),
    ("AndChecker", AndChecker),
    ("OrChecker", OrChecker),
    ("AndOrChecker", AndOrChecker),
    ("ButChecker", ButChecker),
    ("CondChecker", CondChecker),
    ("ConsequenceChecker", ConsequenceChecker),
    ("ParaphraseChecker", ParaphraseChecker),
    ("CondCondChecker", CondCondChecker),
]


def choose_checkers(relevant_checks: list[str], tuple_dir: Path) -> dict[str, Checker]:
    print(f"Relevant checks: {relevant_checks}")
    if relevant_checks[0] == "all":
        relevant_checks = [c[0] for c in checker_classes]

    checkers: dict[str, Checker] = {
        checker_name: cls(path=tuple_dir / f"{checker_name}.jsonl")
        for checker_name, cls in checker_classes
        if checker_name in relevant_checks
    }

    return checkers
