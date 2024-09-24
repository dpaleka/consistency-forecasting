import jsonlines
import json
import warnings
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
from scipy.integrate import solve_ivp
from pathlib import Path
from itertools import product
from abc import ABC, abstractmethod
from typing import Type, Any, List, Self, Callable
from pydantic import BaseModel, field_validator, create_model
from common.datatypes import ForecastingQuestion, Prob, Forecast
from common.utils import (
    write_jsonl_async_from_str,
    update_recursive,
    make_json_serializable,
    delist,
)
from common.path_utils import get_data_path
from common.llm_utils import parallelized_call
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

    def dump_config(self):
        return {
            "name": str(self.name),
            "default_tolerance": self.default_tolerance,
            "frequentist_hparams": self.frequentist_hparams,
            "path": str(self.path),
        }

    @classmethod
    def load_config(cls, config):
        subcls = globals()[config["name"]]
        return subcls(
            default_tolerance=config["default_tolerance"],
            frequentist_hparams=config["frequentist_hparams"],
            path=config["path"],
        )

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
        results = self.instantiate_sync(base_sentences, **kwargs)

        instantiated_with_metadata = []
        for result in results:
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
        if supplied_metadata is None:
            supplied_metadata = {}
        metadata = {"base_sentences": base_sentences}
        update_recursive(metadata, supplied_metadata)
        results = await self.instantiate(base_sentences, **kwargs)

        instantiated_with_metadata = []
        for result in results:
            instantiated_object = result[0] if isinstance(result, tuple) else result

            # Ensure source_question and source_id are included in metadata for each question
            for field, value in instantiated_object.__dict__.items():
                if isinstance(value, ForecastingQuestion):
                    value.metadata = value.metadata or {}
                    question_metadata = supplied_metadata.get(field, {})
                    source_question = question_metadata.get("source_question")
                    source_id = question_metadata.get("source_id")
                    if source_question:
                        value.metadata["source_question"] = source_question
                    if source_id:
                        value.metadata["source_id"] = source_id

            instantiated_with_metadata.append(
                self.TupleFormat_with_metadata(
                    **instantiated_object.dict(), metadata=metadata
                )
            )

        return instantiated_with_metadata

    async def instantiate_and_write(
        self,
        base_sentences: dict[str, ForecastingQuestion],
        supplied_metadata=None,
        **kwargs,
    ):
        if supplied_metadata is None:
            supplied_metadata = {}
        results = await self.instantiate_with_metadata(
            base_sentences, supplied_metadata=supplied_metadata, **kwargs
        )
        if results and not kwargs.get("simulate", False):
            json_list = [result.model_dump_json() for result in results]
            await write_jsonl_async_from_str(self.path, json_list, append=True)
        return results  # necessary to return for instantiate_and_write_many

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
        if overwrite and not kwargs.get("simulate", False):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

        def _instantiate_and_write(base_sentences):
            if isinstance(base_sentences, dict):
                # Old structure
                return self.instantiate_and_write(base_sentences, **kwargs)
            elif isinstance(base_sentences, tuple):
                # Structure with metadata
                questions, metadata = base_sentences
                return self.instantiate_and_write(
                    questions, supplied_metadata=metadata, **kwargs
                )
            else:
                raise ValueError("Unrecognized input format for base_sentences")

        processed_so_far = 0
        written_so_far = 0
        results = []

        while (n_write == -1 or written_so_far < n_write) and processed_so_far < len(
            base_sentencess
        ):
            batch = base_sentencess[
                processed_so_far : processed_so_far + (n_write - written_so_far)
                if n_write != -1
                else None
            ]

            batch_results = await parallelized_call(
                _instantiate_and_write, batch, max_concurrent_queries=10
            )
            processed_so_far += len(batch)
            written_so_far += sum(len(r) for r in batch_results)
            results.extend(batch_results)
        print(
            f"{len(base_sentencess)} base sentences given\nprocessed {processed_so_far}\nwrote {written_so_far}"
        )

        return results

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
        scoring: dict[str, Callable[[Prob], float]] = 1,
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
        scoring: dict[str, Callable[[Prob], float]] = 1,
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

    def de_method(
        self,
        answers: dict[str, Prob],
        scoring: dict[str, Callable[[Prob], float]] = 1,
        euler=False,
        dt=5e-5,
        max_steps=1000,
        tmax=5,
    ) -> tuple[dict[str, Prob], float]:
        """
        Use differential equations to find the best arbitrageur_answers.

        Denote answers = p_i for question i, we will have a differential
        equation for this.
        Initialize p_i with `answers`.

        Then we have a system of differential equations, given by the matrix

        A = np.array(
            [
                [
                    scoring_derivative(p_i) if omega[i] == True
                    else -scoring_derivative(1 - p_i) if omega[i] == False
                    else 0
                ]
                for omega in self.Omega
            ]
        )

        We solve for p_i' = np.linalg.solve(A, [1, 1, ..., 1]) and update p_i and loop until
        det(A) < 0.01.

        """

        p = answers.copy()

        scoring_derivatives = self.get_scoring(
            answers, scoring, return_derivatives=True
        )

        def MATRIX(p):
            if not isinstance(p, dict):
                p = dict(zip(self.TupleFormat.model_fields, p))
            return np.array(
                [
                    [
                        (
                            scoring_derivatives[q](p[q])
                            if a == True  # noqa
                            else (
                                -scoring_derivatives[q](1 - p[q])
                                if a == False  # noqa
                                else 0
                            )
                        )
                        for q, a in omega.items()
                    ]
                    for omega in self.Omega
                ]
            )

        def DET(p):
            if not isinstance(p, dict):
                p = dict(zip(self.TupleFormat.model_fields, p))
            return np.linalg.det(MATRIX(p))

        B = np.array([1] * len(self.Omega))

        def calc_derivs(p):
            if not isinstance(p, dict):
                p = dict(zip(self.TupleFormat.model_fields, p))
            dp_dt_ = np.linalg.solve(MATRIX(p), B)
            dp_dt = dict(zip(self.TupleFormat.model_fields, dp_dt_))
            return dp_dt

        if euler:
            for _ in range(max_steps):
                dp_dt = calc_derivs(p)
                d = DET(p)
                if abs(d) < 0.01:
                    break
                for q in self.TupleFormat.model_fields:
                    p[q] += dp_dt[q] * dt * abs(d)
        else:
            fun = lambda t, p: list(  # noqa
                [
                    calc_derivs(p)[k] for k in self.TupleFormat.model_fields
                ]  # DO NOT use values()
            )
            event = lambda t, p: DET(p)  # noqa
            event.terminal = True

            res = solve_ivp(
                fun,
                [0, tmax],
                list(
                    [answers[k] for k in self.TupleFormat.model_fields]
                ),  # DO NOT use values()
                events=[event],
            )

            p = dict(zip(self.TupleFormat.model_fields, res.y[:, -1]))

        return p, self.min_arbitrage(
            answers=answers, arbitrageur_answers=p, scoring=scoring
        )

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        scoring: dict[str, Callable[[Prob], float]] = 1,
        initial_guess: list[float] | str | None = None,
        euler=False,
        dt: float = 5e-5,
        max_steps: int = 1000,
        tmax=5,
        methods: tuple[str] = ("de",),
    ) -> tuple:
        """Finding the best arbitrageur_answers to maximize the guaranteed minimum
        arbitrage earned for some given forecaster answers.

        Args:
            answers (dict[str, Prob]): Forecaster answers.
            scoring (dict[str, Callable[[Prob], float]], optional): Scoring function. Defaults to np.log.
            initial_guess (list[float] | str | None, optional): Initial guess for the optimization. Defaults to None.
            dt (float, optional): Step size for DE method. Defaults to 5e-5.
            max_steps (int, optional): Maximum steps for DE method. Defaults to 1000.
            tmax (int | float, optional): Maximum time for DE method. Defaults to 5.
            euler (bool, optional): Use Euler method for DE method. Defaults to False.
            methods (tuple[str], optional): Optimization method. Options:
                de -- differential equation method, see Checker.de_method
                Nelder-Mead, L-BFGS-B, trust-exact -- often unreliable, as they are local optimization
                basinhopping -- slow I think? at least for AndChecker, OrChecker, AndOrChecker
                brute -- some syntax error
                shgo, differential_evolution, dual_annealing -- working. shgo takes negligible time but is unreliable
                    for small violations; differential_evolution takes much longer but is more reliable. dual_annealing
                    takes 2x the time as differential_evolution and doesn't seem to hold any advantage over it.
                root -- instead of maximizing min_arbitrage, it finds the values of arbitrageur_answers at which
                    arbitrage(outcome, answers, arbitrageur_answers) are all equal for all outcomes; then picks the
                    arbitrageur_answers for which this (equal) arbitrage is highest. Mostly broken though.
            Defaults to ("de,").

        """
        if "de" in methods:
            return self.de_method(
                answers=answers,
                scoring=scoring,
                dt=dt,
                max_steps=max_steps,
                tmax=tmax,
                euler=euler,
            )

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
        for k, v in answers.items():
            if isinstance(v, Forecast):
                answers[k] = v.prob
        try:
            return self.max_min_arbitrage(answers, **kwargs)[1]
        except ZeroDivisionError as e:
            warnings.warn(f"ZeroDivisionError in arbitrage_violation on {answers}: {e}")
            return str(e)
        except Exception as e:
            warnings.warn(f"Error in arbitrage_violation on {answers}: {e}")
            return str(e)

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        raise NotImplementedError("Subclasses must implement this")

    def violation(
        self,
        answers: dict[str, Any],
        force_pos=True,
        remove_zeros=1e-3,
        metric="default",
        **kwargs,
    ) -> float:
        """Can be re-defined in subclass to use an exact calculation."""
        for k, v in answers.items():
            if isinstance(v, Forecast):
                answers[k] = v.prob
        if metric in ["default", "default_scaled"]:
            if remove_zeros:
                # remove_zeros is an epsilon value to avoid division by zero
                answers = {k: v or remove_zeros for k, v in answers.items()}
            v = self.arbitrage_violation(answers, **kwargs)
            if force_pos and not isinstance(v, str):
                v = max(0, v)  # this also forces np.nan to 0
            if metric == "default_scaled" and not isinstance(v, str):
                v = v / len(answers)
        elif metric == "frequentist":
            v = self.frequentist_violation(answers, **kwargs)
        else:
            raise ValueError(f"Metric {metric} not implemented")

        return v

    def check(
        self,
        answers: dict[str, Any],
        metric: str = "default",
    ) -> bool:
        for k, v in answers.items():
            if isinstance(v, Forecast):
                answers[k] = v.prob
        if metric in ["default", "default_scaled"]:
            viol = self.violation(answers, metric=metric)
            if isinstance(viol, str):
                warnings.warn(f"Error in check: {viol}")
                return None
            return bool(viol < self.default_tolerance)
        elif metric == "frequentist":
            return bool(
                self.frequentist_violation(answers)
                < self.frequentist_hparams["gamma"] * self.frequentist_hparams["sigma"]
            )
        else:
            raise ValueError(f"Metric {metric} not implemented")

    def get_line_obj(self, line: dict[str, Any]) -> "Self.TupleFormat":
        metadata = line.pop("metadata", None)
        line_obj = self.TupleFormat.model_validate(line)
        return line_obj

    def check_from_elicited_probs(
        self,
        answers: dict[str, Prob],
        metric: str | list[str] = "default",
    ) -> dict[str, Any]:
        if isinstance(metric, list):
            return {
                m: self.check_from_elicited_probs(answers, metric=m) for m in metric
            }
        print(f"answers: {answers}\n")
        if any([a is None for a in answers.values()]):
            print("ERROR: Some answers are None!")
            return {"successful_elicitation": False}
        try:
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
        except Exception as e:
            print(
                f"Error in check_from_elicited_probs for "
                f"{answers} with metric {metric}: {e}"
            )
            return {"successful_elicitation": False}

    def check_all_from_elicited_probs(
        self,
        all_answers: list[dict[str, Prob]],
        metric: str | list[str] = "default",
    ) -> list[dict[str, Any]]:
        results = []
        for answers in all_answers:
            result = self.check_from_elicited_probs(answers, metric=metric)
            results.append(result)
        return results

    def test_sync(
        self,
        forecaster: Forecaster,
        tuples: list[dict[str, Any]] | None = None,
        do_check=True,
        **kwargs,
    ) -> list[dict[str, Any]]:
        results = []
        log_path = (
            get_data_path()
            / "check_tuple_logs"
            / f"{self.__class__.__name__}_test_log.jsonl"
        )

        with jsonlines.open(log_path, mode="a") as writer:
            writer.write({"test_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            if tuples is None:
                with open(self.path, "r", encoding="utf-8") as file:
                    data: list[dict[str, Any]] = [json.loads(line) for line in file]
            else:
                data = tuples

            for line in data:
                print(f"START\nline: {line}\n")
                line_obj: "Self.TupleFormat" = self.get_line_obj(line)

                answers_: dict[str, Forecast] = forecaster.elicit(line_obj, **kwargs)

                answers = {q: a.prob for q, a in answers_.items()}
                if do_check:
                    violation_data: dict = self.check_from_elicited_probs(
                        answers,
                        metric=["default", "default_scaled", "frequentist"],
                    )
                else:
                    violation_data = {}

                result = {}
                for question in answers_.keys():
                    result[question] = {
                        "question": line[question],
                        "forecast": {
                            "prob": answers[question],
                            "metadata": make_json_serializable(
                                answers_.get(question).metadata
                            ),
                        },
                    }

                result = {"line": result, "violation_data": violation_data}

                results.append(result)
                writer.write(result)

        return results

    async def test(
        self,
        forecaster: Forecaster,
        tuples: list[dict[str, Any]] | None = None,
        do_check=True,
        **kwargs,
    ) -> list[dict[str, Any]]:
        results = []
        log_path = (
            get_data_path()
            / "check_tuple_logs"
            / f"{self.__class__.__name__}_test_log.jsonl"
        )

        with jsonlines.open(log_path, mode="a") as writer:
            writer.write({"test_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            if tuples is None:
                with open(self.path, "r", encoding="utf-8") as file:
                    data = [json.loads(line) for line in file]
            else:
                data = tuples

            validated_lines: list[BaseModel] = [
                self.get_line_obj(line) for line in data
            ]
            print(validated_lines)
            # TODO: what's going on here? what happens if some lines are not validated?

            print("Starting async elicitation")
            elicit_func = functools.partial(forecaster.elicit_async, **kwargs)
            all_answers_ = await parallelized_call(
                elicit_func,
                validated_lines,
                max_concurrent_queries=10,
            )
            all_answers = [
                {q: a.prob for q, a in answers_.items()} for answers_ in all_answers_
            ]

            if do_check:
                print("Starting checking")
                violations_data = self.check_all_from_elicited_probs(
                    all_answers,
                    metric=["default", "default_scaled", "frequentist"],
                )
            else:
                violations_data = [{} for _ in data]

            for line, answers_, answers, violation_data in zip(
                data, all_answers_, all_answers, violations_data
            ):
                line_obj: "Self.TupleFormat" = self.get_line_obj(line)
                result = {}
                for question in answers_.keys():
                    result[question] = {
                        "question": line[question],
                        "forecast": {
                            "prob": answers[question],
                            "metadata": make_json_serializable(
                                answers_.get(question).metadata
                            ),
                        },
                    }

                result = {"line": result, "violation_data": violation_data}

                results.append(result)
                writer.write(result)

        return results

    @classmethod
    def get_scoring(
        cls,
        answers: dict[str, Prob],
        scoring: Any,
        return_just_log_weights=False,
        return_derivatives=False,
    ) -> dict[str, Callable[[Prob], float]] | dict[str, float] | None:
        # handle None
        if scoring is None:
            scoring = 1.0

        # handle lists
        if isinstance(scoring, list):
            # fill missing values with last element in scoring function list
            if len(scoring) < len(answers):
                scoring = scoring + [scoring[-1]] * (len(answers) - len(scoring))

            # cast to dict
            scoring = {q: scoring[i] for i, q in enumerate(answers.keys())}

        # handle single items
        if not isinstance(scoring, dict):
            scoring = {q: scoring for q in answers.keys()}

        # so far scoring could either be dict[str, Callable[[Prob], float]] or dict[str, float]
        # i.e. either scoring_functions or scoring_weights. Now we calculate both.
        scoring_weights = {}
        scoring_functions = {}
        scoring_derivatives = {}
        for key, scoring_item in scoring.items():
            # if it's a number, it's a weight.
            # we take the scoring function to be weight * log(x)
            if isinstance(scoring_item, (float, int)):
                scoring_weights[key] = scoring_item
                scoring_functions[key] = lambda x, sf=scoring_item: sf * np.log(
                    x
                )  # stupid HACK bc lambda can't take a variable from the outer scope
                scoring_derivatives[key] = lambda x, sf=scoring_item: sf / x

            # if it's a callable, it's a scoring function
            # cannot easily check if it's logarithmic, so we return None
            elif callable(scoring_item):
                scoring_functions[key] = scoring_item
                scoring_weights = None
                scoring_derivatives[key] = None
            else:
                raise ValueError(f"Scoring function {scoring_item} not recognized")

        if return_just_log_weights:
            return scoring_weights
        if return_derivatives:
            return scoring_derivatives
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

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync(base_sentences, **kwargs)
        if not P:
            return []
        not_P = Neg().instantiate_sync(base_sentences, **kwargs)
        if not not_P:
            return []
        P, not_P = delist(P), delist(not_P)
        return self.TupleFormat(P=P.P, not_P=not_P.not_P)

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate(base_sentences, **kwargs)
        if not P:
            return []
        not_P = await Neg().instantiate(base_sentences, **kwargs)
        if not not_P:
            return []
        P, not_P = delist(P), delist(not_P)
        return [self.TupleFormat(P=P.P, not_P=not_P.not_P)]

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        **kwargs,
    ) -> float:
        """Subclassing this one to use the exact formula."""
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

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)
        # Either the verification failed and the list is empty, or there is more than one element which is not expected.
        if (
            isinstance(P_and_Q, list)
            or isinstance(P, list)
            or isinstance(P, list)
            or isinstance(Q, list)
        ):
            return []
        return [self.TupleFormat(P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q)]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = await And().instantiate(base_sentences, **kwargs)
        # Either the verification failed and the list is empty, or there is more than one element which is not expected.
        if (
            isinstance(P_and_Q, list)
            or isinstance(P, list)
            or isinstance(P, list)
            or isinstance(Q, list)
        ):
            return []
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

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        scoring: dict[str, Callable[[Prob], float]] = 1,
        initial_guess: List[float] | str | None = None,
        euler=False,
        dt: float = 0.00005,
        max_steps: int = 1000,
        tmax=5,
        methods: tuple[str] = ("shgo",),
    ) -> tuple:
        """We're subclassing this because DE method doesn't work for this one
        (matrix not square; len(Omega) != len(self.TupleFormat.model_fields))."""
        return super().max_min_arbitrage(
            answers, scoring, initial_guess, euler, dt, max_steps, tmax, methods
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

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_or_Q = Or().instantiate_sync(base_sentences, **kwargs)
        # Either the verification failed and the list is empty, or there is more than one element which is not expected.
        if (
            isinstance(P_or_Q, list)
            or isinstance(P, list)
            or isinstance(P, list)
            or isinstance(Q, list)
        ):
            return []
        return [self.TupleFormat(P=P.P, Q=Q.P, P_or_Q=P_or_Q.P_or_Q)]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)
        # Either the verification failed and the list is empty, or there is more than one element which is not expected.
        if (
            isinstance(P_or_Q, list)
            or isinstance(P, list)
            or isinstance(P, list)
            or isinstance(Q, list)
        ):
            return []
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

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        scoring: dict[str, Callable[[Prob], float]] = 1,
        initial_guess: List[float] | str | None = None,
        euler=False,
        dt: float = 0.00005,
        max_steps: int = 1000,
        tmax=5,
        methods: tuple[str] = ("shgo",),
    ) -> tuple:
        """We're subclassing this because DE method doesn't work for this one
        (matrix not square; len(Omega) != len(self.TupleFormat.model_fields))."""
        return super().max_min_arbitrage(
            answers, scoring, initial_guess, euler, dt, max_steps, tmax, methods
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

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)
        P_or_Q = Or().instantiate_sync(base_sentences, **kwargs)
        # Either the verification failed and the list is empty, or there is more than one element which is not expected.
        if (
            isinstance(P_and_Q, list)
            or isinstance(P_or_Q, list)
            or isinstance(P, list)
            or isinstance(P, list)
            or isinstance(Q, list)
        ):
            return []
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
        # Either the verification failed and the list is empty, or there is more than one element which is not expected.
        if (
            isinstance(P_and_Q, list)
            or isinstance(P_or_Q, list)
            or isinstance(P, list)
            or isinstance(P, list)
            or isinstance(Q, list)
        ):
            return []
        return [
            self.TupleFormat(
                P=P.P, Q=Q.P, P_and_Q=P_and_Q.P_and_Q, P_or_Q=P_or_Q.P_or_Q
            )
        ]

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        scoring: dict[str, Callable[[Prob], float]] = 1,
        initial_guess: List[float] | str | None = None,
        euler=False,
        dt: float = 0.00005,
        max_steps: int = 1000,
        tmax=5,
        methods: tuple[str] = ("shgo",),
    ) -> tuple:
        """We're subclassing this because DE method doesn't work for this one
        (matrix not square; len(Omega) != len(self.TupleFormat.model_fields))."""
        return super().max_min_arbitrage(
            answers, scoring, initial_guess, euler, dt, max_steps, tmax, methods
        )

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

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        not_P = Neg().instantiate_sync({"P": base_sentences["P"]}, **kwargs)

        if isinstance(P, list) or isinstance(not_P, list):
            return []

        Q_and_not_P = And().instantiate_sync(
            {"P": base_sentences["Q"], "Q": not_P.not_P}, **kwargs
        )
        P_or_Q = Or().instantiate_sync(base_sentences, **kwargs)

        # Either the verification failed and the list is empty, or there is more than one element which is not expected.
        if isinstance(Q_and_not_P, list) or isinstance(P_or_Q, list):
            return []
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

        if isinstance(P, list) or isinstance(not_P, list):
            return []

        Q_and_not_P = await And().instantiate(
            {"P": base_sentences["Q"], "Q": not_P.not_P}, **kwargs
        )
        P_or_Q = await Or().instantiate(base_sentences, **kwargs)

        # Either the verification failed and the list is empty, or there is more than one element which is not expected.
        if isinstance(Q_and_not_P, list) or isinstance(P_or_Q, list):
            return []
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

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q_given_P = Conditional().instantiate_sync(base_sentences, **kwargs)
        P_and_Q = And().instantiate_sync(base_sentences, **kwargs)

        # Either the verification failed and the list is empty, or there is more than one element which is not expected.
        if (
            isinstance(P_and_Q, list)
            or isinstance(Q_given_P, list)
            or isinstance(P, list)
        ):
            return []
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

        # Either the verification failed and the list is empty, or there is more than one element which is not expected.
        if (
            isinstance(P_and_Q, list)
            or isinstance(Q_given_P, list)
            or isinstance(P, list)
        ):
            return []
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


class ExpectedEvidenceChecker(Checker):
    num_base_questions = 2

    class TupleFormat(BaseModel):
        P: ForecastingQuestion
        Q: ForecastingQuestion
        P_given_Q: ForecastingQuestion
        P_given_not_Q: ForecastingQuestion

        @field_validator("P", "Q")
        def check_question_type_binary(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value

        @field_validator("P_given_Q", "P_given_not_Q")
        def check_question_type_condbinary(cls, value):
            if value.question_type != "conditional_binary":
                raise ValueError("Question type must be conditional binary")
            return value

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q = Trivial().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        P_given_Q = Conditional().instantiate_sync(
            {"P": base_sentences["Q"], "Q": base_sentences["P"]}, **kwargs
        )
        not_Q = Neg().instantiate_sync({"P": base_sentences["Q"]}, **kwargs)
        if (
            isinstance(P_given_Q, list)
            or isinstance(not_Q, list)
            or isinstance(P, list)
            or isinstance(Q, list)
        ):
            return []
        P_given_not_Q = Conditional().instantiate_sync(
            {"P": not_Q.not_P, "Q": base_sentences["P"]}, **kwargs
        )
        if isinstance(P_given_not_Q, list):
            return []
        return [
            self.TupleFormat(
                P=P.P,
                Q=Q.P,
                P_given_Q=P_given_Q.Q_given_P,
                P_given_not_Q=P_given_not_Q.Q_given_P,
            )
        ]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate({"P": base_sentences["P"]}, **kwargs)
        Q = await Trivial().instantiate({"P": base_sentences["Q"]}, **kwargs)
        P_given_Q = await Conditional().instantiate(
            {"P": base_sentences["Q"], "Q": base_sentences["P"]}, **kwargs
        )
        not_Q = await Neg().instantiate({"P": base_sentences["Q"]}, **kwargs)
        if (
            isinstance(P_given_Q, list)
            or isinstance(not_Q, list)
            or isinstance(P, list)
            or isinstance(Q, list)
        ):
            return []
        P_given_not_Q = await Conditional().instantiate(
            {"P": not_Q.not_P, "Q": base_sentences["P"]}, **kwargs
        )
        if isinstance(P_given_not_Q, list):
            return []
        return [
            self.TupleFormat(
                P=P.P,
                Q=Q.P,
                P_given_Q=P_given_Q.Q_given_P,
                P_given_not_Q=P_given_not_Q.Q_given_P,
            )
        ]

    def check_exact(self, answers: dict[str, Prob]) -> bool:
        return answers in [
            {"P": True, "Q": True, "P_given_Q": True, "P_given_not_Q": None},
            {"P": True, "Q": False, "P_given_Q": None, "P_given_not_Q": True},
            {"P": False, "Q": True, "P_given_Q": False, "P_given_not_Q": None},
            {"P": False, "Q": False, "P_given_Q": None, "P_given_not_Q": False},
        ]

    def frequentist_violation(self, answers: dict[str, Any]) -> float:
        a, b, c, d = (
            answers["P"],
            answers["P_given_Q"],
            answers["P_given_not_Q"],
            answers["Q"],
        )
        denom = (
            a * (1 - a)
            + d**2 * b * (1 - b)
            + (1 - d) ** 2 * c * (1 - c)
            + (b - c) ** 2 * d * (1 - d)
        )
        denom = (denom + self.frequentist_hparams["beta"]) ** 0.5
        num = abs(a - b * d - c * (1 - d))
        v = num / denom
        return v


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
        """Subclassing this one to use the exact formula."""
        if self.must_compute_arbitrage_numerically(answers, **kwargs):
            kwargs["methods"] = ("shgo",)  # DE does not work for this one
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

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = Trivial().instantiate_sync(base_sentences, **kwargs)
        para_P = Paraphrase().instantiate_sync(base_sentences, **kwargs)
        if isinstance(para_P, list):
            return []
        return [self.TupleFormat(P=P.P, para_P=para_P.para_P)]

    async def instantiate(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        P = await Trivial().instantiate(base_sentences, **kwargs)
        para_P = await Paraphrase().instantiate(base_sentences, **kwargs)
        if isinstance(para_P, list):
            return []
        return [self.TupleFormat(P=P.P, para_P=para_P.para_P)]

    def max_min_arbitrage(
        self,
        answers: dict[str, Prob],
        **kwargs,
    ) -> float:
        """Subclassing this one to use the exact formula."""
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

    def instantiate_sync(
        self, base_sentences: dict[str, ForecastingQuestion], **kwargs
    ) -> List["Self.TupleFormat"]:
        base_sentences_PQ = {"P": base_sentences["P"], "Q": base_sentences["Q"]}

        P_obj = Trivial().instantiate_sync({"P": base_sentences["P"]}, **kwargs)
        Q_given_P_obj = Conditional().instantiate_sync(base_sentences_PQ, **kwargs)
        P_and_Q_obj = And().instantiate_sync(base_sentences_PQ, **kwargs)

        if (
            isinstance(P_obj, list)
            or isinstance(Q_given_P_obj, list)
            or isinstance(P_and_Q_obj, list)
        ):
            return []

        P = P_obj.P
        Q_given_P = Q_given_P_obj.Q_given_P
        P_and_Q = P_and_Q_obj.P_and_Q

        R_given_P_and_Q_obj = Conditional().instantiate_sync(
            {"P": P_and_Q, "Q": base_sentences["R"]}, **kwargs
        )
        P_and_Q_and_R_obj = And().instantiate_sync(
            {"P": P_and_Q, "Q": base_sentences["R"]}, **kwargs
        )

        if isinstance(R_given_P_and_Q_obj, list) or isinstance(P_and_Q_and_R_obj, list):
            return []

        R_given_P_and_Q = R_given_P_and_Q_obj.Q_given_P
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
        Q_given_P_obj = await Conditional().instantiate(base_sentences_PQ, **kwargs)
        P_and_Q_obj = await And().instantiate(base_sentences_PQ, **kwargs)

        if (
            isinstance(P_obj, list)
            or isinstance(Q_given_P_obj, list)
            or isinstance(P_and_Q_obj, list)
        ):
            return []

        P = P_obj.P
        Q_given_P = Q_given_P_obj.Q_given_P
        P_and_Q = P_and_Q_obj.P_and_Q

        R_given_P_and_Q_obj = await Conditional().instantiate(
            {"P": P_and_Q, "Q": base_sentences["R"]}, **kwargs
        )
        P_and_Q_and_R_obj = await And().instantiate(
            {"P": P_and_Q, "Q": base_sentences["R"]}, **kwargs
        )

        if isinstance(R_given_P_and_Q_obj, list) or isinstance(P_and_Q_and_R_obj, list):
            return []

        R_given_P_and_Q = R_given_P_and_Q_obj.Q_given_P
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
    ("ExpectedEvidenceChecker", ExpectedEvidenceChecker),
]


def choose_checkers(
    relevant_checks: list[str], tuple_dir: Path | None = None
) -> dict[str, Checker]:
    print(f"Relevant checks: {relevant_checks}")
    if relevant_checks[0] == "all":
        relevant_checks = [c[0] for c in checker_classes]
    elif isinstance(relevant_checks[0], int) or relevant_checks[0] in [
        str(n) for n in range(1, 4)
    ]:  # allow choosing checks with some number of base questions
        relevant_checks = [
            c[0]
            for c in checker_classes
            if c[1].num_base_questions == int(relevant_checks[0])
        ]
    checkers: dict[str, Checker] = {}
    for checker_name, cls in checker_classes:
        if checker_name in relevant_checks:
            path = None if tuple_dir is None else tuple_dir / f"{checker_name}.jsonl"
            checkers[checker_name] = cls(path=path)
    return checkers
