from common.utils import shallow_dict
from common.path_utils import get_data_path
from common.llm_utils import parallelized_call
from .forecaster import Forecaster
from .basic_forecaster import BasicForecaster
from common.datatypes import ForecastingQuestion, Forecast
from static_checks import (
    Checker,
    NegChecker,
    ParaphraseChecker,
    AndOrChecker,
    ButChecker,
    CondChecker,
)
from static_checks.tuple_relevance import (
    get_relevant_questions,
    get_relevant_questions_sync,
)
from generate_related_questions import generate_questions_from_question


class ConsistentForecaster(Forecaster):
    """
    So it's easy to explain for a single consistency check--

    given forecasts F(P1), F(P2) ... F(Pn) (e.g. P1 = P, P2 = ¬P), our arbitrage metric (Checker.max_min_arbitrage() in the code) computes two things:

    Improved forecasts F'(P1), F'(P2) ... F'(P_n) which are consistent
    The profit earned by an arbitrageur who bets these improved forecasts
    So say you have a forecaster F: P -> F(P)

    Then define a new forecaster F' which, given P:

    instantiates P2, ... Pn with Checker.instantiate()
    Queries F(P1), F(P2) ... F(Pn) from F.
    uses Checker.max_min_arbitrage() to calculate F'(P1), F'(P2) ... F'(Pn).

    This F' is now perfectly consistent under Checker.
    Of course:

    1) This isn't necessarily robust to different instantiations. You might be able to instantiate P2 ... Pn slightly differently and F' would be inconsistent under that

    2) This is only for a single Checker. E.g. you could make an F' that is consistent under NegChecker and it won't magically become consistent under any other checkers.

    For multiple checkers, we just sequentially arbitrage on each checker.
    """

    def __init__(
        self,
        model,
        hypocrite: Forecaster = None,
        checks: list[Checker] = None,
        base_data_path=get_data_path()
        / "fq"
        / "real"
        / "questions_cleaned_formatted.jsonl",
        coerce_nonbinary_qs=True,
        use_generate_related_questions=True,  # make sure to this is also set in forecasters.create::make_forecaster
        instantiation_kwargs: dict = None,
        bq_func_kwargs: dict = None,
        **kwargs,
    ):
        self.model = model
        self.hypocrite = hypocrite or BasicForecaster(model)
        self.checks = checks or [
            NegChecker(),
            ParaphraseChecker(),
            AndOrChecker(),
            ButChecker(),
            CondChecker(),
        ]
        self.base_data_path = base_data_path
        self.coerce_nonbinary_qs = coerce_nonbinary_qs
        self.instantiation_kwargs = {"verify_before_instantiation": False} | (
            instantiation_kwargs or {}
        )
        self.bq_func_kwargs = bq_func_kwargs or {}
        self.kwargs = kwargs

        self.use_generate_related_questions = use_generate_related_questions

        self.bq_func_kwargs["cost_log"] = self.kwargs.get("cost_log", None)
        self.bq_func_kwargs["simulate"] = self.kwargs.get("simulate", False)
        self.instantiation_kwargs["cost_log"] = self.kwargs.get("cost_log", None)
        self.instantiation_kwargs["simulate"] = self.kwargs.get("simulate", False)

    def bq_function(
        self,
        sentence: ForecastingQuestion,
        keys: list = None,
        n_relevance: int = 10,
        n_return: int = 1,
        tuple_size=2,
        base_data_path=None,
        **kwargs,
    ) -> dict[str, ForecastingQuestion]:
        """Get relevant questions for the given sentence.
        Args:
        sentence (ForecastingQuestion): Sentence to get relevant questions for.
        keys (list): Keys to use for the relevant questions.
        n_relevance (int): Number of retrieved questions to consider.
        n_return (int): Number of questions to return.
        tuple_size (int): Size of the tuple to get.
        base_data_path (Path): Path to questions to retrieve from.
        """
        if self.use_generate_related_questions:
            raise NotImplementedError(
                "generate_questions_from_question does not have synchronous version"
                "please use --async"
            )
        if keys is None:
            keys = ["P", "Q", "R", "S", "T"]
        if base_data_path is None:
            base_data_path = self.base_data_path
        tup = get_relevant_questions_sync(
            existing_questions=[sentence],
            n_relevance=n_relevance,
            n_return=n_return,
            tuple_size=tuple_size,
            base_data_path=base_data_path,
            **kwargs,
        )[0][0]
        return {k: fq for k, fq in zip(keys, tup)}

    async def bq_function_async(
        self,
        sentence: ForecastingQuestion,
        keys: list = None,
        n_relevance: int = 10,
        n_return: int = 1,
        tuple_size=2,
        base_data_path=None,
        **kwargs,
    ) -> dict[str, ForecastingQuestion]:
        """Get relevant questions for the given sentence.
        Args:
        sentence (ForecastingQuestion): Sentence to get relevant questions for.
        keys (list): Keys to use for the relevant questions.
        n_relevance (int): Number of retrieved questions to consider.
        n_return (int): Number of questions to return.
        tuple_size (int): Size of the tuple to get.
        base_data_path (Path): Path to questions to retrieve from.
        """
        if keys is None:
            keys = ["P", "Q", "R", "S", "T"]
        if base_data_path is None:
            base_data_path = self.base_data_path
        if self.use_generate_related_questions:
            related_questions = await generate_questions_from_question(
                source_question=sentence.title,
                num_questions=tuple_size - 1,
                model=kwargs.get("model", self.model),
                source_body=sentence.body,
                resolve_by=sentence.resolution_date,
                return_fq=True,
            )
            tup = [sentence] + related_questions
        else:
            res = await get_relevant_questions(
                existing_questions=[sentence],
                n_relevance=n_relevance,
                n_return=n_return,
                tuple_size=tuple_size,
                base_data_path=base_data_path,
                **kwargs,
            )
            tup = res[0][0]
        return {k: fq for k, fq in zip(keys, tup)}

    def instantiate_cons_tuples(
        self,
        sentence: ForecastingQuestion,
        bq_func_kwargs: dict = None,
        instantiation_kwargs: dict = None,
        **kwargs,
    ) -> list[dict[str, ForecastingQuestion]]:
        """Instantiate tuples for arbitraging the given sentence.
        Args:
            sentence (ForecastingQuestion): Sentence to instantiate consistent tuples for.
            bq_func_kwargs (dict): Keyword arguments for bq_function.
            instantiation_kwargs (dict): Keyword arguments for instantiation.
        """
        if self.coerce_nonbinary_qs and not sentence.question_type == "binary":
            sentence.question_type = "binary"
        kwargs = self.kwargs | (kwargs or {})
        bq_func_kwargs = self.bq_func_kwargs | (bq_func_kwargs or {})
        instantiation_kwargs = self.instantiation_kwargs | (instantiation_kwargs or {})
        bq_func_kwargs["cost_log"] = kwargs.get("cost_log", None)
        bq_func_kwargs["simulate"] = kwargs.get("simulate", False)
        instantiation_kwargs["cost_log"] = kwargs.get("cost_log", None)
        instantiation_kwargs["simulate"] = kwargs.get("simulate", False)
        # pre-generate bq_tuple for tuple_size=max(check.num_base_questions for check in self.checks)
        seed = 137
        max_tuple_size = max(check.num_base_questions for check in self.checks)
        bq_tuple_max = self.bq_function(
            sentence, tuple_size=max_tuple_size, seed=seed, **bq_func_kwargs
        )
        cons_tuples = []
        checks_so_far = []
        for check in self.checks:
            if check.__class__.__name__ in checks_so_far:
                # if we have multiple checks of the same type, they should have DIFFERENT base questions
                seed += 1
                bq_tuple = self.bq_function(
                    sentence,
                    tuple_size=check.num_base_questions,
                    seed=seed,
                    **bq_func_kwargs,
                )
            else:
                bq_tuple = {
                    k: v
                    for k, v in bq_tuple_max.items()
                    if k in list(bq_tuple_max.keys())[: check.num_base_questions]
                }
            checks_so_far.append(check.__class__.__name__)
            print(f"{bq_tuple=}")
            cons_tuple = check.instantiate_sync(bq_tuple, **instantiation_kwargs)
            print(f"{cons_tuple=}")
            if isinstance(cons_tuple, list):
                if len(cons_tuple) == 0:
                    print(
                        f"Found no valid instantiated cons_tuple for {check.__class__.__name__}"
                    )
                    continue
                cons_tuple = cons_tuple[0]
            cons_tuples.append(cons_tuple)
        return cons_tuples

    async def instantiate_cons_tuples_async(
        self,
        sentence: ForecastingQuestion,
        bq_func_kwargs: dict = None,
        instantiation_kwargs: dict = None,
        **kwargs,
    ) -> list[dict[str, ForecastingQuestion]]:
        """Instantiate tuples for arbitraging the given sentence.

        Args:
            sentence (ForecastingQuestion): Sentence to instantiate consistent tuples for.
            bq_func_kwargs (dict): Keyword arguments for bq_function.
            instantiation_kwargs (dict): Keyword arguments for instantiation.

        """
        if self.coerce_nonbinary_qs and not sentence.question_type == "binary":
            sentence.question_type = "binary"
        kwargs = self.kwargs | (kwargs or {})
        bq_func_kwargs = self.bq_func_kwargs | (bq_func_kwargs or {})
        instantiation_kwargs = self.instantiation_kwargs | (instantiation_kwargs or {})
        bq_func_kwargs["cost_log"] = kwargs.get("cost_log", None)
        bq_func_kwargs["simulate"] = kwargs.get("simulate", False)
        instantiation_kwargs["cost_log"] = kwargs.get("cost_log", None)
        instantiation_kwargs["simulate"] = kwargs.get("simulate", False)

        # pre-generate bq_tuple for tuple_size=max(check.num_base_questions for check in self.checks)
        seed = 137
        max_tuple_size = max(check.num_base_questions for check in self.checks)
        bq_tuple_max = await self.bq_function_async(
            sentence, tuple_size=max_tuple_size, seed=seed, **bq_func_kwargs
        )

        checks_so_far = []
        tasks = []

        def _instantiate_check(tup):
            check, bq_tuple, instantiation_kwargs = tup
            return check.instantiate(bq_tuple, **instantiation_kwargs)

        for check in self.checks:
            if check.__class__.__name__ in checks_so_far:
                seed += 1
                bq_tuple = await self.bq_function_async(
                    sentence,
                    tuple_size=check.num_base_questions,
                    seed=seed,
                    **bq_func_kwargs,
                )
            else:
                bq_tuple = {
                    k: v
                    for k, v in bq_tuple_max.items()
                    if k in list(bq_tuple_max.keys())[: check.num_base_questions]
                }
            checks_so_far.append(check.__class__.__name__)
            tasks.append((check, bq_tuple, instantiation_kwargs))

        cons_tuples = await parallelized_call(_instantiate_check, tasks)

        cons_tuples = [
            cons_tuple[0]
            if isinstance(cons_tuple, list) and len(cons_tuple) > 0
            else cons_tuple
            for cons_tuple in cons_tuples
            if cons_tuple
        ]

        return cons_tuples

    def call(
        self,
        sentence: ForecastingQuestion,
        bq_func_kwargs: dict = None,
        instantiation_kwargs: dict = None,
        only_arbitrage_if_fail=False,
        **kwargs,
    ) -> Forecast:
        """Call ConsistentForecaster by sequentially arbitraging against checks.

        Args:
            sentence (ForecastingQuestion): Sentence to forecast.
            bq_func_kwargs (dict): Keyword arguments for bq_function.
            instantiation_kwargs (dict): Keyword arguments for instantiation.

        Example usage:

        ```python
        cf.call(
            fq,
            bq_func_kwargs={
                "n_relevance": 10,
                "n_return": 1,
                "tuple_size": 2,
                "model": "gpt-4o",
            },
            instantiation_kwargs={
                "model": "gpt-4o",
            },
            model="gpt-4o",
        )
        ```

        """
        metadata = []
        ans_P = self.hypocrite.call(sentence, **kwargs)
        metadata_entry = {
            "name": "P",
            "elicited_prob": ans_P.prob,
            "elicitation_metadata": ans_P.metadata,
        }
        metadata.append(metadata_entry)
        ans_P = ans_P.prob  ###

        cons_tuples = self.instantiate_cons_tuples(
            sentence,
            bq_func_kwargs=bq_func_kwargs,
            instantiation_kwargs=instantiation_kwargs,
            **kwargs,
        )

        P_weight = 1.0
        for check, cons_tuple in zip(self.checks, cons_tuples):
            cons_tuple = shallow_dict(cons_tuple)
            del cons_tuple["P"]
            hypocrite_answers = self.hypocrite.elicit(cons_tuple, **kwargs)

            metadata_entry = {
                "name": check.__class__.__name__,
                "data": {
                    k: {
                        **cons_tuple[k].model_dump(),
                        "elicited_prob": hypocrite_answers[k].prob,
                        "elicitation_metadata": hypocrite_answers[k].metadata,
                    }
                    for k in cons_tuple
                },
            }
            metadata.append(metadata_entry)

            hypocrite_answers = {k: v.prob for k, v in hypocrite_answers.items()}

            hypocrite_answers["P"] = ans_P
            print("HYPOCRISY", hypocrite_answers)
            other = len(cons_tuple) - 1
            cons_answers, v = check.max_min_arbitrage(
                hypocrite_answers, scoring=[P_weight] + [1.0] * other
            )
            P_weight += 1.0 * other
            if v > check.default_tolerance or not only_arbitrage_if_fail:
                ans_P = cons_answers["P"]

        return Forecast(prob=ans_P, metadata=metadata)

    async def call_async(
        self,
        sentence: ForecastingQuestion,
        bq_func_kwargs: dict = None,
        instantiation_kwargs: dict = None,
        only_arbitrage_if_fail=False,
        **kwargs,
    ) -> Forecast:
        """Call ConsistentForecaster by sequentially arbitraging against checks.

        Args:
            sentence (ForecastingQuestion): Sentence to forecast.
            bq_func_kwargs (dict): Keyword arguments for bq_function.
            instantiation_kwargs (dict): Keyword arguments for instantiation.

        Example usage:

        ```python
        cf.call(
            fq,
            bq_func_kwargs={
                "n_relevance": 10,
                "n_return": 1,
                "tuple_size": 2,
                "model": "gpt-4o",
            },
            instantiation_kwargs={
                "model": "gpt-4o",
            },
            model="gpt-4o",
        )
        ```

        """
        metadata = []
        ans_P = await self.hypocrite.call_async(sentence, **kwargs)
        metadata_entry = {
            "name": "P",
            "elicited_prob": ans_P.prob,
            "elicitation_metadata": ans_P.metadata,
        }
        metadata.append(metadata_entry)
        ans_P = ans_P.prob  ###

        cons_tuples = await self.instantiate_cons_tuples_async(
            sentence,
            bq_func_kwargs=bq_func_kwargs,
            instantiation_kwargs=instantiation_kwargs,
            **kwargs,
        )
        P_weight = 1.0
        for check, cons_tuple in zip(self.checks, cons_tuples):
            cons_tuple = shallow_dict(cons_tuple)
            del cons_tuple["P"]
            hypocrite_answers = await self.hypocrite.elicit_async(cons_tuple, **kwargs)

            metadata_entry = {
                "name": check.__class__.__name__,
                "data": {
                    k: {
                        **cons_tuple[k].model_dump(),
                        "elicited_prob": hypocrite_answers[k].prob,
                        "elicitation_metadata": hypocrite_answers[k].metadata,
                    }
                    for k in cons_tuple
                },
            }
            metadata.append(metadata_entry)

            hypocrite_answers = {k: v.prob for k, v in hypocrite_answers.items()}

            hypocrite_answers["P"] = ans_P
            cons_answers, v = check.max_min_arbitrage(
                hypocrite_answers, scoring=[P_weight, 1.0]
            )
            P_weight += 1.0 * (len(cons_tuple) - 1)
            if v > check.default_tolerance or not only_arbitrage_if_fail:
                ans_P = cons_answers["P"]
        return Forecast(prob=ans_P, metadata=metadata)

    @classmethod
    def recursive(
        cls,
        depth: int = 0,
        hypocrite: Forecaster = None,
        **kwargs,
    ) -> "ConsistentForecaster":
        assert isinstance(depth, int) and depth >= 0
        if depth == 0:
            return hypocrite or BasicForecaster()
        else:
            return cls(
                hypocrite=cls.recursive(
                    depth=depth - 1,
                    hypocrite=hypocrite,
                    **kwargs,
                ),
                **kwargs,
            )

    def dump_config(self):
        return {
            "hypocrite": self.hypocrite.dump_config(),
            "checks": [c.dump_config() for c in self.checks],
            "base_data_path": str(self.base_data_path),
            "instantiation_kwargs": self.instantiation_kwargs,
            "bq_func_kwargs": self.bq_func_kwargs,
            "call_kwargs": self.kwargs,
        }

    @classmethod
    def load_config(cls, config):
        return cls(
            hypocrite=Forecaster.load_config(config["hypocrite"]),
            checks=[Checker.load_config(c) for c in config["checks"]],
            base_data_path=config["base_data_path"],
            instantiation_kwargs=config["instantiation_kwargs"],
            bq_func_kwargs=config["bq_func_kwargs"],
            **config["call_kwargs"],
        )
