from common.utils import shallow_dict
from .forecaster import Forecaster
from .basic_forecaster import BasicForecaster
from common.datatypes import ForecastingQuestion
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
from common.path_utils import get_data_path


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
        hypocrite: Forecaster = None,
        checks: list[Checker] = None,
        base_data_path=get_data_path()
        / "fq"
        / "real"
        / "questions_cleaned_formatted.jsonl",
        pregenerate=True,
        coerce_nonbinary_qs=True,
        instantiation_kwargs: dict = None,
        bq_func_kwargs: dict = None,
        **kwargs,
    ):
        self.hypocrite = hypocrite or BasicForecaster()
        self.checks = checks or [
            NegChecker(),
            ParaphraseChecker(),
            AndOrChecker(),
            ButChecker(),
            CondChecker(),
        ]
        self.base_data_path = base_data_path
        self.coerce_nonbinary_qs = coerce_nonbinary_qs
        self.pregenerate = pregenerate
        self.instantiation_kwargs = instantiation_kwargs or {}
        self.bq_func_kwargs = bq_func_kwargs or {}
        self.kwargs = kwargs

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

    def call(
        self,
        sentence: ForecastingQuestion,
        bq_func_kwargs: dict = None,
        instantiation_kwargs: dict = None,
        **kwargs,
    ) -> float:
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

        """
        if self.coerce_nonbinary_qs and not sentence.question_type == "binary":
            sentence.question_type = "binary"
        kwargs = self.kwargs | (kwargs or {})
        bq_func_kwargs = self.bq_func_kwargs | (bq_func_kwargs or {})
        instantiation_kwargs = self.instantiation_kwargs | (instantiation_kwargs or {})
        ans_P = self.hypocrite.call(sentence, **kwargs)

        if self.pregenerate:
            # pre-generate bq_tuple for tuple_size=max(check.num_base_questions for check in self.checks)
            max_tuple_size = max(check.num_base_questions for check in self.checks)
            bq_tuple_max = self.bq_function(
                sentence, tuple_size=max_tuple_size, **bq_func_kwargs
            )

        for check in self.checks:
            if self.pregenerate:
                bq_tuple = {
                    k: v
                    for k, v in bq_tuple_max.items()
                    if k in list(bq_tuple_max.keys())[: check.num_base_questions]
                }
            else:
                bq_tuple = self.bq_function(
                    sentence, tuple_size=check.num_base_questions, **bq_func_kwargs
                )
            cons_tuple = check.instantiate_sync(bq_tuple, **instantiation_kwargs)
            if isinstance(cons_tuple, list):
                cons_tuple = cons_tuple[0]
            cons_tuple = shallow_dict(cons_tuple)
            del cons_tuple["P"]
            hypocrite_answers = self.hypocrite.elicit(cons_tuple, **kwargs)
            hypocrite_answers["P"] = ans_P
            cons_answers, v = check.max_min_arbitrage(hypocrite_answers)
            ans_P = cons_answers["P"]
        return ans_P

    async def call_async(
        self,
        sentence: ForecastingQuestion,
        bq_func_kwargs: dict = None,
        instantiation_kwargs: dict = None,
        **kwargs,
    ) -> float:
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

        """
        if self.coerce_nonbinary_qs and not sentence.question_type == "binary":
            sentence.question_type = "binary"
        kwargs = self.kwargs | (kwargs or {})
        bq_func_kwargs = self.bq_func_kwargs | (bq_func_kwargs or {})
        instantiation_kwargs = self.instantiation_kwargs | (instantiation_kwargs or {})
        ans_P = await self.hypocrite.call_async(sentence, **kwargs)
        if self.pregenerate:
            # pre-generate bq_tuple for tuple_size=max(check.num_base_questions for check in self.checks)
            max_tuple_size = max(check.num_base_questions for check in self.checks)
            bq_tuple_max = await self.bq_function_async(
                sentence, tuple_size=max_tuple_size, **bq_func_kwargs
            )

        for check in self.checks:
            if self.pregenerate:
                bq_tuple = {
                    k: v
                    for k, v in bq_tuple_max.items()
                    if k in list(bq_tuple_max.keys())[: check.num_base_questions]
                }
            else:
                bq_tuple = await self.bq_function_async(
                    sentence, tuple_size=check.num_base_questions, **bq_func_kwargs
                )
            cons_tuple = await check.instantiate(bq_tuple, **instantiation_kwargs)
            if isinstance(cons_tuple, list):
                cons_tuple = cons_tuple[0]
            cons_tuple = shallow_dict(cons_tuple)
            del cons_tuple["P"]
            hypocrite_answers = await self.hypocrite.elicit_async(cons_tuple, **kwargs)
            hypocrite_answers["P"] = ans_P
            cons_answers, v = check.max_min_arbitrage(hypocrite_answers)
            ans_P = cons_answers["P"]
        return ans_P

    def dump_config(self):
        return {
            "hypocrite": self.hypocrite.dump_config(),
            "checks": [c.dump_config() for c in self.checks],
            "base_data_path": str(self.base_data_path),
            "pregenerate": self.pregenerate,
            "instantiation_kwargs": self.instantiation_kwargs,
            "bq_func_kwargs": self.bq_func_kwargs,
            "call_kwargs": self.kwargs,
        }