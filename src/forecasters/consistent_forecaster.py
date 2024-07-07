from .forecaster import Forecaster
from common.datatypes import ForecastingQuestion
from static_checks import Checker
from static_checks.tuple_relevance import (
    get_relevant_questions,
    get_relevant_questions_sync,
)


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
        hypocrite: Forecaster,
        checks: list[Checker] = None,
    ):
        self.hypocrite = hypocrite
        self.checks = checks or []

    def bq_function(
        self, sentence: ForecastingQuestion, keys: dict = None, **kwargs
    ) -> dict:
        """Get relevant questions for the given sentence.

        Args:

        sentence (ForecastingQuestion): Sentence to get relevant questions for.
        keys (dict): Keys to use for the relevant questions.

        Keyword Args:

        n_relevance (int): Number of retrieved questions to consider.
        n_return (int): Number of questions to return.
        tuple_size (int): Size of the tuple to get.
        base_data_path (Path): Path to questions to retrieve from.

        """
        if keys is None:
            keys = ["P", "Q", "R", "S", "T"]
        tup = get_relevant_questions_sync(existing_questions=[sentence], **kwargs)[0][0]
        return {k: fq for k, fq in zip(keys, tup)}

    async def bq_function_async(
        self, sentence: ForecastingQuestion, keys: dict = None, **kwargs
    ) -> dict[str, ForecastingQuestion]:
        """Get relevant questions for the given sentence.

        Args:

        sentence (ForecastingQuestion): Sentence to get relevant questions for.
        keys (dict): Keys to use for the relevant questions.

        Keyword Args:

        n_relevance (int): Number of retrieved questions to consider.
        n_return (int): Number of questions to return.
        tuple_size (int): Size of the tuple to get.
        base_data_path (Path): Path to questions to retrieve from.

        """
        if keys is None:
            keys = ["P", "Q", "R", "S", "T"]
        res = await get_relevant_questions(existing_questions=[sentence], **kwargs)
        tup = res[0][0]
        return {k: fq for k, fq in zip(keys, tup)}

    def call(self, sentence: ForecastingQuestion, bq_func_kwargs, **kwargs) -> float:
        """Call ConsistentForecaster by sequentially arbitraging against checks.

        Args:
            sentence (ForecastingQuestion): Sentence to forecast.
            bq_func_kwargs (dict): Keyword arguments for bq_function.

        """
        ans_P = self.hypocrite.call(sentence, **kwargs)
        for check in self.checks:
            bq_tuple = self.bq_function(
                sentence, tuple_size=check.num_base_questions, **bq_func_kwargs
            )
            cons_tuple = self.hypocrite.instantiate_sync(bq_tuple)
            del cons_tuple["P"]
            hypocrite_answers = self.hypocrite.elicit(cons_tuple)
            hypocrite_answers["P"] = ans_P
            cons_answers, v = check.max_min_arbitrage(hypocrite_answers)
            ans_P = cons_answers["P"]
        return ans_P

    async def call_async(self, sentence: ForecastingQuestion, **kwargs) -> float:
        """Call ConsistentForecaster by sequentially arbitraging against checks.

        Args:
            sentence (ForecastingQuestion): Sentence to forecast.
            bq_func_kwargs (dict): Keyword arguments for bq_function.

        """
        ans_P = await self.hypocrite.call_async(sentence, **kwargs)
        for check in self.checks:
            bq_tuple = await self.bq_function_async(
                sentence, tuple_size=check.num_base_questions, **kwargs
            )
            cons_tuple = await self.hypocrite.instantiate(bq_tuple)
            del cons_tuple["P"]
            hypocrite_answers = await self.hypocrite.elicit_async(cons_tuple)
            hypocrite_answers["P"] = ans_P
            cons_answers, v = await check.max_min_arbitrage(hypocrite_answers)
            ans_P = cons_answers["P"]
        return ans_P
