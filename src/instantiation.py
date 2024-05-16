import itertools as it
import jsonlines
import asyncio




# MODEL = "gpt-3.5-turbo"
MODEL = "gpt-4-turbo"
#MODEL = 'gpt-4o'

# from static_checks.MiniInstantiator import MiniInstantiator
from static_checks.Checker import (
    NegChecker,
    AndChecker,
    OrChecker,
    AndOrChecker,
    ButChecker,
    CondChecker,
    ConsequenceChecker,
    ParaphraseChecker,
    SymmetryAndChecker,
    SymmetryOrChecker,
    CondCondChecker,
)
from static_checks.tuple_relevance import relevance
from common.datatypes import ForecastingQuestion
import random

neg_checker = NegChecker()
and_checker = AndChecker()
or_checker = OrChecker()
andor_checker = AndOrChecker()
but_checker = ButChecker()
cond_checker = CondChecker()
cons_checker = ConsequenceChecker()
para_checker = ParaphraseChecker()
symmand_checker = SymmetryAndChecker()
symmor_checker = SymmetryOrChecker()
condcond_checker = CondCondChecker()

from common.path_utils import get_data_path


async def load_data(
    file,
    n=10,  # we have to decide this now because we can't score relevance for everything
):
    bqs = []
    for line in jsonlines.open(file):
        try:
            bq = ForecastingQuestion(**line)
            bqs.append(bq)
        except Exception as e:
            print(e)
            continue
    base_questions_p = [{"P": P} for P in bqs]
    base_questions_pq = [{"P": P, "Q": Q} for P, Q in it.combinations(bqs, 2)]
    base_questions_pqr = [
        {"P": P, "Q": Q, "R": R} for P, Q, R in it.combinations(bqs, 3)
    ]
    random.shuffle(base_questions_p)
    random.shuffle(base_questions_pq)
    random.shuffle(base_questions_pqr)

    base_questions_p = base_questions_p[:n]
    base_questions_pq = base_questions_pq[:n]
    base_questions_pqr = base_questions_pqr[:n]

    tasks_pq = [relevance(tup) for tup in base_questions_pq]
    tasks_pqr = [relevance(tup) for tup in base_questions_pqr]

    print("Calculating relevance scores for 2-ples")
    relevances_pq = await asyncio.gather(*tasks_pq)

    print("Calculating relevance scores for 3-ples")
    relevances_pqr = await asyncio.gather(*tasks_pqr)

    base_questions_with_relevance_pq = list(zip(base_questions_pq, relevances_pq))
    base_questions_with_relevance_pqr = list(zip(base_questions_pqr, relevances_pqr))

    base_questions_with_relevance_pq.sort(
        key=lambda x: x[1]["relevance"]["score"], reverse=True
    )
    base_questions_with_relevance_pqr.sort(
        key=lambda x: x[1]["relevance"]["score"], reverse=True
    )

    return (
        base_questions_p,
        base_questions_with_relevance_pq,
        base_questions_with_relevance_pqr,
    )


async def instantiate(path, n_relevance=10, length=3):
    base_questions_p, base_questions_pq, base_questions_pqr = await load_data(
        path,
        n=n_relevance,
    )
    # fmt: off
    await neg_checker.instantiate_and_write_many(
        base_questions_p[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    await and_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    await or_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    await andor_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    await but_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    await cond_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    await cons_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    await para_checker.instantiate_and_write_many(
        base_questions_p[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    await symmand_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    await symmor_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    await condcond_checker.instantiate_and_write_many(
        base_questions_pqr[:length],
        model=MODEL,
        overwrite=True,
        validate_before=True,
        n_validation=3,
    )
    # fmt: on


asyncio.run(
    instantiate(
        path=get_data_path() / "fq" / "real" / "questions_cleaned_formatted.jsonl"
    )
)
