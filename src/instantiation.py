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


def load_data(file):
    bqs = []
    for line in jsonlines.open(file):
        try:
            # line["resolution_date"] = re.split(":", line["resolution_date"], 1)[1]
            # line["resolution_date"] = line["resolution_date"].strip()
            # line["resolution_date"] = parser.parse(line["resolution_date"])
            # line["question_type"] = line["question_type"].lower()
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
    return base_questions_p, base_questions_pq, base_questions_pqr


async def instantiate(path, length=1):
    base_questions_p, base_questions_pq, base_questions_pqr = load_data(path)
    # fmt: off
    await neg_checker.instantiate_and_write_many(
        base_questions_p[:length],
        model=MODEL,
        overwrite=True,
    )
    await and_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
    )
    await or_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
    )
    await andor_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
    )
    await but_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
    )
    await cond_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
    )
    await cons_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
    )
    await para_checker.instantiate_and_write_many(
        base_questions_p[:length],
        model=MODEL,
        overwrite=True,
    )
    await symmand_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
    )
    await symmor_checker.instantiate_and_write_many(
        base_questions_pq[:length],
        model=MODEL,
        overwrite=True,
    )
    await condcond_checker.instantiate_and_write_many(
        base_questions_pqr[:length],
        model=MODEL,
        overwrite=True,
    )
    # fmt: on


asyncio.run(
    instantiate(
        path=get_data_path() / "fq" / "real" / "questions_cleaned_formatted.jsonl"
    )
)
