import os
import re
import itertools as it
import jsonlines
import asyncio
from dateutil import parser
from static_checks.MiniInstantiator import *
from static_checks.Checker import *
from common.datatypes import *

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


def load_data():
    bqs = []
    for line in jsonlines.open("src/data/politics_qs_1_formated.jsonl"):
        try:
            line["resolution_date"] = re.split(":", line["resolution_date"], 1)[1]
            line["resolution_date"] = line["resolution_date"].strip()
            line["resolution_date"] = parser.parse(line["resolution_date"])
            line["question_type"] = line["question_type"].lower()
            bq = ForecastingQuestion(**line)
            bqs.append(bq)
        except:
            continue
    return bqs


bqs = load_data()

base_questions_p = [{"P": P} for P in bqs]
base_questions_pq = [{"P": P, "Q": Q} for P, Q in it.combinations(bqs, 2)]
base_questions_pqr = [{"P": P, "Q": Q, "R": R} for P, Q, R in it.combinations(bqs, 3)]


async def instantiate(length=10):
    await neg_checker.instantiate_and_write_many(
        base_questions_p[:length], model="gpt-3.5-turbo", overwrite=True
    )
    await and_checker.instantiate_and_write_many(
        base_questions_pq[:length], model="gpt-3.5-turbo", overwrite=True
    )
    await or_checker.instantiate_and_write_many(
        base_questions_pq[:length], model="gpt-3.5-turbo", overwrite=True
    )
    await andor_checker.instantiate_and_write_many(
        base_questions_pq[:length], model="gpt-3.5-turbo", overwrite=True
    )
    await but_checker.instantiate_and_write_many(
        base_questions_pq[:length], model="gpt-3.5-turbo", overwrite=True
    )
    await cond_checker.instantiate_and_write_many(
        base_questions_pq[:length], model="gpt-3.5-turbo", overwrite=True
    )
    await cons_checker.instantiate_and_write_many(
        base_questions_pq[:length], model="gpt-3.5-turbo", overwrite=True
    )
    await para_checker.instantiate_and_write_many(
        base_questions_pq[:length], model="gpt-3.5-turbo", overwrite=True
    )
    await symmand_checker.instantiate_and_write_many(
        base_questions_pq[:length], model="gpt-3.5-turbo", overwrite=True
    )
    await symmor_checker.instantiate_and_write_many(
        base_questions_pq[:length], model="gpt-3.5-turbo", overwrite=True
    )
    await condcond_checker.instantiate_and_write_many(
        base_questions_pqr[:length], model="gpt-3.5-turbo", overwrite=True
    )

asyncio.run(instantiate())

# asyncio.run(
#     neg_checker.instantiate_and_write_many(base_questions_p, model="gpt-3.5-turbo")
# )
# asyncio.run(
#     and_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo")
# )
# asyncio.run(
#     or_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo")
# )
# asyncio.run(
#     andor_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo")
# )
# asyncio.run(
#     but_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo")
# )
# asyncio.run(
#     cond_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo")
# )
# asyncio.run(
#     cons_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo")
# )
# asyncio.run(
#     para_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo")
# )
# asyncio.run(
#     symmand_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo")
# )
# asyncio.run(
#     symmor_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo")
# )
# asyncio.run(
#     condcond_checker.instantiate_and_write_many(
#         base_questions_pqr, model="gpt-3.5-turbo"
#     )
# )
