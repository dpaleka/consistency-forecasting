"""TEST INSTANTIATORS"""

# from common.utils import *
# from common.llm_utils import *
from datetime import datetime
from static_checks.MiniInstantiator import *
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
from common.path_utils import get_data_path
import pytest


neg_checker = NegChecker(path=f"{get_data_path()}/test/instantiation/NegChecker.jsonl")
and_checker = AndChecker(path=f"{get_data_path()}/test/instantiation/AndChecker.jsonl")
or_checker = OrChecker(path=f"{get_data_path()}/test/instantiation/OrChecker.jsonl")
andor_checker = AndOrChecker(
    path=f"{get_data_path()}/test/instantiation/AndOrChecker.jsonl"
)
but_checker = ButChecker(path=f"{get_data_path()}/test/instantiation/ButChecker.jsonl")
cond_checker = CondChecker(
    path=f"{get_data_path()}/test/instantiation/CondChecker.jsonl"
)
consequence_checker = ConsequenceChecker(
    path=f"{get_data_path()}/test/instantiation/ConsequenceChecker.jsonl"
)
paraphrase_checker = ParaphraseChecker(
    path=f"{get_data_path()}/test/instantiation/ParaphraseChecker.jsonl"
)
symmetry_and_checker = SymmetryAndChecker(
    path=f"{get_data_path()}/test/instantiation/SymmetryAndChecker.jsonl"
)
symmetry_or_checker = SymmetryOrChecker(
    path=f"{get_data_path()}/test/instantiation/SymmetryOrChecker.jsonl"
)
condcond_checker = CondCondChecker(
    path=f"{get_data_path()}/test/instantiation/CondCondChecker.jsonl"
)


base_question = ForecastingQuestion(
    title="Will Jimmy Neutron be US president in 2025?",
    body="Resolves YES if Jimmy Neutron is president by the end of 2025. Resolves NO otherwise.",
    resolution_date=datetime(2025, 12, 31),
    question_type="binary",
    data_source="synthetic",
    url="https://jimmyneutron.com",
)
base_question2 = ForecastingQuestion(
    title="Will I eat a blueberry tomorrow?",
    body="Resolves YES if I eat a blueberry tomorrow. Resolves NO otherwise.",
    resolution_date=datetime(2025, 12, 31),
    question_type="binary",
    data_source="synthetic",
    url="https://blueberrycorporation.org",
)
base_question3 = ForecastingQuestion(
    title="Will it rain tomorrow?",
    body="Resolves YES if it rains tomorrow. Resolves NO otherwise.",
    resolution_date=datetime(2025, 12, 31),
    question_type="binary",
    data_source="synthetic",
    url="https://weather.com",
)

base_questions_p = [{"P": base_question}]
base_questions_pq = [{"P": base_question, "Q": base_question2}]
base_questions_pqr = [{"P": base_question, "Q": base_question2, "R": base_question3}]
base_questions_pqrs = [base_questions_pqr[0]]

# asyncio.run(neg_checker.instantiate_and_write_many(base_questions_p, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(and_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(or_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(andor_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(but_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(cond_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(consequence_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(paraphrase_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(symmetry_and_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(symmetry_or_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))


@pytest.mark.asyncio
async def test_andor_checker_instantiation():
    await andor_checker.instantiate_and_write_many(
        base_questions_pq, model="gpt-3.5-turbo", overwrite=False
    )
