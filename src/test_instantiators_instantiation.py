"""TEST INSTANTIATORS"""

# from common.utils import *
# from common.llm_utils import *
import asyncio
from static_checks.MiniInstantiator import *
from static_checks.Checker import *
from common.datatypes import *


neg_checker = NegChecker(path="src/data/test/NegChecker.jsonl")
and_checker = AndChecker(path="src/data/test/AndChecker.jsonl")
or_checker = OrChecker(path="src/data/test/OrChecker.jsonl")
andor_checker = AndOrChecker(path="src/data/test/AndOrChecker.jsonl")
but_checker = ButChecker(path="src/data/test/ButChecker.jsonl")
cond_checker = CondChecker(path="src/data/test/CondChecker.jsonl")
cons_checker = ConsequenceChecker(path="src/data/test/ConsequenceChecker.jsonl")
para_checker = ParaphraseChecker(path="src/data/test/ParaphraseChecker.jsonl")
symmand_checker = SymmetryAndChecker(path="src/data/test/SymmetryAndChecker.jsonl")
symmor_checker = SymmetryOrChecker(path="src/data/test/SymmetryOrChecker.jsonl")
condcond_checker = CondCondChecker(path="src/data/test/CondCondChecker.jsonl")


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

base_questions_p = [{"P" :base_question}]
base_questions_pq = [{"P" :base_question, "Q": base_question2}]
base_questions_pqr = [{"P" :base_question, "Q": base_question2, "R": base_question3}]
base_questions_pqrs = [base_questions_pqr[0], base_questions_pqr[0], base_questions_pqr[0]]

# asyncio.run(neg_checker.instantiate_and_write_many(base_questions_p, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(and_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(or_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(andor_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(but_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(cond_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(cons_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(para_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(symmand_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
# asyncio.run(symmor_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo", overwrite=False))
asyncio.run(condcond_checker.instantiate_and_write_many(base_questions_pqrs, model="gpt-3.5-turbo", overwrite=False))