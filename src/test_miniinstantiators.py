from static_checks.BaseChecker import Trivial, Neg, And, Or, Paraphrase
from common.datatypes import *
import asyncio

# mini_trivial = Trivial()
# mini_neg = Neg()
#mini_and = And()
mini_or = Or()
# mini_para = Paraphrase()


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
    url="https://jimmyneutron.com",
)

# base_questions = [{"P" :base_question}]
base_questions = [{"P" :base_question, "Q": base_question2}]

async def foo():
    x = await mini_or.instantiate(base_questions[0])
    print(x)

asyncio.run(foo())