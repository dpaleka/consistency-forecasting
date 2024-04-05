from static_checks.NegChecker import Trivial, Neg
from common.datatypes import *
import asyncio

minitrivial = Trivial()
minineg = Neg()

base_question = ForecastingQuestion(
    title="Will Jimmy Neutron be US president in 2025?",
    body="Resolves YES if Jimmy Neutron is president by the end of 2025. Resolves NO otherwise.",
    resolution_date=datetime(2025, 12, 31),
    question_type="binary",
    data_source="synthetic",
    url="https://jimmyneutron.com",
)
base_questions = [{"P" :base_question}]

async def foo():
    x = await minineg.instantiate(base_questions[0])
    print(x)

asyncio.run(foo())