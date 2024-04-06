"""TEST INSTANTIATORS"""

# from common.utils import *
# from common.llm_utils import *
import asyncio
from static_checks.MiniInstantiator import *
from static_checks.Checker import *
from common.datatypes import *


neg_checker = NegChecker()
and_checker = AndChecker()
or_checker = OrChecker()

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

base_questions_p = [{"P" :base_question}]
base_questions_pq = [{"P" :base_question, "Q": base_question2}]

#asyncio.run(neg_checker.instantiate_and_write_many(base_questions_p, model="gpt-3.5-turbo"))
asyncio.run(or_checker.instantiate_and_write_many(base_questions_pq, model="gpt-3.5-turbo"))