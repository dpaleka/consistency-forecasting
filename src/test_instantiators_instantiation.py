"""TEST INSTANTIATORS"""

# from common.utils import *
# from common.llm_utils import *
from static_checks import *
from common.datatypes import *
import asyncio

neg = NegChecker()

base_question = ForecastingQuestion(
    title="Will Jimmy Neutron be US president in 2025?",
    body="Resolves YES if Jimmy Neutron is president by the end of 2025. Resolves NO otherwise.",
    resolution_date=datetime(2025, 12, 31),
    question_type="binary",
    data_source="synthetic",
    url="https://jimmyneutron.com",
)
base_questions = [{"P" :base_question}]

asyncio.run(neg.instantiate_and_write_many(base_questions, model="gpt-3.5-turbo"))