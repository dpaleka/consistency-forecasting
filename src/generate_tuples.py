#%%

from static_checks import *
from common.datatypes import *
import asyncio

base_question = Sentence(
    title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
    body="Resolves YES if the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025. Resolves NO otherwise.",
    resolution_date=datetime(2025, 1, 1),
    question_type=QuestionType("binary"),
    data_source="https://www.coindesk.com/price/bitcoin",
    url="https://www.coindesk.com/price/bitcoin",
    metadata={"currency": "USD"},
    resolution="YES",
)

#asyncio.run(ButNotChecker().instantiate_and_write_many(base_questions_combos))
asyncio.run(NegChecker().instantiate_and_write_many([[base_question]], model="gpt-3.5-turbo"))
#asyncio.run(ParaphrasalChecker().instantiate_and_write_many(base_questions_combos, model="gpt-3.5-turbo"))
    
#%%