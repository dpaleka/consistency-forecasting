from static_checks.MiniInstantiator import (
    Trivial,
    Neg,
    And,
    Or,
    Paraphrase,
    Conditional,
    Consequence,
)
from common.datatypes import ForecastingQuestion
from datetime import datetime
import asyncio
import pytest

base_question = ForecastingQuestion(
    title="Will Jimmy Neutron be US president in 2025?",
    body="Resolves YES if Jimmy Neutron is president by the end of 2025. Resolves NO otherwise.",
    resolution_date=datetime(2025, 12, 31),
    question_type="binary",
    data_source="synthetic",
    url="https://jimmyneutron.com",
    metadata={},
)

base_question2 = ForecastingQuestion(
    title="Will I eat a blueberry tomorrow?",
    body="Resolves YES if I eat a blueberry tomorrow. Resolves NO otherwise.",
    resolution_date=datetime(2025, 12, 31),
    question_type="binary",
    data_source="synthetic",
    url="https://jimmyneutron.com",
    metadata={},
)

base_questions_p = {"P": base_question}
base_questions_pq = {"P": base_question, "Q": base_question2}


@pytest.mark.asyncio
async def test_trivial():
    mini_trivial = Trivial()
    x = await mini_trivial.instantiate(base_questions_p)
    print("Trivial result:", x)


@pytest.mark.asyncio
async def test_neg():
    mini_neg = Neg()
    x = await mini_neg.instantiate(base_questions_p)
    print("Neg result:", x)


@pytest.mark.asyncio
async def test_and():
    mini_and = And()
    x = await mini_and.instantiate(base_questions_pq)
    print("And result:", x)


@pytest.mark.asyncio
async def test_or():
    mini_or = Or()
    x = await mini_or.instantiate(base_questions_pq)
    print("Or result:", x)


@pytest.mark.asyncio
async def test_paraphrase():
    mini_para = Paraphrase()
    x = await mini_para.instantiate(base_questions_p)
    print("Paraphrase result:", x)


@pytest.mark.asyncio
async def test_conditional():
    mini_cond = Conditional()
    x = await mini_cond.instantiate(base_questions_pq)
    print("Conditional result:", x)


@pytest.mark.asyncio
async def test_consequence():
    mini_cons = Consequence()
    x = await mini_cons.instantiate(base_questions_p)
    print("Consequence result:", x)


if __name__ == "__main__":
    asyncio.run(test_trivial())
    asyncio.run(test_neg())
    asyncio.run(test_and())
    asyncio.run(test_or())
    asyncio.run(test_paraphrase())
    asyncio.run(test_conditional())
    asyncio.run(test_consequence())
