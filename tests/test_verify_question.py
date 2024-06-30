import pytest
import uuid
from datetime import datetime
from common.datatypes import ForecastingQuestion, VerificationResult
from question_generators.question_formatter import verify_question_all_methods


async def assert_verification_result(
    question: ForecastingQuestion, expected_valid: bool
):
    result = await verify_question_all_methods(question)
    assert isinstance(result, VerificationResult)
    assert result.valid == expected_valid


@pytest.mark.asyncio
async def test_verify_question_all_methods():
    # Test case 1: Valid question
    spacex_valid_q = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will SpaceX successfully land humans on Mars by 2030?",
        body="This question will resolve as Yes if SpaceX successfully lands at least one human on the surface of Mars before January 1, 2031. The landing must be confirmed by at least two reputable space agencies (e.g., NASA, ESA, Roscosmos) or through clear video evidence broadcast live from Mars.",
        resolution_date=datetime(2030, 12, 31),
        question_type="binary",
        data_source="metaculus",
    )
    await assert_verification_result(spacex_valid_q, True)

    # Test case 2: Invalid question (vague resolution criteria)
    ai_invalid_q = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will AI surpass human intelligence by 2050?",
        body="This question will resolve as Yes if AI becomes smarter than humans.",
        resolution_date=datetime(2050, 12, 31),
        question_type="binary",
        data_source="synthetic",
    )
    await assert_verification_result(ai_invalid_q, False)

    # Test case 3: Invalid question (inconsistent resolution date)
    inconsistent_date_question = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will the 2024 Olympics start as scheduled, on July 26?",
        body="This question will resolve as Yes if the opening ceremony of the 2024 Summer Olympics in Paris starts as originally scheduled on July 26, 2024, and finishes before sunrise in Paris on July 27.",
        resolution_date=datetime(
            2024, 12, 31
        ),  # Inconsistent date, should be close to the resolution date
        question_type="binary",
        data_source="synthetic",
    )
    await assert_verification_result(inconsistent_date_question, False)

    # Test case 4: Question with known smell
    ragnarok_question = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Ragnark Question Series: Will Fenrir devour Odin?",
        body="This question is part of the Ragnark Question Series. It will resolve as Yes if...",
        resolution_date=datetime(2100, 12, 31),
        question_type="binary",
        data_source="metaculus",
    )
    await assert_verification_result(ragnarok_question, False)

    # Test case 5: Question with resolution date in the past
    past_date_question = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will the UK leave the European Union?",
        body="This question will resolve as Yes if the United Kingdom formally leaves the European Union before January 1, 2021.",
        resolution_date=datetime(2020, 12, 31),
        question_type="binary",
        data_source="synthetic",
    )
    await assert_verification_result(past_date_question, True)
