import pytest
import uuid
from datetime import datetime
from common.datatypes import ForecastingQuestion, VerificationResult
from question_generators.question_formatter import (
    verify_question_all_methods,
    verify_question_llm,
)
import os
from dotenv import load_dotenv

load_dotenv()


model = "gpt-4o-2024-05-13"


async def assert_verification_result(
    question: ForecastingQuestion, expected_valid: bool, comment: str = ""
):
    result = await verify_question_all_methods(question, model=model)
    print(result)
    assert isinstance(result, VerificationResult)
    assert result.valid == expected_valid, f"Comment: {comment}"


pytest.mark.expensive = pytest.mark.skipif(
    os.getenv("TEST_FQ_VERIFICATION", "True").lower() == "false",
    reason="Skipping expensive verification tests",
)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_spacex_valid_question():
    spacex_valid_q = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will SpaceX successfully land humans on Mars by 2030?",
        body="This question will resolve as Yes if SpaceX successfully lands at least one human on the surface of Mars before January 1, 2031. The landing must be confirmed by at least two reputable space agencies (e.g., NASA, ESA, Roscosmos) or through clear and untampered video evidence broadcast live from Mars.",
        resolution_date=datetime(2030, 12, 31),
        question_type="binary",
        data_source="metaculus",
    )
    await assert_verification_result(spacex_valid_q, True)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_ai_invalid_question():
    ai_invalid_q = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will AI surpass human intelligence by 2050?",
        body="This question will resolve as Yes if AI becomes smarter than humans.",
        resolution_date=datetime(2050, 12, 31),
        question_type="binary",
        data_source="synthetic",
    )
    await assert_verification_result(ai_invalid_q, False)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_inconsistent_date_question():
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


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_past_invalid_question():
    past_invalid_question = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will the UK leave the European Union?",
        body="This question will resolve as Yes if the United Kingdom formally leaves the European Union before January 1, 2021.",
        resolution_date=datetime(2020, 12, 31),
        question_type="binary",
        data_source="synthetic",
    )
    # Not sure what to do with this one
    # await assert_verification_result(past_invalid_question, False)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_nobel_invalid_question():
    nobel_invalid_q = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will the discovery of dark matter lead to a Nobel Prize in Physics that significantly impacts the global economy by 2040?",
        body="This question will resolve as Yes if, by December 31, 2040, any of the following conditions are met:\n"
        "A Nobel Prize in Physics is awarded for the discovery of dark matter, and within five years of the award, "
        "there is a measurable and significant impact on the global economy as reported by at least three major "
        "financial institutions or economic research bodies. Significant impact is defined as a change in global GDP "
        "by at least 0.5%% attributable directly to technologies or innovations derived from the discovery of "
        "dark matter.\n"
        "A Nobel Prize in Physics is awarded for the discovery of dark matter, and it leads to the development "
        "of a new industry or the significant expansion of an existing industry, with a global market value exceeding "
        "$100 billion within five years of the award.\n"
        "In the absence of a Nobel Prize, if an equivalent prestigious international award is given for the "
        "discovery of dark matter and meets the above economic criteria, it will also be considered for resolution.\n"
        "Fallback scenario: If there is controversy or ambiguity regarding the discovery of dark matter or its "
        "economic impact, a panel of experts from leading scientific and economic institutions will be convened to "
        "make a determination based on available evidence and expert consensus.",
        resolution_date=datetime(2040, 12, 31),
        question_type="binary",
        data_source="synthetic",
    )
    await assert_verification_result(
        nobel_invalid_q,
        False,
        comment="We don't want multiple different main pathways to Yes resolution for 'Will event X happen' in resolution criteria. It's better to simplify, have one key variable that resolves the question, and then patch any edge cases.\n"
        "'In the absence of a Nobel Prize, if an equivalent prestigious international award is given for the discovery of dark matter and meets the above economic criteria, it will also be considered for resolution.' is just wrong.\n"
        "Too many resolution criteria, and the Nobel Prize one contradicts the title.",
    )


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_republican_valid_question():
    republican_valid_q = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will the Republican Party propose an increase of at least 20%% in NASA's budget for space exploration in their 2024 platform?",
        body="The question will resolve as Yes if the Republican Party's official platform for the 2024 United States "
        "presidential election includes a proposal that specifies an increase in NASA's budget for space exploration "
        "by at least 20%% compared to the fiscal year 2023 budget. The budget increase must be explicitly allocated "
        "for space exploration activities, including but not limited to manned and unmanned missions, research and "
        "development of space technologies, and exploration of the Moon, Mars, or other celestial bodies.\n\nThe "
        "resolution will rely on the official platform document released by the Republican Party, as well as any "
        "official statements or press releases from the party's leadership. In the event of ambiguity or lack of "
        "clarity in the official documents, statements from the chairperson of the Republican National Committee "
        "(RNC) or the party's presidential nominee will be considered authoritative.\n\n"
        "If the party's platform is not publicly available by October 1, 2024, or if the platform does not "
        "explicitly address NASA's budget for space exploration, the question will resolve as No.",
        resolution_date=datetime(2024, 10, 1),
        question_type="binary",
        data_source="synthetic",
    )
    await assert_verification_result(republican_valid_q, True)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_building_valid_question():
    building_valid_q = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will there be a breakthrough in engineering that allows for the construction of a building over 2,000 meters tall by 2030?",
        body="This question will resolve as Yes if, by December 31, 2030, there is a verifiable report or announcement by a recognized authority in the field of engineering or architecture (such as the Council on Tall Buildings and Urban Habitat or a similar authoritative body) that a building measuring over 2,000 meters in height has been built. If construction is still underway at the time of resolution, this resolves Yes if the height built so far (as reported by the aforementioned bodies) is over 2,000 meters at the time of resolution, and No otherwise. In the event of conflicting reports, the resolution will be based on the preponderance of evidence from credible sources.",
        resolution_date=datetime(2030, 12, 31, 23, 59, 59),
        question_type="binary",
        data_source="synthetic",
    )
    await assert_verification_result(building_valid_q, True)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_market_probability_in_body():
    market_prob_question = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will SpaceX successfully land humans on Mars by 2030?",
        body="This question will resolve as Yes if SpaceX successfully lands at least one human on the surface of Mars before January 1, 2031. The current market probability for this event is 30%.",
        resolution_date=datetime(2030, 12, 31),
        question_type="binary",
        data_source="synthetic",
    )
    await assert_verification_result(
        market_prob_question,
        False,
        comment="The body should not contain market probabilities.",
    )


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_spacex_invalid_title():
    spacex_valid_q = ForecastingQuestion(
        id=uuid.uuid4(),
        title="SpaceX is cool",
        body="This question will resolve as Yes if SpaceX successfully lands at least one human on the surface of Mars before January 1, 2031. The landing must be confirmed by at least two reputable space agencies (e.g., NASA, ESA, Roscosmos) or through clear and untampered video evidence broadcast live from Mars.",
        resolution_date=datetime(2030, 12, 31),
        question_type="binary",
        data_source="metaculus",
    )
    await assert_verification_result(spacex_valid_q, False)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_spacex_invalid_nonrelevant_body():
    spacex_valid_q = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will SpaceX successfully land humans on Mars by 2030?",
        body="This question will resolve as Yes if Trump successfully lands at least one human on the surface of Mars before January 1, 2031. The landing must be confirmed by at least two reputable space agencies (e.g., NASA, ESA, Roscosmos) or through clear and untampered video evidence broadcast live from Mars.",
        resolution_date=datetime(2030, 12, 31),
        question_type="binary",
        data_source="metaculus",
    )
    await assert_verification_result(spacex_valid_q, False)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_spacex_invalid_date():
    spacex_valid_q = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will SpaceX successfully land humans on Mars by 2030?",
        body="This question will resolve as Yes if SpaceX successfully lands at least one human on the surface of Mars before January 1, 2031. The landing must be confirmed by at least two reputable space agencies (e.g., NASA, ESA, Roscosmos) or through clear and untampered video evidence broadcast live from Mars.",
        resolution_date=datetime(2070, 12, 31),
        question_type="binary",
        data_source="metaculus",
    )
    await assert_verification_result(spacex_valid_q, False)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_verify_ai_person_of_the_year_valid_question():
    ai_person_of_the_year_valid_question = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will an AI win TIME's Person of the Year by 2030?",
        body="This question will resolve as Yes if an artificial intelligence system (or a group of AI systems, or a group of entities at least one of which is an AI system) wins TIME's Person of the Year by 31 December 2030. In case TIME's Person of the Year stops existing before resolution is triggered, the question will resolve as No. The entity needs to be described by TIME as an AI system.",
        resolution_date=datetime(2030, 12, 31),
        question_type="binary",
        data_source="synthetic",
    )
    await assert_verification_result(ai_person_of_the_year_valid_question, True)


@pytest.mark.asyncio
async def test_verify_invalid_title_single_call(mocker):
    """
    Test that the verify_question_llm function calls verify_title before verify_body,
    and fails immediately if the title is invalid.
    """
    invalid_title_question = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Will an AI win TIME's Person of the Year by 2030?",
        body="This question will resolve as Yes if an artificial intelligence system (or a group of AI systems, or a group of entities at least one of which is an AI system) wins TIME's Person of the Year by 31 December 2030. In case TIME's Person of the Year stops existing before resolution is triggered, the question will resolve as No. The entity needs to be described by TIME as an AI system.",
        resolution_date=datetime(2030, 12, 31),
        question_type="binary",
        data_source="synthetic",
    )

    # Mock the verify_title function
    mock_verify_title = mocker.patch(
        "question_generators.question_formatter.verify_title",
        new_callable=mocker.AsyncMock,
    )
    mock_verify_title.return_value = VerificationResult(
        valid=False, reasoning="Invalid title"
    )

    mock_verify_body = mocker.patch(
        "question_generators.question_formatter.verify_body",
        new_callable=mocker.AsyncMock,
    )
    mock_verify_body.return_value = VerificationResult(
        valid=True, reasoning="Valid body"
    )

    # Call the function
    result = await verify_question_llm(invalid_title_question, datetime.now())

    # Assert that the result is as expected
    assert result.valid is False
    assert "Title validation failed" in result.reasoning

    # Assert that verify_title was called only once
    mock_verify_title.assert_called_once()

    # Assert that verify_body was not called
    mock_verify_body.assert_not_called()
