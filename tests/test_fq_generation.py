import sys
from common.path_utils import get_src_path

sys.path.append(str(get_src_path()))


import pytest
import asyncio
from datetime import datetime
from uuid import UUID
from fq_generation.fq_body_generator import from_string
from common.utils import normalize_date_format
from common.datatypes import ForecastingQuestion


@pytest.mark.asyncio
async def test_from_string_basic():
    question = "Will AI surpass human intelligence by 2030?"
    data_source = "manifold"
    result = await from_string(question, data_source, fill_in_body=True)

    assert isinstance(result, ForecastingQuestion)
    assert result.title == question
    assert result.data_source == data_source
    assert isinstance(result.id, UUID)
    assert isinstance(result.resolution_date, datetime)
    assert result.question_type == "binary"
    assert result.body is not None
    assert result.url is None
    assert result.metadata is None
    assert result.resolution is None


@pytest.mark.asyncio
async def test_from_string_with_optional_params():
    question = "Will SpaceX launch Starship to Mars by 2025?"
    data_source = "manifold"
    url = "https://example.com"
    metadata = {"category": "space"}
    body = "This question resolves positively if SpaceX successfully launches Starship to Mars by December 31, 2025."
    date = "31/12/2025"

    result = await from_string(
        question,
        data_source,
        url=url,
        metadata=metadata,
        body=body,
        date=date,
        question_type="binary",
        fill_in_body=False,
    )

    assert isinstance(result, ForecastingQuestion)
    assert result.title == question
    assert result.data_source == data_source
    assert result.url == url
    assert result.metadata == metadata
    assert result.body == body
    assert result.resolution_date == datetime(2025, 12, 31)
    assert result.question_type == "binary"


@pytest.mark.asyncio
async def test_from_string_fill_in_body():
    question = "Will quantum computers break RSA-2048 encryption by 2030?"
    data_source = "manifold"

    result = await from_string(question, data_source, fill_in_body=True)

    assert isinstance(result, ForecastingQuestion)
    assert result.body is not None
    assert len(result.body) > 0


@pytest.mark.asyncio
async def test_from_string_no_body_no_fill():
    question = "Will fusion energy become commercially viable by 2040?"
    data_source = "manifold"

    with pytest.raises(
        ValueError, match="No question body provided and fill_in_body is False"
    ):
        await from_string(question, data_source, fill_in_body=False)


@pytest.mark.asyncio
async def test_from_string_date_dmy():
    question = "Will we colonize Mars by 2025?"
    data_source = "manifold"
    date_to_be_dmy = "12/07/2025"

    result = await from_string(
        question, data_source, date=date_to_be_dmy, fill_in_body=True
    )

    assert isinstance(result, ForecastingQuestion)
    assert isinstance(result.resolution_date, datetime)
    assert normalize_date_format(date_to_be_dmy) == datetime.strptime(
        date_to_be_dmy, "%d/%m/%Y"
    )
    assert result.resolution_date == datetime.strptime(date_to_be_dmy, "%d/%m/%Y")
    assert result.resolution_date != datetime.strptime(date_to_be_dmy, "%m/%d/%Y")


if __name__ == "__main__":
    asyncio.run(pytest.main([__file__]))
