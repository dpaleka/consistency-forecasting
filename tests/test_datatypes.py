import sys
import os
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from common.datatypes import ForecastingQuestion, Prob
import pytest


def test_prob_valid():
    assert Prob(prob=0.0).prob == 0.0
    assert Prob(prob=1.0).prob == 1.0
    assert Prob(prob=0.5).prob == 0.5


def test_prob_invalid():
    with pytest.raises(ValueError):
        Prob(prob=-0.1)
    with pytest.raises(ValueError):
        Prob(prob=1.1)


def test_forecasting_question_creation():
    fq = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Test Title",
        body="Test Body",
        question_type="binary",
        resolution_date=datetime(2024, 1, 1),
        data_source="synthetic",
        url="http://example.com",
        metadata={"topics": ["test"]},
        resolution=True,
    )
    assert str(fq.id) != "test_id"  # id should be a valid UUID
    assert fq.title == "Test Title"
    assert fq.question_type == "binary"
    assert fq.resolution_date == datetime(2024, 1, 1)
    assert fq.data_source == "synthetic"
    assert fq.url == "http://example.com"
    assert fq.metadata == {"topics": ["test"]}
    assert fq.resolution is True  # resolution should be a boolean


def test_to_dict():
    fq = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Test Title",
        body="Test Body",
        question_type="binary",
        resolution_date=datetime(2024, 1, 1),
        data_source="synthetic",
        url="http://example.com",
        metadata={"topics": ["test"]},
        resolution=True,
    )
    expected_dict = {
        "id": str(fq.id),  # id should be a valid UUID
        "title": "Test Title",
        "body": "Test Body",
        "resolution_date": fq.resolution_date.isoformat(),
        "question_type": "binary",
        "data_source": "synthetic",
        "url": "http://example.com",
        "metadata": {"topics": ["test"]},
        "created_date": None,
        "resolution": True,  # resolution should be a boolean
    }
    dumped_dict = fq.model_dump()
    dumped_dict["id"] = str(dumped_dict["id"])  # Ensure 'id' is a string
    dumped_dict["resolution_date"] = dumped_dict[
        "resolution_date"
    ].isoformat()  # Ensure 'resolution_date' is a string
    assert dumped_dict == expected_dict
