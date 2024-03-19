from datetime import datetime
from common.datatypes import ForecastingQuestion, QuestionType, Prob
import pytest


def test_prob_valid():
    assert Prob(0.0) == 0.0
    assert Prob(1.0) == 1.0
    assert Prob(0.5) == 0.5

def test_prob_invalid():
    with pytest.raises(ValueError):
        Prob(-0.1)
    with pytest.raises(ValueError):
        Prob(1.1)

def test_forecasting_question_creation():
    fq = ForecastingQuestion(
        id="test_id",
        title="Test Title",
        body="Test Body",
        question_type=QuestionType("binary"),
        resolution_date=datetime(2024, 1, 1),
        data_source="synthetic",
        url="http://example.com",
        metadata={"topics": ["test"]},
        resolution="resolved"
    )
    assert fq.id == "test_id"
    assert fq.title == "Test Title"
    assert fq.question_type == "binary"
    assert fq.resolution_date == datetime(2024, 1, 1)
    assert fq.data_source == "synthetic"
    assert fq.url == "http://example.com"
    assert fq.metadata == {"topics": ["test"]}
    assert fq.resolution == "resolved"

def test_to_dict():
    fq = ForecastingQuestion(
        id="test_id",
        title="Test Title",
        body="Test Body",
        question_type=QuestionType("binary"),
        resolution_date=datetime(2024, 1, 1),
        data_source="synthetic",
        url="http://example.com",
        metadata={"topics": ["test"]},
        resolution="resolved"
    )
    expected_dict = {
        "id": "test_id",
        "title": "Test Title",
        "body": "Test Body",
        "resolution_date": "2024-01-01T00:00:00",
        "question_type": "binary",
        "data_source": "synthetic",
        "url": "http://example.com",
        "metadata": {"topics": ["test"]},
        "resolution": "resolved"
    }
    assert fq.to_dict() == expected_dict
