import pytest
from fq_generation.multi_to_binary import (
    reformat_metaculus_question,
    reformat_metaculus_question_sync,
)


@pytest.mark.asyncio
async def test_reformat_metaculus_question_with_change():
    title = "Who will win the 2024 U.S. presidential election? (Donald Trump)"
    body = None
    expected_title = "Will Donald Trump win the 2024 U.S. presidential election?"
    expected_body = None

    result = await reformat_metaculus_question(
        title, body, model="gpt-4o-mini-2024-07-18"
    )
    assert result["title"] == expected_title
    assert result["body"] == expected_body
    assert result["did_change"] is True


def test_reformat_metaculus_question_sync_with_change():
    title = "Who will win the 2024 U.S. presidential election? (Donald Trump)"
    body = None
    expected_title = "Will Donald Trump win the 2024 U.S. presidential election?"
    expected_body = None

    result = reformat_metaculus_question_sync(
        title, body, model="gpt-4o-mini-2024-07-18"
    )
    assert result["title"] == expected_title
    assert result["body"] == expected_body
    assert result["did_change"] is True


@pytest.mark.asyncio
async def test_reformat_metaculus_question_no_change():
    title = "Will the sun rise tomorrow?"
    body = None

    result = await reformat_metaculus_question(
        title, body, model="gpt-4o-mini-2024-07-18"
    )
    assert result["title"] == title
    assert result["body"] == body
    assert result["did_change"] is False


def test_reformat_metaculus_question_sync_no_change():
    title = "Will the sun rise tomorrow?"
    body = None

    result = reformat_metaculus_question_sync(
        title, body, model="gpt-4o-mini-2024-07-18"
    )
    assert result["title"] == title
    assert result["body"] == body
    assert result["did_change"] is False
