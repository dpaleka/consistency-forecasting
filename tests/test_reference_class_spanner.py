"""
Test individual outputs and overall pipline for NewsAPI -> FQs
"""

import sys
from common.path_utils import get_src_path

sys.path.append(str(get_src_path()))

import pytest
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from common.datatypes import ForecastingQuestion
from generate_fqs_using_reference_class import BinaryFQReferenceClassSpanner

load_dotenv()

sample_forecasting_question_dicts = [
    {
        "id": "18b81ba1-bd49-48c2-874b-48cd149f6eb2",
        "title": "What is the probability that a woman will be elected as the President of the United States between 2024 and 2030?",
        "body": "This question will resolve as Yes if, between January 20, 2025, and January 20, 2031, a woman is inaugurated as the President of the United States. The resolution will be based on the official inauguration records provided by the U.S. National Archives or any other authoritative governmental source. In the event of a dispute regarding the election outcome, the question will resolve based on the decision of the U.S. Supreme Court. If a woman assumes the presidency without an inauguration due to unforeseen circumstances (e.g., succession following the incapacitation or death of a sitting president), this will also result in a Yes resolution, provided that she serves as president for at least 24 hours.",
        "resolution_date": "2023-11-05T00:00:00Z",
        "question_type": "binary",
        "data_source": "synthetic",
        "url": None,
        "metadata": None,
        "resolution": None,
    },
    {
        "id": "5d8a3198-dd56-4a7b-ac3b-b464cadc94e4",
        "title": "What is the probability that the United Kingdom will have a new Prime Minister by January 1, 2028?",
        "body": "This question will resolve as Yes if, by January 1, 2028, an individual other than the current Prime Minister as of the date of this question's creation is officially serving as the Prime Minister of the United Kingdom. The change in leadership must be confirmed by an official announcement from the UK government or a reputable news source such as the BBC. Interim or acting Prime Ministers will count towards resolution only if they are officially carrying out the duties of the Prime Minister and are not merely temporary placeholders during a brief transition period. In the event of a dispute over the legitimacy of the leadership change, the resolution will be based on the recognition by the UK Parliament. If the current Prime Minister is re-elected or re-appointed after a general election or a party leadership contest, this will not count as a 'new' Prime Minister for the purposes of this question.",
        "resolution_date": "2028-01-01T00:00:00Z",
        "question_type": "binary",
        "data_source": "synthetic",
        "url": None,
        "metadata": None,
        "resolution": None,
    },
    {
        "id": "114dda02-c4ca-432b-99a3-0d6687c9a55e",
        "title": "What is the probability that Angela Merkel will no longer be the Chancellor of Germany by the end of 2028?",
        "body": "This question will resolve as Yes if Angela Merkel is not serving as the Chancellor of Germany at any point from the start of the year 2028 until the end of the year 2028. The resolution will be based on official statements from the German Federal Government or credible news sources confirming that Angela Merkel has ceased to hold the office of Chancellor, whether due to resignation, impeachment, term limits, or any other reason. In the event of temporary incapacitation or delegation of duties without formal resignation, the question will not resolve as Yes unless the incapacitation or delegation spans the entirety of the year 2028. If Angela Merkel's status as Chancellor is contested or unclear due to political or legal disputes, the question will resolve based on the de facto control of the office as recognized by the majority of EU member states.",
        "resolution_date": "2028-12-31T00:00:00Z",
        "question_type": "binary",
        "data_source": "synthetic",
        "url": None,
        "metadata": None,
        "resolution": None,
    },
]


@pytest.mark.asyncio
async def test_reference_class_spanned_questions():
    num_spanned_questions = 7
    tasks = []
    for source_fq_dict in sample_forecasting_question_dicts:
        source_fq = ForecastingQuestion(**source_fq_dict)
        tasks.append(
            BinaryFQReferenceClassSpanner.generate_spanned_fqs(
                source_fq,
                "openai/gpt-4o-2024-08-06",
                num_spanned_questions,
                datetime(2024, 7, 1),
                "basic",
            )
        )

    results = await asyncio.gather(*tasks)

    for result in results:
        assert len(result) >= num_spanned_questions


if __name__ == "__main__":
    asyncio.run(pytest.main([__file__]))
