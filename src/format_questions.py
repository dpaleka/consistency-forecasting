import asyncio
import json
from typing import List, Optional
from dataclasses import asdict
from common.datatypes import ForecastingQuestion
from question_generators import question_formater
from common.utils import write_jsonl_async


async def process_questions_from_file(file_path: str, data_source: str, max_questions: Optional[int]) -> List[ForecastingQuestion]:
    with open(file_path, 'r') as file:
        questions = json.load(file)
    
    max_questions = max_questions if max_questions else len(questions)
    tasks = []

    for question in questions[:max_questions]:
        task = asyncio.create_task(question_formater.from_string(question, data_source))
        tasks.append(task)

    forecasting_questions = await asyncio.gather(*tasks)
    return forecasting_questions

async def main():
    file_path = 'data/politics_qs_2.json'
    data_source = 'synthetic'
    max_questions = 5

    forecasting_questions = await process_questions_from_file(file_path, data_source, max_questions)

    data_to_write = [fq.dict() for fq in forecasting_questions]
    for data in data_to_write:
        data["id"] = str(data["id"])
        data["resolution_date"] = str(data["resolution_date"])

    await write_jsonl_async('data/politics_qs_2_formated.jsonl', data_to_write, append=False)


if __name__ == "__main__":
    asyncio.run(main())