import os
import asyncio
import argparse
from common.path_utils import get_data_path
from common.utils import write_jsonl_async_from_str
from common.datatypes import QuestionGenerationResponse
from question_generators.utils import deduplicate
import json
from common.llm_utils import answer


source_questions = [
    (
        "Will global average temperatures rise by more than 2 degrees Celsius above pre-industrial levels by 2040?"
    ),
    ("Will the United States have a female president before 2028?"),
    ("Will the World Health Organization declare a new pandemic before 2025?"),
    ("Will a team from Africa win the FIFA World Cup before 2040?"),
    ("Will quantum computing be commercially available to the public before 2030?"),
    ("Will NASA discover definitive evidence of life on another planet by 2035?"),
]


prompt = """Objective: Generate a set of forecasting questions for a forecasting market site like Metaculus or PredictIt. I will provide a source question. 
Your task is to generate {num_questions} new related questions that are logically related to the provided source question. 
Each new question should be suitable for probabilistic evaluation and should logically combine with the source question in a meaningful way. 

Guidelines:
- The new questions should explore related scenarios, alternate outcomes, consequences and prerequisites of the source question.
- Consider alternate outcomes, timelines, or deeper implications that are still connected to the theme of the source question.
- Each question should be binary and can be answered with a probability between 0 and 1. 

The source question will optionally include a body (detailed resolution criteria). If the source question has a body, use it to inform the generation of related questions.
You still need to generate only single sentences, not detailed resolution criteria.

Example 1:
---
Source question: Will Europa be the first place humanity will discover extraterrestrial life, if it is discovered before 2045? 
Source question body: This question will resolve as Yes if humanity discovers convincing evidence of life on Europa before January 1, 2045 and does so before detecting extra terrestrial life anywhere else in the universe. If extraterrestrial life is found convincingly elsewhere prior to on Europa, this question will resolve as No. If no extraterrestrial life is found before 2045, this question will be Annulled.
Note: extraterrestrial life must be (a) living currently and (b) highly unlikely to be a result of contamination by Earth spacecraft. This leaves open the possibility of life transported from Earth via other non-human-engineered means.

=> Related questions:
- Will we find life on Mars by 2050?
- Will a sample-return mission from Europa confirm the presence of life by 2050?
- Will life be discovered on Enceladus before 2045?
---


Example 2 (without body):
---
Source question: Will North Korea engage in a significant diplomatic negotiation with a Western country by 2025?

=> Related questions:
- Will North Korea significantly reform its legal system by 2030?
- Will North Korea join an international climate change agreement by 2030?
- Will North Korea open its borders for international tourism by 2030?
---

Now generate {num_questions} related questions for the source question:

Source question: {source_question}
{body_prompt}

=> Related questions:
"""

body_prompt = """
Source question body: {source_body}
"""


# class QuestionGenerationResponse(BaseModel):
#     question_1: SyntheticTagQuestion
#     question_2: SyntheticTagQuestion
#     question_3: SyntheticTagQuestion


def load_questions_from_jsonl(file_path):
    if not os.path.exists(file_path):
        return {}

    questions_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            json_data = json.loads(line)
            category = json_data["category"]
            question_tuple = (
                json_data["title"],
                json_data["category"],
                json_data["tags"],
            )
            if category not in questions_dict:
                questions_dict[category] = [question_tuple]
            else:
                questions_dict[category].append(question_tuple)

    return questions_dict


def get_titles_from_fq(file_path, use_body=False):
    titles = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                if "title" in data:
                    if use_body:
                        titles.append({"title": data["title"], "body": data["body"]})
                    else:
                        titles.append({"title": data["title"]})
    except Exception as e:
        print(f"An error occurred: {e}")
    return titles


async def generate_questions_from_question(
    source_question, model, num_questions, source_body=None
):
    question_prompt = prompt.format(
        source_question=source_question,
        num_questions=num_questions,
        body_prompt=body_prompt.format(source_body=source_body) if source_body else "",
    )

    generated_questions = await answer(
        prompt=question_prompt,
        preface=None,
        response_model=QuestionGenerationResponse,
        model=model,
    )

    # Ensure each generated question has the source question field populated
    for question in generated_questions.questions:
        question.source_question = source_question

    return generated_questions.questions
    # question_prompt = prompt.format(source_question=source_question)
    # generated_questions = await answer(
    #     prompt=question_prompt,
    #     preface=None,
    #     response_model=QuestionGenerationResponse,
    #     model=model,
    # )

    # # print(generated_questions)

    # # Fill in the source_question manually if it's not provided by the LLM
    # for field_name, synthetic_question in generated_questions.model_dump().items():
    #     print(synthetic_question)
    #     # Instantiate SyntheticRelQuestion from the dictionary
    #     synthetic_question = SyntheticRelQuestion.model_validate(synthetic_question)
    #     # Check if the field contains a SyntheticRelQuestion by re-instantiating it from the data
    #     synthetic_question.source_question = source_question
    #     # Set the updated SyntheticRelQuestion back to the original response object
    #     setattr(generated_questions, field_name, synthetic_question)

    # return [
    #     q
    #     for q in [
    #         generated_questions.question_1,
    #         generated_questions.question_2,
    #         generated_questions.question_3,
    #     ]
    # ]


async def generate_questions(
    input_file, output_file, model, max_questions, questions_per_source, use_body
):
    questions = get_titles_from_fq(input_file, use_body)

    print(f"len of questions: {len(questions)}")

    if len(questions) == 0:
        questions = source_questions

    questions = questions[:max_questions]

    tasks = [
        generate_questions_from_question(
            question["title"],
            source_body=question["body"] if use_body else None,
            model=model,
            num_questions=questions_per_source,
        )
        for question in questions
    ]
    results = await asyncio.gather(*tasks)
    generated_questions = [item for sublist in results for item in sublist]

    print(f"len of generated questions: {len(generated_questions)}")

    deduplicated_questions = await deduplicate(generated_questions)
    print(
        f"len deduplicated questions: {len(deduplicated_questions)}, len of generated questions: {len(generated_questions)}"
    )
    deduplicated_questions = [q.model_dump_json() for q in deduplicated_questions]

    # add output parameter
    await write_jsonl_async_from_str(output_file, deduplicated_questions, append=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        default=get_data_path() / "fq" / "real" / "questions_cleaned_formatted.jsonl",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=get_data_path() / "other" / "from_related.jsonl",
    )
    parser.add_argument("--model", "-m", type=str, default="gpt-4-0125-preview")
    parser.add_argument("--n_max_questions", "-n", type=int, default=50)
    parser.add_argument("--questions_per_source", "-q", type=int, default=5)
    parser.add_argument("--use_body", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        generate_questions(
            input_file=args.input_file,
            output_file=args.output_file,
            model=args.model,
            max_questions=args.n_max_questions,
            questions_per_source=args.questions_per_source,
            use_body=args.use_body,
        )
    )