import os
import asyncio
import argparse
from common.path_utils import get_data_path
from common.utils import write_jsonl_async_from_str
from common.datatypes import QuestionGenerationResponse3
from fq_generation.utils import deduplicate
import json
from common.llm_utils import answer
import random
from datetime import datetime


# If not None, this will be the target resolution date for the questions.
# It also changes the resolution date in some of the initail questions if its before 2030.
# Eg: for a very long term question, it may be better to add new questions.
TARGET_RESOLUTION_DATE = None  # "2030-01-01 00:00:00"

target_resolution_date = (
    datetime(2030, 1, 1)
    if TARGET_RESOLUTION_DATE is None
    else datetime.strftime(TARGET_RESOLUTION_DATE)
)

example_datetime = (
    "2030-01-01"
    if target_resolution_date > datetime(2030, 1, 1)
    else target_resolution_date.strftime("%Y-%m-%d")
)


all_categories = [
    "Artificial Intelligence",
    "Computing and Math",
    "Cryptocurrencies",
    "Economy & Business",
    "Elections",
    "Environment & Climate",
    "Geopolitics",
    "Health & Pandemics",
    "Law",
    "Natural Sciences",
    "Nuclear Technology & Risks",
    "Politics",
    "Social Sciences",
    "Space",
    "Sports & Entertainment",
    "Technology",
]

topics = [
    "United States",
    "Medicine",
    "Virology",
    "COVID-19",
    "China",
    "Russia",
    "Physics",
    "Epidemiology",
    "United Kingdom",
    "Donald Trump",
    "Ukraine",
    "Machine Learning",
    "European Union",
    "Joe Biden",
    "Virginia",
    "Immunology",
    "OpenAI",
    "Centers for Disease Control and Prevention",
    "Environmental Science",
    "Biology",
    "Infectious Disease",
    "Israel",
    "Genetics",
    "Democratic Party (US)",
    "Republican Party (US)",
    "Astronomy",
    "Biotechnology",
    "Food and Drug Administration",
    "Elon Musk",
    "Russo-Ukrainian War",
    "India",
    "COVID-19 Vaccine",
    "Chemistry",
    "Scott Alexander",
    "World Health Organization",
    "Philosophy",
    "SpaceX Reusable Launch System",
    "NASA",
    "Information Technology",
    "Bureau of Labor Statistics",
    "US dollar",
    "Energy",
    "Taiwan",
    "Engineering",
    "SpaceX",
    "Microbiology",
    "France",
    "Germany",
    "Climate change denial",
    "US Senate",
    "Ethics",
    "Iran",
    "2020 US presidential election",
    "DeepMind",
    "Robotics",
    "Supreme Court of the US",
    "California",
    "2024 US Elections",
    "US House of Representatives",
    "Atmospheric science",
    "NATO",
    "Pharmacology",
    "Google",
    "North Korea",
    "Amazon",
    "US Congress",
    "Microsoft",
    "Aerospace engineering",
    "Climatology",
    "United Nations",
    "Virginia Department of Health",
    "Japan",
    "Tesla",
    "Spain",
    "Switzerland",
    "Italy",
    "Brazil",
    "Australia",
    "Canada",
    "Mexico",
    "South Korea",
    "Sweden",
    "Norway",
    "Particle physics",
    "String theory",
    "Black holes",
    "Dark matter",
    "Quantum computing",
    "Basketball",
    "Soccer",
    "Cricket",
    "Tennis",
    "Baseball",
    "Golf",
    "Formula 1",
    "Olympics",
    "FIFA",
    "NBA",
    "NFL",
    "MLB",
    "NHL",
]

initial_questions = [
    (
        "Will SpaceX successfully land at least one crewed spacecraft on the surface of Mars before January 1, 2030?",
        "Space",
        "SpaceX,Mars",
    ),
    (
        "Will any country from the following list: Belgium, France, Italy, Luxembourg, Netherlands, or Germany, officially withdraw its membership from the European Union before January 1, 2027?",
        "Geopolitics",
        "European Union",
    ),
    (
        "Will the global human population fall below 1 billion individuals for any reason before January 1, 2100?",
        "Natural Sciences",
        "Extinction",
    ),
    (
        "Will the net worth of the world's richest individual, as reported by Forbes or Bloomberg, be equal to or greater than 2% of the United States' GDP, as reported by the World Bank, on December 31, 2033?",
        "Economy & Business",
        "Wealth",
    ),
    (
        "Will an artificial intelligence system pass a formally administered Turing Test, as judged by a panel of at least 5 experts in the field, before January 1, 2026?",
        "Artificial Intelligence",
        "AI,Turing Test",
    ),
    (
        "Will the daily transaction volume of any cryptocurrency, as reported by CoinMarketCap or a similar reputable source, exceed that of the US dollar for at least 7 consecutive days before {example_datetime}?",
        "Cryptocurrencies",
        "Bitcoin,US dollar",
    ),
    (
        "Will the 5-year average global temperature, as reported by NASA's Goddard Institute for Space Studies, exceed pre-industrial levels by more than 2 degrees Celsius at any point before January 1, 2041?",
        "Environment & Climate",
        "Global warming",
    ),
    (
        "Will a woman be sworn in as President of the United States before {example_datetime}?",
        "Elections",
        "US Politics,Presidential Election",
    ),
    (
        "Will the World Health Organization officially declare a Public Health Emergency of International Concern (PHEIC) for a novel pathogen before January 1, 2026?",
        "Health & Pandemics",
        "WHO,Pandemic",
    ),
    (
        "Will India's annual GDP, as reported by the World Bank, exceed China's annual GDP for at least one calendar year before January 1, 2041?",
        "Economy & Business",
        "India,China,GDP",
    ),
    (
        "Will an international treaty specifically addressing cyber warfare be ratified by at least 50 UN member states before {example_datetime}?",
        "Law",
        "Cyber warfare",
    ),
    (
        "Will a men's national football team from an African country win the FIFA World Cup before January 1, 2041?",
        "Sports & Entertainment",
        "FIFA,World Cup",
    ),
    (
        "Will at least one company offer commercially available quantum computing services to the general public, with a minimum of 1000 qubits, before {example_datetime}?",
        "Technology",
        "Quantum computing",
    ),
    (
        "Will NASA or any other space agency publicly announce the discovery of fossilized or living microorganisms on another planet, moon, or asteroid, confirmed by peer-reviewed studies, before January 1, 2036?",
        "Space",
        "NASA,Extraterrestrial life",
    ),
    (
        "Will Russia and Ukraine sign and ratify a comprehensive peace treaty, ending all ongoing military conflicts between the two nations, before January 1, 2026?",
        "Geopolitics",
        "Russia,Ukraine",
    ),
    (
        "Will a new infectious disease with a confirmed case fatality rate of over 10%, based on WHO data from at least 1000 cases, emerge and spread to at least 3 countries before {example_datetime}?",
        "Health & Pandemics",
        "Infectious Disease",
    ),
    (
        "Will the total number of operational satellites in Earth orbit, as tracked by the United Nations Office for Outer Space Affairs, exceed 50,000 before {example_datetime}?",
        "Technology",
        "Satellites",
    ),
    (
        "Will any G7 country pass legislation allowing genetic editing of human embryos for non-medical trait enhancement before {example_datetime}?",
        "Law",
        "Genetic editing",
    ),
    (
        "Will the global mean sea level, as measured by satellite altimetry and reported by NASA or NOAA, rise by more than 10 centimeters above the 2020 average level before {example_datetime}?",
        "Environment & Climate",
        "Sea level rise",
    ),
]

resolution_message = (
    ""
    if TARGET_RESOLUTION_DATE is None
    else f"The question should resolve on the date {example_datetime}"
)

# Prompt for generating forecasting questions
prompt = """
You are tasked with generating forecasting questions for a platform similar to Metaculus or PredictIt.
Based on the provided category and tags, create questions that can be answered with a probability between 0 and 1.
Be precise and avoid vague language or qualifiers. Ensure each question has a clear resolution criteria and timeframe.

Tip 1: For each tag, generate a relevant question if the tag is pertinent to the category. If the tag is not relevant, generate a general question about the category.

Tip 2: Avoid using qualifiers like "significant" or "substantial". Instead, use specific, measurable criteria.

Tip 3: Include a clear anchor for quality or quantity in each question. For example, instead of asking if something will "improve", ask if it will "increase by at least X%".

Tip 4: Specify a clear timeframe for each question. Use precise dates or periods rather than vague terms like "soon" or "in the near future".

Tip 5: Ensure that the resolution criteria for each question are unambiguous and can be objectively determined.

Tip 6: If appropriate, include specific numerical thresholds or ranges in your questions to make them more precise.

Tip 7: Consider including a reliable source for resolving the question, such as a specific government agency or reputable organization.

Provide your generated questions in the following XML structure:

<forecasting_questions>
  <question1>
    <title>[Insert concise question title here]</title>
    <details>[Provide detailed resolution criteria, including specific metrics, thresholds, and timeframes]</details>
    <resolution_date>[Specify the date by which the question should be resolved]</resolution_date>
  </question1>
  <question2>
    [Repeat structure for additional questions]
  </question2>
  [Add more question elements as needed]
</forecasting_questions>

Ensure each question is well-formed, precise, and adheres to the principles of good forecasting questions.

<example>
{example_1}
</example>

<example>
{example_2}
</example>

<example>
{example_3}
</example>

<example>
{example_4}
</example>

<example>
{example_5}
</example>

<example>
{example_6}
</example>

Generate forecasting questions based on the following category and tags:

Category: {category}
Tags: {tags}
"""


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


def get_example_question(initial_questions, questions):
    if random.randint(0, 1) or len(questions) == 0:
        return random.choice(initial_questions)
    else:
        return random.choice(questions)


async def generate_questions_for_category(
    initial_questions, questions_dict, model, categories=None
):
    if categories is None:
        categories = all_categories
    category = random.choice(categories)
    tags = random.sample(topics, 3)

    example_1, example_2 = random.sample(initial_questions, 2)

    if (
        category in questions_dict
        and len(questions_dict[category]) >= 2
        and len(categories) > 1
    ):
        example_5, example_6 = random.sample(questions_dict[category], 2)
        other_categories = [cat for cat in questions_dict if cat != category]
        cat1, cat2 = random.sample(other_categories, 2)
        example_3 = random.choice(questions_dict[cat1])
        example_4 = random.choice(questions_dict[cat2])
    elif len(questions_dict) == 0:
        (
            example_1,
            example_2,
            example_3,
            example_4,
            example_5,
            example_6,
        ) = random.sample(initial_questions, 6)
    else:
        chosen_categories = random.choices(list(questions_dict.keys()), k=4)
        example_3 = random.choice(questions_dict[chosen_categories[0]])
        example_4 = random.choice(questions_dict[chosen_categories[1]])
        example_5 = random.choice(questions_dict[chosen_categories[2]])
        example_6 = random.choice(questions_dict[chosen_categories[3]])

    question_prompt = prompt.format(
        example_1=example_1[0],
        example_2=example_2[0],
        example_3=example_3[0],
        example_4=example_4[0],
        example_5=example_5[0],
        example_6=example_6[0],
        category=category,
        tags=",".join(tags),
    )
    generated_questions = await answer(
        prompt=question_prompt,
        preface=None,
        response_model=QuestionGenerationResponse3,
        model=model,
    )

    return [
        q
        for q in [
            generated_questions.question_1,
            generated_questions.question_2,
            generated_questions.question_3,
        ]
    ]


async def generate_questions(file_path, model, n=3, categories=None):
    questions = load_questions_from_jsonl(file_path)

    print(f"len of questions: {len(questions)}")
    print(f"len of initial questions: {len(initial_questions)}")

    tasks = [
        generate_questions_for_category(initial_questions, questions, model, categories)
        for _ in range(n)
    ]
    results = await asyncio.gather(*tasks)

    generated_questions = [item for sublist in results for item in sublist]

    print(f"len of generated questions: {len(generated_questions)}")

    deduplicated_questions = await deduplicate(generated_questions)
    print(
        f"len deduplicated questions: {len(deduplicated_questions)}, len of generated questions: {len(generated_questions)}"
    )
    deduplicated_questions = [q.model_dump_json() for q in deduplicated_questions]
    await write_jsonl_async_from_str(file_path, deduplicated_questions, append=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        "-f",
        type=str,
        default=get_data_path() / "other" / "intermediate_questions.jsonl",
    )
    parser.add_argument("--model", "-m", type=str, default="gpt-4o")
    parser.add_argument("--n", "-n", type=int, default=3)
    parser.add_argument("--categories", "-c", type=str, nargs="+", default=None)
    args = parser.parse_args()
    asyncio.run(
        generate_questions(
            file_path=args.file_path,
            model=args.model,
            n=args.n,
            categories=args.categories,
        )
    )
