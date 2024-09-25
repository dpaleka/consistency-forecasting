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
from typing import List, Dict


# If not None, this will be the target resolution date for the questions.
# It also changes the resolution date in some of the initail questions if its before 2030.
# Eg: for a very long term question, it may be better to add new questions.
TARGET_RESOLUTION_DATE = "2028-01-01"  # "2030-01-01 00:00:00"

target_resolution_date = (
    datetime(2030, 1, 1)
    if TARGET_RESOLUTION_DATE is None
    else datetime.strptime(TARGET_RESOLUTION_DATE, "%Y-%m-%d")
)

example_datetime = (
    "2030-01-01"
    if target_resolution_date > datetime(2030, 1, 1)
    else target_resolution_date.strftime("%Y-%m-%d")
)

CREATION_DATETIME = "2024-10-01"  # Default: "2024-10-01 00:00:00"


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
    {
        "title": "Will SpaceX successfully land at least one crewed spacecraft on the surface of Mars before July 4, 2030?",
        "body": "This question will be resolved as 'Yes' if SpaceX successfully lands at least one crewed spacecraft on the surface of Mars before January 1, 2030, as confirmed by official SpaceX announcements and independent verification. The spacecraft must carry at least one human crew member who survives the landing.",
        "category": "Space",
        "tags": ["SpaceX", "Mars"],
        "resolution_datetime": "01-01-2030",
        "created_date": CREATION_DATETIME if CREATION_DATETIME is not None else "01-09-2023"
    },
    {
        "title": "Will any country from the following list: Belgium, France, Italy, Luxembourg, Netherlands, or Germany, officially withdraw its membership from the European Union before January 1, 2027?",
        "body": "This question will be resolved as 'Yes' if any of the listed countries (Belgium, France, Italy, Luxembourg, Netherlands, or Germany) officially notifies the European Council of its intention to withdraw from the EU and completes the withdrawal process before January 1, 2027, as confirmed by official EU documentation.",
        "category": "Geopolitics",
        "tags": ["European Union"],
        "resolution_datetime": "01-01-2027",
        "created_date": CREATION_DATETIME if CREATION_DATETIME is not None else "01-10-2023"
    },
    {
        "title": "Will the global human population fall below 1 billion individuals for any reason before January 1, 2100?",
        "body": "This question will be resolved as 'Yes' if the global human population, as estimated by the United Nations Population Division or another reputable international organization, falls below 1 billion individuals at any point before January 1, 2100, regardless of the cause.",
        "category": "Natural Sciences",
        "tags": ["Extinction"],
        "resolution_datetime": "01-01-2100",
        "created_date": CREATION_DATETIME if CREATION_DATETIME is not None else "01-11-2023"
    },
    {
        "title": "Will the net worth of the world's richest individual, as reported by Forbes or Bloomberg, be equal to or greater than 2% of the United States' GDP, as reported by the World Bank, on December 31, 2033?",
        "body": "This question will be resolved as 'Yes' if on December 31, 2033, the net worth of the world's richest individual, as reported by either Forbes or Bloomberg, is equal to or greater than 2% of the United States' GDP for the most recent year available, as reported by the World Bank.",
        "category": "Economy & Business",
        "tags": ["Wealth"],
        "resolution_datetime": "31-12-2033",
        "created_date": CREATION_DATETIME if CREATION_DATETIME is not None else "01-03-2024"
    },
    {
        "title": "Will an artificial intelligence system pass a formally administered Turing Test, as judged by a panel of at least 5 experts in the field, before January 1, 2026?",
        "body": "This question will be resolved as 'Yes' if before January 1, 2026, an artificial intelligence system passes a formally administered Turing Test, judged by a panel of at least 5 recognized experts in the field of AI. The test must be conducted under controlled conditions and the results must be published in a peer-reviewed scientific journal.",
        "category": "Artificial Intelligence",
        "tags": ["AI", "Turing Test"],
        "resolution_datetime": "01-01-2026",
        "created_date": CREATION_DATETIME if CREATION_DATETIME is not None else "01-04-2024"
    },
    {
        "title": "Will the 5-year average global temperature, as reported by NASA's Goddard Institute for Space Studies, exceed pre-industrial levels by more than 2 degrees Celsius at any point by January 1, 2041?",
        "body": "This question will be resolved as 'Yes' if at any point between January 9, 2023 and January 1, 2041, the 5-year average global temperature, as reported by NASA's Goddard Institute for Space Studies (GISS), exceeds pre-industrial levels by more than 2 degrees Celsius. The pre-industrial baseline will be defined as the average temperature from 1850-1900, as commonly used in climate science.",
        "category": "Environment & Climate",
        "tags": ["Global warming"],
        "resolution_datetime": "01-01-2041",
        "created_date": "01-09-2023"
    },
    {
        "title": "Will a woman be sworn in as President of the United States before {example_datetime}?",
        "body": "This question will be resolved as 'Yes' if a woman takes the oath of office as President of the United States in an official ceremony between January 9, 2023 and {example_datetime}. This includes both elected presidents and those who assume office due to succession.",
        "category": "Elections",
        "tags": ["US Politics", "Presidential Election"],
        "resolution_datetime": "{example_datetime}",
        "created_date": "01-09-2023"
    },
    {
        "title": "Will the World Health Organization officially declare a Public Health Emergency of International Concern (PHEIC) for a novel pathogen by and January 1, 2026?",
        "body": "This question will be resolved as 'Yes' if the World Health Organization (WHO) officially declares a Public Health Emergency of International Concern (PHEIC) for a pathogen that is novel (i.e., not previously identified in humans) between January 5, 2024 and January 1, 2026. The declaration must be made through official WHO channels and documented on their website.",
        "category": "Health & Pandemics",
        "tags": ["WHO", "Pandemic"],
        "resolution_datetime": "01-01-2026",
        "created_date": "01-05-2024"
    },
    {
        "title": "Will India's annual GDP, as reported by the World Bank, exceed China's annual GDP for at least one calendar year by and January 1, 2041?",
        "body": "This question will be resolved as 'Yes' if India's annual Gross Domestic Product (GDP), as reported by the World Bank in its official statistics, exceeds China's annual GDP for at least one full calendar year between January 6, 2024 and January 1, 2041. The comparison will be based on the most recent data available from the World Bank at the time of resolution.",
        "category": "Economy & Business",
        "tags": ["India", "China", "GDP"],
        "resolution_datetime": "01-01-2041",
        "created_date": "01-06-2024"
    },
    {
        "title": "Will an international treaty specifically addressing cyber warfare be ratified by at least 50 UN member states by {example_datetime}?",
        "body": "This question will be resolved as 'Yes' if an international treaty specifically addressing cyber warfare is ratified by at least 50 member states of the United Nations between January 9, 2024 and {example_datetime}. The treaty must be registered with the United Nations Treaty Collection and explicitly address regulations, definitions, or restrictions related to cyber warfare.",
        "category": "Law",
        "tags": ["Cyber warfare"],
        "resolution_datetime": "{example_datetime}",
        "created_date": "01-09-2024"
    },
    {
        "title": "Will a men's national football team from an African country win the FIFA World Cup between January 9, 2024 and January 1, 2041?",
        "body": "This question will be resolved as 'Yes' if a men's national football team representing a country from the African continent wins the FIFA World Cup final between January 9, 2024 and January 1, 2041. The victory must be officially recognized by FIFA and recorded in their official World Cup records.",
        "category": "Sports & Entertainment",
        "tags": ["FIFA", "World Cup"],
        "resolution_datetime": "01-01-2041",
        "created_date": "01-09-2024"
    },
    {
        "title": "Will at least one company offer commercially available quantum computing services to the general public, with a minimum of 1000 qubits, before {example_datetime}?",
        "body": "This question will be resolved as 'Yes' if before {example_datetime}, at least one company offers quantum computing services commercially to the general public, featuring a quantum computer with a minimum of 1000 qubits. The service must be publicly announced, commercially available (not just for research purposes), and the qubit count must be verified by independent experts or reputable tech publications.",
        "category": "Technology",
        "tags": ["Quantum computing"],
        "resolution_datetime": "{example_datetime}",
    },
    {
        "title": "Will NASA or any other space agency publicly announce the discovery of fossilized or living microorganisms on another planet, moon, or asteroid, confirmed by peer-reviewed studies, before January 1, 2036?",
        "body": "This question will be resolved as 'Yes' if, between January 9, 2023 and January 1, 2036, NASA or any other national or international space agency publicly announces the discovery of fossilized or living microorganisms on another planet, moon, or asteroid. The discovery must be confirmed by peer-reviewed studies published in reputable scientific journals and acknowledged by the broader scientific community.",
        "category": "Space",
        "tags": ["NASA", "Extraterrestrial life"],
        "resolution_datetime": "01-01-2036",
        "created_date": "01-09-2023"
    },
    {
        "title": "Will Russia and Ukraine sign and ratify a comprehensive peace treaty, ending all ongoing military conflicts between the two nations, before January 1, 2026?",
        "body": "This question will be resolved as 'Yes' if, between July 1, 2024 and January 1, 2026, Russia and Ukraine sign and ratify a comprehensive peace treaty that ends all ongoing military conflicts between the two nations. The treaty must be officially recognized by both governments, ratified according to their respective legal processes, and result in a complete cessation of hostilities.",
        "category": "Geopolitics",
        "tags": ["Russia", "Ukraine"],
        "resolution_datetime": "01-01-2026",
        "created_date": "01-07-2024"
    },
    {
        "title": "Will a new infectious disease with a confirmed case fatality rate of over 10%, based on WHO data from at least 1000 cases, emerge and spread to at least 3 countries before {example_datetime}?",
        "body": "This question will be resolved as 'Yes' if, between the creation date of this question and {example_datetime}, a new infectious disease emerges with a confirmed case fatality rate of over 10%, based on World Health Organization (WHO) data from at least 1000 confirmed cases, and spreads to at least 3 countries. The disease must be novel (not previously identified in humans) and its spread must be officially reported by the WHO or national health authorities.",
        "category": "Health & Pandemics",
        "tags": ["Infectious Disease"],
        "resolution_datetime": "{example_datetime}"
    },
    {
        "title": "Will the total number of operational satellites in Earth orbit, as tracked by the United Nations Office for Outer Space Affairs, exceed 50,000 before {example_datetime}?",
        "body": "This question will be resolved as 'Yes' if, between the creation date of this question and {example_datetime}, the total number of operational satellites in Earth orbit, as tracked and reported by the United Nations Office for Outer Space Affairs (UNOOSA), exceeds 50,000 at any point. The count must be based on official UNOOSA data and include all types of operational satellites in Earth orbit.",
        "category": "Technology",
        "tags": ["Satellites"],
        "resolution_datetime": "{example_datetime}"
    },
    {
        "title": "Will any G7 country pass legislation allowing genetic editing of human embryos for non-medical trait enhancement before {example_datetime}?",
        "body": "This question will be resolved as 'Yes' if, between January 9, 2024 and {example_datetime}, any G7 country (Canada, France, Germany, Italy, Japan, the United Kingdom, or the United States) passes legislation that explicitly allows genetic editing of human embryos for non-medical trait enhancement. The legislation must be officially enacted and published in the country's official legal records.",
        "category": "Law",
        "tags": ["Genetic editing"],
        "resolution_datetime": "{example_datetime}",
        "created_date": "01-09-2024"
    },
    {
        "title": "Will the global mean sea level, as measured by satellite altimetry and reported by NASA or NOAA, rise by more than 10 centimeters above the 2020 average level before {example_datetime}?",
        "body": "This question will be resolved as 'Yes' if the global mean sea level, as measured by satellite altimetry and reported by either NASA or NOAA, rises by more than 10 centimeters above the 2020 average level at any point before {example_datetime}. The measurement must be based on the official data releases from either NASA or NOAA, using their standard methodologies for calculating global mean sea level.",
        "category": "Environment & Climate",
        "tags": ["Sea level rise"],
        "resolution_datetime": "{example_datetime}",
    },
        {
        "title": "Will a human set foot on Mars in 2035?",
        "body": "This question will be resolved as 'Yes' if a human being physically steps onto the surface of Mars at any point during the calendar year 2035 (from January 1, 2035, to December 31, 2035, inclusive). The event must be officially confirmed by a reputable space agency or verified by multiple independent sources.",
        "category": "Space Exploration",
        "tags": ["Mars", "Space Travel"],
        "resolution_datetime": "01-01-2036",
        "created_date": "01-10-2024"
    },
    {
        "title": "Will the global average temperature increase exceed 1.5°C above pre-industrial levels in 2030?",
        "body": "This question will be resolved as 'Yes' if the global average temperature increase, as reported by the World Meteorological Organization (WMO) or a similarly authoritative body, exceeds 1.5°C above pre-industrial levels for the calendar year 2030. The measurement should be based on the annual global temperature anomaly for 2030 compared to the pre-industrial baseline.",
        "category": "Climate & Environment",
        "tags": ["Global Warming", "Climate Change"],
        "resolution_datetime": "07-01-2031",
        "created_date": "01-10-2024"
    },
    {
        "title": "Will a cryptocurrency become legal tender in a G7 country in 2028?",
        "body": "This question will be resolved as 'Yes' if any cryptocurrency (e.g., Bitcoin, Ethereum) is officially adopted as legal tender by any G7 country (Canada, France, Germany, Italy, Japan, the United Kingdom, or the United States) during the calendar year 2028. The adoption must be enacted into law and come into effect within 2028.",
        "category": "Economy & Finance",
        "tags": ["Cryptocurrency", "G7"],
        "resolution_datetime": "01-01-2029",
        "created_date": "01-10-2024"
    },
    {
        "title": "Will lab-grown meat products account for more than 10% of the global meat market share in 2040?",
        "body": "This question will be resolved as 'Yes' if lab-grown meat products (also known as cultured meat or in vitro meat) account for more than 10% of the global meat market share by value or volume in the calendar year 2040. The market share data should be sourced from a reputable market research firm or industry association.",
        "category": "Food & Agriculture",
        "tags": ["Lab-grown Meat", "Food Technology"],
        "resolution_datetime": "07-01-2041",
        "created_datetime": "01-10-2024"
    },
    {
        "title": "Will a non-English language film win the Academy Award for Best Picture in 2026?",
        "body": "This question will be resolved as 'Yes' if a film primarily in a language other than English wins the Academy Award for Best Picture at the Academy Awards ceremony held in 2026 (which will honor films released in 2025). The win must be officially announced by the Academy of Motion Picture Arts and Sciences.",
        "category": "Entertainment",
        "tags": ["Academy Awards", "World Cinema"],
        "resolution_datetime": "04-01-2026",
        "created_datetime": "01-10-2024"
    }
]

resolution_message = (
    ""
    if TARGET_RESOLUTION_DATE is None
    else f"""The question should resolve on the date {target_resolution_date}. Ensure the resolution criteria are clearly defined for this specific date.
You can ask wether an event will happen before this date, or in the year of this date, but the resolution should be on this date.""" 
)

creation_message = (
    ""
    if CREATION_DATETIME is None
    else f"""The question should be created with the creation date {CREATION_DATETIME} in mind.
Avoid questions that could already be resolved by the creation date. 
The question body should specify that for a positive resolution, the event must occur between the creation date and the resolution date."""
)

# Prompt for generating forecasting questions
general_task_description = (
    "\nYou are tasked with generating forecasting questions for a platform similar to Metaculus or PredictIt.\n" +
    "Based on the provided category, tags, and creation_date, create questions that can be answered with a probability between 0 and 1.\n" +
    "Be precise and avoid vague language or qualifiers. Ensure each question has a clear resolution criteria and timeframe.\n" +
    creation_message +
    resolution_message
)

content_instructions = """Tip 1: For each tag, generate a relevant question if the tag is pertinent to the category. If the tag is not relevant, generate a general question about the category.
Tip 2: Avoid using qualifiers like "significant", "substantial", or "major". Instead, use specific, measurable criteria.
Tip 3: Include a clear anchor for quality or quantity in each question. For example, instead of asking if something will "improve", ask if it will "increase by at least X%".
Tip 4: Specify a clear timeframe for each question. Use precise dates or periods rather than vague terms like "soon" or "in the near future".
Tip 5: Ensure that the resolution criteria for each question are unambiguous and can be objectively determined.
Tip 6: If appropriate, include specific numerical thresholds or ranges in your questions to make them more precise.
Tip 7: Consider including a reliable source for resolving the question, such as a specific government agency or reputable organization.
Tip 8: Form questions with the creation_date in mind. The event in question should occur within the interval between the creation_date and the resolution_date. This means the event could have already happened by the creation_date or be set to occur close to it.
Tip 9: Ensure that the resolution datetime specified in the body of the question matches exactly with the resolution datetime field. Consistency between these two is crucial."""

def generate_examples_string(examples):
    examples_string = ""

    for example in examples:
        if example is None:
            continue
        
        example_string = "<example>\n"
        example_string += f"    <title>{example['title']}</title>\n"
        
        if example.get('body'):
            example_string += f"    <body>{example['body']}</body>\n"
        
        example_string += f"    <category>{example['category']}</category>\n"
        
        if example.get('tags'):
            example_string += f"    <tags>{example['tags']}</tags>\n"
        
        example_string += f"    <resolution_datetime>{example['resolution_datetime']}</resolution_datetime>\n"
        
        if example.get('creation_datetime'):
            example_string += f"    <creation_datetime>{example['creation_datetime']}</creation_datetime>\n"
        
        example_string += "</example>\n"
        examples_string += example_string

    return examples_string

final_instruction = (
    "\nGenerate forecasting questions based on the following category, tags, and creation date:\n"
    "Category: {category}\n"
    "Tags: {tags}\n" + 
    creation_message +
    resolution_message
)

def generate_prompt(examples: List[Dict[str, str]], tags: List[str], category: str):
    examples_string = generate_examples_string(examples)
    return (
        "\n" +
        general_task_description +
        content_instructions +
        examples_string +
        final_instruction.format(category=category, tags=", ".join(tags))
    )

def load_questions_from_jsonl(file_path):
    if not os.path.exists(file_path):
        return {}
    
    questions_dict = {}
    
    with open(file_path, "r") as file:
        for line in file:
            json_data = json.loads(line)
            category = json_data["category"]
            
            question_dict = {
                "title": json_data["title"],
                "body": json_data.get("body", ""),  
                "tags": json_data["tags"],
                "category": category,
                "resolution_datetime": json_data.get("resolution_datetime", "")  
            }
            
            if category not in questions_dict:
                questions_dict[category] = [question_dict]
            else:
                questions_dict[category].append(question_dict)
    
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

    example_1, example_2, example_7, example_8 = random.sample(initial_questions, 4)

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
            example_7,
            example_8
        ) = random.sample(initial_questions, 8)
    else:
        chosen_categories = random.choices(list(questions_dict.keys()), k=4)
        example_3 = random.choice(questions_dict[chosen_categories[0]])
        example_4 = random.choice(questions_dict[chosen_categories[1]])
        example_5 = random.choice(questions_dict[chosen_categories[2]])
        example_6 = random.choice(questions_dict[chosen_categories[3]])

    question_prompt = generate_prompt(
        [example_1, example_2, example_3, example_4, example_5, example_6, example_7, example_8],
        tags,
        category
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
