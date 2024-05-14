from typing import Optional
import asyncio
from common.path_utils import get_data_path
from common.utils import write_jsonl_async, write_jsonl_async_from_str
from common.datatypes import SyntheticTagQuestion
from question_generators.utils import deduplicate
import json
from common.llm_utils import answer
import random
from pydantic import BaseModel


file_path = get_data_path()/'other'/'high-quality-questions-all-domains.jsonl'
model = "gpt-4-0125-preview"

categories = [
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
    "Technology"
]

topics = [
    "United States", "Medicine", "Virology", "COVID-19", "China", "Russia", "Physics",
    "Epidemiology", "United Kingdom", "Donald Trump", "Ukraine", "Machine Learning", "European Union",
    "Joe Biden", "Virginia", "Immunology", "OpenAI", "Centers for Disease Control and Prevention",
    "Environmental Science", "Biology", "Infectious Disease", "Israel", "Genetics", "Democratic Party (US)",
    "Republican Party (US)", "Astronomy", "Biotechnology", "Food and Drug Administration", "Elon Musk",
    "Russo-Ukrainian War", "India", "COVID-19 Vaccine", "Chemistry", "Scott Alexander", "World Health Organization",
    "Philosophy", "SpaceX Reusable Launch System", "NASA", "Information Technology", "Bureau of Labor Statistics",
    "US dollar", "Energy", "Taiwan", "Engineering", "SpaceX", "Microbiology", "France", "Germany",
    "Climate change denial", "US Senate", "Ethics", "Iran", "2020 US presidential election", "DeepMind",
    "Robotics", "Supreme Court of the US", "California", "2024 US Elections", "US House of Representatives",
    "Atmospheric science", "NATO", "Pharmacology", "Google", "North Korea", "Amazon", "US Congress",
    "Microsoft", "Aerospace engineering", "Climatology", "United Nations", "Virginia Department of Health", "Japan", "Tesla"
    "Spain", "Switzerland", "Italy", "Brazil", "Australia", "Canada", "Mexico", "South Korea", "Sweden", "Norway",
    "Particle physics", "String theory", "Black holes", "Dark matter", "Quantum computing", "Basketball", "Soccer",
    "Cricket", "Tennis", "Baseball", "Golf", "Formula 1", "Olympics", "FIFA", "NBA", "NFL", "MLB", "NHL",
]

initial_questions = [
    ("Will SpaceX land people on Mars before 2030?", "Space", "SpaceX,Mars"),
    ("Will any of Belgium, France, Italy, Luxembourg, Netherlands, and/or Germany leave the EU before 2027?", "Geopolitics", "European Union"),
    ("Will humans go extinct before 2100?", "Natural Sciences", "Extinction"),
    ("Will the richest person in the world in 2033 have a net worth equivalent to or greater than 2% of the United States' GDP at the time?", "Economy & Business", "Wealth"),
    ("Will artificial intelligence pass the Turing Test by 2025?", "Artificial Intelligence", "AI,Turing Test"),
    ("Will a major cryptocurrency outperform the US dollar in daily transaction volume by 2030?", "Cryptocurrencies", "Bitcoin,US dollar"),
    ("Will global average temperatures rise by more than 2 degrees Celsius above pre-industrial levels by 2040?", "Environment & Climate", "Global warming"),
    ("Will the United States have a female president before 2028?", "Elections", "US Politics,Presidential Election"),
    ("Will the World Health Organization declare a new pandemic before 2025?", "Health & Pandemics", "WHO,Pandemic"),
    ("Will India's GDP surpass China's at any point before 2040?", "Economy & Business", "India,China,GDP"),
    ("Will there be a legally binding international treaty on cyber warfare signed by over 50 countries by 2030?", "Law", "Cyber warfare"),
    ("Will a team from Africa win the FIFA World Cup before 2040?", "Sports & Entertainment", "FIFA,World Cup"),
    ("Will quantum computing be commercially available to the public before 2030?", "Technology", "Quantum computing"),
    ("Will NASA discover definitive evidence of life on another planet by 2035?", "Space", "NASA,Extraterrestrial life"),
    ("Will Russia and Ukraine sign a permanent peace treaty before 2025?", "Geopolitics", "Russia,Ukraine"),
    ("Will a new infectious disease with a fatality rate over 10% emerge before 2030?", "Health & Pandemics", "Infectious Disease"),
    ("Will the total number of operational satellites exceed 10,000 by 2025?", "Technology", "Satellites"),
    ("Will genetic editing in humans for enhancing traits (not medical) be legalized in any G7 country by 2030?", "Law", "Genetic editing"),
    ("Will the global sea level rise by more than 10 centimeters above 2020 levels before 2030?", "Environment & Climate", "Sea level rise")
]

prompt = """
I want you to help me generate some forecasting questions for a forecasting market site like metaculus.
I am going to provide you with a category and some tags. Generate some questions that are answerable with a probability between 0 and 1.
Generate one question for each tag, if the tag is relevant to the category. If it is not relevant, generate a question about the category in general.

Examples:

{example_1}

{example_2}

{example_3}

{example_4}

{example_5}

{example_6}
----

Category: {category}
Tags: {tags}
"""


class QuestionGenerationResponse(BaseModel):
    question_1: SyntheticTagQuestion
    question_2: SyntheticTagQuestion
    question_3: SyntheticTagQuestion

def load_questions_from_jsonl(file_path):
    questions_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line) 
            category = json_data['category']  
            question_tuple = (json_data['title'], json_data['category'], json_data['tags'])  
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


async def generate_questions_for_category(initial_questions, questions_dict):
    category = random.choice(list(questions_dict.keys()))
    tags = random.sample(topics, 3)

    example_1, example_2 = random.sample(initial_questions, 2)

    if category in questions_dict and len(questions_dict[category]) >= 2:
        example_5, example_6 = random.sample(questions_dict[category], 2)
        other_categories = [cat for cat in questions_dict if cat != category]
        cat1, cat2 = random.sample(other_categories, 2)
        example_3 = random.choice(questions_dict[cat1])
        example_4 = random.choice(questions_dict[cat2])
    else:
        chosen_categories = random.sample(list(questions_dict.keys()), 4)
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
        tags=",".join(tags)
    )
    generated_questions = await answer(model, question_prompt, response_model=QuestionGenerationResponse)

    return [q for q in [generated_questions.question_1, generated_questions.question_2, generated_questions.question_3]]



async def generate_questions(n=3):
    questions = load_questions_from_jsonl(file_path)
    tasks = [generate_questions_for_category(initial_questions, questions) for _ in range(n)]
    results = await asyncio.gather(*tasks)
    
    generated_questions = [item for sublist in results for item in sublist]
    
    deduplicated_questions = await deduplicate(generated_questions)
    print(f"len deduplicated questions: {len(deduplicated_questions)}, len of generated questions: {len(generated_questions)}")
    deduplicated_questions = [q.model_dump_json() for q in deduplicated_questions]
    await write_jsonl_async_from_str(file_path, deduplicated_questions, append=True)


if __name__ == "__main__":
    asyncio.run(generate_questions(180))



