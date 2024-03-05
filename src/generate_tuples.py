from static_checks import NegationChecker
import asyncio
from common.utils import write_jsonl_async
from common.llm_utils import parallelized_call

base_questions = [
    "What is the probability that the Democratic party will win the US Presidential election in 2024?",
    "What is the probability that Ebola will be eradicated by 2030?",
    "Will Ebola be erradicated by 2030?",
    "Will the Democratic party win the US Presidential election in 2024?"
    "Will New York City have a skyscraper a mile tall by 2030?",
    "Will the US have a skyscraper a mile tall by 2030?",
    "Will there be a remake of the movie Titanic by 2030?",
]

base_questions2 = [
    "What is the probability of achieving net-zero carbon emissions globally by 2050?",
    "Will artificial intelligence pass the Turing Test by 2025?",
    "What are the chances that self-driving cars will be the standard mode of transportation in major cities by 2035?",
    "Is it likely that a cure for Alzheimer's disease will be discovered by 2040?",
    "Will the global population reach 10 billion by 2050?",
    "What is the probability of a manned Mars landing by 2040?",
    "Can renewable energy sources supply 80% of the world's energy needs by 2040?",
    "Will quantum computing become mainstream in technology by 2030?",
    "Is it probable that the Arctic will have ice-free summers by 2040?",
    "What are the chances of a significant asteroid impact on Earth in the next 100 years?",
    "Will lab-grown meat replace traditional livestock farming as the primary source of protein worldwide by 2040?",
    "What is the likelihood of a global water crisis by 2040?",
    "Will virtual reality classrooms become the predominant form of education by 2030?",
    "Is there a high probability of discovering extraterrestrial life by 2050?",
    "Will the Great Barrier Reef be fully restored by 2050?",
]


negation_checker = NegationChecker()
model = "gpt-3.5-turbo"

async def instantiate_and_write(question: str):
    result = await negation_checker.instantiate_async(question, model)
    await write_jsonl_async(f"negation-{model}.jsonl", [result], append=True)

async def main():
    await parallelized_call(instantiate_and_write, base_questions)

if __name__ == "__main__":
    asyncio.run(main())