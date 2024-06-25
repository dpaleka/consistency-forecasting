import asyncio
import time
from datetime import datetime
from uuid import uuid4
from forecasters.advanced_forecaster import AdvancedForecaster
from common.datatypes import ForecastingQuestion

async def process_question(forecaster, question, idx=0):
    start_time = time.time()
    prediction = await forecaster.call_async(question, idx=idx)
    end_time = time.time()
    execution_time = end_time - start_time
    return question.title, prediction, execution_time

async def measure_forecaster_performance():
    t00 = time.time()
    forecaster = AdvancedForecaster()

    questions = [
        ForecastingQuestion(
            id=uuid4(),
            title=f"Will the price of {crypto} exceed ${price} by the end of 2024?",
            body="The question will be resolved as 'Yes' if the price, as reported by CoinMarketCap, exceeds the specified amount at any point before 11:59 PM UTC on December 31, 2024. Otherwise, it will be resolved as 'No'.",
            resolution_date=datetime(2024, 12, 31, 23, 59, 59),
            question_type="binary",
            data_source="synthetic",
            metadata={
                "background_info": "Cryptocurrencies are known for their price volatility."
            },
        )
        for crypto, price in [
            ("Bitcoin", 50000),
            ("Ethereum", 5000),
            #("Cardano", 5),
            #("Dogecoin", 1),
            #("Ripple", 2),
            #("Solana", 300),
            #("Polkadot", 50),
            #("Chainlink", 30),
            #("Litecoin", 200),
            #("Uniswap", 20)
        ]
    ]

    tasks = [process_question(forecaster, question,idx=idx) for idx, question in enumerate(questions)]
    results = await asyncio.gather(*tasks)

    for title, prediction, execution_time in results:
        print(f"Question: {title}")
        print(f"Prediction: {prediction}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print("-" * 50)

    total_time = sum(result[2] for result in results)
    average_time = total_time / len(results)
    total_time = time.time() - t00
    print(f"Average execution time: {average_time:.2f} seconds, Total time: {total_time:.2f} seconds")

# Run the async function
asyncio.run(measure_forecaster_performance())