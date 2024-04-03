import asyncio
from datetime import datetime
from datasets import load_dataset
from forecasters import Forecaster, COT_Forecaster, BasicForecaster

model = "gpt-3.5-turbo"
# model = "gpt-4-1106-preview"


def brier(prob1, prob2):
    return (prob1 - prob2) ** 2 if prob1 is not None and prob2 is not None else None


async def calculate_brier_score_async(
    forecaster: Forecaster, dataset_name: str, date: datetime
) -> float:
    dataset = load_dataset(dataset_name)
    split = "train" if "train" in dataset else list(dataset.keys())[0]

    resolutions, questions = zip(
        *[
            (example["resolution"], example["question"])
            for example in dataset[split]
            if datetime.fromisoformat(example["date_close"]) > date
        ]
    )
    print(f"len(resolutions): {len(resolutions)}")

    # Use the async elicitation method
    forecasts = await forecaster.elicit_async(
        {(i, q): q for i, q in enumerate(questions)}, model=model, verbose=False
    )

    # Calculate Brier score
    scores = [brier(resolutions[i], forecasts[(i, q)]) for i, q in enumerate(questions)]
    scores = [s for s in scores if s is not None]

    return sum(scores) / len(scores) if scores else None


# 3 of may 2023
date = datetime(2023, 5, 3)

dataset_name = "YuehHanChen/forecasting"


async def main():
    basic_forecaster = BasicForecaster()
    r = await calculate_brier_score_async(
        basic_forecaster, dataset_name=dataset_name, date=date
    )
    print(f"basic_forecaster: {r}")
    reasoning_forecaster = COT_Forecaster()
    r = await calculate_brier_score_async(
        reasoning_forecaster, dataset_name=dataset_name, date=date
    )
    print(f"reasoning_forecaster: {r}")


if __name__ == "__main__":
    asyncio.run(main())
