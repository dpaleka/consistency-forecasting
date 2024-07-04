# %%

from forecasters.basic_forecaster import BasicForecaster, ForecastingQuestion
from forecasters.cot_forecaster import COT_Forecaster
from datetime import datetime

f = COT_Forecaster()

x=f.call(
    ForecastingQuestion(
        title="Will this market resolve YES?",
        body="This market will resolve YES precisely if your forecast for this market is less than 0.5, and NO if your forecast is greater than or equal to 0.5.",
        resolution_date=datetime(2022, 1, 1),
        question_type="binary",
    )
)
print(x)

# %%
