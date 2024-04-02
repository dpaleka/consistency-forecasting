from forecasters.basic_forecaster import BasicForecaster
from common.datatypes import ForecastingQuestion, Prob
from datetime import datetime

bf = BasicForecaster()
fq = ForecastingQuestion(
    title="Will the sun rise tomorrow?",
    body="I'm asking a simple question.",
    resolution_date=datetime.now(),
    question_type="binary"
)
resp = bf.call(fq)
print(resp.prob)