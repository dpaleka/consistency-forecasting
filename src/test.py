from forecasters.basic_forecaster import BasicForecaster
from common.llm_utils import Example, answer_sync, query_api_chat_sync
from common.datatypes import ForecastingQuestion, Prob, Prob_cot
from datetime import datetime

# bf = BasicForecaster()
# fq = ForecastingQuestion(
#     title="Will the sun rise tomorrow?",
#     body="I'm asking a simple question.",
#     resolution_date=datetime.now(),
#     question_type="binary"
# )
# resp = bf.call(fq)
# print(resp.prob)

# v=answer_sync(
#     prompt="Will the sun rise tomorrow?",
#     preface="You are an informed and well-calibrated forecaster. I need you to give me "
#             "your best probability estimate for the following sentence or question resolving YES. "
#             "Your answer should be a float between 0 and 1, with nothing else in your response.",
#     examples=[Example("Will Manhattan have a skyscraper a mile tall by 2030?", "0.03")],
#     response_model=Prob,
#     n=3,
# )
# print(v)

# v = query_api_chat_sync(
#     messages=[
#         {"role" : "system", "content" : "You are a helpful assistant."},
#         {"role" : "user", "content" : "Will the sun rise tomorrow?"},
#     ],
#     model="gpt-4",
#     response_model=Prob,
#     n = 3,
# )

# print(v)