from forecasters.basic_forecaster import BasicForecaster
from common.llm_utils import Example, answer_sync, query_api_chat_sync
from common.datatypes import ForecastingQuestion, Prob, Prob_cot, PlainText
from datetime import datetime
import inspect

bf = BasicForecaster()
fq = ForecastingQuestion(
    title="Will the sun rise tomorrow?",
    body="I'm asking a simple question.",
    resolution_date=datetime.now(),
    question_type="binary"
)
print(fq.model_dump_json(indent=4,include={'title', 'body'}))
resp = bf.call(fq)
print(resp.prob)

def answer_sync_func_1():
    v=answer_sync(
        prompt="Will the sun rise tomorrow?",
        preface="You are an informed and well-calibrated forecaster. I need you to give me "
                "your best probability estimate for the following sentence or question resolving YES. "
                "Your answer should be a float between 0 and 1, with nothing else in your response.",
        examples=[Example("Will Manhattan have a skyscraper a mile tall by 2030?", "0.03")],
        response_model=Prob_cot,
        n=1,
    )
    return v

#print(inspect.getsource(answer_sync_func_1), "\n", answer_sync_func_1(), "\n")

def query_api_chat_sync_func_1():
    v = query_api_chat_sync(
        messages=[
            {"role" : "system", "content" : "You are a helpful assistant."},
            {"role" : "user", "content" : "Will the sun rise tomorrow?"},
        ],
        model="gpt-4",
        response_model=Prob_cot,
        n = 2,
    )
    return v

#print(inspect.getsource(query_api_chat_sync_func_1), "\n", query_api_chat_sync_func_1(), "\n")

from forecasters.reasoning_forecaster import ReasoningForecaster
rf = ReasoningForecaster()
fq = ForecastingQuestion(
    title="Will the sun rise tomorrow?",
    body="I'm asking a simple question.",
    resolution_date=datetime.now(),
    question_type="binary"
)
resp = rf.call(fq, verbose=True)
print(resp)
