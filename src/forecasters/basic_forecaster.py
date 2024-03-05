from .forecaster import Forecaster, Prob
import re
from common.llm_utils import query_api_chat_sync, query_api_chat


class BasicForecaster(Forecaster):

    def __init__(self):
        self.system_prompt = "You are an informed and well-calibrated forecaster. I need you to give me your best probability estimate for the following sentence or question resolving YES."+\
            " Your answer should be a float between 0 and 1, with nothing else in your response."
        self.example = [{"role": "user", "content": "Will manhattan have a skyscrapper a mile tall by 2030."}, {"role": "system", "content": "0.04"}]
        self.initial_messages = [{"role": "system", "content": self.system_prompt}] + self.example

    
    def call(self, sentence: str, model: str = "gpt-4-1106-preview") -> Prob:
        messages = self.initial_messages + [{"role": "user", "content": sentence}]
        response = query_api_chat_sync(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        return self.extract_prob(response)

    async def call_async(self, sentence: str, model: str = "gpt-4-1106-preview") -> Prob:
        messages = self.initial_messages + [{"role": "user", "content": sentence}]
        response = await query_api_chat(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        return self.extract_prob(response)

    def extract_prob(self, s: str) -> float:
        pattern = r"-?\d*\.?\d+"
        match = re.search(pattern, s)
        if match:
            try:
                return Prob(float(match.group()))
            except Exception as e:
                #TODO: log error
                return None
        else:
            return None




