"""Where f(x) is the model,
R(x1, *xi)
    := x1 == (xi for some i); xi => ¬xj for i,j != 1
S(f(x1), *f(xi))
    := f(x1) = sum f(xi)"""

import re
from common.llm_utils import answer_sync, answer
from forecasters import SentencesTemplate, ProbsTemplate
from .BaseChecker import BaseChecker

class DisjointSpanningChecker(BaseChecker):
    preface = (
        "You are a helpful assistant. I will give you a question, q -- "
        "I want you to give me a comprehensive and disjoint list of ways "
        "in which q could be true. I.e. all of the ways must be mutually exclusive "
        "(no two of them can be true at once), and together they must cover all the ways "
        "in which q could be true. Make sure your response is in the form of a list of questions, "
        " in exactly the following format: "
        "q1: bla bla bla \n\n q2: bla bla bla \n\n q3: bla bla bla \n\n ... \n\n qn: bla bla bla."
    )

    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)

    def instantiate_sync(self, base_sentence: str, **kwargs) -> SentencesTemplate:
        response = answer_sync(
            prompt = base_sentence, 
            preface = self.preface,
            **kwargs)
        
        response = re.findall(r'q\d:.*?\n\n', response, re.DOTALL) # or response = response.split("\n\n")
        response = [x.split(":") for x in response]
        response = {x[0].strip(): x[1].strip() for x in response}
        # or, this returns a list:
        # response = re.findall(r'(?<=q\d:).*?(?=\n\n)', response, re.DOTALL)
        sentences = {"Q": base_sentence}
        sentences.update(response)
        return sentences
    
    async def instantiate(self, base_sentence: str, **kwargs) -> SentencesTemplate:
        response = await answer(
            prompt = base_sentence, 
            preface = self.preface,
            **kwargs)
        
        response = re.findall(r'q\d:.*?\n\n', response, re.DOTALL) # or response = response.split("\n\n")
        response = [x.split(":") for x in response]
        response = {x[0].strip(): x[1].strip() for x in response}
        # or, this returns a list:
        # response = re.findall(r'(?<=q\d:).*?(?=\n\n)', response, re.DOTALL)
        sentences = {"Q": base_sentence}
        sentences.update(response)
        return sentences

    def violation(self, answers: ProbsTemplate) -> float:
        return abs(sum([v for k, v in answers.items() if k != "Q"]) - answers["Q"])