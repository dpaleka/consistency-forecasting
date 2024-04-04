from common.llm_utils import answer_sync, answer, Example
from common.datatypes import *
from .BaseChecker import BaseChecker
from pydantic import BaseModel, field_validator

class NegChecker(BaseChecker):
    """Where f(x) is the forecaster,
    R(x1, x2)       :=  x2 == ¬x1
    S(f(x1), f(x2)) :=  f(x1) + f(x2) = 1
    """
    
    class BaseQuestionFormat(BaseModel):
        P : ForecastingQuestion
    
        @field_validator("P")
        def check_question_type(cls, value):
            if value.question_type != "binary":
                raise ValueError("Question type must be binary")
            return value
    
    
        
    
    preface = (
        "You are a helpful assistant. I will give you a forecasting question with Yes/No "
        "answer, in the format: \n\n"
        "TITLE: [Title of the question]\n"
        "RESOLUTION DATE: [Date of resolution]\n"
        "DETAILS: [Question body with details]\n\n"
        "You should then give me the NEGATION of the question, i.e. the question that "
        "would be answered YES if the original question would be answered NO, and vice "
        "versa. You MUST give me your answer in the same format as specified below. "
        "Demorgan's laws should be followed with and/or negation. Avoid using the word "
        "'won't'."
    )

    examples = [
        Example(
            user=(
                "TITLE: Will the price of Bitcoin be above $100,000 on 1st January 2025?\n"
                "RESOLUTION DATE: 1st January 2025\n"
                "DETAILS: Resolves YES if the spot price of Bitcoin against USD is more than 100,000 "
                "on 1st January 2025. Resolves NO otherwise."
            ),
            assistant=(
                "TITLE: Will the price of Bitcoin be less than or equal to $100,000 on 1st January 2025?\n"
                "RESOLUTION DATE: 1st January 2025\n"
                "DETAILS: Resolves YES if the spot price of Bitcoin against USD is less than or equal to "
                "100,000 on 1st January 2025. Resolves NO otherwise."
            ),
        )
    ]

    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)

    def instantiate_sync(self, base_sentence: ForecastingQuestion, **kwargs) -> ForecastingQuestionTuple:
        prompt = self.stack(base_sentence)
        response = answer_sync(
            prompt=prompt, preface=self.preface, examples=self.examples, **kwargs
        )
        sentences = {"P": base_sentence, "notP": ForecastingQuestion.from_str(response, question_type=base_sentence.question_type)}
        return sentences

    async def instantiate(self, base_sentence: ForecastingQuestion, **kwargs) -> ForecastingQuestionTuple:
        prompt = self.stack(base_sentence)
        response = await answer(
            prompt=prompt, preface=self.preface, examples=self.examples, **kwargs
        )
        sentences = {"P": base_sentence, "notP": ForecastingQuestion.from_str(response, question_type=base_sentence.question_type)}
        return sentences

    def violation(self, answers: ProbsTuple) -> float:
        return abs(answers["P"] + answers["notP"] - 1)
