from datetime import datetime
from dateutil import parser
import re
from uuid import uuid4, UUID
from pydantic import BaseModel, ConfigDict, field_validator

# class Prob(float):
#     def __new__(cls, value):
#         if not (0.0 <= value <= 1.0):
#             raise ValueError("Probability must be between 0 and 1.")
#         return super(Prob, cls).__new__(cls, value)

class Prob(BaseModel):
    prob : float
    
    @field_validator('prob')
    @classmethod
    def validate_prob(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return v

class ForecastingQuestion(BaseModel):
    title: str
    body: str
    resolution_date: datetime
    question_type: str
    data_source: str = None
    url: str = None
    metadata: dict = None
    resolution: str = None

    def __str__(self):
        return (
            f"TITLE: {self.title}\n"
            f"RESOLUTION DATE: {self.resolution_date}\n"
            f"DETAILS:\n{self.body}"
        )
    
    @field_validator('question_type')
    def validate_question_type(cls, v):
        if v not in ['binary', 'conditional_binary']:
            raise ValueError("Question type must be either 'binary' or 'conditional_binary'")
        return v

    @property
    def expected_answer_type(self) -> BaseModel:
        exp_answers = {
            'binary': Prob,
            'conditional_binary': Prob
        }
        return exp_answers[self.question_type]


ForecastingQuestionTuple = dict[str, ForecastingQuestion]
ProbsTuple = dict[str, Prob]